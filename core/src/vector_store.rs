//! Exact vector storage alongside FTS5. Conan uses packed bits (the caller's
//! packbits order is preserved); WeMM uses cosine distance, not similarity.
//! Cache hits are exposed as `Some(bytes)` so callers can count hits/misses.
use anyhow::{bail, Result};
use rusqlite::{params, OptionalExtension};
use std::sync::OnceLock;

use crate::store::{now, Store};

pub const CONAN_BITS: usize = 1792;
pub const CONAN_BYTES: usize = CONAN_BITS / 8;
pub const WEMM_DIM: usize = 1024;
pub const CACHE_TTL_MILLIS: i64 = 7 * 24 * 60 * 60 * 1000;

pub(crate) fn register_extension() -> Result<()> {
    static REGISTERED: OnceLock<i32> = OnceLock::new();
    let code = *REGISTERED.get_or_init(|| {
        // sqlite-vec documents this auto-extension cast. SQLite calls the static
        // entry point with its extension API on every subsequently opened connection.
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
                unsafe extern "C" fn(),
                unsafe extern "C" fn(
                    *mut rusqlite::ffi::sqlite3,
                    *mut *mut std::os::raw::c_char,
                    *const rusqlite::ffi::sqlite3_api_routines,
                ) -> std::os::raw::c_int,
            >(sqlite_vec::sqlite3_vec_init)))
        }
    });
    if code != rusqlite::ffi::SQLITE_OK {
        bail!("sqlite-vec auto-extension registration failed: {code}");
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub enum Vector<'a> {
    Conan(&'a [u8]),
    Wemm(&'a [f32]),
}

impl Vector<'_> {
    fn encode(self, leg: &str) -> Result<Vec<u8>> {
        match (leg, self) {
            ("conan", Self::Conan(bits)) if bits.len() == CONAN_BYTES => Ok(bits.to_vec()),
            ("wemm", Self::Wemm(values)) if values.len() == WEMM_DIM => {
                if values.iter().any(|v| !v.is_finite()) || values.iter().all(|v| *v == 0.0) {
                    bail!("WeMM cosine vectors must be finite and nonzero");
                }
                Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
            }
            ("conan", Self::Conan(bits)) => bail!("Conan requires {CONAN_BYTES} bytes ({CONAN_BITS} bits), got {}", bits.len()),
            ("wemm", Self::Wemm(values)) => bail!("WeMM requires {WEMM_DIM} floats, got {}", values.len()),
            ("conan" | "wemm", _) => bail!("vector type does not match leg {leg}"),
            _ => bail!("unknown vector leg: {leg}"),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct VectorMatch {
    pub chunk_id: i64,
    pub distance: f64,
}

/// Replace one vector atomically. Missing chunks are rejected to avoid orphans.
pub fn write_vector(store: &Store, leg: &str, chunk_id: i64, vector: Vector<'_>) -> Result<()> {
    let bytes = vector.encode(leg)?;
    let mut db = store.connection();
    let tx = db.transaction()?;
    let exists: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM chunks WHERE id=?)", [chunk_id], |r| r.get(0))?;
    if !exists {
        bail!("cannot write vector for missing chunk {chunk_id}");
    }
    // vec0 does not support SQLite's ordinary ON CONFLICT upsert syntax.
    let (delete, insert) = match leg {
        "conan" => ("DELETE FROM vec_conan WHERE rowid=?", "INSERT INTO vec_conan(rowid,embedding) VALUES (?,vec_bit(?))"),
        "wemm" => ("DELETE FROM vec_wemm WHERE rowid=?", "INSERT INTO vec_wemm(rowid,embedding) VALUES (?,?)"),
        _ => unreachable!("encode validated the leg"),
    };
    tx.execute(delete, [chunk_id])?;
    tx.execute(insert, params![chunk_id, bytes])?;
    tx.commit()?;
    Ok(())
}

/// Exact top-N, ascending distance with chunk id as a deterministic tie breaker.
/// Scalar distance functions scan vec0 rows: unlike a bounded KNN candidate set,
/// this also resolves ties at the N boundary deterministically. WeMM is cosine.
pub fn query_top_n(store: &Store, leg: &str, query_vector: Vector<'_>, n: usize) -> Result<Vec<VectorMatch>> {
    let bytes = query_vector.encode(leg)?;
    let limit = i64::try_from(n)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let sql = match leg {
        "conan" => "SELECT rowid,vec_distance_hamming(embedding,vec_bit(?)) AS distance FROM vec_conan ORDER BY distance,rowid LIMIT ?",
        "wemm" => "SELECT rowid,vec_distance_cosine(embedding,?) AS distance FROM vec_wemm ORDER BY distance,rowid LIMIT ?",
        _ => unreachable!("encode validated the leg"),
    };
    let db = store.connection();
    let mut stmt = db.prepare(sql)?;
    let rows = stmt.query_map(params![bytes, limit], |r| Ok(VectorMatch { chunk_id: r.get(0)?, distance: r.get(1)? }))?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

/// Also called automatically by the chunks-delete trigger on removal/reindex.
pub fn delete_vectors_for_chunk(store: &Store, chunk_id: i64) -> Result<()> {
    let mut db = store.connection();
    let tx = db.transaction()?;
    tx.execute("DELETE FROM vec_conan WHERE rowid=?", [chunk_id])?;
    tx.execute("DELETE FROM vec_wemm WHERE rowid=?", [chunk_id])?;
    tx.commit()?;
    Ok(())
}

/// `Some` is a cache hit; `None` is absent or expired. Storage errors propagate.
pub fn get_cached_embedding(store: &Store, chunk_hash: &str, model: &str) -> Result<Option<Vec<u8>>> {
    Ok(store.connection().query_row(
        "SELECT vector FROM embedding_cache WHERE chunk_hash=? AND model=? AND created_at>?",
        params![chunk_hash, model, now() - CACHE_TTL_MILLIS], |r| r.get(0),
    ).optional()?)
}

/// Store raw packed-bit bytes or f32 little-endian bytes, keyed by model identity.
/// Callers should include the model revision in `model` when changing encoders.
pub fn put_cached_embedding(store: &Store, chunk_hash: &str, model: &str, vector_bytes: &[u8]) -> Result<()> {
    store.connection().execute(
        "INSERT INTO embedding_cache(chunk_hash,model,vector,created_at) VALUES (?,?,?,?) ON CONFLICT(chunk_hash,model) DO UPDATE SET vector=excluded.vector,created_at=excluded.created_at",
        params![chunk_hash, model, vector_bytes, now()],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(count: i64) -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("index.sqlite")).unwrap();
        {
            let db = store.connection();
            db.execute("INSERT INTO documents(id,root,path,name,hash,mtime) VALUES(1,'test','test.md','Test','hash',0)", []).unwrap();
            for id in 1..=count {
                db.execute("INSERT INTO chunks(id,document_id,heading,ranges,start_line,end_line) VALUES(?,1,'Test','[]',1,1)", [id]).unwrap();
                db.execute("INSERT INTO chunk_fts(rowid,title,heading,body) VALUES(?,'Test','','vector coexistence')", [id]).unwrap();
            }
        }
        (dir, store)
    }

    fn random(seed: &mut u64) -> u32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 32) as u32
    }

    #[test]
    fn conan_matches_exact_hamming_and_fts_survives() {
        // fusion_fixture.json contains ranked ids, not embeddings. Reproducible,
        // full-dimensional synthetic packed vectors avoid heavy npz loading and
        // dependence on external fixture paths. Rust supplies the exact reference.
        let (_dir, store) = fixture(80);
        let mut seed = 42;
        let mut vectors: Vec<Vec<u8>> = (0..80).map(|_| (0..CONAN_BYTES).map(|_| random(&mut seed) as u8).collect()).collect();
        vectors[79] = vectors[7].clone(); // Tie at the top-N boundary.
        let query = &vectors[7];
        for (i, v) in vectors.iter().enumerate() {
            write_vector(&store, "conan", i as i64 + 1, Vector::Conan(v)).unwrap();
        }
        let mut expected: Vec<_> = vectors.iter().enumerate().map(|(i, v)| {
            let distance: u32 = v.iter().zip(query).map(|(a, b)| (a ^ b).count_ones()).sum();
            (i as i64 + 1, distance as f64)
        }).collect();
        expected.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        for n in [0, 1, 2, 10, 50, 100] {
            let actual = query_top_n(&store, "conan", Vector::Conan(query), n).unwrap();
            assert_eq!(actual.iter().map(|m| (m.chunk_id, m.distance)).collect::<Vec<_>>(), expected[..n.min(80)]);
        }
        let count: i64 = store.connection().query_row("SELECT count(*) FROM chunk_fts WHERE chunk_fts MATCH 'coexistence'", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 80);
    }

    #[test]
    fn wemm_matches_exact_cosine() {
        let (_dir, store) = fixture(80);
        let mut seed = 123;
        let mut vectors: Vec<Vec<f32>> = (0..80).map(|_| (0..WEMM_DIM).map(|_| (random(&mut seed) as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32).collect()).collect();
        vectors[79] = vectors[7].clone();
        let query = &vectors[7];
        for (i, v) in vectors.iter().enumerate() {
            write_vector(&store, "wemm", i as i64 + 1, Vector::Wemm(v)).unwrap();
        }
        // Independent f64 cosine reference on full-dimensional floats.
        let mut expected: Vec<_> = vectors.iter().enumerate().map(|(i, v)| {
            let dot: f64 = v.iter().zip(query).map(|(&a, &b)| a as f64 * b as f64).sum();
            let a: f64 = v.iter().map(|&x| (x as f64).powi(2)).sum();
            let b: f64 = query.iter().map(|&x| (x as f64).powi(2)).sum();
            (i as i64 + 1, 1.0 - dot / (a * b).sqrt())
        }).collect();
        expected.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        for n in [0, 1, 2, 10, 50, 100] {
            let actual = query_top_n(&store, "wemm", Vector::Wemm(query), n).unwrap();
            assert_eq!(actual.iter().map(|m| m.chunk_id).collect::<Vec<_>>(), expected.iter().take(n).map(|m| m.0).collect::<Vec<_>>());
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!((actual.distance - expected.1).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn validation_upsert_and_explicit_delete() {
        let (_dir, store) = fixture(1);
        let bits = [0; CONAN_BYTES];
        let floats = [1.0; WEMM_DIM];
        for (leg, vector) in [
            ("conan", Vector::Conan(&bits[..CONAN_BYTES - 1])),
            ("conan", Vector::Conan(&[0; CONAN_BYTES + 1])),
            ("wemm", Vector::Wemm(&floats[..WEMM_DIM - 1])),
            ("wemm", Vector::Wemm(&[1.0; WEMM_DIM + 1])),
            ("conan", Vector::Wemm(&floats)),
            ("wemm", Vector::Conan(&bits)),
            ("sag", Vector::Conan(&bits)),
            ("wemm", Vector::Wemm(&[0.0; WEMM_DIM])),
            ("wemm", Vector::Wemm(&[f32::NAN; WEMM_DIM])),
            ("wemm", Vector::Wemm(&[f32::INFINITY; WEMM_DIM])),
        ] {
            assert!(write_vector(&store, leg, 1, vector).is_err());
            assert!(query_top_n(&store, leg, vector, 10).is_err());
        }
        assert!(write_vector(&store, "conan", 999, Vector::Conan(&bits)).is_err());
        write_vector(&store, "conan", 1, Vector::Conan(&bits)).unwrap();
        write_vector(&store, "conan", 1, Vector::Conan(&[255; CONAN_BYTES])).unwrap();
        assert_eq!(query_top_n(&store, "conan", Vector::Conan(&bits), 10).unwrap(), vec![VectorMatch { chunk_id: 1, distance: CONAN_BITS as f64 }]);
        write_vector(&store, "wemm", 1, Vector::Wemm(&floats)).unwrap();
        write_vector(&store, "wemm", 1, Vector::Wemm(&[-1.0; WEMM_DIM])).unwrap();
        let hits = query_top_n(&store, "wemm", Vector::Wemm(&floats), 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert!((hits[0].distance - 2.0).abs() < 1e-5);
        delete_vectors_for_chunk(&store, 1).unwrap();
        delete_vectors_for_chunk(&store, 1).unwrap();
        assert!(query_top_n(&store, "conan", Vector::Conan(&bits), 10).unwrap().is_empty());
        assert!(query_top_n(&store, "wemm", Vector::Wemm(&floats), 10).unwrap().is_empty());
    }

    #[test]
    fn removal_reindex_and_cascades_delete_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let folder = dir.path().join("library");
        std::fs::create_dir(&folder).unwrap();
        let path = folder.join("test.md");
        std::fs::write(&path, "# Test\nVector coexistence content.").unwrap();
        let store = Store::open(&dir.path().join("index.sqlite")).unwrap();
        let root = store.add_root(&folder).unwrap();
        let bits = [0; CONAN_BYTES];
        let floats = [1.0; WEMM_DIM];
        for action in 0..4 {
            store.index_file(&root, &path, 512).unwrap();
            let id: i64 = store.connection().query_row("SELECT id FROM chunks LIMIT 1", [], |r| r.get(0)).unwrap();
            write_vector(&store, "conan", id, Vector::Conan(&bits)).unwrap();
            write_vector(&store, "wemm", id, Vector::Wemm(&floats)).unwrap();
            match action {
                0 => {
                    std::fs::write(&path, "# Changed\nNew vector coexistence content.").unwrap();
                    store.index_file(&root, &path, 512).unwrap();
                }
                1 => store.invalidate_index().unwrap(),
                2 => store.remove_indexed(path.to_str().unwrap()).unwrap(),
                _ => store.remove_root(&root).unwrap(),
            }
            assert!(query_top_n(&store, "conan", Vector::Conan(&bits), 10).unwrap().is_empty());
            assert!(query_top_n(&store, "wemm", Vector::Wemm(&floats), 10).unwrap().is_empty());
        }
    }

    #[test]
    fn cache_reuses_bytes_separates_models_refreshes_and_expires() {
        let (_dir, store) = fixture(1);
        let mut calls = 0;
        let bytes = vec![0x5a; CONAN_BYTES];
        for _ in 0..2 {
            let result = if let Some(cached) = get_cached_embedding(&store, "hash", "conan").unwrap() { cached } else {
                calls += 1;
                put_cached_embedding(&store, "hash", "conan", &bytes).unwrap();
                bytes.clone()
            };
            assert_eq!(result, bytes);
        }
        assert_eq!(calls, 1);
        assert!(get_cached_embedding(&store, "other-hash", "conan").unwrap().is_none());
        let float_bytes: Vec<_> = [1.25_f32; WEMM_DIM].iter().flat_map(|v| v.to_le_bytes()).collect();
        put_cached_embedding(&store, "hash", "wemm", &float_bytes).unwrap();
        assert_eq!(get_cached_embedding(&store, "hash", "wemm").unwrap(), Some(float_bytes));
        store.connection().execute("UPDATE embedding_cache SET created_at=? WHERE model='conan'", [now() - CACHE_TTL_MILLIS - 1]).unwrap();
        assert!(get_cached_embedding(&store, "hash", "conan").unwrap().is_none());
        put_cached_embedding(&store, "hash", "conan", &bytes).unwrap();
        put_cached_embedding(&store, "hash", "conan", &bytes).unwrap();
        assert_eq!(get_cached_embedding(&store, "hash", "conan").unwrap(), Some(bytes));
        let counts: (i64, i64) = store.connection().query_row("SELECT count(*),count(DISTINCT model) FROM embedding_cache", [], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
        assert_eq!(counts, (2, 2));
        store.invalidate_index().unwrap();
        assert!(get_cached_embedding(&store, "hash", "conan").unwrap().is_some());
        store.clear_all_data().unwrap();
        assert!(get_cached_embedding(&store, "hash", "conan").unwrap().is_none());
    }

    #[test]
    fn existing_database_gains_tables_and_persists_vectors_cache_and_fts() {
        let (dir, store) = fixture(1);
        store.connection().execute_batch("DROP TRIGGER chunks_delete_vectors; DROP TABLE vec_conan; DROP TABLE vec_wemm; DROP TABLE embedding_cache;").unwrap();
        drop(store);
        let path = dir.path().join("index.sqlite");
        let store = Store::open(&path).unwrap();
        write_vector(&store, "conan", 1, Vector::Conan(&[0; CONAN_BYTES])).unwrap();
        write_vector(&store, "wemm", 1, Vector::Wemm(&[1.0; WEMM_DIM])).unwrap();
        put_cached_embedding(&store, "hash", "model", &[1, 2, 3]).unwrap();
        drop(store);
        let store = Store::open(&path).unwrap();
        assert_eq!(query_top_n(&store, "conan", Vector::Conan(&[0; CONAN_BYTES]), 10).unwrap()[0].chunk_id, 1);
        assert_eq!(query_top_n(&store, "wemm", Vector::Wemm(&[1.0; WEMM_DIM]), 10).unwrap()[0].chunk_id, 1);
        assert_eq!(get_cached_embedding(&store, "hash", "model").unwrap(), Some(vec![1, 2, 3]));
        // Auto-extension applies to independently opened read-only connections too.
        let reader = rusqlite::Connection::open_with_flags(&path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let count: i64 = reader.query_row("SELECT count(*) FROM vec_conan", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 1);
        let count: i64 = reader.query_row("SELECT count(*) FROM chunk_fts WHERE chunk_fts MATCH 'coexistence'", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 1);
    }
}
