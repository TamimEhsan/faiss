# Exposed interfaces (`IndexIVFPQ`, `IndexIVFRaBitQ`)

[← index](README.md)

Both derive `IndexIVF → Index`, so most of the API is inherited and identical;
only a thin layer is type-specific. C++ names; SWIG exposes the same in Python.
Lists below are generated from the installed `faiss` (`dir(cls)`), de-noised of
SWIG `_c`/`_ex` shims and `thisown`.

## Shared — from `Index` (generic vector-DB API)

`train`, `add`, `add_with_ids`, `add_sa_codes`, `assign`, `search`, `search1`,
`search_and_reconstruct`, `search_subset`, `range_search`, `remove_ids`,
`reconstruct`, `reconstruct_n`, `reconstruct_batch`, `reset`, `compute_residual`,
`compute_residual_n`, `merge_from`, `check_compatible_for_merge`,
`get_distance_computer`, `sa_code_size`, `sa_encode`, `sa_decode`.
Fields: `d`, `ntotal`, `is_trained`, `metric_type`, `metric_arg`, `verbose`.
(Persistence is via the free functions `faiss.write_index` / `faiss.read_index`.)

Declared in `faiss/Index.h`.

## Shared — from `IndexIVF` (IVF core)

`add_core`, `encode_vectors`, `decode_vectors`, `encode_listno`, `decode_listno`,
`coarse_code_size`, `train_encoder`, `train_encoder_num_vectors`, `train_q1`,
`search_preassigned`, `range_search_preassigned`, `search_and_return_codes`,
`get_InvertedListScanner`, `get_CodePacker`, `get_list_size`,
`reconstruct_from_offset`, `update_vectors`, `copy_subset_to`, `check_ids_sorted`,
`make_direct_map`, `set_direct_map_type`, `replace_invlists`.
Fields: `quantizer`, `nlist`, `nprobe`, `max_codes`, `invlists`, `code_size`,
`by_residual`, `direct_map`, `own_fields`, `own_invlists`, `parallel_mode`,
`quantizer_trains_alone`, `clustering_index`, `cp`.
Diagnostics on `invlists`: `imbalance_factor()`, `print_stats()`.

Declared in `faiss/IndexIVF.h`.

## `IndexIVFPQ`-specific (`faiss/IndexIVFPQ.h`)

`pq` (the `ProductQuantizer`: `pq.M`, `pq.nbits`, `pq.ksub`, `pq.dsub`,
`pq.centroids`, …), `add_core_o`, `encode`, `encode_multiple`, `decode_multiple`,
`find_duplicates`, `precompute_table`.
Fields: `precomputed_table`, `use_precomputed_table`, `scan_table_threshold`,
`do_polysemous_training`, `polysemous_training`, `polysemous_ht`.
Search params: `IVFPQSearchParameters`.

## `IndexIVFRaBitQ`-specific (`faiss/IndexIVFRaBitQ.h`)

`rabitq` (the `RaBitQuantizer`), and the field `qb` (query-quantization bits;
`0` = raw fp32 query). It also overrides `get_distance_computer`,
`encode_vectors`/`decode_vectors`, `add_core`, `reconstruct_from_offset`,
`sa_decode`.
Search params: `IVFRaBitQSearchParameters` (`qb`, `centered`, + IVF fields).

## Contrast

| | IVF-PQ | IVF-RaBitQ |
|---|---|---|
| add / add_with_ids / remove_ids / update_vectors | ✅ | ✅ |
| `search` + selector (`sel`) | ✅ | ✅ |
| codebook knobs | rich (`pq`, polysemous, precomputed tables, duplicates) | minimal (`rabitq`, `qb`, `centered`) |
| type-specific members (`dir` diff) | 13 | 2 (`rabitq`, `qb`) |
| dedicated search params | `IVFPQSearchParameters` | `IVFRaBitQSearchParameters` |
| `by_residual` toggle effect | real (origin vs residual encoding) | ignored (always cluster-centroid) |

## Regenerate

```python
import faiss
base = set(dir(faiss.IndexIVF))
for name in ["IndexIVFPQ", "IndexIVFRaBitQ"]:
    cls = getattr(faiss, name)
    own = sorted(m for m in dir(cls)
                 if m not in base and not m.startswith("_"))
    print(name, "specific:", own)
```
