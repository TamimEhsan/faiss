# Memory management (growth & reclamation)

[← index](README.md)

## The underlying primitive: everything is `std::vector`

Every in-RAM container is ultimately a `std::vector<T>`, often wrapped in
`MaybeOwnedVector` whose `resize`/`clear` just delegate to an internal
`owned_data` vector (`faiss/impl/maybe_owned_vector.h:260-267`). So memory
behavior is **std::vector semantics**:

- **Grow** (`resize` larger): if the new size exceeds capacity, allocate a new
  larger contiguous buffer (geometric growth, typically **2×**), move/copy
  existing elements, free the old buffer. Amortized O(1) per element, with
  occasional O(n) realloc and a transient peak holding both buffers.
- **Shrink** (`resize` smaller): destroys tail elements but **keeps capacity** —
  memory is *not* returned to the allocator (FAISS rarely calls `shrink_to_fit`).
- Buffers are contiguous; offsets/pointers into them are invalidated on realloc.

## IVF

**Add → grow per inverted list.** `ArrayInvertedLists` keeps two
`MaybeOwnedVector`s *per list*, `ids` and `codes`
(`faiss/invlists/InvertedLists.h:264-266`). `add_entries` appends by growing them
(`InvertedLists.cpp:279-296`):

```cpp
size_t o = ids[list_no].size();
ids[list_no].resize(o + n_entry);                 // geometric realloc
codes[list_no].resize((o + n_entry) * code_size);
memcpy(...);                                       // id (8 B) + code (code_size B)
```

- Memory grows **independently per list** — `nlist` separate vector pairs, each
  doubling on its own. `add_core` batches in 65536-vector chunks but still appends
  per list.
- Per stored vector: 8 B (`idx_t` id) + `code_size` B (PQ ≈ `M` B; RaBitQ
  `(d+7)/8` + factors).
- Many small lists ⇒ more capacity overshoot + per-vector bookkeeping than one big
  array.

**Delete → logical shrink, no RAM returned.** `remove_ids` does swap-with-last
then `invlists->resize(list_no, smaller)` (`InvertedLists.cpp:319-322`); capacity
is retained and slots are reused by later adds. Reclaim only by rebuild or
`write_index` + `read_index`.

## HNSW

Add touches **two** structures (`faiss/IndexHNSW.cpp:378-394`):

**1. The `storage` index (the vectors).** `IndexFlatCodes` holds one big
contiguous `codes` vector, grown once per batch (`IndexFlatCodes.cpp:33`):

```cpp
codes.resize((ntotal + n) * code_size);           // single big realloc + copy
```

A large add can reallocate and copy the entire buffer — a real transient spike at
scale.

**2. The graph (CSR-style, `faiss/impl/HNSW.h:114-131`).**
- `levels` (`vector<int>`): one `push_back` per node (`HNSW.cpp:220`).
- `offsets` (`vector<size_t>`): prefix offsets, one `push_back` per node
  (`HNSW.cpp:230`): `offsets[i+1] = offsets[i] + cum_nb_neighbors(level+1)`.
- `neighbors` (`MaybeOwnedVector<int32>`): one flat array of all neighbor slots,
  grown to the new total `neighbors.resize(offsets.back(), -1)`
  (`HNSW.cpp:232`), pre-filled with `-1` then populated by `add_with_locks`.

Per node the neighbor budget is **fixed at insertion**: `2·M` slots on level 0,
`M` on each higher level. Most nodes are level-0 only ⇒ ~`2·M·4` bytes of graph
memory each. This fixed layout is why it's a flat CSR array (no per-node dynamic
lists).

**Delete → none.** `remove_ids` throws; no shrink path, and the positional CSR
layout can't drop a node without a full rebuild.

## Summary

| | Grow on add | Delete behavior | Returns RAM on delete? |
|---|---|---|---|
| IVF codes/ids | per-list `std::vector` resize (≈2×) | swap-with-last + `resize(smaller)` | ❌ capacity kept; reused |
| HNSW storage (codes) | one contiguous `std::vector` resize | — (unsupported) | ❌ |
| HNSW graph (neighbors/offsets/levels) | flat CSR arrays; fixed `2M`/`M` slots per node | — (unsupported) | ❌ |

**Implications.** Adds amortize cheaply but cause periodic realloc spikes (largest
for the single big HNSW/Flat `codes` buffer). Deletes never shrink process RSS —
reclaim by rebuilding or round-tripping `write_index`/`read_index`. If the final
size is known, pre-sizing (build once with all vectors) avoids repeated reallocs.

## Disk-based loading: mmap & `OnDiskInvertedLists`

The same `MaybeOwnedVector` that enables heap-owned arrays also enables
**memory-mapped (view) arrays** — the basis of FAISS's two disk mechanisms.

### Owned vs view

`MaybeOwnedVector` has two modes (`faiss/impl/maybe_owned_vector.h:34-46`):

- **owned** (`is_owned = true`): a normal heap `std::vector` — mutable, growable.
- **view** (`is_owned = false`): `view_data`/`view_size` pointing **into** an
  `mmap`'d region, holding a `shared_ptr<MaybeOwnedVectorOwner>` to keep the
  mapping alive (`mapped_io`, `faiss/impl/mapped_io.h:20-31`).

Views are **read-only**: `resize`/`clear`/`insert`/`erase` assert `is_owned`
(`maybe_owned_vector.h:261-264`), so a mapped array cannot grow or shrink.

### Mechanism 1 — zero-copy mmap read (`IO_FLAG_MMAP_IFC`)

```python
index = faiss.read_index("idx.index", faiss.IO_FLAG_MMAP_IFC)
```

`read_index` wraps the file in a `MappedFileIOReader` only for `IO_FLAG_MMAP_IFC`
(`faiss/impl/index_read.cpp:2855`, `2871`); big arrays read via `read_vector`
become views (`index_read.cpp:152-196`). Works for **any** index type, including
**HNSW** — `read_HNSW` reads `hnsw.neighbors` and the storage `codes` through the
view path (`index_read.cpp:1258`), so the two largest arrays are mapped (small
metadata like `levels`/`offsets` is still copied to heap). The older `IO_FLAG_MMAP`
(`= IO_FLAG_SKIP_IVF_DATA | magic`, `faiss/index_io.h:67`) is the IVF-OnDisk combo,
not the generic path.

Because the arrays are views, an mmap-loaded index is **serve-only**: `add` /
`remove_ids` would try to `resize` a viewed array and throw. (HNSW has no delete
anyway.)

### Mechanism 2 — `OnDiskInvertedLists` (IVF only)

`faiss/invlists/OnDiskInvertedLists.{h,cpp}` keeps the IVF inverted lists in an
on-disk (mmap'd) file instead of RAM. Unlike generic mmap read, it's designed for
**building/merging** indexes that don't fit in RAM (via `merge_ondisk` + shards)
and searching them on disk. Triggered with `IO_FLAG_ONDISK_SAME_DIR` /
`IO_FLAG_MMAP`.

These are the two disk-based options; `IO_FLAG_MMAP_IFC` is general and read-only,
`OnDiskInvertedLists` is IVF-specific and supports on-disk build/merge.

## Source references

| What | Location |
|---|---|
| `MaybeOwnedVector::resize` | `faiss/impl/maybe_owned_vector.h:260-267` |
| invlists per-list `ids`/`codes` | `faiss/invlists/InvertedLists.h:264-266` |
| `ArrayInvertedLists::add_entries` | `faiss/invlists/InvertedLists.cpp:279-296` |
| `ArrayInvertedLists::resize` (shrink) | `faiss/invlists/InvertedLists.cpp:319-322` |
| `IndexFlatCodes::add` (storage grow) | `faiss/IndexFlatCodes.cpp:28-35` |
| HNSW graph arrays | `faiss/impl/HNSW.h:114-131` |
| `prepare_level_tab` (levels/offsets/neighbors grow) | `faiss/impl/HNSW.cpp:211-235` |
| owned vs view modes | `faiss/impl/maybe_owned_vector.h:34-46` |
| view is read-only (resize assert) | `faiss/impl/maybe_owned_vector.h:261-264` |
| `mapped_io` (mmap owner + reader) | `faiss/impl/mapped_io.h:20-51` |
| mmap reader enabled (`IO_FLAG_MMAP_IFC`) | `faiss/impl/index_read.cpp:2855`, `2871` |
| view substitution in `read_vector` | `faiss/impl/index_read.cpp:152-196` |
| `read_HNSW` (neighbors via view path) | `faiss/impl/index_read.cpp:1258` |
| IO flags | `faiss/index_io.h:51-71` |
| `OnDiskInvertedLists` | `faiss/invlists/OnDiskInvertedLists.{h,cpp}` |
