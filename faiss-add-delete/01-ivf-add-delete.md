# IVF add & delete machinery

[← index](README.md)

Both IVF-PQ and IVF-RaBitQ derive from `IndexIVF` and reuse the same three
components:

- **Coarse quantizer** → which inverted list a vector belongs to.
- **`InvertedLists`** → per-list storage of `(id, code)` pairs.
- **`DirectMap`** → optional `external id → (list_no, offset)` map enabling
  reconstruct/remove/update.

## Add path

`IndexIVF::add` → `add_with_ids` → `add_core` (`faiss/IndexIVF.cpp:187-288`):

1. **Coarse assignment** — `quantizer->assign(n, x, coarse_idx)` maps each vector
   to a list in `[0, nlist)` (`IndexIVF.cpp:192-195`).
2. **Trained check** — `add_core` asserts `is_trained` (`IndexIVF.cpp:239`); a
   fresh index throws here until `train()` runs.
3. **Batching** — adds > 65536 are chunked to bound memory.
4. **Encode** — `encode_vectors(n, x, coarse_idx, flat_codes)`
   (`IndexIVF.cpp:250-251`), an abstract method each subclass implements
   (see [IVF-PQ](03-ivf-pq.md) / [IVF-RaBitQ](04-ivf-rabitq.md)).
5. **Append** — per entry, `invlists->add_entry(list_no, id, code)` returns the
   append `offset`; `DirectMapAdd` records it. ID is the caller's `xids[i]` or
   auto `ntotal + i` (`IndexIVF.cpp:253-278`). Lists are sharded across threads by
   `list_no % nthreads` to avoid contention.
6. `ntotal += n`.

Per-entry storage in `ArrayInvertedLists` (`faiss/invlists/InvertedLists.h:264-266`,
`InvertedLists.cpp:279-296`): two parallel arrays per list — `ids[list_no]`
(8-byte `idx_t`) and `codes[list_no]` (`code_size` bytes). `add_entries` `resize`s
and `memcpy`s onto the end, returning the starting offset.

## Delete path — `remove_ids`

`IndexIVF::remove_ids` (`faiss/IndexIVF.cpp:1252-1256`) delegates to
`DirectMap::remove_ids(sel, invlists)` (`faiss/invlists/DirectMap.cpp:157-231`),
then decrements `ntotal`. Behavior depends on the **DirectMap mode**:

- **`NoMap` (default)** — exhaustive parallel scan of every list. For each list, a
  two-pointer **swap-with-last** compaction: if `sel.is_member(id)`, overwrite it
  with the list's last entry (`update_entry`) and shrink; else advance. Then
  `resize` each list down (`DirectMap.cpp:164-190`). O(ntotal).
- **`Hashtable`** — direct lookup per id, swap-with-last, fix the moved entry's
  hashtable record, `resize` (`DirectMap.cpp:191-227`). **Only works with
  `IDSelectorArray`** (throws otherwise) and **not** with `BlockInvertedLists`.
- **`Array`** — `FAISS_THROW_MSG("remove not supported with this direct_map
  format")` (`DirectMap.cpp:228`).

**Key consequence — internal positions are not stable.** Swap-with-last can move a
surviving vector to a different offset. External ids stay valid (that's what
`DirectMap` tracks), but anything relying on raw list offsets must be refreshed.

## DirectMap modes (`faiss/invlists/DirectMap.h:38-49`)

- `NoMap` — no id map (default). Reconstruct/update unavailable; remove works via
  full scan.
- `Array` — `vector<idx_t>` indexed by id; requires **sequential ids** (i.e. `add`
  without explicit ids). O(1) lookup, but **remove is disallowed**.
- `Hashtable` — `unordered_map<id, lo>` for arbitrary ids; supports remove (with
  `IDSelectorArray`) and update.

`lo` packs `(list_no, offset)` into a uint64: `lo_build = list<<32 | offset`
(`DirectMap.h:23-33`). Enable via `make_direct_map()` (→ Array) or
`set_direct_map_type(Hashtable)` (`IndexIVF.cpp:290-300`).

## Update — `update_vectors` (`faiss/IndexIVF.cpp:1258-1283`)

Requires a DirectMap. `Hashtable`: remove-then-add. `Array`: in-place via
`DirectMap::update_codes` (`DirectMap.cpp:233`) — re-assign list, swap-out old,
append new — so the sequential id space stays hole-free.

## InvertedLists backends and remove support

| Backend | Add | Remove |
|---|---|---|
| `ArrayInvertedLists` (RAM) | ✅ | ✅ swap-with-last |
| `OnDiskInvertedLists` (mmap) | ✅ | ✅ but slow (shrinks); poor under parallel |
| `BlockInvertedLists` (fast-scan) | ✅ | ✅ via own `remove_ids`; **incompatible with Hashtable removal** |
| `ReadOnlyInvertedLists` | ❌ throws | ❌ throws |

## Source references

| What | Location |
|---|---|
| `IndexIVF::add` / `add_with_ids` / `add_core` | `faiss/IndexIVF.cpp:187` / `191` / `213` |
| `is_trained` guard in add | `faiss/IndexIVF.cpp:239` |
| `encode_vectors` call | `faiss/IndexIVF.cpp:250` |
| per-entry append | `faiss/IndexIVF.cpp:253-278` |
| `ArrayInvertedLists::add_entries` / `resize` | `faiss/invlists/InvertedLists.cpp:279` / `319` |
| invlists fields (`ids`, `codes`) | `faiss/invlists/InvertedLists.h:264-266` |
| `IndexIVF::remove_ids` | `faiss/IndexIVF.cpp:1252` |
| `DirectMap::remove_ids` | `faiss/invlists/DirectMap.cpp:157` |
| `DirectMap::update_codes` | `faiss/invlists/DirectMap.cpp:233` |
| `DirectMap::Type` enum | `faiss/invlists/DirectMap.h:38-49` |
| `lo_build`/`lo_listno`/`lo_offset` | `faiss/invlists/DirectMap.h:23-33` |
| `make_direct_map` / `set_direct_map_type` | `faiss/IndexIVF.cpp:290-300` |
| `IndexIVF::update_vectors` | `faiss/IndexIVF.cpp:1258` |
