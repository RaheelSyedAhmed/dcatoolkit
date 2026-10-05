# Changelog

## 0.4.0
This release updates the minimum Python version from 3.10 to 3.12 and requires biotite 1.7 or later, which brings performance improvements and type annotations. There are no other API changes. Users on Python 3.10 or 3.11 automatically keep receiving 0.3.x from `pip`.

### Changed
- **Requires Python ≥ 3.12 and biotite ≥ 1.7.** Tested with biotite 1.7.1, NumPy 2.5, pandas 3.0, and SciPy 1.18.
- PDB files load about 6× faster with biotite 1.7.
- `StructureInformation.fetch_pdb()` raises a clear `TypeError` if RCSB returns binary data, and invalid PDB IDs raise `biotite.database.RequestError`.

## 0.3.0

This release makes contact search much faster, adds support for AlphaFold3 models, fixes several bugs, and tightens up the `Pairs` and `DirectInformationData` APIs. It contains **breaking changes**; see [Upgrading from 0.2.x](#upgrading-from-02x) below.

Tested on Python 3.10, 3.11, and 3.12 (with biotite 1.2, 1.6, and 1.7 respectively).

### Highlights
- **Faster, lower-memory contact search.** `get_contacts()` now uses a KD-tree instead of a full distance matrix: about 20× faster on typical structures, and large complexes that previously needed gigabytes of memory now need megabytes. Results are unchanged.
- **AlphaFold3 models load.** `MMCIFInformation` can now read AlphaFold3 mmCIF output, which lacks the one-letter sequence record the reader previously required.
- **numpydoc docstrings** throughout, validated against the numpydoc standard.

### Added
- `Pairs.to_ndarray()` converts a structured pairs array into a plain `(n, 2)` int array.
- `Pairs.load_from_file(..., delimiter=...)`.
- `ResidueAlignment.load_from_align_file(..., valid_residues=...)`.
- `MSATools.load_from_file()` and `MSATools.write()` accept `pathlib.Path`.

### Changed
- `MMCIFInformation.get_min_dist_atom_info()` is about 30–60× faster.
- `get_contacts(..., auth_seq_id=True)` is up to about 2.5× faster on large or multi-model structures.
- Errors are now `ValueError` / `TypeError` with descriptive messages, instead of bare `Exception` or unclear unpacking errors (e.g. malformed align files, wrongly shaped arrays).
- **Dependencies:** `scikit-learn` is no longer a dependency (it was unused). `matplotlib` moved to an optional extra used by the plotting example: `pip install "dcatoolkit[plot]"`.

### Fixed
- `MMCIFInformation.get_start_res_id()` raised `KeyError` for every chain except the first.
- `Pairs.mirror_diagonal()` could fill fields other than `residue1`/`residue2` (e.g. `DI`) with garbage values.
- `DirectInformationData.load_from_DI_file()` / `load_from_dca_output()` returned a 0-dimensional array for one-line files.
- `DirectInformationData.write_DI_data()` raised `IndexError` for empty input; it now writes an empty file.
- `MSATools.load_from_file()` kept `\r` characters from Windows line endings for in-memory sources.

### Upgrading from 0.2.x
Most code keeps working. Check for these patterns:

| If your code... | Change it to... |
|---|---|
| Indexes `Pairs(...).pairs` like a plain array (`[:, 0]`, `[:, 1]`) | `.pairs['residue1']` / `.pairs['residue2']`, or `Pairs.to_ndarray(p.pairs)`. `.pairs` is now always a structured array; 0.2.x returned a plain array from files and stored arrays as given. |
| Calls `find_DI_with_residues(c1, c2, 300, arr1, arr2)` | `find_DI_with_residues(c1, c2, arr1, arr2, max_rank=300)`. `max_rank` is now keyword-only and optional; drop a positional `None`. |
| Calls `Pairs.mirror_pairs(pairs, True)` | `Pairs.mirror_pairs(pairs)`. The `mirror` parameter was removed; for a conditional mirror use `Pairs.get_pairs(pairs, mirror=flag)`. |
| Passes plain arrays to `DirectInformationData(...)` | `DirectInformationData.load_as_ndarray(array)`. The constructor now requires a structured array with `residue1`, `residue2`, and `DI` fields. |
| Passes plain arrays to `mirror_diagonal`, `mirror_pairs`, `get_pairs`, or `find_DI_with_residues` | Structured arrays with `residue1`/`residue2` fields, e.g. `Pairs(ndarr=array).pairs`. |
| Reads a DI field or third column from `Pairs` | Use `DirectInformationData` for DI values. `Pairs` keeps only `residue1`/`residue2`. |

Other behavior changes:
- `Pairs` rejects non-whole-number residue values (e.g. `12.7`, NaN, inf) instead of silently truncating them. Whole-number floats such as `12.0` are fine.
- `find_DI_with_residues(..., max_rank=0)` now returns no results (it previously meant "no limit").
- `ResidueAlignment(..., valid_residues=[])` now maps no residues. Previously an empty list behaved like `None`.
- `ResidueAlignment.load_from_align_file()` raises `ValueError` for files that don't contain exactly two complete 4-line entries, including files with trailing partial entries that previously loaded silently.
- `MSATools.write()` with a `pathlib.Path` now writes the file (0.2.x silently wrote nothing), and unsupported destinations raise `TypeError`.
- `domain_to_protein` / `protein_to_domain` values and `get_contacts()` results are plain Python `int`s instead of NumPy integers, and `MMCIFInformation.full_sequences` values are `str` instead of `ProteinSequence`. Comparisons and lookups behave the same.

### Coming next
The next release will require **Python ≥ 3.12** and **biotite ≥ 1.7**. Users on Python 3.10 or 3.11 will keep receiving 0.3.x from `pip`.
