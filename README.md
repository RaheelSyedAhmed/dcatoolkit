# dcatoolkit
 Collection of useful modules and representations for managing DCA output data.

**Documentation:** https://dcatoolkit.readthedocs.io

## Installation

```bash
pip install dcatoolkit

# optional: adds matplotlib for the plotting example"
pip install "dcatoolkit[plot]"  
```

Requires Python 3.12+.
Upgrading from 0.2.x? See the [changelog](https://github.com/RaheelSyedAhmed/dcatoolkit/blob/main/CHANGELOG.md).

## Major Sections
### Representations
  * Use Pairs to load lists, tuples, sets, and ndarrays with the correct orientation of elements. This will allow you to store integer pairs, in the form of structured arrays with fields `residue1` and `residue2` that can be mirrored (where y becomes x and vice versa) and subset.
  * Use DirectInformationData to create structured ndarrays with `residue1`, `residue2`, and `DI` fields that can be sorted by `DI`, mapped to a protein with a ResidueAlignment, and used to generate output for other programs (including UCSF Chimera)
  * Use ResidueAlignment to generate a reference map. Indices of one sequence of characters can be linked to their corresponding indices of the other sequence of characters. The dictionaries produced, domain-to-protein and protein-to-domain, allow for forward mapping and backmapping.
  * Use StructureInformation to find contacts in a protein structure and find atomic information related to specific pairs of interest. It can read in PDBx/mmCIF and PDB files or fetch them from RCSB and find contacts between residues.
### Analytics
  * Use MSATools to load in Multiple Sequence Alignment (MSA) data and provide functionality including generating frequency statistics on "gappiness" in the MSA and filtering and cleaning MSAs.


## Quick Start
```python
from dcatoolkit import DirectInformationData, ResidueAlignment, StructureInformation

# Rank DI pairs and map them from MSA (domain) numbering onto the protein sequence.
di = DirectInformationData.load_from_DI_file("my_dca_output.DI")
alignment = ResidueAlignment.load_from_align_file("my_domain.align")
top_pairs = di.get_ranked_mapped_pairs(alignment, alignment, number=50)

# Compare the top pairs with residue contacts in a structure.
structure = StructureInformation.fetch_pdb("2KLL")
contacts = structure.get_contacts(ca_only=False, threshold=8, chain1="A", chain2="A")
hits = [pair for pair in top_pairs.tolist() if pair in contacts]
```

## Diagram of Hidden Markov Model & Direct Coupling Analysis Pipeline
<p align="center">
  <img src="https://github.com/user-attachments/assets/4768e08f-d513-4dbf-abc5-c80c1b3d42aa"/>
</p>

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/RaheelSyedAhmed/dcatoolkit.git
cd dcatoolkit
uv sync  # Installs dcatoolkit and the dev group (pytest, ruff), which uv includes by default.
```

Add `--group docs` to also install the documentation tools (Sphinx), and `--extra plot` for matplotlib, used by the plotting example. Use `uv sync --no-dev` to install only dcatoolkit and its dependencies, without pytest or ruff.

Run the test suite (requires internet to fetch from RCSB):

```bash
uv run pytest
```

Lint:

```bash
uv run ruff check
```
