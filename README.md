# dcatoolkit
 Collection of useful modules and representations for managing DCA output data.

## Installation

```bash
pip install dcatoolkit
```

Requires Python 3.10+.

## Major Sections
### Representations
  * Use Pairs to load lists, tuples, sets, and ndarrays with the correct orientation of elements. This will allow you to yield integer pairs that can be mirrored (where y becomes x and vice versa) and to subset various pairs.
  * Use DirectInformationData to create 3-column structured ndarrays that can be sorted by "DI", mapped to a protein with a ResidueAlignment, and used to generate output for other programs (including UCSF Chimera)
  * Use ResidueAlignment to generate a reference map. Indices of one sequence of characters can be linked to their corresponding indices of the other sequence of characters. The dictionaries produced, domain-to-protein and protein-to-domain, allow for forward mapping and backmapping.
  * Use StructureInformation to find contacts in a protein structure and find atomic information related to specific pairs of interest.
### Analytics
  * Use MSATools to load in Multiple Sequence Alignment (MSA) data and provide functionality including generating frequency statistics on "gappiness" in the MSA and filtering and cleaning MSAs.


## Diagram of Hidden Markov Machine & Direct Coupling Analysis Pipeline
<p align="center">
  <img src="https://github.com/user-attachments/assets/4768e08f-d513-4dbf-abc5-c80c1b3d42aa"/>
</p>

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/RaheelSyedAhmed/dcatoolkit.git
cd dcatoolkit
uv sync --all-extras
```

Run the test suite:

```bash
uv run pytest
```
