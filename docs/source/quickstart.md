# Quick start

A typical workflow ranks the Direct Information (DI) pairs from a DCA run, maps them from MSA (domain) numbering onto a protein sequence, and compares them with residue contacts in a structure.

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

What each step does:

- {meth}`~dcatoolkit.DirectInformationData.load_from_DI_file` loads a three-column DI file (residue 1, residue 2, DI score).
- {meth}`~dcatoolkit.ResidueAlignment.load_from_align_file` reads an alignment of the HMM domain to the protein sequence, which maps MSA positions to protein residues.
- {meth}`~dcatoolkit.DirectInformationData.get_ranked_mapped_pairs` drops pairs within 4 residues of each other, ranks the rest by DI, maps them onto the protein, and returns the top `number` pairs.
- {meth}`~dcatoolkit.StructureInformation.fetch_pdb` downloads a structure from RCSB; use {meth}`~dcatoolkit.StructureInformation.read_mmCIF_file` or {meth}`~dcatoolkit.StructureInformation.read_pdb_file` for local files, including AlphaFold3 models.
- {meth}`~dcatoolkit.MMCIFInformation.get_contacts` returns the residue pairs with any atoms within `threshold` Ångströms. It uses the structure's label (mmCIF) residue numbering by default, or author numbering with `auth_seq_id=True`; use whichever numbering the protein sequence in your align file follows.

See the {doc}`api` for every class and method.
