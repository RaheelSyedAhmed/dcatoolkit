import pandas as pd
from typing import Optional

class ResidueAlignment:
    """
    A representation of a residue alignment, often from a query HMM to a protein structure target sequence.

    Parameters
    ----------
    domain_name : str
        The name of the query HMM.
    protein_name : str
        The name of the target protein sequence.
    domain_start : int
        The starting index of the domain alignment in the query HMM.
    protein_start : int
        The starting index of the domain alignment in the protein target sequence.
    domain_text : str
        The sequence of the domain in the query HMM corresponding to this alignment.
    protein_text : str
        The sequence of the protein target sequence corresponding to this alignment.
    valid_residues : list of tuple of int, str, optional
        A list of tuples that contain first residue index then residue name (e.g. ``[(1, 'A'), (2, 'W'), (3, 'C')]``). If None, this will imply all residues are valid and should be mapped sequentially from `protein_start`. An empty list of valid residues implies that none of the residues are mapped to each other.

    Attributes
    ----------
    domain_name : str
        The name of the query HMM.
    protein_name : str
        The name of the target protein sequence.
    valid_residues : list of tuple of int, str or None
        The `valid_residues` supplied, if any.
    reference_mapping : pandas.DataFrame
        The representation of the mapping where a row constitutes one aligned position, with the columns ``domain_index``, ``domain_residue``, ``protein_residue``, and ``protein_index``. Indices are ``pd.NA`` at gaps or where no valid residue matched.
    domain_to_protein : dict of {int : int}
        A dictionary allowing for mapping from indices corresponding to the query HMM and Multiple Sequence Alignment to the protein target sequence.
    protein_to_domain : dict of {int : int}
        A dictionary allowing for mapping from indices corresponding to the protein target sequence to the query HMM and Multiple Sequence Alignment.
    """
    _INVALID_CHARS = {".", "_", "-"}
    def __init__(self, domain_name: str, protein_name: str, domain_start: int, protein_start: int, domain_text: str, protein_text: str, valid_residues: Optional[list[tuple[int, str]]]=None) -> None:
        self.domain_name = domain_name
        self.protein_name = protein_name
        self.valid_residues = valid_residues
        self._set_reference_mapping(domain_start, protein_start, domain_text, protein_text, valid_residues)

    def _row_stream(self, domain_start: int, protein_start: int, domain_text: str, protein_text: str, valid_residues: Optional[list[tuple[int, str]]]):
        """
        Generates one mapping row per aligned position in `domain_text` and `protein_text`, resolving each side's residue index independently.

        Parameters
        ----------
        domain_start : int
            The starting index of the domain alignment in the query HMM.
        protein_start : int
            The starting index of the domain alignment in the protein target sequence.
        domain_text : str
            The sequence of the domain in the query HMM corresponding to this alignment.
        protein_text : str
            The sequence of the protein target sequence corresponding to this alignment.
        valid_residues : list of tuple of int, str, optional
            List of valid residues, non-missing residues in a structure, in the format of ``(seq_id, residue_name)``. When supplied, the protein index is resolved by scanning forward through this list, starting at list position ``protein_start - 1``, for a matching residue letter (permanently skipping any that don't match) rather than a simple sequential count. When None, the protein index is assigned by incrementing `protein_start` for every non-gap protein residue.

        Yields
        ------
        tuple of (int or pandas.NA), str, str, (int or pandas.NA)
            One row per aligned position: the domain index (``pd.NA`` if the domain character is a gap), the domain residue, the protein residue, and the protein index (``pd.NA`` if the protein character is a gap, or if `valid_residues` was supplied but exhausted without a match).
        """
        pointer = protein_start - 1
        for domain_aa, protein_aa in zip(domain_text, protein_text):
            if domain_aa in ResidueAlignment._INVALID_CHARS:
                domain_index = pd.NA
            else:
                domain_index = domain_start
                domain_start += 1

            if protein_aa in ResidueAlignment._INVALID_CHARS:
                protein_index = pd.NA
            elif valid_residues is not None:
                protein_index = pd.NA
                while pointer < len(valid_residues):
                    prot_index, valid_residue = valid_residues[pointer]
                    pointer += 1
                    if protein_aa.lower() == valid_residue.lower():
                        protein_index = prot_index
                        break
            else:
                protein_index = protein_start
                protein_start += 1

            yield domain_index, domain_aa, protein_aa, protein_index

    def _set_reference_mapping(self, domain_start: int, protein_start: int, domain_text: str, protein_text: str, valid_residues: Optional[list[tuple[int, str]]]) -> None:
        """
        Set values for the ``reference_mapping`` attribute and the mapping dictionaries, ``domain_to_protein`` and ``protein_to_domain``.

        Parameters
        ----------
        domain_start : int
            The starting index of the domain alignment in the query HMM.
        protein_start : int
            The starting index of the domain alignment in the protein target sequence.
        domain_text : str
            The sequence of the domain in the query HMM corresponding to this alignment.
        protein_text : str
            The sequence of the protein target sequence corresponding to this alignment.
        valid_residues : list of tuple of int, str, optional
            List of valid residues, non-missing residues in a structure, in the format of ``(seq_id, residue_name)``. These are iteratively selected in the order of the sequence to map to. If None, the protein index is assigned sequentially starting at `protein_start` instead.

        See Also
        --------
        _row_stream : Produces the rows of ``reference_mapping``.
        """
        self.reference_mapping = pd.DataFrame(
            self._row_stream(domain_start, protein_start, domain_text, protein_text, valid_residues),
            columns=['domain_index', 'domain_residue', 'protein_residue', 'protein_index'],
        )
        self.reference_mapping = self.reference_mapping.astype({'domain_index': pd.Int32Dtype(), 'protein_index': pd.Int32Dtype(), 'domain_residue': pd.StringDtype(), 'protein_residue': pd.StringDtype()})
        reference_mapping_notna = self.reference_mapping.dropna()

        # tolist() gives plain Python ints rather than np.int32 scalars, so the dicts print cleanly and build/look up faster.
        domain_indices = reference_mapping_notna.domain_index.tolist()
        protein_indices = reference_mapping_notna.protein_index.tolist()
        self.domain_to_protein = dict(zip(domain_indices, protein_indices))
        self.protein_to_domain = dict(zip(protein_indices, domain_indices))

    @staticmethod
    def load_from_align_file(align_filepath: str, valid_residues: Optional[list[tuple[int, str]]]=None) -> 'ResidueAlignment':
        """
        Generate ResidueAlignment from a standard align file generated from HMM scan.

        Parameters
        ----------
        align_filepath : str
            Filepath of the align file generated from a scan file produced via hmmscan.
        valid_residues : list of tuple of int, str, optional
            A list of tuples that contain first residue index then residue name (e.g. ``[(1, 'A'), (2, 'W'), (3, 'C')]``), passed through to the ResidueAlignment constructor. If None, this will imply all residues are valid and should be mapped sequentially from the protein start index. An empty list of valid residues implies that none of the residues are mapped to each other.

        Returns
        -------
        ResidueAlignment
            ResidueAlignment with domain and protein starting indices and corresponding sequence texts.

        Raises
        ------
        ValueError
            If the file doesn't contain exactly 2 complete entries of 4 non-blank lines each.

        Notes
        -----
        The align file holds two entries, the domain (HMM) first and the protein second, each made of 4 non-blank lines: name, start index, aligned sequence text, and end index. Blank lines between entries are ignored. For example::

            Domain_name
            1
            XXXXXXXXXXXXXXXXXXXX
            20

            Protein_name
            70
            XXXXXXXXXXXXXXXXXXXX
            89
        """
        # Read the alignment file and parse the important information from each alignment entry.
        alignment_entries = ResidueAlignment._read_align_file(align_filepath)
        if len(alignment_entries) != 2:
            raise ValueError(f"Expected 2 complete alignment entries (4 non-blank lines each) in {align_filepath}, found {len(alignment_entries)}.")
        hmm_entry, protein_entry = alignment_entries
        domain_name, domain_start, domain_text, _ = hmm_entry
        protein_name, protein_start, protein_text, _ = protein_entry
    
        # Convert to ints for iteration
        domain_start = int(domain_start)
        protein_start = int(protein_start)

        return ResidueAlignment(domain_name, protein_name, domain_start, protein_start, domain_text, protein_text, valid_residues)

    @staticmethod
    def _read_align_file(align_filepath: str) -> list[list[str]]:
        """
        Reads a standard align file, where a scan file is selected for a particular domain and processed into an align file format. See `load_from_align_file()` for the format.

        Parameters
        ----------
        align_filepath : str
            Filepath and filename of alignment file that contains information on the domain / protein of interest and its mapping to a protein's structural sequence.

        Returns
        -------
        alignment_entries : list of list of str
            List of entries, each a list of its 4 non-blank lines (one entry corresponds to the HMM produced sequence and its indices, and one corresponds to the protein's sequence and its indices).

        Raises
        ------
        ValueError
            If the file ends partway through an entry (its non-blank line count isn't a multiple of 4).
        """
        with open(align_filepath, 'r') as fs:
            alignment_entries: list[list[str]] = []
            current_entry: list[str] = []
            line_count = 0
            for line in fs:
                line = line.strip()
                if line != '':
                    line_count += 1
                    current_entry.append(line)
                if line_count == 4:
                    line_count = 0
                    alignment_entries.append(current_entry)
                    current_entry: list[str] = []
        # Leftover lines mean the file ended partway through an entry; don't silently drop them.
        if current_entry:
            raise ValueError(f"Incomplete alignment entry at the end of {align_filepath}: expected 4 non-blank lines, found {len(current_entry)} ({current_entry}).")
        return alignment_entries
    
    def __str__(self) -> str:
        """
        Returns string representation of the ResidueAlignment pandas DataFrame in tab-separated value (TSV) format.

        Returns
        -------
        str
            The ``reference_mapping`` DataFrame exported to TSV format via ``DataFrame.to_csv(sep="\\t")``, including the row index.
        """
        return self.reference_mapping.to_csv(sep="\t")