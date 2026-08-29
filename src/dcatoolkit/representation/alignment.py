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
        A list of tuples that contain first residue index then residue name (e.g. [(1, 'A'), (2, 'W'), (3, 'C')]  )

    Attributes
    ----------
    reference_mapping : pandas.DataFrame
        The representation of the mapping where a row constitutes a residue pair and its indices in the format: 'domain_index', 'domain_residue', 'protein_residue', 'protein_index'.
    domain_to_protein : dict[int, int]
        A dictionary allowing for mapping from indices corresponding to the query HMM and Multiple Sequence Alignment to the protein target sequence.
    protein_to_domain : dict[int, int]
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
        Generates one mapping row per aligned position in domain_text/protein_text, resolving each side's residue index independently.

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
            List of valid residues, non-missing residues in a structure, in the format of (seq_id, residue_name). When supplied, protein_index is resolved by scanning forward through this list for a matching residue letter (permanently skipping any that don't match) rather than a simple sequential count. When None, protein_index is assigned by incrementing protein_start for every non-gap protein residue.

        Yields
        ------
        tuple of (int or pandas.NA), str, str, (int or pandas.NA)
            One row per aligned position: domain_index (pd.NA if the domain character is a gap), domain_residue, protein_residue, and protein_index (pd.NA if the protein character is a gap, or if `valid_residues` was supplied but exhausted without a match).
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
            elif valid_residues:
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
        Set values for reference_mapping and mapping dictionaries, domain_to_protein and protein_to_domain.

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
            List of valid residues, non-missing residues in a structure, in the format of (seq_id, residue_name). These are iteratively selected in the order of the sequence to map to. If None, protein_index is assigned sequentially starting at protein_start instead.

        Returns
        -------
        None
        """
        self.reference_mapping = pd.DataFrame(
            self._row_stream(domain_start, protein_start, domain_text, protein_text, valid_residues),
            columns=['domain_index', 'domain_residue', 'protein_residue', 'protein_index'],
        )
        self.reference_mapping = self.reference_mapping.astype({'domain_index': pd.Int32Dtype(), 'protein_index': pd.Int32Dtype(), 'domain_residue': pd.StringDtype(), 'protein_residue': pd.StringDtype()})
        reference_mapping_notna = self.reference_mapping.dropna()

        self.domain_to_protein = dict(zip(reference_mapping_notna.domain_index, reference_mapping_notna.protein_index))
        self.protein_to_domain = dict(zip(reference_mapping_notna.protein_index, reference_mapping_notna.domain_index))

    @staticmethod
    def load_from_align_file(align_filepath: str) -> 'ResidueAlignment':
        """
        Generate ResidueAlignment from a standard align file generated from HMM scan.

        Parameters
        ----------
        align_filepath : str
            Filepath of the align file generated from a scan file produced via hmmscan.

        Returns
        -------
        ResidueAlignment
            ResidueAlignment with domain and protein starting indices and corresponding sequence texts.

        File Format
        -----------
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
        hmm_entry, protein_entry = alignment_entries
        domain_name, domain_start, domain_text, _ = hmm_entry
        protein_name, protein_start, protein_text, _ = protein_entry
    
        # Convert to ints for iteration
        domain_start = int(domain_start)
        protein_start = int(protein_start)

        return ResidueAlignment(domain_name, protein_name, domain_start, protein_start, domain_text, protein_text)

    @staticmethod
    def _read_align_file(align_filepath: str) -> list[list[str]]:
        """
        Reads standard align file, where a scan file is selected for a particular domain and processed into an align file format. Details are present in produce_align_from_scan().
        
        Parameters
        ----------
        align_filepath : str
            Filepath and filename of alignment file that contains information on the domain / protein of interest and its mapping to a protein's structural sequence
        
        Returns
        -------
        alignment_entries : list of list of strings
            list of associated lines (one which corresponds to the HMM produced sequence and its indices and one that corresponds to the protein's seqeuence and its indices), which are also contained in a list.
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
        return alignment_entries
    
    def __str__(self) -> str:
        """
        Returns string representation of the ResidueAlignment pandas DataFrame in tab-separated value (tsv) format.
        
        Returns
        -------
        str
            reference_mapping pandas DataFrame exported to TSV format via the to_csv(sep="\t") function from pandas.
        """
        return self.reference_mapping.to_csv(sep="\t")