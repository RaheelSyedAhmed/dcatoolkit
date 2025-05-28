import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb
import biotite.database.rcsb as rcsb
from biotite.sequence import ProteinSequence

from collections.abc import Iterable
from typing import Optional, Union, Literal, overload
import numpy.typing as npt


class Pairs:
    """
    Object that contains a representation (as an ndarray) of pairs of entities that are related. This may extend to Direct Information Pairs or Structural contacts, where each residue is one component of the pair.
    
    Note
    ----
    Either a filepath or a ndarr has to be specified in order to produce a Pairs representation.

    Parameters
    ----------
    filepath : str, optional
        Filepath of the pairs in tabular representation, separated by whitespace between the pair components and newlines between each pair.
    ndarr : numpy.ndarray, optional
        Populated ndarray that contains pair information.
    delimiter : str, optional
        String used to specify separator between two pairs. See numpy.loadtxt() for details.
    
    Attributes
    ----------
    pairs : numpy.ndarray
        Ndarray representation of pairs supplied by the user. This is produced via the np.loadtxt() function.
    """
    def __init__(self, filepath: Optional[str]=None, ndarr: Optional[npt.NDArray]=None, delimiter: Optional[str]=None) -> None:
        if (filepath is not None and ndarr is not None) or (filepath is None and ndarr is None):
            raise Exception("Please specify either a filepath or a NumPy array to populate your pairs.")
        elif filepath is not None:
            if delimiter:
                self.pairs = np.loadtxt(filepath, dtype=int, delimiter=delimiter)
            else:
                self.pairs = np.loadtxt(filepath, dtype=int)
        elif ndarr is not None:
            self.pairs = ndarr
    
    @staticmethod
    def load_from_file(filepath: str):
        """
        Loads file containing whitespace-delimited data in columns of residues being column 1 and column 2.

        Parameters
        ----------
        filepath : str
            Filepath with residue columns corresponding to the indices of first and second components (proteins, chains, etc.) constituting a pair.

        Returns
        -------
        Pairs
            Pairs object with a loaded, structured ndarray with dtype=[('residue1', int), ('residue2', int)]
        """
        return Pairs(ndarr=np.loadtxt(filepath, dtype=[('residue1', int), ('residue2', int)]))

    @staticmethod
    def load_from_ndarray(ndarray: Union[npt.NDArray, Iterable[Iterable]]):
        """
        Loads 2d ndarray of residue pairs in columnar format into Pairs object.

        Parameters
        ----------
        ndarray : numpy.ndarray or Iterable of Iterable (excluding dict)
            Unstructured ndarray or iterable of iterables with pairs of residue indices with residue1 and residue 2 in separate columns or as two separate elements.

        Returns
        -------
        Pairs
            Pairs object with a loaded, structured ndarray with dtype=[('residue1', int), ('residue2', int)]
        """
        return Pairs(ndarr=np.array([tuple(x) for x in ndarray], dtype={'names': ('residue1', 'residue2'), 'formats': (int, int)}))
    
    @staticmethod
    def mirror_diagonal(pairs: npt.NDArray) -> npt.NDArray:
        """ 
        Flip 2D ndarray with 2 columns columnwise. Flips pair positions for diagonal-mirrored representation.

        Parameters
        ----------
        pairs : numpy.ndarray
            2d array with (n, 2) shape.

        Returns
        -------
        numpy.ndarray 
            Values flipped along the column axis.
        """
        if pairs.dtype.names is not None:
            return np.array([tuple(x)[::-1] for x in pairs], dtype={'names': ('residue1', 'residue2'), 'formats': (int, int)})
        else:
            return np.flip(pairs, axis=1)
    
    @staticmethod
    def subset_pairs(pairs: npt.NDArray, number : Optional[int]=None) -> npt.NDArray:
        """
        Picks out a subset of 'number' pairs if a number is supplied. Otherwise, returns all pairs.

        Parameters
        ----------
        pairs : numpy.ndarray
            Ndarray to select number of rows from.
        number :  int, optional
            Specific number of rows of pairs to subset.

        Returns
        -------
        numpy.ndarray
            Subset of pairs from rows 0 to number.
        pairs : numpy.ndarray
            All pairs specified from the parameters section.
        """
        if number is not None:
            return pairs[:number, ]
        else:
            return pairs
    
    @staticmethod
    def mirror_pairs(pairs: npt.NDArray, mirror: bool=False) -> npt.NDArray:
        """
        Produces combined array of pairs and potentially their mirrored representation.

        Parameters
        ----------
        pairs : numpy.ndarray
            Ndarray to mirror and vertically append if mirror is set to True.
        mirror : bool
            Whether or not to append mirrored representation of pairs to the original pairs ndarray.

        Returns
        -------
        mirrored_ndarray : numpy.ndarray
            combined ndarray of pairs and mirrored pairs.
        pairs : numpy.ndarray
            The original pairs specified from the parameters section.
        """
        if mirror:
            if pairs.dtype.names is not None:
                unstruc_pairs = [tuple(x) for x in pairs]
                unstruc_pairs_mirrored = [tuple(x)[::-1] for x in pairs]
                return np.array(unstruc_pairs + unstruc_pairs_mirrored, dtype=[('residue1', int), ('residue2', int)])     
            else:
                return np.vstack((pairs, Pairs.mirror_diagonal(pairs)))
        else:
            return pairs
    
    @staticmethod
    def get_pairs(pairs: npt.NDArray, mirror: bool=False, number: Optional[int]=None) -> npt.NDArray:
        """
        Returns pairs based on user specification, offering options to produce mirrored representation of pairs and to select a specific number of pairs.

        Parameters
        ----------
        pairs : numpy.ndarray
            ndarray of pairs to select from or to mirror.
        mirror : bool
            Whether or not to append mirrored representation of pairs to the original pairs ndarray.
        number : int
            Specific number of rows of pairs to subset.

        Returns
        -------
        numpy.ndarray
            mirrored, subset version of pairs produced via subset_pairs() and mirror_pairs() on pairs.
        """
        # Check to see if user requested mirrored pairs, if so, add in pairs that are mirrored across diagonal
        pairs = Pairs.subset_pairs(pairs, number)
        if mirror:
            pairs = Pairs.mirror_pairs(pairs, mirror)
        return pairs

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
    valid_residues : list of tuple of int, str
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
    def __init__(self, domain_name: str, protein_name: str, domain_start: int, protein_start: int, domain_text: str, protein_text: str, valid_residues: Optional[list[tuple[int, str]]]=None) -> None:
        self.domain_name = domain_name
        self.protein_name = protein_name
        self.valid_residues = valid_residues
        if valid_residues:
            self._set_restricted_reference_mapping(domain_start, protein_start, domain_text, protein_text, valid_residues)
        else:
            self._set_reference_mapping(domain_start, protein_start, domain_text, protein_text)
    
    def _set_restricted_reference_mapping(self, domain_start: int, protein_start: int, domain_text: str, protein_text: str, valid_residues: list[tuple[int, str]]) -> None:
        """
        Set values for reference_mapping and mapping dictionaries, domain_to_protein and protein_to_domain.

        Parameters
        ----------
        valid_residues : list of tuple of int, str
            List of valid residues, non-missing residues in a structure, in the format of (seq_id, residue_name). These are iteratively selected in the order of the sequence to map to.

        Notes
        -----
        For details on `domain_start`, `protein_start`, `domain_text`, `protein_text`, please refer to the `ResidueAlignment` docstring.

        Returns
        -------
        None
        """
        invalid_chars = [".", "_", "-"]
        # Convert text to list variant for iteration
        domain_sequence = list(domain_text)
        protein_sequence = list(protein_text)
        # Store mapping values per iteration here.
        mapping_entries = []
        
        # Go through aligned sequences and append data concerning domain index, domain residue, protein residue, and protein index per valid aligned residues.
        for domain_aa, protein_aa in zip(domain_sequence, protein_sequence):
            mapping_entry = []
            if domain_aa not in invalid_chars:
                mapping_entry.append(domain_start)
                domain_start += 1
            else:
                mapping_entry.append(pd.NA)
            mapping_entry.append(domain_aa)

            if protein_aa not in invalid_chars:
                while len(valid_residues) >= protein_start:
                    prot_index, valid_residue = valid_residues.pop(protein_start-1)
                    if protein_aa.lower() == valid_residue.lower():
                        mapping_entry.append(protein_aa)
                        mapping_entry.append(prot_index)
                        break
                else:
                    # Default case for if valid residues are missing towards the end.
                    mapping_entry.append(protein_aa)
                    mapping_entry.append(pd.NA)
            else:
                mapping_entry.append(protein_aa)
                mapping_entry.append(pd.NA)
            mapping_entries.append(mapping_entry)

        self.reference_mapping = pd.DataFrame(mapping_entries, columns=['domain_index', 'domain_residue', 'protein_residue', 'protein_index'])
        self.reference_mapping = self.reference_mapping.astype({'domain_index': pd.Int32Dtype(), 'protein_index': pd.Int32Dtype(), 'domain_residue': pd.StringDtype(), 'protein_residue': pd.StringDtype()})
        reference_mapping_notna = self.reference_mapping.dropna()
        
        self.domain_to_protein = dict(zip(reference_mapping_notna.domain_index, reference_mapping_notna.protein_index))
        self.protein_to_domain = dict(zip(reference_mapping_notna.protein_index, reference_mapping_notna.domain_index))

    def _set_reference_mapping(self, domain_start: int, protein_start: int, domain_text: str, protein_text: str) -> None:
        """
        Set values for reference_mapping and mapping dictionaries, domain_to_protein and protein_to_domain.

        Note
        ----
        For details on `domain_start`, `protein_start`, `domain_text`, `protein_text`, please refer to the `ResidueAlignment` docstring.

        Returns
        -------
        None
        """
        invalid_chars = [".", "_", "-"]
        # Convert text to list variant for iteration
        domain_sequence = list(domain_text)
        protein_sequence = list(protein_text)
        # Store mapping values per iteration here.
        mapping_entries = []

        for domain_aa, protein_aa in zip(domain_sequence, protein_sequence):
            mapping_entry = []
            # Check to see if domain residue is valid, if so, we can assign the proper index.
            if domain_aa not in invalid_chars:
                mapping_entry.append(domain_start)
                domain_start += 1
            else:
                mapping_entry.append(pd.NA)
            # Assign the values of the residues mapped together.
            mapping_entry.append(domain_aa)
            mapping_entry.append(protein_aa)
            # Check to see if protein residue is valid, if so, we can assign the proper index.
            if protein_aa not in invalid_chars:
                mapping_entry.append(protein_start)
                protein_start += 1
            else:
                mapping_entry.append(pd.NA)
            # Store the resulting mapping in the reference map.
            mapping_entries.append(mapping_entry)
        self.reference_mapping = pd.DataFrame(mapping_entries, columns=['domain_index', 'domain_residue', 'protein_residue', 'protein_index'])
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

class DirectInformationData:
    """
    Representation and interface for Direct Information data including residue indices for a pair and its corresponding DI value represented as a 3-column ndarray.

    Parameters
    ----------
    structured_ndarray : numpy.ndarray
        Ndarray with the shape (n,3) with dtype={'names': ('residue1', 'residue2', 'DI'), 'formats': (int, int, float, float)}

    Attributes
    ----------
    DI_data : numpy.ndarray
        The structured_ndarray in the parameters section where column 1 corresponds to a pair's first residue, column 2 corresponds to the pair's second residue, and column 3 corresponds to the Direct Information of the pair.
    """
    def __init__(self, structured_ndarray: npt.NDArray) -> None:
        self.DI_data = structured_ndarray

    @staticmethod    
    def load_from_dca_output(dca_filepath: str) -> 'DirectInformationData':
        """
        Function to generate a DirectInformationData object from the direct output of the MATLab dca function.

        Parameters
        ----------
        dca_filepath : str
            Filepath of the DCA output to be read and compiled into a structured ndarray. DCA output is a 4 column text file with the following columns: (residue 1, residue 2, Mutual Information, Direct Information).

        Returns
        -------
        DirectInformationData
            DirectInformationData object with named structured array containing residue indices and the DI value of the pair.
        """
        file_data = np.loadtxt(dca_filepath, dtype={'names': ('residue1', 'residue2', 'MI', 'DI'), 'formats': (int, int, float, float)})
        return DirectInformationData(file_data[['residue1', 'residue2', 'DI']])

    @staticmethod
    def load_from_DI_file(DI_filepath: str) -> 'DirectInformationData':
        """
        Function to generate a DirectInformationData object from the modified DI-only version of the DCA output generated via the MATLab dca function.

        Parameters
        ----------
        DI_filepath : str
            Filepath of the DI file to be read and compile into a structured ndarray. DI file is a 3 column text file with the following columns: (residue 1, residue 2, Direct Information).
        
        Returns
        -------
        DirectInformationData
            DirectInformationData object with named structured array containing residue indices and the DI value of the pair.
        """
        return DirectInformationData(np.loadtxt(DI_filepath, dtype={'names': ('residue1', 'residue2', 'DI'), 'formats': (int, int, float)}))

    @staticmethod
    def load_as_ndarray(ndarray: Union[npt.NDArray, Iterable[Iterable]]) -> 'DirectInformationData':
        """
        Function to generate a Direct Information object from a ndarray.

        Parameters
        ----------
        ndarray : numpy.ndarray or Iterable of Iterable (excluding dict)
            An ndarray of shape (n,3) where its columns are (residue 1, residue 2, and Direct Information). Can also be parsed from an iterable of iterable provided that the aforementioned format is followed.

        Returns
        -------
        DirectInformationData
            DirectInformationData object with named structured array containing residue indices and the DI value of the pair.
        """
        
        if isinstance(ndarray, np.ndarray) and ndarray.shape[1] != 3:
            raise Exception("Dimensions of numpy array supplied are different from what is expected. Please supply residue1, residue2, and DI column in int, int, float format and with shape of (n, 3).")
        # Structured ndarrays require list of tuples for conversion.
        DI_data = np.array([tuple(x) for x in ndarray], dtype={'names': ('residue1', 'residue2', 'DI'), 'formats': (int, int, float)})
        
        return DirectInformationData(DI_data)
    
    def get_ranked_mapped_pairs(self, RA1: ResidueAlignment, RA2: ResidueAlignment, pairs_only: bool=True, mirror: bool=False, number: Optional[int]=None) -> npt.NDArray:
        """
        Uses DirectInformationData and Pairs interface methods to obtain ranked, mapped residues. See rank_pairs() function and map_DIs() function for details on rank and mapping. Residue Alignments can be the same for intra-domain / intra-protein mapping.

        Parameters
        ----------
        RA1 : ResidueAlignment
            The ResidueAlignment used for mapping the first column of residues to the appropriate target sequence.
        RA2 : ResidueAlignment
            The ResidueAlignment used for mapping the second column of residues to the appropriate target sequence.
        pairs_only : bool
            True if the final ndarray should contain only columns 1 and 2, corresponding to the residues that constitute the pair. This would drop the DI column.
        mirror : bool
            See Pairs.mirror_pairs() or get_pairs() for details. NOTE: This option is overriden entirely if pairs_only is False. If true, this will produce an ndarray that has the original residue indices and repeated residue indices but with residue 1 and residue 2 switched. This is useful for plotting across the upper diagonal of a contact map.
        number : int, None
            Number of ranked, mapped pairs to return.

        Returns
        -------
        numpy.ndarray
            Structured ndarray with columns residue 1, residue 2 and optionally DI. Only has specified number of pairs if `number` is specified and mirrored pairs if `mirror` is True and pairs_only is False.
        
        Notes
        -----
        ResidueAlignments contain dictionaries like domain_to_protein to map residues produced via Direct Coupling Analysis (DCA) on an MSA generated in context to an HMM. The residues are mapped to a protein structure via alignment of the HMM hit / domain to the protein sequence.
        """
        ranked_pairs = DirectInformationData.rank_pairs(DirectInformationData.nonlocal_pairs(self.DI_data))
        ranked_mapped_pairs = DirectInformationData.map_DIs(ranked_pairs, RA1, RA2)
        if pairs_only:
            return Pairs.get_pairs(ranked_mapped_pairs[['residue1', 'residue2']], mirror=mirror, number=number)
        else:
            return Pairs.get_pairs(ranked_mapped_pairs, mirror=False, number=number)
    
    @staticmethod
    def map_DIs(DI_data : npt.NDArray, RA1: ResidueAlignment, RA2: ResidueAlignment) -> npt.NDArray:
        """
        Uses domain-to-protein mappings present in the Residue Alignments provided to generate mapped representations of the residues from the DI_data structured ndarray provided.

        Parameters
        ----------
        DI_data : numpy.ndarray
            Structured ndarray that contains columns "residue1" and "residue2".
        RA1 : ResidueAlignment
            The ResidueAlignment used for mapping the first column of residues to the appropriate target sequence.
        RA2 : ResidueAlignment
            The ResidueAlignment used for mapping the second column of residues to the appropriate target sequence.
        
        Returns
        -------
        mappable_DI_data : numpy.ndarray
            DI_data that has been mapped to the target sequence specified in the generation of the corresponding ResidueAlignments.

        Note:
            Residues that do not map to the target sequence of the ResidueAlignment are dropped.
        """
        # Alternative approach is to just use get function instead of [x] and default to np.nan and drop nans row-wise.
        mapping_key_mask = (np.isin(DI_data['residue1'], list(RA1.domain_to_protein.keys()))) & (np.isin(DI_data['residue2'], list(RA2.domain_to_protein.keys())))
        mappable_DI_data = DI_data[mapping_key_mask]
        if len(mappable_DI_data) == 0:
            return mappable_DI_data
        else:
            mappable_DI_data['residue1'] = np.vectorize(lambda x: RA1.domain_to_protein[x])(mappable_DI_data['residue1'])
            mappable_DI_data['residue2'] = np.vectorize(lambda x: RA2.domain_to_protein[x])(mappable_DI_data['residue2'])
            return mappable_DI_data
    
    @staticmethod
    def rank_pairs(DI_data: npt.NDArray) -> npt.NDArray:
        """
        Sorts a structured ndarray of pairs information to order the ndarray by Direct Information (DI) score.

        Parameters
        ----------
        DI_data : numpy.ndarray
            Structured ndarray that contains columns with names "residue1", "residue2", and "DI" (Direct Information)
        
        Returns
        -------
        numpy.ndarray
            Structured ndarray sorted upon column that is named DI in descending order.
        """
        # [::-1] reverses the order from ascending DI Score to descending DI score.
        return np.sort(DI_data, order='DI')[::-1]
    
    @staticmethod
    def nonlocal_pairs(DI_data: npt.NDArray) -> npt.NDArray:
        """
        Subsets a structured ndarray of pairs information to find nonlocal pairs, where residue interactions are likely not involved in secondary structure formation i.e. helices and sheet interactions. Nonlocal pairs must be at least 4 residues apart.

        Parameters
        ----------
        DI_data : numpy.ndarray
            Structured ndarray that contains (at least) the first and second columns with the names "residue1" and "residue2" respectively.
        
        Returns
        -------
        numpy.ndarray
            Structured ndarray of DI pairs where residue 1 and residue 2 are at least 4 residues apart.
        """
        return DI_data[abs(DI_data['residue1'] - DI_data['residue2']) > 4]
    
    @staticmethod
    def find_DI_with_residues(critical_residues_1 : Iterable[int], critical_residues_2 : Iterable[int], max_rank: Optional[int]=None, *mapped_resi_arrs: Iterable[npt.NDArray]) -> list[tuple[list, int]]:
        """
        Function that takes an n number of ranked, mapped DI pairs and checks to see if they're in a list of potential residue indices.
        
        Parameters
        ----------
        critical_residues_1 : collections.abc.Iterable of int
            Specific residue indices that a DI pair will be compared to. If the first residue of the DI pair is not one of these indices, it will not be appended to results.
        crtical_residues_2 : collections.abc.Iterable of int
            Specific residue indices that a DI pair will be compared to. If the second residue of the DI pair is not one of these indices, it will not be appended to results.
        threshold : int, optional
            Maximum "rank" of the DI pair considered.
        *mapped_resi_arrs : tuple of numpy.ndarray
            Tuple of ranked, mapped pairs that are compared to critical residue indices and appended to results if in those indices and within threshold.
        
        Returns
        -------
        results : list of tuple of list of int, int
            Results which consist of tuples where the first element is a list of residue1, residue2, and DI score, whereas the second element is the rank.
        """
        results = []
        for mapped_resi_arr in mapped_resi_arrs:
            # count_rank represents the rank of the DI pair being evaluated, iterating over every new row considered.
            count_rank = 0
            for row in mapped_resi_arr:
                row_as_list = list(row)
                count_rank += 1
                if max_rank:
                    if row_as_list[0] in critical_residues_1 and row_as_list[1] in critical_residues_2 and count_rank <= max_rank:
                        results.append((row_as_list, count_rank))
                else:
                    if row_as_list[0] in critical_residues_1 and row_as_list[1] in critical_residues_2:
                        results.append((row_as_list, count_rank))
        return results

    @staticmethod
    def get_dist_commands(model1: str | int, model2: str | int, chain1: str, chain2: str, pairs: npt.NDArray, ca_only: bool=True, auth_res_ids: bool=False) -> list[str]:
        """
        Get UCSF Chimera commands for displaying distance commands for usage in displaying distances between residue pairs. Options are present for alpha-carbon to alpha-carbon distance or for specified atom to specified atom distance.
        
        Parameters
        ----------
        model1 : str, int
            Number of model corresponding to the structure containing the first column of residues.
        model2 : str, int
            Number of model corresponding to the structure containing the second column of residues.
        chain1 : str
            The chain present in the structure in model1 containing the first column of residues.
        chain2 : str
            The chain present in the structure in model2 containing the second column of residues.
        pairs : numpy.ndarray
            Structured ndarray that contains (at least) the first and second columns with the names "residue1" and "residue2" respectively. `ca_only` should be set to false and an additional "atom_name1" and "atom_name2" column should be added if atoms are specified per pair.
        ca_only : bool
            True if distance commands are for displaying distances between the two alpha-carbons of the residue pair. If false, specific_atom_names is used in lieu of "CA" as an atom identifier.
        auth_res_ids : bool
            If true, use the "auth_residue1" and "auth_residue2" columns instead of "residue1" and "residue2" columns. These residue ids correspond to the auth protein residue ids.
        
        Returns
        -------
        distance_commands : list of str
            List of distance commands generated between two residues with model and chain information needed, either between two alpha-carbons or the specified atoms.
        
        Note
        ----
        model1 and model2 can be equivalent if both columns involve residues referenced by the same model. The same would apply for chains if the residues are present on the same chain. 
        """
        distance_commands: list[str] = []
        for i in range(np.shape(pairs)[0]):
            if auth_res_ids:
                residue1 = pairs[i]['auth_residue1']
                residue2 = pairs[i]['auth_residue2']
            else:
                residue1 = pairs[i]['residue1']
                residue2 = pairs[i]['residue2']
            if ca_only:
                distance_commands.append(f"distance #{model1}:{residue1}.{chain1}@CA #{model2}:{residue2}.{chain2}@CA;")
            else:
                atom1 = pairs[i]['atom_name1']
                atom2 = pairs[i]['atom_name2']
                distance_commands.append(f"distance #{model1}:{residue1}.{chain1}@{atom1} #{model2}:{residue2}.{chain2}@{atom2};")
        return distance_commands

    @staticmethod
    def write_DI_data(filepath: str, pairs: npt.NDArray, delimiter: str="\t", fmt: tuple[str, str, str] | tuple[str, str]=('%d', '%d', '%.3f')) -> None:
        """
        Writes pairs ndarray to file with specified delimiter between the pairs' row elements, i.e. residue 1, residue 2, and DI score.

        Parameters
        ----------
        filepath : str
            Path of the file to write DirectInformation data to.
        pairs : numpy.ndarray
            Ndarray of at-least pairs information (residue 1, residue 2) and optionally Direct Information to write to a file via numpy.savetxt().
        delimiter : str
            Delimiter to separate columns of the pairs ndarray when writing to a file.
        fmt : tuple of str, default=('%d', '%d', '%.3f)
            format passed as an argument to numpy.savetxt() to define type of column and output format. Set to ('%d', '%d') if only two columns are present in the ndarray.

        Returns
        -------
        None
        """
        if len(pairs[0]) == 2:
            fmt = ('%d', '%d')
        np.savetxt(filepath, pairs, delimiter=delimiter, fmt=fmt)

class StructureInformation:
    """
    Information regarding a protein structure, obtained from a protein structure file.

    Uses fetch_pdb() to pull protein structure information from RCSB. Uses read_x_file() to supply a filepath to pull protein structure information from a file.
    """
    @overload
    @staticmethod
    def fetch_pdb(pdb_id: str, struc_format: Literal["mmcif"], model_num: int=1) -> 'MMCIFInformation':
        ...

    @overload
    @staticmethod
    def fetch_pdb(pdb_id: str, struc_format: Literal["pdb"], model_num: int=1) -> 'PDBInformation':
        ...


    @staticmethod
    def fetch_pdb(pdb_id: str, struc_format: Literal["mmcif", "pdb"]="mmcif", model_num: int=1) -> Union['MMCIFInformation', 'PDBInformation']:
        """
        Fetches PDB as mmCIF file from RCSB and compiles the information into a StructureInformation instance.

        Parameters
        ----------
        pdb_id : str
            PDB ID to be fetched from the RCSB database.
        struc_format : str
            The format of the file to pull from the RCSB database.
        model_num : int
            The model number to access from the PDB to ensure an AtomArray is returned containing the atom information of the protein structure.
            
        Returns
        -------
        StructureInformation
            StructureInformation generated from pdbx.get_structure() function using the pdbx file fetched from RCSB.
        
        Raises
        ------
        TypeError
            Fetched data was not found and returned None instead.
        ValueError
            Structure format may be invalid (not PDBx/mmCIF or PDB).
        """
        fetched_data = rcsb.fetch(pdb_id, struc_format)
        if fetched_data is None:
            raise TypeError("RCSB fetch failed. Try fetch again.")
        elif struc_format == "mmcif":
            pdbx_file = pdbx.CIFFile.read(fetched_data)
            return MMCIFInformation(pdbx.get_structure(pdbx_file=pdbx_file, model=model_num, use_author_fields=False), pdbx_file, model_num)
        elif struc_format == "pdb":
            pdb_file = pdb.PDBFile.read(fetched_data)
            return PDBInformation(pdb.get_structure(pdb_file=pdb_file, model=model_num), pdb_file=pdb_file, model_num=model_num)
        else:
            raise ValueError(f"struc_format {struc_format} is not valid or currently supported by DCA Toolkit")
    @staticmethod
    def read_mmCIF_file(pdbx_filepath: str, model_num: int=1) -> 'MMCIFInformation':
        """
        Reads PDB mmCIF file from filepath and compiles the information into a CIFInformation instance.

        Parameters
        ----------
        pdbx_filepath : str
            Filepath of the PDB mmCIF file to be read.
        model_num : int
            The model number to access from the PDB to ensure an AtomArray is returned containing the atom information of the protein structure.

        Returns
        -------
        CIFInformation
            CIFInformation generated from pdbx.get_structure() function using the PDBx file read from the pdbx_filepath.
        """
        pdbx_file = pdbx.CIFFile.read(pdbx_filepath)
        return MMCIFInformation(pdbx.get_structure(pdbx_file, model=model_num, use_author_fields=False), pdbx_file, model_num)
    
    @staticmethod
    def read_pdb_file(pdb_filepath: str, model_num: int=1) -> 'PDBInformation':
        """
        Reads PDB file from filepath and compiles the information into a PDBInformation instance.

        Parameters
        ----------
        pdb_filepath : str
            Filepath of the PDB mmCIF file to be read.
        model_num : int
            The model number to access from the PDB to ensure an AtomArray is returned containing the atom information of the protein structure.

        Returns
        -------
        PDBInformation
            PDBInformation generated from pdb.get_structure() function using the PDB file read from the pdb_filepath.
        """
        pdb_file = pdb.PDBFile.read(pdb_filepath)
        return PDBInformation(pdb.get_structure(pdb_file, model=model_num), pdb_file, model_num)
    
    @staticmethod
    def write_contacts_set(filepath : str, contacts_set : set[tuple[int, int]]) -> None:
        """
        Write the contacts generated from get_contacts or general set of tuples of pairs.

        Parameters
        ----------
        filepath : str
            Path of file to output contacts_set to.
        contacts_set : set of tuple of int, int
            Set of tuples of pairs that represent contacts.
        
        Returns
        -------
        None
        """
        contacts_list = list(sorted(contacts_set))
        with open(filepath, 'w') as fs:
            for pair in contacts_list:
                fs.write(str(pair[0]) + "\t" + str(pair[1]) + "\n")

class MMCIFInformation(StructureInformation):
    """
    Information regarding a protein structure, obtained from a protein structure file.

    Parameters
    ----------
    structure : biotite.structure.AtomArray
        Structure obtained from an RCSB entry with a provided pdbx/mmcif file with a specified model number.
    pdbx_file : biotite.io.pdbx.CIFFile
        mmCIF file that contains generic information and atomic information of the protein structure categorized into mmCIF blocks.
    model_num : int
        The model number to access from the PDB to ensure an AtomArray is returned containing the atom information of the protein structure.

    Attributes
    ----------
    self.full_sequences : dict of str, biotite.sequence.ProteinSequence
        The full protein sequences from the pdbx file used to generate the structure stored in a dictionary where auth_chain_id is the key and the ProteinSequence object is the value.
    self.non_missing_sequences : dict of str, biotite.sequence.ProteinSequence
        The protein sequences, without missing residues, compiled in the structure of the StructureInformation instance stored in a dictionary where chain_id is the key and the sequence string is the value.
    self.atom_data : numpy.ndarray, optional
        Entries in the format 'ATOM', residue index, chain ID, auth residue index, auth chain ID, model number
    self.het_atom_data : numpy.ndarray, optional
        Array of entries in the format 'HETATM', residue index, chain ID, auth residue index, auth chain ID, model number
    self.unique_chains : numpy.ndarray, optional
        Array of unique asym_id entries which corresponds to unique chain IDs.
    self.chain_auth_dict : dict of str, str, optional
        Uses chain id as a key and provides auth chain id as a value.
    self.auth_chain_dict : dict of str, str, optional
        Uses auth chain id as a key and provides original/label chain id as a value. 
    self.res_auth_dict : dict of str, tuple of int, int or optional 
        Uses chain id as a key and an array of residue index and auth residue index as a value.
    """
    def __init__(self, structure, pdbx_file: pdbx.CIFFile, model_num: int):
        self.structure = structure
        self.pdbx_file = pdbx_file
        self.model_num = model_num
        self.full_sequences = pdbx.get_sequence(pdbx_file)
        non_hetero_structure = self.structure[self.structure.hetero == False]
        self.non_missing_sequences = {str(chain): str(sequence) for (chain, sequence) in list(zip(struc.get_chains(non_hetero_structure), struc.to_sequence(non_hetero_structure)[0]))}
        self._generate_auth_info()

    def _generate_auth_info(self) -> None:
        """
        Ran as part of constructor function. Generates information needed to access auth information including auth_seq_id and auth_asym_id, which correspond to alternative chain ids and alternative residue indices.
        
        Note
        ----
        See attributes for details.
        
        Returns
        -------
        None
        """
        if len(self.pdbx_file.keys()) > 0:
            self.first_block = list(self.pdbx_file)[0]
            atom_site_category = self.pdbx_file[self.first_block].get('atom_site')
            self.chain_auth_dict: dict[str, str] = {}
            self.auth_chain_dict: dict[str, str] = {}
            if atom_site_category:
                categories = ['group_PDB', 'label_seq_id', 'label_asym_id', 'auth_seq_id', 'auth_asym_id', 'pdbx_PDB_model_num']
                atom_site_data = np.column_stack([atom_site_category[category].as_array() for category in categories])
                _, idx = np.unique(atom_site_data, axis=0, return_index=True)
                atom_site_data = atom_site_data[np.sort(idx)]
                atom_data = atom_site_data[atom_site_data[:,0] == "ATOM"]
                self.unique_chains = np.unique(atom_data[:,2])
                for unique_chain in self.unique_chains:
                    unique_entry = atom_data[atom_data[:,2] == unique_chain][0]
                    self.chain_auth_dict[unique_entry[2]] = unique_entry[4]
                    self.auth_chain_dict[unique_entry[4]] = unique_entry[2]
                self.atom_site_df = pd.DataFrame(np.column_stack([atom_site_category[category].as_array() for category in atom_site_category.keys()]), columns=atom_site_category.keys())
                type_conversion_dict = {'label_seq_id': 'int64', 'auth_seq_id': 'int64', 'id': 'int64', 'Cartn_x': 'float', 'Cartn_y': 'float','Cartn_z': 'float', 'B_iso_or_equiv': 'float'}
                self.atom_df = self.atom_site_df[self.atom_site_df['group_PDB'] == 'ATOM'].astype(type_conversion_dict)

    def get_start_res_id(self, chain_id: str, get_auth_res_ids: bool=False, auth_chain_id_supplied: bool=False) -> int:
        """
        Gets starting residue id of the specified chain excluding heteroatom group entries.

        Parameters
        ----------
        chain_id : str
            The chain id supplied and selected for from the structure.
        get_auth_res_ids : bool
            True if you want alt_ids for residues indices, False if cif residue indexing is needed.
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.            

        Returns
        -------
        int
            The residue id of the first atom in the chain provided.
        """
        if auth_chain_id_supplied:
            chain_df = self.atom_df[self.atom_df['auth_asym_id'] == chain_id]
        else:
            chain_df = self.atom_df[self.atom_df['label_asym_id'] == chain_id]
        if get_auth_res_ids:
            return chain_df['auth_seq_id'][0]
        else:
            return chain_df['label_seq_id'][0]

    def get_full_sequence(self, chain_id: str, auth_chain_id_supplied: bool=False) -> str:
        """
        Get full sequence, including missing residues, from the specified chain off of RCSB.

        Parameters
        ----------
        chain_id : str
            Chain id supplied. The full sequence, including missing residues, of this chain will be returned.
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.

        Returns
        -------
        str
            The full sequence, including missing residues, of the chain specified.
        """
        if auth_chain_id_supplied:
            return str(self.full_sequences[chain_id])
        else:
            return str(self.full_sequences[self.chain_auth_dict[chain_id]])
        
    def get_non_missing_sequence(self, chain_id: str, auth_chain_id_supplied: bool=False) -> str:
        """
        Get sequence, including only non-missing residues, from the specified chain off of RCSB.

        Parameters
        ----------
        chain_id : str
            Chain id supplied. The full sequence, including only non-missing residues, of this chain will be returned.
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.

        Returns
        -------
        str
            The full sequence, including only non-missing residues, of the chain specified.
        """
        if auth_chain_id_supplied:
            original_chain_id = self.auth_chain_dict[chain_id]
            return self.non_missing_sequences[original_chain_id]
        else:
            return self.non_missing_sequences[chain_id]
        
    def get_chain_specific_structure(self, ca_only: bool, chain_id: str, remove_hetero=True, auth_chain_id_supplied: bool=False):
        """
        Subsets structure attribute to select for chain specific portions of the structure.

        Parameters
        ----------
        ca_only : bool
            If true, the structure will also be subsetted for atom entries where the atom_name annotation is "CA" (referring to alpha-carbons)
        chain_id : str
            The name of the chain to be selected for within the structure.
        remove_hetero : bool, default=True
            If true, the structure will also be subsetted for atom entries where the hetero annotation is False, thus removing heteroatoms.
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.

        Returns
        -------
        tuple of biotite.structure.AtomArray, biotite.structure.AtomArray
            Two AtomArrays that refer to atoms in the first chain and second chain, respectively without accounting for the presence of heteroatoms if `remove_hetero` is True.
        """
        if auth_chain_id_supplied:
            chain_id = self.auth_chain_dict[chain_id]
        selected_structure = self.structure
        if remove_hetero:
            # Remove hetero atoms via hetero column of structure ndarray
            selected_structure = self.structure[self.structure.hetero == False]
        if ca_only:
            # Consider selection of alpha-carbon atoms only
            selected_structure = selected_structure[selected_structure.atom_name == "CA"]
        chain_structure = selected_structure[selected_structure.chain_id == chain_id]
        return chain_structure
    
    def get_chain_site_data(self, ca_only: bool, chain_id: str, remove_hetero=True, auth_chain_id_supplied: bool=False):
        """
        Subsets the atom_site dataframe to get atom information where the conditions are met.

        Parameters
        ----------
        ca_only : bool
            If true, the dataframe will also be subsetted for atom entries where the label_atom_id annotation is "CA" (referring to alpha-carbons)
        chain_id : str
            The name of the chain to be selected for within the dataframe.
        remove_hetero : bool, default=True
            If true, the dataframe will also be subsetted for atom entries where the group_PDB annotation is ATOM rather than HETATM, thus removing heteroatoms.
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.
        """
        atom_df = self.atom_df.copy()
        if ca_only:
            atom_df = atom_df[atom_df['label_atom_id'] == 'CA']
        if remove_hetero:
            atom_df = atom_df[atom_df['group_PDB'] == 'ATOM']
        if auth_chain_id_supplied:
            return atom_df[atom_df['auth_asym_id'] == chain_id]
        else:
            return atom_df[atom_df['label_asym_id'] == chain_id]
        
    def get_seq_id_mapping(self, chain_id: str, seq_to_auth: bool, auth_chain_id_supplied: bool=False) -> dict[int, int]:
        """
        Gets mapping from auth seq ids to label seq ids or vice-versa.

        Parameters
        ----------
        chain_id : str
            Chain id of the chain addressed for determining residue index mappings.
        seq_to_auth : bool
            If True, this indicates the mapping uses the label_seq_id as a key and the auth_seq_id as a value. Otherwise, keys and values are switched.
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.

        Returns
        -------
        dict of int, int
            Dictionary with either label seq id or auth seq id as a key and the other as a value. The directionality is dependent on seq_to_auth.
        """
        chain_df = self.get_chain_site_data(ca_only=True, chain_id=chain_id, remove_hetero=True, auth_chain_id_supplied=auth_chain_id_supplied)
        if seq_to_auth: 
            return dict(zip(chain_df['label_seq_id'], chain_df['auth_seq_id']))
        else:
            return dict(zip(chain_df['auth_seq_id'], chain_df['label_seq_id']))
    
    def get_valid_chain_residues(self, chain_id: str, auth_seq_id: bool=False, auth_chain_id_supplied: bool=False) -> list[tuple[int, str]]:
        """
        Gets valid indexing for residues of a specified chain. This is directly analogous to get_non_missing_sequence, does not contain missing residues, and provides the corresponding indices as well.

        Parameters
        ----------
        chain_id : str
            Chain id of the chain to be selected from the structure. This chain's sequence and corresponding residue indices are what are exclusively selected for.
        auth_seq_id: bool
            If True, the seq_ids that are the first element of the tuples in the returned list are auth_seq_ids. 
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.
        
        Returns
        -------
        list of tuple of int, str
            A list of residue information in sequential order reflecting the structure. The list consists of tuple elements where each tuple is the residue index and its corresponding one-letter amino acid.
        """
        chain_structure = self.get_chain_specific_structure(ca_only=True, chain_id=chain_id, remove_hetero=True, auth_chain_id_supplied=auth_chain_id_supplied)
        res_ids = chain_structure.res_id.tolist()
        res_names = chain_structure.res_name
        if auth_seq_id:
            seq_id_mapping = self.get_seq_id_mapping(chain_id=chain_id, seq_to_auth=True, auth_chain_id_supplied=auth_chain_id_supplied)
            auth_res_ids = [seq_id_mapping[res_id] for res_id in res_ids]
            return list(zip(auth_res_ids, map(lambda symbol: ProteinSequence.convert_letter_3to1(symbol), res_names)))
        else:
            return list(zip(res_ids, map(lambda symbol: ProteinSequence.convert_letter_3to1(symbol), res_names)))

    def generate_dist_matrix(self, ca_only: bool, chain1: str, chain2: str, auth_chain_id_supplied: bool=False):
        """
        Generates distance matrix between two chains in the structure attribute.

        Parameters
        ----------
        ca_only : bool
            If True, only atoms that have the name "CA" are selected in the chains the distance matrix is calculated between.
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the first column of residues in the structure.
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.

        Returns
        -------
        tuple of biotite.structure.AtomArray, biotite.structure.AtomArray, numpy.ndarray
            Tuple containing the chain 1 structure, the chain 2 structure, and the distance matrix of chain 1 and chain 2's pairwise distances.
        """
        chain1_structure = self.get_chain_specific_structure(ca_only=ca_only, chain_id=chain1, remove_hetero=True, auth_chain_id_supplied=auth_chain_id_supplied)
        chain2_structure = self.get_chain_specific_structure(ca_only=ca_only, chain_id=chain2, remove_hetero=True, auth_chain_id_supplied=auth_chain_id_supplied)
        dist_matrix = cdist(chain1_structure.coord, chain2_structure.coord)
        return (chain1_structure, chain2_structure, dist_matrix)

    def get_min_dist_atom_info(self, pairs: npt.NDArray, chain1: str, chain2: str, auth_chain_id_supplied: bool=False) -> npt.NDArray:
        """
        Generate a ndarray of residue ids and their corresponding atom names such that the distance is the minimum between the initial residues provided.

        Parameters
        ----------
        pairs : numpy.ndarray
            Pairs structured ndarray with "residue1" and "residue2" columns.
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the second column of residues in the structure.
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.
            
        Returns
        -------
        min_dist_pairs_atoms_arr : numpy.ndarray
            Structured ndarray that has residue indices, auth residue indices (corresponding to the protein numbering), and atomic names in the format {'names': ['residue1','residue2','auth_residue1','auth_residue2','atom_name1','atom_name2'], 'formats': [int,int,int,int,'<U10','<U10']}
        """
        chain1_structure = self.get_chain_specific_structure(ca_only=False, chain_id=chain1, remove_hetero=True, auth_chain_id_supplied=auth_chain_id_supplied)
        chain2_structure = self.get_chain_specific_structure(ca_only=False, chain_id=chain2, remove_hetero=True, auth_chain_id_supplied=auth_chain_id_supplied)
        min_dist_pairs_atoms = []
        for row in pairs:
            # Obtain structure information for chains 1 and 2
            chain1_res1_structure = chain1_structure[chain1_structure.res_id == row['residue1']]
            chain2_res2_structure = chain2_structure[chain2_structure.res_id == row['residue2']]
            
            # Calculate a distance matrix and find the indices of the minimal value in the matrix
            dist_matrix = cdist(chain1_res1_structure.coord, chain2_res2_structure.coord)
            
            ind = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            # Use the indices to access the atom in the atom array and get the correct atom name.
            # Generate the auth ids of the residues in the pairs ndarray
            seq_mapping_chain1 = self.get_seq_id_mapping(chain_id=chain1, seq_to_auth=True, auth_chain_id_supplied=auth_chain_id_supplied)
            seq_mapping_chain2 = self.get_seq_id_mapping(chain_id=chain2, seq_to_auth=True, auth_chain_id_supplied=auth_chain_id_supplied)
            auth_res_id1 = seq_mapping_chain1[row['residue1']]
            auth_res_id2 = seq_mapping_chain2[row['residue2']]
            min_dist_pairs_atoms.append((row['residue1'], row['residue2'], auth_res_id1, auth_res_id2, chain1_res1_structure[ind[0]].atom_name, chain2_res2_structure[ind[1]].atom_name))
        min_dist_pairs_atoms_arr = np.array(min_dist_pairs_atoms, dtype={'names': ['residue1','residue2','auth_residue1','auth_residue2','atom_name1','atom_name2'], 'formats': [int,int,int,int,'<U10','<U10']})
        return min_dist_pairs_atoms_arr

    def get_contacts(self, ca_only: bool, threshold: float, chain1: str, chain2: str, auth_seq_id: bool=False, auth_chain_id_supplied: bool=False) -> set[tuple[int, int]]:
        """
        Get contacts from the structure attribute where the distance between two residues is less than the threshold.

        Parameters
        ----------
        ca_only : bool
            If true, only consider alpha-carbon to alpha-carbon distances. 
        threshold : float
            Maximum distance to consider between two atoms.
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the second column of residues in the structure.
        auth_seq_id : bool
            True if you want auth_seq_ids for residues indices, False if cif residue indexing is needed.
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.

        Returns
        -------
        contacts_set : set of tuple of ints
            Set of contacts, tuples with "residue1" and "residue2" from the structure that are within the distance threshold.
        """
        chain1_structure, chain2_structure, dist_matrix = self.generate_dist_matrix(ca_only, chain1, chain2, auth_chain_id_supplied=auth_chain_id_supplied)
        seq_mapping_chain1 = self.get_seq_id_mapping(chain_id=chain1, seq_to_auth=True, auth_chain_id_supplied=auth_chain_id_supplied)
        seq_mapping_chain2 = self.get_seq_id_mapping(chain_id=chain2, seq_to_auth=True, auth_chain_id_supplied=auth_chain_id_supplied)
        thresh_ind = np.argwhere(dist_matrix <= threshold)
        contacts_set = set()
        for indices in thresh_ind:
            chain1_atom = chain1_structure[indices[0]]
            chain2_atom = chain2_structure[indices[1]]
            res1 = chain1_atom.res_id
            res2 = chain2_atom.res_id
            if not(chain1==chain2 and res1 >= res2):
                if auth_seq_id:
                    contacts_set.add((seq_mapping_chain1[res1], seq_mapping_chain2[res2]))
                else:
                    contacts_set.add((res1, res2))
        return contacts_set

class PDBInformation(StructureInformation):
    """
    Information regarding a protein structure, obtained from a protein structure file.

    Parameters
    ----------
    structure : biotite.structure.AtomArray
        Structure obtained from an RCSB entry with a provided pdbx/mmcif file with a specified model number.
    pdb_file : biotite.io.pdb.PDBFile
        mmCIF file that contains generic information and atomic information of the protein structure categorized into mmCIF blocks.
    model_num : int
        The model number to access from the PDB to ensure an AtomArray is returned containing the atom information of the protein structure.

    Attributes
    ----------
    self.non_missing_sequences : dict of str, biotite.sequence.ProteinSequence
        The protein sequences, without missing residues, compiled in the structure of the StructureInformation instance stored in a dictionary where chain_id is the key and the sequence string is the value.
    
    """
    def __init__(self, structure, pdb_file: pdb.PDBFile, model_num: int):
        self.structure = structure
        self.pdb_file = pdb_file
        self.model_num = model_num
        non_hetero_structure = self.structure[self.structure.hetero == False]
        self.non_missing_sequences = {str(chain): str(sequence) for (chain, sequence) in list(zip(struc.get_chains(non_hetero_structure), struc.to_sequence(non_hetero_structure)[0]))}
        self.unique_chains = struc.get_chains(non_hetero_structure)

    def get_start_res_id(self, chain_id: str) -> int:
        """
        Gets starting residue id of the specified chain excluding heteroatom group entries.

        Parameters
        ----------
        chain_id : str
            The chain id supplied and selected for from the structure.

        Returns
        -------
        int
            The residue id of the first atom in the chain provided.
        """
        non_hetero_structure = self.structure[self.structure.hetero == False]
        if chain_id in self.unique_chains:
            return non_hetero_structure[non_hetero_structure.chain_id == chain_id][0].res_id
        else:
            raise ValueError("Chain supplied not found in structure.")

    def get_non_missing_sequence(self, chain_id: str) -> str:
        """
        Get sequence, including only non-missing residues, from the specified chain.

        Parameters
        ----------
        chain_id : str
            Chain id supplied. The full sequence, including only non-missing residues, of this chain will be returned.

        Returns
        -------
        str
            The full sequence, including only non-missing residues, of the chain specified.
        """
        return self.non_missing_sequences[chain_id]
    
    def get_chain_specific_structure(self, ca_only: bool, chain_id: str, remove_hetero=True):
        """
        Subsets structure attribute to select for chain specific portions of the structure.

        Parameters
        ----------
        ca_only : bool
            If true, the structure will also be subsetted for atom entries where the atom_name annotation is "CA" (referring to alpha-carbons)
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the second column of residues in the structure.
        remove_hetero : bool, default=True
            If true, the structure will also be subsetted for atom entries where the hetero annotation is False, thus removing heteroatoms.

        Returns
        -------
        tuple of biotite.structure.AtomArray, biotite.structure.AtomArray
            Two AtomArrays that refer to atoms in the first chain and second chain, respectively without accounting for the presence of heteroatoms if `remove_hetero` is True.
        """
        selected_structure = self.structure
        if remove_hetero:
            # Remove hetero atoms via hetero column of structure ndarray
            selected_structure = self.structure[self.structure.hetero == False]
        if ca_only:
            # Consider selection of alpha-carbon atoms only
            selected_structure = selected_structure[selected_structure.atom_name == "CA"]
        chain_structure = selected_structure[selected_structure.chain_id == chain_id]
        return chain_structure
    
    def get_valid_chain_residues(self, chain_id: str) -> list[tuple[int, str]]:
        """
        Gets valid indexing for residues of a specified chain. This is directly analogous to get_non_missing_sequence, does not contain missing residues, and provides the corresponding indices as well.

        Parameters
        ----------
        chain_id : str
            Chain id of the chain to be selected from the structure. This chain's sequence and corresponding residue indices are what are exclusively selected for.
        
        Returns
        -------
        list of tuple of int, str
            A list of residue information in sequential order reflecting the structure. The list consists of tuple elements where each tuple is the residue index and its corresponding one-letter amino acid.
        """
        chain_structure = self.get_chain_specific_structure(ca_only=True, chain_id=chain_id, remove_hetero=True)
        return list(zip(chain_structure.res_id.tolist(), map(lambda symbol: ProteinSequence.convert_letter_3to1(symbol), chain_structure.res_name)))

    def generate_dist_matrix(self, ca_only: bool, chain1: str, chain2: str):
        """
        Generates distance matrix between two chains in the structure attribute.

        Parameters
        ----------
        ca_only : bool
            If True, only atoms that have the name "CA" are selected in the chains the distance matrix is calculated between.
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the first column of residues in the structure.

        Returns
        -------
        tuple of biotite.structure.AtomArray, biotite.structure.AtomArray, numpy.ndarray
            Tuple containing the chain 1 structure, the chain 2 structure, and the distance matrix of chain 1 and chain 2's pairwise distances.
        """
        chain1_structure = self.get_chain_specific_structure(ca_only=ca_only, chain_id=chain1, remove_hetero=True)
        chain2_structure = self.get_chain_specific_structure(ca_only=ca_only, chain_id=chain2, remove_hetero=True)
        dist_matrix = cdist(chain1_structure.coord, chain2_structure.coord)
        return (chain1_structure, chain2_structure, dist_matrix)

    def get_min_dist_atom_info(self, pairs: npt.NDArray, chain1: str, chain2: str) -> npt.NDArray:
        """
        Generate a ndarray of residue ids and their corresponding atom names such that the distance is the minimum between the initial residues provided.

        Parameters
        ----------
        pairs : numpy.ndarray
            Pairs structured ndarray with "residue1" and "residue2" columns.
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the second column of residues in the structure.
            
        Returns
        -------
        min_dist_pairs_atoms_arr : numpy.ndarray
            Structured ndarray that has residue indices, auth residue indices (corresponding to the protein numbering), and atomic names in the format {'names': ['residue1','residue2','auth_residue1','auth_residue2','atom_name1','atom_name2'], 'formats': [int,int,int,int,'<U10','<U10']}
        """
        chain1_structure = self.get_chain_specific_structure(ca_only=False, chain_id=chain1, remove_hetero=True)
        chain2_structure = self.get_chain_specific_structure(ca_only=False, chain_id=chain2, remove_hetero=True)
        min_dist_pairs_atoms = []
        for row in pairs:
            # Obtain structure information for chains 1 and 2
            chain1_res1_structure = chain1_structure[chain1_structure.res_id == row['residue1']]
            chain2_res2_structure = chain2_structure[chain2_structure.res_id == row['residue2']]
            
            # Calculate a distance matrix and find the indices of the minimal value in the matrix
            dist_matrix = cdist(chain1_res1_structure.coord, chain2_res2_structure.coord)
            
            ind = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            min_dist_pairs_atoms.append((row['residue1'], row['residue2'], row['residue1'], row['residue2'], chain1_res1_structure[ind[0]].atom_name, chain2_res2_structure[ind[1]].atom_name))
        min_dist_pairs_atoms_arr = np.array(min_dist_pairs_atoms, dtype={'names': ['residue1','residue2','auth_residue1','auth_residue2','atom_name1','atom_name2'], 'formats': [int,int,int,int,'<U10','<U10']})
        return min_dist_pairs_atoms_arr    

    def get_contacts(self, ca_only: bool, threshold: float, chain1: str, chain2: str) -> set[tuple[int, int]]:
        """
        Get contacts from the structure attribute where the distance between two residues is less than the threshold.

        Parameters
        ----------
        ca_only : bool
            If true, only consider alpha-carbon to alpha-carbon distances. 
        threshold : float
            Maximum distance to consider between two atoms.
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the second column of residues in the structure.

        Returns
        -------
        contacts_set : set of tuple of ints
            Set of contacts, tuples with "residue1" and "residue2" from the structure that are within the distance threshold.
        """
        
        chain1_structure, chain2_structure, dist_matrix = self.generate_dist_matrix(ca_only, chain1, chain2)
        thresh_ind = np.argwhere(dist_matrix <= threshold)
        contacts_set = set()
        for indices in thresh_ind:
            chain1_atom = chain1_structure[indices[0]]
            chain2_atom = chain2_structure[indices[1]]
            res1 = chain1_atom.res_id
            res2 = chain2_atom.res_id
            if not(chain1==chain2 and res1 >= res2):
                contacts_set.add((res1, res2))
        return contacts_set