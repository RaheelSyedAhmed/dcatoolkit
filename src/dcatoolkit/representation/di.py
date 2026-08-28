import numpy as np
from typing import Optional, Union
from collections.abc import Iterable
import numpy.typing as npt
from .pairs import Pairs
from .alignment import ResidueAlignment


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
        # Boolean-mask indexing already returns a copy, but that's made explicit here since the in-place
        # field assignment below would silently corrupt the caller's DI_data if this ever became a view.
        mappable_DI_data = DI_data[mapping_key_mask].copy()
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