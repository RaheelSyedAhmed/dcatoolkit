from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from .alignment import ResidueAlignment
from .pairs import Pairs


class DirectInformationData:
    """
    Representation and interface for Direct Information data including residue indices for a pair and its corresponding DI value, stored as a structured ndarray.

    Parameters
    ----------
    structured_ndarray : numpy.ndarray
        Structured ndarray of shape ``(n,)``, one record per pair, with at least the fields ``residue1`` (int), ``residue2`` (int), and ``DI`` (float). Additional fields are allowed and kept. To build one from a plain ``(n, 3)`` ndarray, use `load_as_ndarray()`.

    Attributes
    ----------
    DI_data : numpy.ndarray
        The `structured_ndarray` from the parameters section, where ``residue1`` is a pair's first residue, ``residue2`` is the pair's second residue, and ``DI`` is the Direct Information of the pair.

    Raises
    ------
    ValueError
        If `structured_ndarray` is not a structured ndarray or lacks any of the ``residue1``, ``residue2``, or ``DI`` fields.
    """
    def __init__(self, structured_ndarray: npt.NDArray) -> None:
        if structured_ndarray.dtype.names is None:
            raise ValueError(f"structured_ndarray must be a structured array with residue1/residue2/DI fields, got a plain array with dtype {structured_ndarray.dtype}.")
        missing = {'residue1', 'residue2', 'DI'} - set(structured_ndarray.dtype.names)
        if missing:
            raise ValueError(f"structured_ndarray is missing required fields: {missing}")
        self.DI_data = structured_ndarray

    @staticmethod    
    def load_from_dca_output(dca_filepath: str) -> 'DirectInformationData':
        """
        Function to generate a DirectInformationData object from the direct output of the MATLAB dca function.

        Parameters
        ----------
        dca_filepath : str
            Filepath of the DCA output to be read and compiled into a structured ndarray. DCA output is a 4 column text file with the following columns: (residue 1, residue 2, Mutual Information, Direct Information).

        Returns
        -------
        DirectInformationData
            DirectInformationData object with a structured ndarray containing the ``residue1``, ``residue2``, and ``DI`` fields.
        """
        file_data = np.loadtxt(dca_filepath, dtype={'names': ('residue1', 'residue2', 'MI', 'DI'), 'formats': (int, int, float, float)}, ndmin=1)
        return DirectInformationData(file_data[['residue1', 'residue2', 'DI']])

    @staticmethod
    def load_from_DI_file(DI_filepath: str) -> 'DirectInformationData':
        """
        Function to generate a DirectInformationData object from the modified DI-only version of the DCA output generated via the MATLAB dca function.

        Parameters
        ----------
        DI_filepath : str
            Filepath of the DI file to be read and compiled into a structured ndarray. DI file is a 3 column text file with the following columns: (residue 1, residue 2, Direct Information).
        
        Returns
        -------
        DirectInformationData
            DirectInformationData object with a structured ndarray containing the ``residue1``, ``residue2``, and ``DI`` fields.
        """
        return DirectInformationData(np.loadtxt(DI_filepath, dtype={'names': ('residue1', 'residue2', 'DI'), 'formats': (int, int, float)}, ndmin=1))

    @staticmethod
    def load_as_ndarray(ndarray: npt.NDArray | Iterable[Iterable]) -> 'DirectInformationData':
        """
        Function to generate DirectInformationData from a plain ndarray or an iterable of pairs.

        Parameters
        ----------
        ndarray : numpy.ndarray or Iterable of Iterable (excluding dict)
            A plain ndarray of shape ``(n, 3)`` whose columns are residue 1, residue 2, and Direct Information. Can also be parsed from an iterable of iterables, each with exactly those three values.

        Returns
        -------
        DirectInformationData
            DirectInformationData object with a structured ndarray containing the ``residue1``, ``residue2``, and ``DI`` fields.

        Raises
        ------
        ValueError
            If `ndarray` is a structured ndarray (use the DirectInformationData constructor instead), or a plain ndarray that is not 2D with exactly 3 columns.
        """
        
        if isinstance(ndarray, np.ndarray):
            if ndarray.dtype.names is not None:
                raise ValueError("ndarray is a structured numpy array. If this structured array has columns for 'residue1', 'residue2', and 'DI', please use the DirectInformationData constructor directly. Otherwise, please convert to an unstructured numpy array before using this function.")
            elif ndarray.ndim != 2 or ndarray.shape[1] != 3:
                raise ValueError(f"Dimensions of numpy array supplied are different from what is expected. Please supply residue1, residue2, and DI column in int, int, float format and with shape of (n, 3), got {ndarray.shape}.")
            else:
                DI_data = np.zeros(len(ndarray), dtype={'names': ('residue1', 'residue2', 'DI'), 'formats': (int, int, float)})
                DI_data['residue1'] = ndarray[:, 0]
                DI_data['residue2'] = ndarray[:, 1]
                DI_data['DI'] = ndarray[:, 2]
                return DirectInformationData(DI_data)
        else:
            # Structured ndarrays require list of tuples for conversion.
            DI_data = np.array([tuple(x) for x in ndarray], dtype={'names': ('residue1', 'residue2', 'DI'), 'formats': (int, int, float)})
            return DirectInformationData(DI_data)
    
    def get_ranked_mapped_pairs(self, RA1: ResidueAlignment, RA2: ResidueAlignment, pairs_only: bool=True, mirror: bool=False, number: int | None=None) -> npt.NDArray:
        """
        Uses DirectInformationData and Pairs interface methods to obtain ranked, mapped residues that are further than 4 residues apart. Residue Alignments can be the same for intra-domain / intra-protein mapping.

        Parameters
        ----------
        RA1 : ResidueAlignment
            The ResidueAlignment used for mapping the first column of residues to the appropriate target sequence.
        RA2 : ResidueAlignment
            The ResidueAlignment used for mapping the second column of residues to the appropriate target sequence.
        pairs_only : bool, default True
            If True, the final ndarray contains only the ``residue1`` and ``residue2`` fields, dropping the ``DI`` field.
        mirror : bool, default False
            If True, the original pairs are followed by the same pairs with residue 1 and residue 2 switched, which is useful for plotting across the upper diagonal of a contact map. Ignored if `pairs_only` is False. See `Pairs.get_pairs()` for details.
        number : int, optional
            Number of ranked, mapped pairs to return. If None, all pairs are returned.

        Returns
        -------
        numpy.ndarray
            Structured ndarray with the ``residue1`` and ``residue2`` fields, plus ``DI`` if `pairs_only` is False. Only has `number` pairs if `number` is specified, and mirrored pairs if `mirror` and `pairs_only` are both True.

        See Also
        --------
        nonlocal_pairs : Removes pairs within 4 residues of each other.
        rank_pairs : Ranks pairs by DI in descending order.
        map_DIs : Maps residues through the ResidueAlignments.

        Notes
        -----
        ResidueAlignments contain dictionaries like ``domain_to_protein`` to map residues produced via Direct Coupling Analysis (DCA) on an MSA generated in context to an HMM. The residues are mapped to a protein structure via alignment of the HMM hit / domain to the protein sequence.
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
        Uses domain-to-protein mappings present in the Residue Alignments provided to generate mapped representations of the residues from the `DI_data` structured ndarray provided.

        Parameters
        ----------
        DI_data : numpy.ndarray
            Structured ndarray that contains the ``residue1`` and ``residue2`` fields.
        RA1 : ResidueAlignment
            The ResidueAlignment used for mapping the first column of residues to the appropriate target sequence.
        RA2 : ResidueAlignment
            The ResidueAlignment used for mapping the second column of residues to the appropriate target sequence.
        
        Returns
        -------
        mappable_DI_data : numpy.ndarray
            `DI_data` that has been mapped to the target sequence specified in the generation of the corresponding ResidueAlignments. Other fields (e.g. ``DI``) are kept.

        Notes
        -----
        Residues that do not map to the target sequence of the ResidueAlignment are dropped.
        """
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
            Structured ndarray that contains the ``residue1``, ``residue2``, and ``DI`` (Direct Information) fields.

        Returns
        -------
        numpy.ndarray
            Structured ndarray sorted by the ``DI`` field in descending order.
        """
        # [::-1] reverses the order from ascending DI Score to descending DI score.
        return np.sort(DI_data, order='DI')[::-1]
    
    @staticmethod
    def nonlocal_pairs(DI_data: npt.NDArray) -> npt.NDArray:
        """
        Subsets a structured ndarray of pairs information to find nonlocal pairs, where residue interactions are likely not involved in secondary structure formation i.e. helices and sheet interactions. Nonlocal pairs must be greater than 4 residues apart.

        Parameters
        ----------
        DI_data : numpy.ndarray
            Structured ndarray that contains (at least) the ``residue1`` and ``residue2`` fields.

        Returns
        -------
        numpy.ndarray
            Structured ndarray of DI pairs where ``residue1`` and ``residue2`` are greater than 4 residues apart.
        """
        return DI_data[abs(DI_data['residue1'] - DI_data['residue2']) > 4]
    
    @staticmethod
    def find_DI_with_residues(critical_residues_1 : Iterable[int], critical_residues_2 : Iterable[int], *mapped_resi_arrs: npt.NDArray, max_rank: int | None=None) -> list[tuple[list, int]]:
        """
        Searches one or more ranked, mapped DI arrays for pairs whose ``residue1`` and ``residue2`` are within `critical_residues_1` and `critical_residues_2`, respectively.

        Parameters
        ----------
        critical_residues_1 : collections.abc.Iterable of int
            Specific residue indices that a DI pair will be compared to. If the first residue of the DI pair is not one of these indices, it will not be appended to results.
        critical_residues_2 : collections.abc.Iterable of int
            Specific residue indices that a DI pair will be compared to. If the second residue of the DI pair is not one of these indices, it will not be appended to results.
        *mapped_resi_arrs : numpy.ndarray
            One or more ranked, mapped structured ndarrays with the ``residue1`` and ``residue2`` fields that are compared to critical residue indices and appended to results if in those indices and within `max_rank`. Rank restarts at 1 for each array.
        max_rank : int, optional
            Keyword-only. Maximum "rank", or position by score in descending order when sorted, of the DI pair considered, counted from 1. If None, all pairs are considered.

        Returns
        -------
        results : list of tuple of list of int, int
            Results which consist of tuples where the first element is a list of the row's fields (e.g. ``residue1``, ``residue2``, and ``DI``), whereas the second element is the rank.

        Raises
        ------
        TypeError
            If a mapped array isn't a numpy.ndarray, e.g. when `max_rank` is passed positionally as in older versions.
        """
        # Older versions took max_rank positionally before the arrays; catch that call style with a clear message.
        for mapped_resi_arr in mapped_resi_arrs:
            if not isinstance(mapped_resi_arr, np.ndarray):
                raise TypeError(f"Expected numpy.ndarray mapped arrays, got {type(mapped_resi_arr).__name__}. max_rank is keyword-only, e.g. find_DI_with_residues(residues1, residues2, arr1, arr2, max_rank=300).")
        critical_residues_1 = set(critical_residues_1)
        critical_residues_2 = set(critical_residues_2)
        results = []
        for mapped_resi_arr in mapped_resi_arrs:
            # count_rank represents the rank of the DI pair being evaluated, counted from 1 for every new row considered.
            for count_rank, row in enumerate(mapped_resi_arr, start=1):
                if max_rank is not None and count_rank > max_rank:
                    break
                if row['residue1'] in critical_residues_1 and row['residue2'] in critical_residues_2:
                    results.append((list(row), count_rank))
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
            The chain present in the structure in `model1` containing the first column of residues.
        chain2 : str
            The chain present in the structure in `model2` containing the second column of residues.
        pairs : numpy.ndarray
            Structured ndarray that contains (at least) the ``residue1`` and ``residue2`` fields. If atoms are specified per pair, `ca_only` should be set to False and the ``atom_name1`` and ``atom_name2`` fields should be present, e.g. from `get_min_dist_atom_info()`.
        ca_only : bool, default True
            If True, distance commands are between the two alpha-carbons of the residue pair. If False, the atom names in the ``atom_name1`` and ``atom_name2`` fields of `pairs` are used instead of ``"CA"``.
        auth_res_ids : bool, default False
            If True, use the ``auth_residue1`` and ``auth_residue2`` fields instead of the ``residue1`` and ``residue2`` fields. These residue ids correspond to the auth protein residue ids.

        Returns
        -------
        distance_commands : list of str
            List of distance commands generated between two residues with model and chain information needed, either between two alpha-carbons or the specified atoms.

        Notes
        -----
        `model1` and `model2` can be equivalent if both columns involve residues referenced by the same model. The same would apply for chains if the residues are present on the same chain.
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
            Ndarray of at least pairs information (residue 1, residue 2) and optionally Direct Information to write to a file via `numpy.savetxt()`. Either a structured ndarray (e.g. from `get_ranked_mapped_pairs()`) or a plain ``(n, 2)`` or ``(n, 3)`` ndarray. An empty ndarray writes an empty file.
        delimiter : str, default ``'\\t'``
            Delimiter to separate columns of the pairs ndarray when writing to a file.
        fmt : tuple of str, default ``('%d', '%d', '%.3f')``
            Format passed as an argument to `numpy.savetxt()` to define type of column and output format. Replaced with ``('%d', '%d')`` if `pairs` has only two fields or columns.
        """
        # Count columns from the dtype/shape rather than pairs[0], so an empty pairs ndarray writes an empty file instead of raising.
        n_columns = len(pairs.dtype.names) if pairs.dtype.names is not None else pairs.shape[1]
        if n_columns == 2:
            fmt = ('%d', '%d')
        np.savetxt(filepath, pairs, delimiter=delimiter, fmt=fmt)