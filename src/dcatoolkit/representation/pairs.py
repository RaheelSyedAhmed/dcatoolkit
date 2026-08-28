import numpy as np
from typing import Optional, Union
from collections.abc import Iterable
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
    _DTYPE = [('residue1', int), ('residue2', int)]

    def __init__(self, filepath: Optional[str]=None, ndarr: Optional[npt.NDArray]=None, delimiter: Optional[str]=None) -> None:
        if (filepath is not None and ndarr is not None) or (filepath is None and ndarr is None):
            raise Exception("Please specify either a filepath or a NumPy array to populate your pairs.")
        elif filepath is not None:
            if delimiter:
                self.pairs = np.loadtxt(filepath, dtype=Pairs._DTYPE, delimiter=delimiter, ndmin=1)
            else:
                self.pairs = np.loadtxt(filepath, dtype=Pairs._DTYPE, ndmin=1)
        elif ndarr is not None:
            self.pairs = Pairs._normalize(ndarr)

    @staticmethod
    def _normalize(ndarr: npt.NDArray) -> npt.NDArray:
        """
        Coerces a plain (n, 2) int ndarray into the structured residue1/residue2 dtype used throughout Pairs. Ndarrays that are already structured are passed through unchanged.

        Parameters
        ----------
        ndarr : numpy.ndarray
            Either a plain (n, 2) int ndarray or an ndarray already carrying the structured residue1/residue2 dtype.

        Returns
        -------
        numpy.ndarray
            Structured ndarray with dtype=[('residue1', int), ('residue2', int)]
        """
        if ndarr.dtype.names is not None:
            if 'residue1' not in ndarr.dtype.names or 'residue2' not in ndarr.dtype.names:
                raise ValueError(f"Structured ndarray must contain 'residue1' and 'residue2' fields, got {ndarr.dtype.names}.")
            return ndarr
        structured = np.zeros(len(ndarr), dtype=Pairs._DTYPE)
        structured['residue1'] = ndarr[:, 0]
        structured['residue2'] = ndarr[:, 1]
        return structured

    @staticmethod
    def to_ndarray(pairs: npt.NDArray) -> npt.NDArray:
        """
        Converts a structured residue1/residue2 ndarray into a plain (n, 2) int ndarray, dropping any other fields (e.g. a DI score column). Ndarrays that are already unstructured are passed through unchanged.

        Parameters
        ----------
        pairs : numpy.ndarray
            Structured ndarray carrying at least residue1/residue2 fields, or an already-plain (n, 2) ndarray.

        Returns
        -------
        numpy.ndarray
            Plain (n, 2) int ndarray with residue1 in column 0 and residue2 in column 1.
        """
        if pairs.dtype.names is None:
            return pairs
        return np.column_stack([pairs['residue1'], pairs['residue2']])

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
        return Pairs(ndarr=np.loadtxt(filepath, dtype=Pairs._DTYPE, ndmin=1))

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
        if isinstance(ndarray, np.ndarray):
            return Pairs(ndarr=ndarray)
        return Pairs(ndarr=np.array([tuple(x) for x in ndarray], dtype=Pairs._DTYPE))

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
        mirrored = np.empty_like(pairs)
        mirrored['residue1'] = pairs['residue2']
        mirrored['residue2'] = pairs['residue1']
        return mirrored
    
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
            return np.concatenate([pairs, Pairs.mirror_diagonal(pairs)])
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