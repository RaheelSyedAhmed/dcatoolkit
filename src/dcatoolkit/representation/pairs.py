from collections.abc import Iterable

import numpy as np
import numpy.typing as npt


class Pairs:
    """
    Object that contains a representation (as an ndarray) of pairs of entities that are related. This may extend to Direct Information Pairs or Structural contacts, where each residue is one component of the pair.

    Parameters
    ----------
    filepath : str, optional
        Filepath of the pairs in tabular representation, with residue 1 and residue 2 in the first two columns and one pair per line. Any further columns (e.g. a DI score) are dropped. An empty file produces an empty Pairs.
    ndarr : numpy.ndarray, optional
        Populated ndarray that contains pair information. See `_normalize()` for the accepted forms.
    delimiter : str, optional
        String used to separate the columns within a line of the file. Defaults to whitespace. See `numpy.loadtxt()` for details.

    Attributes
    ----------
    pairs : numpy.ndarray
        Structured ndarray with ``dtype=[('residue1', int), ('residue2', int)]`` holding the pairs supplied by the user.

    Raises
    ------
    ValueError
        If both or neither of `filepath` and `ndarr` are given.

    Notes
    -----
    Exactly one of `filepath` or `ndarr` must be specified in order to produce a Pairs representation.
    """
    _DTYPE = np.dtype([('residue1', int), ('residue2', int)])

    def __init__(self, filepath: str | None=None, ndarr: npt.NDArray | None=None, delimiter: str | None=None) -> None:
        if (filepath is not None and ndarr is not None) or (filepath is None and ndarr is None):
            raise ValueError("Please specify either a filepath or a NumPy array to populate your pairs.")
        elif filepath is not None:
            # Load as plain floats so _normalize handles the int conversion and checks, same as ndarray input.
            self.pairs = Pairs._normalize(np.loadtxt(filepath, delimiter=delimiter or None, ndmin=2))
        elif ndarr is not None:
            self.pairs = Pairs._normalize(ndarr)

    @staticmethod
    def _normalize(ndarr: npt.NDArray) -> npt.NDArray:
        """
        Coerces a plain or structured ndarray into a new structured ndarray with only the ``residue1`` and ``residue2`` fields used throughout Pairs. All Pairs inputs (ndarrays, iterables, and files) pass through here, so this is the single place residue values are converted and checked. Any other fields or columns (e.g. a DI score) are dropped.

        Parameters
        ----------
        ndarr : numpy.ndarray
            Either a plain ``(n, k)`` ndarray with ``k >= 2``, where residue 1 and residue 2 are the first two columns, or a structured ndarray carrying at least the ``residue1`` and ``residue2`` fields. Residue values may be ints or whole-number floats (e.g. ``12.0``, as in an all-float DI array). An empty plain ndarray produces an empty result.

        Returns
        -------
        numpy.ndarray
            Structured ndarray with ``dtype=[('residue1', int), ('residue2', int)]``.

        Raises
        ------
        ValueError
            If a structured ndarray lacks the ``residue1`` or ``residue2`` field, a plain ndarray is not 2D with at least 2 columns, or any residue value is not a whole number (including NaN and inf).
        """
        if ndarr.dtype.names is not None:
            if 'residue1' not in ndarr.dtype.names or 'residue2' not in ndarr.dtype.names:
                raise ValueError(f"Structured ndarray must contain 'residue1' and 'residue2' fields, got {ndarr.dtype.names}.")
            residue1, residue2 = ndarr['residue1'], ndarr['residue2']
        else:
            # Empty input (e.g. no contacts) is valid; it can arrive as shape (0,) from an empty list or (0, 1) from an empty file.
            if ndarr.size == 0:
                ndarr = ndarr.reshape(0, 2)
            if ndarr.ndim != 2 or ndarr.shape[1] < 2:
                raise ValueError(f"Plain ndarray must have shape (n, k) with k>=2, got {ndarr.shape}.")
            residue1, residue2 = ndarr[:, 0], ndarr[:, 1]
        structured = np.zeros(len(residue1), dtype=Pairs._DTYPE)
        with np.errstate(invalid='ignore'):
            structured['residue1'] = residue1
            structured['residue2'] = residue2

        # Float residues are allowed (e.g. a plain DI array is all floats), but only if they survive the int cast unchanged.
        has_float_residues = residue1.dtype.kind == 'f' or residue2.dtype.kind == 'f'
        if has_float_residues and not (np.array_equal(structured['residue1'], residue1) and np.array_equal(structured['residue2'], residue2)):
            raise ValueError("Residue indices must be whole numbers, but non-integer values were found.")
        return structured

    @staticmethod
    def to_ndarray(pairs: npt.NDArray) -> npt.NDArray:
        """
        Converts a structured ndarray with ``residue1`` and ``residue2`` fields into a plain ``(n, 2)`` int ndarray, dropping any other fields (e.g. ``DI``). Ndarrays that are already unstructured are passed through unchanged.

        Parameters
        ----------
        pairs : numpy.ndarray
            Structured ndarray carrying at least the ``residue1`` and ``residue2`` fields, or an already-plain ``(n, 2)`` ndarray.

        Returns
        -------
        numpy.ndarray
            Plain ``(n, 2)`` int ndarray with ``residue1`` in column 0 and ``residue2`` in column 1.
        """
        if pairs.dtype.names is None:
            return pairs
        return np.column_stack([pairs['residue1'], pairs['residue2']])

    @staticmethod
    def load_from_file(filepath: str, delimiter: str | None=None) -> 'Pairs':
        """
        Loads file containing delimited data in columns of residues being column 1 and column 2. Any further columns (e.g. a DI score) are dropped, and an empty file produces an empty Pairs.

        Parameters
        ----------
        filepath : str
            Filepath with residue columns corresponding to the indices of first and second components (proteins, chains, etc.) constituting a pair.
        delimiter : str, optional
            String used to separate the columns within a line of the file. Defaults to whitespace. See `numpy.loadtxt()` for details.

        Returns
        -------
        Pairs
            Pairs object with a loaded, structured ndarray with ``dtype=[('residue1', int), ('residue2', int)]``.
        """
        return Pairs(filepath=filepath, delimiter=delimiter)

    @staticmethod
    def load_from_ndarray(ndarray: npt.NDArray | Iterable[Iterable]) -> 'Pairs':
        """
        Loads a 2D ndarray, structured ndarray, or iterable of residue pairs in columnar format into a Pairs object. Any values after the first two in each pair (e.g. a DI score) are dropped, and an empty input produces an empty Pairs.

        Parameters
        ----------
        ndarray : numpy.ndarray or Iterable of Iterable (excluding dict)
            Ndarray or iterable of iterables with pairs of residue indices, with residue 1 and residue 2 in separate columns or as the first two elements. A structured ndarray needs the ``residue1`` and ``residue2`` fields. Iterables (including generators) are fully loaded into memory before conversion.

        Returns
        -------
        Pairs
            Pairs object with a loaded, structured ndarray with ``dtype=[('residue1', int), ('residue2', int)]``.
        """
        # Build a plain array (no int dtype) so _normalize handles the int conversion and checks, same as file and ndarray input.
        if not isinstance(ndarray, np.ndarray):
            ndarray = np.array(list(ndarray))
        return Pairs(ndarr=ndarray)

    @staticmethod
    def mirror_diagonal(pairs: npt.NDArray) -> npt.NDArray:
        """
        Flips pair positions for diagonal-mirrored representation, e.g. ``(1, 2)`` becomes ``(2, 1)``.

        Parameters
        ----------
        pairs : numpy.ndarray
            Structured ndarray with at least the ``residue1`` and ``residue2`` fields.

        Returns
        -------
        numpy.ndarray
            A new structured ndarray with ``residue1`` and ``residue2`` swapped; other fields are unchanged.
        """
        mirrored = pairs.copy()
        mirrored['residue1'] = pairs['residue2']
        mirrored['residue2'] = pairs['residue1']
        return mirrored
    
    @staticmethod
    def subset_pairs(pairs: npt.NDArray, number : int | None=None) -> npt.NDArray:
        """
        Picks out the first `number` pairs if `number` is supplied. Otherwise, returns all pairs.

        Parameters
        ----------
        pairs : numpy.ndarray
            Ndarray to select rows from.
        number : int, optional
            Specific number of rows of pairs to subset. If None, all pairs are returned.

        Returns
        -------
        numpy.ndarray
            The first `number` rows of `pairs`, or all of `pairs` if `number` is None.
        """
        if number is not None:
            return pairs[:number, ]
        else:
            return pairs
    
    @staticmethod
    def mirror_pairs(pairs: npt.NDArray) -> npt.NDArray:
        """
        Produces combined array of pairs followed by their mirrored representation, e.g. ``[(1, 2), (3, 4)]`` becomes ``[(1, 2), (3, 4), (2, 1), (4, 3)]``.

        Parameters
        ----------
        pairs : numpy.ndarray
            Structured ndarray to mirror and vertically append to, with at least the ``residue1`` and ``residue2`` fields.

        Returns
        -------
        numpy.ndarray
            Combined ndarray of the original pairs followed by the mirrored pairs, with twice as many rows as `pairs`.

        See Also
        --------
        mirror_diagonal : Produces the mirrored copy that is appended.
        """
        return np.concatenate([pairs, Pairs.mirror_diagonal(pairs)])
    
    @staticmethod
    def get_pairs(pairs: npt.NDArray, mirror: bool=False, number: int | None=None) -> npt.NDArray:
        """
        Returns pairs based on user specification, offering options to produce mirrored representation of pairs and to select a specific number of pairs.

        Parameters
        ----------
        pairs : numpy.ndarray
            Structured ndarray of pairs with the ``residue1`` and ``residue2`` fields to select from or to mirror.
        mirror : bool, default False
            Whether or not to append the mirrored representation of the pairs to the original pairs ndarray.
        number : int, optional
            Specific number of rows of pairs to subset. If None, all pairs are kept.

        Returns
        -------
        numpy.ndarray
            The first `number` pairs, followed by their mirrored copies if `mirror` is True. With `mirror`, the result has ``2 * number`` rows.

        See Also
        --------
        subset_pairs : Selects the first `number` pairs.
        mirror_pairs : Appends the mirrored copies.
        """
        pairs = Pairs.subset_pairs(pairs, number)
        # Check to see if user requested mirrored pairs, if so, add in pairs that are mirrored across diagonal
        if mirror:
            pairs = Pairs.mirror_pairs(pairs)
        return pairs