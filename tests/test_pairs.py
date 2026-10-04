import os
import tempfile

import numpy as np
import pytest
from context import Pairs


def as_tuples(structured_array):
    return [tuple(row) for row in structured_array]


def test_normalize_plain_array_to_structured():
    p = Pairs(ndarr=np.array([[1, 2], [3, 4]]))
    assert p.pairs.dtype.names == ('residue1', 'residue2')
    assert as_tuples(p.pairs) == [(1, 2), (3, 4)]

def test_normalize_structured_passthrough():
    structured = np.array([(5, 6), (7, 8)], dtype=Pairs._DTYPE)
    p = Pairs(ndarr=structured)
    assert as_tuples(p.pairs) == [(5, 6), (7, 8)]

def test_normalize_structured_drops_extra_fields():
    # Pairs holds residue pairs only -- a DI field alongside residue1/residue2 is dropped.
    extra = np.array([(1, 2, 0.9), (3, 4, 0.5)], dtype={'names': ('residue1', 'residue2', 'DI'), 'formats': (int, int, float)})
    p = Pairs(ndarr=extra)
    assert p.pairs.dtype.names == ('residue1', 'residue2')
    assert as_tuples(p.pairs) == [(1, 2), (3, 4)]

def test_normalize_rejects_wrong_structured_field_names():
    wrong = np.array([(1, 2)], dtype=[('foo', int), ('bar', int)])
    with pytest.raises(ValueError):
        Pairs(ndarr=wrong)

def test_normalize_rejects_1d_array():
    with pytest.raises(ValueError):
        Pairs(ndarr=np.array([1, 2]))

def test_normalize_rejects_single_column_array():
    with pytest.raises(ValueError):
        Pairs(ndarr=np.array([[1], [2], [3]]))

def test_normalize_structured_input_is_copied():
    structured = np.array([(1, 2), (3, 4)], dtype=Pairs._DTYPE)
    p = Pairs(ndarr=structured)
    structured['residue1'][0] = 99
    assert as_tuples(p.pairs) == [(1, 2), (3, 4)]

def test_constructor_rejects_both_or_neither_source():
    with pytest.raises(ValueError):
        Pairs()
    with pytest.raises(ValueError):
        Pairs(filepath='unused.txt', ndarr=np.array([[1, 2]]))

def test_normalize_accepts_whole_number_floats():
    # A plain DI-shaped array is all floats, so residue 12 arrives as 12.0.
    p = Pairs(ndarr=np.array([[12.0, 30.0, 0.9], [5.0, 7.0, 0.5]]))
    assert as_tuples(p.pairs) == [(12, 30), (5, 7)]

def test_normalize_rejects_fractional_plain_residues():
    with pytest.raises(ValueError):
        Pairs(ndarr=np.array([[12.7, 30.0]]))

def test_normalize_rejects_fractional_structured_residues():
    floats = np.array([(1.0, 2.5)], dtype=[('residue1', float), ('residue2', float)])
    with pytest.raises(ValueError):
        Pairs(ndarr=floats)

@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -np.inf])
def test_normalize_rejects_non_finite_residues(bad_value):
    with pytest.raises(ValueError):
        Pairs(ndarr=np.array([[bad_value, 2.0]]))

def test_normalize_drops_extra_plain_columns_intentionally():
    # Pairs is residue-pair-specific: extra columns on a plain array are dropped, not rejected.
    p = Pairs(ndarr=np.array([[1, 2, 999], [3, 4, 888]]))
    assert as_tuples(p.pairs) == [(1, 2), (3, 4)]

def test_to_ndarray_structured_to_plain():
    p = Pairs(ndarr=np.array([(1, 2), (3, 10)], dtype=Pairs._DTYPE))
    plain = Pairs.to_ndarray(p.pairs)
    assert np.array_equal(plain, np.array([[1, 2], [3, 10]]))

def test_to_ndarray_plain_passthrough():
    plain_in = np.array([[1, 2], [3, 4]])
    assert np.array_equal(Pairs.to_ndarray(plain_in), plain_in)

def test_to_ndarray_drops_di_column():
    di_shaped = np.array([(1, 2, 0.9), (3, 4, 0.5)], dtype={'names': ('residue1', 'residue2', 'DI'), 'formats': (int, int, float)})
    plain = Pairs.to_ndarray(di_shaped)
    assert np.array_equal(plain, np.array([[1, 2], [3, 4]]))

def test_mirror_diagonal():
    pairs = np.array([(1, 2), (3, 4)], dtype=Pairs._DTYPE)
    mirrored = Pairs.mirror_diagonal(pairs)
    assert as_tuples(mirrored) == [(2, 1), (4, 3)]

def test_mirror_diagonal_preserves_extra_fields():
    # DI is symmetric, so each DI value should follow its pair through the flip.
    di_shaped = np.array([(1, 2, 0.9), (3, 4, 0.5)], dtype={'names': ('residue1', 'residue2', 'DI'), 'formats': (int, int, float)})
    mirrored = Pairs.mirror_diagonal(di_shaped)
    assert as_tuples(mirrored) == [(2, 1, 0.9), (4, 3, 0.5)]

def test_mirror_pairs_appends_mirrored_copy():
    pairs = np.array([(1, 2), (3, 4)], dtype=Pairs._DTYPE)
    combined = Pairs.mirror_pairs(pairs)
    assert as_tuples(combined) == [(1, 2), (3, 4), (2, 1), (4, 3)]

def test_get_pairs_subset_and_mirror():
    # get_pairs subsets first, then mirrors what's left -- so the third pair should be
    # dropped by the number=2 subset before mirroring doubles the remaining two.
    pairs = np.array([(1, 2), (3, 10), (5, 6)], dtype=Pairs._DTYPE)
    result = Pairs.get_pairs(pairs, mirror=True, number=2)
    assert as_tuples(result) == [(1, 2), (3, 10), (2, 1), (10, 3)]

def test_load_from_file_single_row_not_collapsed():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write('1 2\n')
        path = f.name
    try:
        p = Pairs.load_from_file(path)
        assert p.pairs.shape == (1,)
        assert as_tuples(p.pairs) == [(1, 2)]
    finally:
        os.remove(path)

def test_constructor_filepath_single_row_not_collapsed():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write('1 2\n')
        path = f.name
    try:
        p = Pairs(filepath=path)
        assert p.pairs.shape == (1,)
        assert as_tuples(p.pairs) == [(1, 2)]
    finally:
        os.remove(path)

def test_load_from_file_empty_file_gives_empty_pairs(tmp_path):
    # No contacts is a legitimate result, so an empty file should give an empty Pairs, not an error.
    path = tmp_path / 'empty.txt'
    path.write_text('')
    with pytest.warns(UserWarning):
        p = Pairs.load_from_file(str(path))
    assert p.pairs.shape == (0,)
    assert p.pairs.dtype.names == ('residue1', 'residue2')

def test_load_from_file_drops_di_column(tmp_path):
    path = tmp_path / 'di.txt'
    path.write_text('1 2 0.9\n3 4 0.5\n')
    p = Pairs.load_from_file(str(path))
    assert as_tuples(p.pairs) == [(1, 2), (3, 4)]

def test_load_from_file_rejects_fractional_residues(tmp_path):
    path = tmp_path / 'bad.txt'
    path.write_text('1.7 2\n')
    with pytest.raises(ValueError):
        Pairs.load_from_file(str(path))

def test_constructor_filepath_custom_delimiter(tmp_path):
    path = tmp_path / 'pairs.csv'
    path.write_text('1,2\n3,4\n')
    p = Pairs(filepath=str(path), delimiter=',')
    assert as_tuples(p.pairs) == [(1, 2), (3, 4)]

def test_load_from_file_custom_delimiter(tmp_path):
    path = tmp_path / 'pairs.csv'
    path.write_text('1,2\n3,4\n')
    p = Pairs.load_from_file(str(path), delimiter=',')
    assert as_tuples(p.pairs) == [(1, 2), (3, 4)]

def test_load_from_ndarray_delegates_for_real_ndarray():
    p = Pairs.load_from_ndarray(np.array([[1, 2], [3, 4]]))
    assert as_tuples(p.pairs) == [(1, 2), (3, 4)]

def test_load_from_ndarray_iterable_fallback():
    p = Pairs.load_from_ndarray([(1, 2), (3, 10)])
    assert as_tuples(p.pairs) == [(1, 2), (3, 10)]

def test_load_from_ndarray_iterable_drops_di_column():
    p = Pairs.load_from_ndarray([(1, 2, 0.9), (3, 4, 0.5)])
    assert as_tuples(p.pairs) == [(1, 2), (3, 4)]

def test_load_from_ndarray_iterable_rejects_fractional_residues():
    with pytest.raises(ValueError):
        Pairs.load_from_ndarray([(1.7, 2)])

def test_load_from_ndarray_empty_iterable_gives_empty_pairs():
    p = Pairs.load_from_ndarray([])
    assert p.pairs.shape == (0,)
    assert p.pairs.dtype.names == ('residue1', 'residue2')

def test_load_from_ndarray_generator_fallback():
    gen = ((x, x + 1) for x in [9, 11])
    p = Pairs.load_from_ndarray(gen)
    assert as_tuples(p.pairs) == [(9, 10), (11, 12)]
