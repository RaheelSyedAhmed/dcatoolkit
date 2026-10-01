from context import DirectInformationData, ResidueAlignment
import numpy as np
import pytest


DI_DTYPE = {'names': ('residue1', 'residue2', 'DI'), 'formats': (int, int, float)}


def as_tuples(structured_array):
    return [tuple(row) for row in structured_array]


def test_init_rejects_plain_array():
    with pytest.raises(ValueError):
        DirectInformationData(np.array([1, 2, 3]))

def test_init_rejects_missing_fields():
    incomplete = np.array([(1, 2)], dtype={'names': ('residue1', 'residue2'), 'formats': (int, int)})
    with pytest.raises(ValueError):
        DirectInformationData(incomplete)

def test_init_accepts_correct_structured_array():
    correct = np.array([(1, 2, 0.9)], dtype=DI_DTYPE)
    d = DirectInformationData(correct)
    assert as_tuples(d.DI_data) == [(1, 2, 0.9)]

def test_init_accepts_extra_fields():
    extra = np.array([(1, 2, 0.9, 'x')], dtype={'names': ('residue1', 'residue2', 'DI', 'extra'), 'formats': (int, int, float, '<U1')})
    d = DirectInformationData(extra)
    assert d.DI_data['residue1'][0] == 1

def test_load_as_ndarray_rejects_structured_input():
    structured = np.array([(1, 2, 0.9)], dtype=DI_DTYPE)
    with pytest.raises(ValueError):
        DirectInformationData.load_as_ndarray(structured)

def test_load_as_ndarray_rejects_wrong_column_count():
    with pytest.raises(ValueError):
        DirectInformationData.load_as_ndarray(np.array([[1, 2], [3, 4]]))

def test_load_as_ndarray_rejects_1d_array():
    with pytest.raises(ValueError):
        DirectInformationData.load_as_ndarray(np.array([1, 2, 3]))

def test_load_as_ndarray_accepts_nested_single_row():
    d = DirectInformationData.load_as_ndarray(np.array([[5, 10, 0.9]]))
    assert as_tuples(d.DI_data) == [(5, 10, 0.9)]

def test_load_as_ndarray_vectorized_plain_path():
    d = DirectInformationData.load_as_ndarray(np.array([[1, 2, 0.9], [3, 4, 0.5]]))
    assert as_tuples(d.DI_data) == [(1, 2, 0.9), (3, 4, 0.5)]

def test_load_as_ndarray_iterable_fallback():
    d = DirectInformationData.load_as_ndarray([(1, 2, 0.9), (3, 4, 0.5)])
    assert as_tuples(d.DI_data) == [(1, 2, 0.9), (3, 4, 0.5)]

def test_map_dis_does_not_mutate_caller_data():
    DI_data = np.array([(1, 2, 0.9), (3, 4, 0.5)], dtype=DI_DTYPE)
    original_copy = DI_data.copy()
    RA1 = ResidueAlignment('d', 'p', 1, 100, 'AC', 'AC')
    RA2 = ResidueAlignment('d', 'p', 1, 200, 'AC', 'AC')
    DirectInformationData.map_DIs(DI_data, RA1, RA2)
    assert np.array_equal(DI_data, original_copy)

def test_map_dis_correctly_maps_residues():
    DI_data = np.array([(1, 2, 0.9)], dtype=DI_DTYPE)
    RA1 = ResidueAlignment('d', 'p', 1, 100, 'AC', 'AC')
    RA2 = ResidueAlignment('d', 'p', 1, 200, 'AC', 'AC')
    mapped = DirectInformationData.map_DIs(DI_data, RA1, RA2)
    assert mapped['residue1'][0] == 100
    assert mapped['residue2'][0] == 201

def test_find_di_with_residues_no_max_rank():
    arr = np.array([(i, i + 1, 1.0 - i * 0.01) for i in range(10)], dtype=DI_DTYPE)
    results = DirectInformationData.find_DI_with_residues(list(range(10)), list(range(1, 11)), None, arr)
    assert len(results) == 10
    assert all(rank == i + 1 for i, (_, rank) in enumerate(results))

def test_find_di_with_residues_max_rank_stops_early():
    arr = np.array([(i, i + 1, 1.0 - i * 0.01) for i in range(20)], dtype=DI_DTYPE)
    results = DirectInformationData.find_DI_with_residues(list(range(20)), list(range(1, 21)), 5, arr)
    assert len(results) == 5
    assert all(rank <= 5 for _, rank in results)

def test_find_di_with_residues_multiple_arrays_reset_rank():
    arr1 = np.array([(1, 2, 0.9)], dtype=DI_DTYPE)
    arr2 = np.array([(100, 101, 0.5)], dtype=DI_DTYPE)
    results = DirectInformationData.find_DI_with_residues([1, 100], [2, 101], None, arr1, arr2)
    assert [rank for _, rank in results] == [1, 1]


def test_load_from_DI_file_single_row_not_collapsed(tmp_path):
    path = tmp_path / 'one.DI'
    path.write_text('1 10 0.5\n')
    di = DirectInformationData.load_from_DI_file(str(path))
    assert di.DI_data.shape == (1,)
    assert as_tuples(di.DI_data) == [(1, 10, 0.5)]

def test_load_from_dca_output_single_row_not_collapsed(tmp_path):
    path = tmp_path / 'one.info'
    path.write_text('1 10 0.1 0.5\n')
    di = DirectInformationData.load_from_dca_output(str(path))
    assert di.DI_data.shape == (1,)
    assert as_tuples(di.DI_data) == [(1, 10, 0.5)]

def test_write_DI_data_empty_pairs_writes_empty_file(tmp_path):
    path = tmp_path / 'empty.DI'
    DirectInformationData.write_DI_data(str(path), np.zeros(0, dtype=DI_DTYPE))
    assert path.read_text() == ''

def test_write_DI_data_pairs_only_uses_int_format(tmp_path):
    # The pairs_only view from get_ranked_mapped_pairs has two fields, so the DI format must be dropped.
    path = tmp_path / 'pairs.DI'
    arr = np.array([(1, 10, 0.5), (2, 20, 0.25)], dtype=DI_DTYPE)
    DirectInformationData.write_DI_data(str(path), arr[['residue1', 'residue2']])
    assert path.read_text().split() == ['1', '10', '2', '20']

def test_find_di_with_residues_rejects_array_as_max_rank():
    # Forgetting max_rank shifts the first mapped array into its slot; that should fail loudly.
    arr = np.array([(1, 2, 0.9)], dtype=DI_DTYPE)
    with pytest.raises(TypeError):
        DirectInformationData.find_DI_with_residues([1], [2], arr)

def test_find_di_with_residues_max_rank_zero_returns_nothing():
    arr = np.array([(1, 2, 0.9)], dtype=DI_DTYPE)
    assert DirectInformationData.find_DI_with_residues([1], [2], 0, arr) == []

def test_find_di_with_residues_uses_field_names_not_positions():
    # DI first: positional row[0] would be the DI score, so matching must go by field name.
    odd = np.array([(0.9, 1, 2)], dtype=[('DI', float), ('residue1', int), ('residue2', int)])
    results = DirectInformationData.find_DI_with_residues([1], [2], None, odd)
    assert [rank for _, rank in results] == [1]
