from context import ResidueAlignment
import pandas as pd
import pytest

test_cases_description = """
Test Case 1
First:  MA.KLT
Second: MAAKLT

Test Case 2
First:  Q..EWLP
Second: QATEWLP

Test Case 3
First:  D.G.H.V
Second: DAGAHAV

Test Case 4
First:  TAKAPF
Second: TAMAPF

Test Case 5
First:  FACARAA
Second: FACAR--

Test Case 6
First:  ALMAY
Second: ALMAY

Test Case 7
First:  VAIDTSK
Second: VAID--K

Test Case 8
First:  NG.TA
Second: NG-TA

Test Case 9
First:  STWLPL
Second: SAWLPL

Test Case 10
First:  HCSTCRAAC
Second: H--TAR--C
"""


test_cases : list[tuple[int, int, str, str]] = [
    (47, 15, 'MA.KLT', 'MAAKLT'),
    (39, 3, 'Q..EWLP', 'QATEWLP'),
    (8, 45, 'D.G.H.V', 'DAGAHAV'),
    (26, 19, 'TAKAPF', 'TAMAPF'),
    (33, 9, 'FACARAA', 'FACAR--')
]
test_cases_validation = [
    (42, 28, 'ALMAY', 'ALMAY'),
    (14, 11, 'VAIDTSK', 'VAID--K'),
    (10, 5, 'NG.TA', 'NG-TA'),
    (2, 44, 'STWLPL', 'SAWLPL'),
    (17, 34, 'HCSTCRAAC', 'H--TAR--C')
]

test_answers = [
    [(47, 48, pd.NA, 49, 50, 51), tuple('MA.KLT'), tuple('MAAKLT'), (15,16,17,18,19,20)],
    [(39, pd.NA, pd.NA, 40, 41, 42, 43), tuple('Q..EWLP'), tuple('QATEWLP'), (3,4,5,6,7,8,9)],
    [(8, pd.NA, 9, pd.NA, 10, pd.NA, 11), tuple('D.G.H.V'), tuple('DAGAHAV'), (45,46,47,48,49,50,51)],
    [(26, 27, 28, 29, 30, 31), tuple('TAKAPF'), tuple('TAMAPF'), (19,20,21,22,23,24)],
    [(33, 34, 35, 36, 37, 38, 39), tuple('FACARAA'), tuple('FACAR--'), (9,10,11,12,13,pd.NA,pd.NA)],
    [(42, 43, 44, 45, 46), tuple('ALMAY'), tuple('ALMAY'), (28,29,30,31,32)],
    [(14, 15, 16, 17, 18, 19, 20), tuple('VAIDTSK'), tuple('VAID--K'), (11, 12, 13, 14, pd.NA, pd.NA, 15)],
    [(10,11,pd.NA,12,13), tuple('NG.TA'), tuple('NG-TA'), (5,6,pd.NA,7,8)],
    [(2,3,4,5,6,7), tuple('STWLPL'), tuple('SAWLPL'), (44,45,46,47,48,49)],
    [(17,18,19,20,21,22,23,24,25), tuple('HCSTCRAAC'), tuple('H--TAR--C'), (34, pd.NA, pd.NA, 35,36,37, pd.NA, pd.NA, 38)]
]

def test_residue_alignments():
    test_num = 0
    for test_num, test_case in enumerate(test_cases):
        domain_start, protein_start, first_seq, second_seq = test_case
        module_result = list(ResidueAlignment(f"Test_{test_num}", f"Test {test_num}", domain_start, protein_start, first_seq, second_seq).reference_mapping.itertuples(index=False, name=None))
        answer = list(zip(*test_answers[test_num]))
        assert module_result == answer
    for test_num, test_case in enumerate(test_cases_validation, start=test_num+1):
        domain_start, protein_start, first_seq, second_seq = test_case
        module_result = list(ResidueAlignment(f"Test_{test_num}", f"Test {test_num}", domain_start, protein_start, first_seq, second_seq).reference_mapping.itertuples(index=False, name=None))
        answer = list(zip(*test_answers[test_num]))
        assert module_result == answer

def test_reference_mapping_restricted_skips_gap_then_exhausts():
    # Can handle excess residues, but not missing any ones that are supposed to be there.
    # (8, 'R') doesn't match 'F', so it's skipped and 'F' resyncs on (12, 'F') across the 9-11 structural gap;
    # 'T' then finds valid_residues exhausted and gets no protein_index.
    valid_residues = [(5, 'M'), (6, 'A'), (7, 'A'), (8, 'R'), (12, 'F')]
    ra = ResidueAlignment('name1', 'name2', 1, 1, 'MAAFT', 'MAAFT', valid_residues=valid_residues)
    assert ra.domain_to_protein == {1: 5, 2: 6, 3: 7, 4: 12}
    assert pd.isna(ra.reference_mapping.iloc[4]['protein_index'])

def test_reference_mapping_base_case_with_gaps():
    # Gaps on both sides: domain 'C' (2) aligns to a protein gap, protein 'C' (2) aligns to a domain gap.
    # Neither should map to the other since they're different alignment columns.
    ra = ResidueAlignment('dom', 'prot', 1, 1, 'AC-D', 'A-CD')
    assert ra.domain_to_protein == {1: 1, 3: 3}
    assert ra.protein_to_domain == {1: 1, 3: 3}

def test_mapping_dicts_use_plain_python_ints():
    ra = ResidueAlignment('dom', 'prot', 1, 300, 'AC', 'AC')
    for mapping in (ra.domain_to_protein, ra.protein_to_domain):
        assert all(type(k) is int and type(v) is int for k, v in mapping.items())

def test_reference_mapping_base_case_sequential():
    ra = ResidueAlignment('dom', 'prot', 5, 9, 'ACD', 'ACD')
    assert ra.domain_to_protein == {5: 9, 6: 10, 7: 11}

def test_reference_mapping_restricted_repeated_residues():
    # Regression test: a naive re-copy of valid_residues per outer-loop iteration would
    # collapse every position onto the first matching residue (1,2,3 -> 10,10,10) instead
    # of correctly advancing through valid_residues as each one is consumed.
    valid_residues = [(10, 'A'), (11, 'A'), (12, 'A')]
    ra = ResidueAlignment('dom', 'prot', 1, 1, 'AAA', 'AAA', valid_residues=valid_residues)
    assert ra.domain_to_protein == {1: 10, 2: 11, 3: 12}
    # The caller's original list must not be mutated.
    assert valid_residues == [(10, 'A'), (11, 'A'), (12, 'A')]

def test_reference_mapping_restricted_skips_mismatched_residues():
    # valid_residues contains (101, 'X'), which never appears in protein_text at all.
    # The algorithm should skip over it and resynchronize on the next match.
    valid_residues = [(100, 'A'), (101, 'X'), (102, 'C'), (103, 'D')]
    ra = ResidueAlignment('dom', 'prot', 1, 1, 'ACD', 'ACD', valid_residues=valid_residues)
    assert ra.domain_to_protein == {1: 100, 2: 102, 3: 103}

def test_reference_mapping_restricted_with_gaps():
    valid_residues = [(1, 'A'), (2, 'C'), (3, 'D')]
    ra = ResidueAlignment('dom', 'prot', 1, 1, 'AC-D', 'A-CD', valid_residues=valid_residues)
    assert ra.domain_to_protein == {1: 1, 3: 3}

def test_reference_mapping_restricted_empty_valid_residues_maps_nothing():
    # An empty list means no resolved residues, so nothing should map (unlike None, which numbers sequentially).
    ra = ResidueAlignment('dom', 'prot', 1, 1, 'AC', 'AC', valid_residues=[])
    assert ra.domain_to_protein == {}
    assert ra.reference_mapping['protein_index'].isna().all()

def test_reference_mapping_restricted_exhausted_falls_back_to_na():
    # valid_residues runs out before the alignment does; the remaining protein_index values should be NA.
    ra = ResidueAlignment('dom', 'prot', 1, 1, 'AC', 'AC', valid_residues=[(1, 'A')])
    assert ra.domain_to_protein == {1: 1}
    assert pd.isna(ra.reference_mapping.iloc[1]['protein_index'])

ALIGN_TEXT = 'Dom\n1\nAC\n2\n\nProt\n5\nAC\n6\n'

def write_align(tmp_path, text):
    path = tmp_path / 'test.align'
    path.write_text(text)
    return str(path)

def test_load_from_align_file_valid(tmp_path):
    ra = ResidueAlignment.load_from_align_file(write_align(tmp_path, ALIGN_TEXT))
    assert ra.domain_to_protein == {1: 5, 2: 6}

def test_load_from_align_file_passes_valid_residues(tmp_path):
    # ALIGN_TEXT's protein side is 'AC' starting at 5; (6, 'X') isn't in the text, so it's skipped and 'C' resyncs on (7, 'C').
    valid_residues = [(1, 'Z'), (2, 'Z'), (3, 'Z'), (4, 'Z'), (5, 'A'), (6, 'X'), (7, 'C')]
    ra = ResidueAlignment.load_from_align_file(write_align(tmp_path, ALIGN_TEXT), valid_residues=valid_residues)
    assert ra.domain_to_protein == {1: 5, 2: 7}

def test_load_from_align_file_rejects_extra_entry(tmp_path):
    with pytest.raises(ValueError, match='Expected 2 complete alignment entries'):
        ResidueAlignment.load_from_align_file(write_align(tmp_path, ALIGN_TEXT + '\nExtra\n1\nAC\n2\n'))

def test_load_from_align_file_rejects_truncated_entry(tmp_path):
    with pytest.raises(ValueError, match='Incomplete alignment entry'):
        ResidueAlignment.load_from_align_file(write_align(tmp_path, 'Dom\n1\nAC\n2\n\nProt\n5\nAC\n'))

def test_load_from_align_file_rejects_trailing_partial_entry(tmp_path):
    # Two complete entries plus leftover lines used to load silently, ignoring the leftovers.
    with pytest.raises(ValueError, match='Incomplete alignment entry'):
        ResidueAlignment.load_from_align_file(write_align(tmp_path, ALIGN_TEXT + '\nExtra\n1\n'))
