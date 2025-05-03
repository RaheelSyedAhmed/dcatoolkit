from context import ResidueAlignment
import pandas as pd

test_cases = """
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


test_cases = [
    (47, 15, 'MA.KLT', 'MAAKLT'),
    (39, 3, 'Q..EWLP', 'QATEWLP'),
    (8, 45, 'D.G.H.V', 'DAGAHAV'),
    (26, 19, 'TAKAPF', 'TAMAPF'),
    (33, 9, 'FACARAA', 'FACAR--')
]
test_cases_validation = [
    (42, 28, 'ALMAY', 'ALMAY', ),
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

# Can handle excess residues, but not missing any ones that are supposed to be there.
print(ResidueAlignment('name1', 'name2', 1, 1, 'MAAFT', 'MAAFT', valid_residues=[(5, 'M'), (6, 'A'), (7, 'A'), (8, 'R'), (12, 'F')]))