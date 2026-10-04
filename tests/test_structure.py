import pytest
from context import MMCIFInformation


@pytest.fixture(scope="module")
def hqz():
    # 1HQZ has 9 protein chains (A-I); chain B's atoms start at row 1089, not row 0.
    return MMCIFInformation.read_mmCIF_file("tests/pdb_info/1hqz.cif")

def test_get_start_res_id_non_first_chain(hqz):
    # Regression test: label-based [0] indexing raised KeyError for every chain after the first.
    assert hqz.get_start_res_id('B') == 3
    assert hqz.get_start_res_id('B', get_auth_res_ids=True) == 3

def test_get_start_res_id_with_auth_chain_id(hqz):
    auth_chain_b = hqz.chain_auth_dict['B']
    assert hqz.get_start_res_id(auth_chain_b, auth_chain_id_supplied=True) == hqz.get_start_res_id('B')

def test_get_start_res_id_every_chain(hqz):
    # Each chain's start should match the first residue of that chain's structure.
    for chain in hqz.unique_chains:
        expected = hqz.get_chain_specific_structure(ca_only=False, chain_id=chain).res_id[0]
        assert hqz.get_start_res_id(chain) == expected

def test_alphafold3_model_loads_full_sequences():
    # AlphaFold3 mmCIF output has no entity_poly.pdbx_seq_one_letter_code_can, so full sequences come from entity_poly_seq.
    af3 = MMCIFInformation.read_mmCIF_file("examples/files/fold_cry1ab_ec12_model_0.cif")
    assert len(af3.get_full_sequence('A')) == 607
    assert af3.get_full_sequence('A').startswith("MDNNPNINECIP")
    assert len(af3.get_full_sequence('B')) == 112

def test_full_sequence_fallback_matches_get_sequence(hqz):
    # Forcing the fallback on an RCSB file should rebuild exactly what biotite's get_sequence returns.
    import biotite.structure.io.pdbx as pdbx
    pdbx_file = pdbx.CIFFile.read("tests/pdb_info/1hqz.cif")
    del pdbx_file.block["entity_poly"]["pdbx_seq_one_letter_code_can"]
    assert hqz._read_full_sequences(pdbx_file) == hqz.full_sequences
