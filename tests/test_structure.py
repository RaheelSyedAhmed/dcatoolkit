from context import MMCIFInformation
import pytest


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
