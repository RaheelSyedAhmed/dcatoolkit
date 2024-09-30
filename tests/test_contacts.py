from context import MMCIFInformation, PDBInformation
from pathlib import Path
import biotite.structure.io.pdbx as pdbx
import biotite.database.rcsb as rcsb

pdb_id_chain_map = {}
with open("tests/pdb_info/pdb_ids.txt", "r") as fs:
    pdb_id_data = fs.read().splitlines()
    for line in pdb_id_data:
        pdb_id, chain1, chain2 = line.split()
        if len(chain1.split("/")) > 1:
            chain1, auth_chain1 = chain1.split("/")
        else:
            auth_chain1 = chain1
        if len(chain2.split("/")) > 1:
            chain2, auth_chain2 = chain2.split("/")
        else:
            auth_chain2 = chain2
        pdb_id_chain_map[pdb_id] = (chain1, auth_chain1, chain2, auth_chain2)        

def read_contacts(input_filepath: str) -> set[tuple[int, int]]:
    with open(input_filepath, 'r') as fs:
        data = fs.read().splitlines()
    results = set()
    for pair in data:
        res1, res2 = pair.split()
        res1 = int(res1)
        res2 = int(res2)
        if res1 < res2:
            results.add((int(res1), int(res2)))
    return results
    #return set([(int(x.split()[0]), int(x.split()[1])) for x in data])

def drop_inord_res(contacts: set[tuple[int, int]]) -> set[tuple[int, int]]:
    results = set()
    for contact in contacts:
        if contact[0] < contact[1]:
            results.add(contact)
    return results

def compare(*args) -> None:
    for arg1 in args:
        for arg2 in args:
            assert arg1 == arg2

def check_contacts(test_CA: bool, threshold: float):
    cif_files = list(Path("tests/pdb_info").glob("*.cif"))
    if test_CA:
        print("Testing CA")
    else:
        print("Testing AA")

    for cif_file in cif_files:
        pdb_id = cif_file.stem.upper()
        if test_CA:
            corresponding_file = f"tests/pdb_info/monomer_calpha_{pdb_id}_10.0"
        else:
            corresponding_file = f"tests/pdb_info/monomer_allatom_{pdb_id}_8.0"
        
        cif_file_contacts = read_contacts(corresponding_file)
        chain1, auth_chain1, chain2, auth_chain2 = pdb_id_chain_map[pdb_id]
        fetch_cif_contacts = {(int(x[0]), int(x[1])) for x in MMCIFInformation.fetch_pdb(pdb_id).get_contacts(test_CA, threshold, chain1, chain2, auth_contacts=True)}
        read_cif_contacts = {(int(x[0]), int(x[1])) for x in MMCIFInformation.read_mmCIF_file(str(cif_file)).get_contacts(test_CA, threshold, chain1, chain2, auth_contacts=True)}
        fetch_authchain_cif_contacts = {(int(x[0]), int(x[1])) for x in MMCIFInformation.fetch_pdb(pdb_id).get_contacts(test_CA, threshold, auth_chain1, auth_chain2, auth_contacts=True, auth_chain_id_supplied=True)}
        read_authchain_cif_contacts = {(int(x[0]), int(x[1])) for x in MMCIFInformation.read_mmCIF_file(str(cif_file)).get_contacts(test_CA, threshold, auth_chain1, auth_chain2, auth_contacts=True, auth_chain_id_supplied=True)}
        fetch_pdb_contacts = {(int(x[0]), int(x[1])) for x in PDBInformation.fetch_pdb(pdb_id, struc_format="pdb").get_contacts(test_CA, threshold, auth_chain1, auth_chain2, auth_contacts=True)}
        read_pdb_contacts = {(int(x[0]), int(x[1])) for x in PDBInformation.read_pdb_file(f"tests/pdb_info/{pdb_id.lower()}.pdb").get_contacts(test_CA, threshold, auth_chain1, auth_chain2, auth_contacts=True)}
        
        
        fetch_cif_contacts = drop_inord_res(fetch_cif_contacts)
        fetch_authchain_cif_contacts = drop_inord_res(fetch_authchain_cif_contacts)
        read_cif_contacts = drop_inord_res(read_cif_contacts)
        read_authchain_cif_contacts = drop_inord_res(read_authchain_cif_contacts)
        fetch_pdb_contacts = drop_inord_res(fetch_pdb_contacts)
        read_pdb_contacts = drop_inord_res(read_pdb_contacts)
        
        if pdbx.get_model_count(pdbx.CIFFile.read(rcsb.fetch(pdb_id, format="mmcif"))) <= 1:
            compare(fetch_cif_contacts, fetch_pdb_contacts, read_cif_contacts, read_pdb_contacts)
            print(f"{pdb_id} has no difference between cif and pdb reading.")

            compare(fetch_cif_contacts, fetch_authchain_cif_contacts, read_cif_contacts, read_authchain_cif_contacts)
            print(f"{pdb_id} has no issue reading with auth chains and asym chains.")

            compare(fetch_cif_contacts, cif_file_contacts)
            print(f"{pdb_id} has no difference between cif and interface_contacts_X.py reading.")
        else:
            compare(fetch_cif_contacts, fetch_pdb_contacts, read_cif_contacts, read_pdb_contacts)
            print(f"{pdb_id} has no difference between cif and pdb reading.")

            compare(fetch_cif_contacts, fetch_authchain_cif_contacts, read_cif_contacts, read_authchain_cif_contacts)
            print(f"{pdb_id} has no issue reading with auth chains and asym chains.")
        
def test_contacts():
    check_contacts(test_CA=True, threshold=10)
    check_contacts(test_CA=False, threshold=8)