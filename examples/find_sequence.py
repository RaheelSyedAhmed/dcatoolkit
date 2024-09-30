from context import StructureInformation


struc_1hqz = StructureInformation.fetch_pdb("1hqz")
print(struc_1hqz.get_full_sequence('A'))
print(struc_1hqz.get_full_sequence('1', auth_chain_id_supplied=True))
print(struc_1hqz.get_non_missing_sequence('A'))
print(struc_1hqz.get_non_missing_sequence('1', auth_chain_id_supplied=True))
