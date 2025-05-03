from context import StructureInformation

struc_1hqz = StructureInformation.fetch_pdb("1hqz", 'mmcif')
# print(struc_1hqz.res_auth_dict)
print(struc_1hqz.chain_auth_dict)
print(struc_1hqz.auth_chain_dict)

#print(struc_1hqz.get_full_sequence('2', True))
#print(struc_1hqz.get_non_missing_sequence('2', True))
#print(struc_1hqz.get_full_sequence('B'))
#print(struc_1hqz.get_non_missing_sequence('B'))
#print(sorted(struc_1hqz.get_contacts(False, 8, '1', '1', False, True)))

print(StructureInformation.fetch_pdb("5XTD", 'mmcif').get_seq_id_mapping('E', True, True))