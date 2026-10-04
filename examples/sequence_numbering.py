from context import MMCIFInformation

pdb_ids = ['1pzs', '3ddv', '6avj', '3d7i', '4OO8']

for index, AA in MMCIFInformation.fetch_pdb("4OO8", "mmcif").get_valid_chain_residues("A"):
   if index > 30:
       break
   else:
       print(index, AA)