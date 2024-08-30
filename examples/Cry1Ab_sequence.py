from src.dcatoolkit.representation import DirectInformationData, StructureInformation, ResidueAlignment


struc_6dj4 = StructureInformation.fetch_pdb("6DJ4")
print(struc_6dj4.get_full_sequence('A'))
print(struc_6dj4.get_non_missing_sequence('A'))
