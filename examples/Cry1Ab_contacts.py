import numpy as np
import numpy.typing as npt
from collections.abc import Iterable
from src.dcatoolkit.representation import DirectInformationData, StructureInformation, ResidueAlignment

# Reading in top 5 scoring AlphaFold3 models produced.
btr1_complex_0 = StructureInformation.read_pdb_mmCIF("examples/files/fold_cry1ab_ec12_model_0.cif")
btr1_complex_1 = StructureInformation.read_pdb_mmCIF("examples/files/fold_cry1ab_ec12_model_1.cif")
btr1_complex_2 = StructureInformation.read_pdb_mmCIF("examples/files/fold_cry1ab_ec12_model_2.cif")
btr1_complex_3 = StructureInformation.read_pdb_mmCIF("examples/files/fold_cry1ab_ec12_model_3.cif")
btr1_complex_4 = StructureInformation.read_pdb_mmCIF("examples/files/fold_cry1ab_ec12_model_4.cif")

# Getting contacts between proteins A and B
contacts_0 = btr1_complex_0.get_contacts(ca_only=False, threshold=8, chain1='A', chain2='B', auth_contacts=True)
contacts_1 = btr1_complex_1.get_contacts(ca_only=False, threshold=8, chain1='A', chain2='B', auth_contacts=True)
contacts_2 = btr1_complex_2.get_contacts(ca_only=False, threshold=8, chain1='A', chain2='B', auth_contacts=True)
contacts_3 = btr1_complex_3.get_contacts(ca_only=False, threshold=8, chain1='A', chain2='B', auth_contacts=True)
contacts_4 = btr1_complex_4.get_contacts(ca_only=False, threshold=8, chain1='A', chain2='B', auth_contacts=True)

# Selecting the residue 1 indices of those contacts
cry1_contacts_0 = [x[0] for x in contacts_0]
cry1_contacts_1 = [x[0] for x in contacts_1]
cry1_contacts_2 = [x[0] for x in contacts_2]
cry1_contacts_3 = [x[0] for x in contacts_3]
cry1_contacts_4 = [x[0] for x in contacts_4]

# Example of selecting common contacts in contacts produced by StructureInformation.
common_contacts = {tuple(x) for x in contacts_2.intersection(contacts_4).intersection(contacts_1)}

# Loading in DI file information and getting the top ranked and mapped to AF3 structures of the BT-R1 complex. 
btr1_complex_DI = DirectInformationData.load_from_DI_file("examples/files/Cry1Ab_EC12_ranked.DI")
M_R = btr1_complex_DI.get_ranked_mapped_pairs(ResidueAlignment.load_from_align_file('examples/files/Endotoxin_M_AF_align'), ResidueAlignment.load_from_align_file('examples/files/EC12_AF_align'), False)
C_R = btr1_complex_DI.get_ranked_mapped_pairs(ResidueAlignment.load_from_align_file('examples/files/Endotoxin_C_AF_align'), ResidueAlignment.load_from_align_file('examples/files/EC12_AF_align'), False)

# Providing critical residue indices of the receptor protein (that corresponds to the second column of residues)
contacts_critical = {15,16,17,18,19,20,103,104,105,106,107,108}

# Printing out results, the DI pairs constricted to the residue indices specified and within the rank threshold of 300.
print(DirectInformationData.find_DI_with_residues(cry1_contacts_0, contacts_critical, 300, M_R, C_R))
print()
print(DirectInformationData.find_DI_with_residues(cry1_contacts_1, contacts_critical, 300, M_R, C_R))
print()
print(DirectInformationData.find_DI_with_residues(cry1_contacts_2, contacts_critical, 300, M_R, C_R))
print()
print(DirectInformationData.find_DI_with_residues(cry1_contacts_3, contacts_critical, 300, M_R, C_R))
print()
print(DirectInformationData.find_DI_with_residues(cry1_contacts_4, contacts_critical, 300, M_R, C_R))

# Writing out information of contacts within a threshold to a file.
StructureInformation.write_contacts_set("examples/outputs/btr1_complex_contacts_0",btr1_complex_0.get_contacts(ca_only=False, threshold=8, chain1='A', chain2='B', auth_contacts=True))
