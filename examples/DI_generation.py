import numpy as np
from src.dcatoolkit.representation import DirectInformationData, StructureInformation

# Pulling a CISD3 protein structure and supplying an ndarray of indices and values to serve as a DI ndarray.
s = StructureInformation.fetch_pdb("6AVJ")
ndarray_test = np.array([(1,2,5.0), (3,4,6.0)])
print(ndarray_test)
print(ndarray_test.shape)
for row in (DirectInformationData.load_as_ndarray(ndarray_test).DI_data):
    print(row)

# Getting the minimally distant atoms in the structure considering only the residues involved and represented in the DI pairs provided.
full_atom_info = s.get_min_dist_atom_info(DirectInformationData.load_as_ndarray(ndarray_test).DI_data, 'A', 'B')
print(full_atom_info)

# Get UCSF Chimera distance commands from the supplied model numbers, chain IDs, atomic info, and further specifiers.
print(' '.join(DirectInformationData.get_dist_commands('0', '0', 'A', 'B', full_atom_info, ca_only=False, auth_res_ids=True)))