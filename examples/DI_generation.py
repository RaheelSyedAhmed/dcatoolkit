import numpy as np
from ..dcatoolkit.representation import DirectInformationData, StructureInformation

s = StructureInformation.fetch_pdb("6AVJ")
ndarray_test = np.array([(1,2,5.0), (3,4,6.0)])
print(ndarray_test)
print(ndarray_test.shape)
for row in (DirectInformationData.load_as_ndarray(ndarray_test).DI_data):
    print(row)

full_atom_info = s.get_min_dist_atom_info(DirectInformationData.load_as_ndarray(ndarray_test).DI_data, 'A', 'B')
print(full_atom_info)

print(' '.join(DirectInformationData.get_dist_commands('0', '0', 'A', 'B', full_atom_info, ca_only=False, auth_res_ids=True)))