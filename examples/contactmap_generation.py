import matplotlib.pyplot as plt
from context import StructureInformation

# Fetching the CISD3 protein structure from RCSB and getting the all-atom contacts within 8 Angstrom of each other
s = StructureInformation.fetch_pdb("6AVJ", "mmcif")
contacts_set = s.get_contacts(ca_only=False, threshold=8, chain1="A", chain2="A", auth_seq_id=False)

# Plotting the residues on a contact map.
x, y = zip(*contacts_set)
figure = plt.figure()
plt.scatter(x,y, s=2)
plt.show()
