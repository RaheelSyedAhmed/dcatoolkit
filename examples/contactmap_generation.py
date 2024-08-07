import matplotlib.pyplot as plt
from src.dcatoolkit.representation import DirectInformationData, StructureInformation

s = StructureInformation.fetch_pdb("6AVJ")
contacts_set = s.get_contacts(ca_only=False, threshold=8, chain1="A", chain2="A", auth_contacts=False)


x, y = zip(*contacts_set)
figure = plt.figure()
plt.scatter(x,y, s=2)
plt.show()
