import numpy as np
from src.dcatoolkit.representation import DirectInformationData, StructureInformation

# Fetching IL33 protein and getting alpha-carbon contacts and all-atom contacts
struc_2kll = StructureInformation.fetch_pdb("2kll")
ca_2kll_contacts = struc_2kll.get_contacts(ca_only=True, threshold=10, chain1='A', chain2='A', auth_contacts=True)
aa_2kll_contacts = struc_2kll.get_contacts(ca_only=False, threshold=8, chain1='A', chain2='A', auth_contacts=True)
#print(ca_2kll_contacts)
#print(aa_2kll_contacts)

# Produce the contacts where residue 1 is always less than residue 2 (useful for same chain contacts)
aa_2kll_contacts_unmirrored = {tuple(sorted(x)) for x in aa_2kll_contacts}
StructureInformation.write_contacts_set("examples/outputs/2kll_ca.txt", ca_2kll_contacts)
StructureInformation.write_contacts_set("examples/outputs/2kll_aa.txt", aa_2kll_contacts_unmirrored)

# Write example original contacts as sets to avoid redundancy and to compare.
check_set = set()
with open("examples/files/2kll_ca_10", 'r') as fs:
    for line in fs:
        check_set.add(tuple([int(x) for x in line.split()]))
StructureInformation.write_contacts_set("examples/outputs/orig_2kll_ca_10",check_set)

check_set = set()
with open("examples/files/2kll_aa_8", 'r') as fs:
    for line in fs:
        check_set.add(tuple([int(x) for x in line.split()]))
StructureInformation.write_contacts_set("examples/outputs/orig_2kll_aa_8",check_set)
