import subprocess

threshold_AA = float(8)
threshold_CA = float(10)


with open("pdb_ids.txt", 'r') as fs:
    data = fs.read().splitlines()

for line in data:
    pdb_id, chain1, chain2 = (line.split())
    orig_chain1 = chain1.split("/")[0]
    auth_chain1 = chain1.split("/")[-1]
    orig_chain2 = chain2.split("/")[0]
    auth_chain2 = chain2.split("/")[-1]
    awk_AA = "awk '{if(NR>1) print $3,$8}'"+ f" contactmap_allatom_{pdb_id.lower()}_{auth_chain1}{auth_chain2}_{threshold_AA} > monomer_allatom_{pdb_id}_{threshold_AA}"
    awk_CA = "awk '{if(NR>1) print $1,$4}'"+ f" contactmap_calpha_{pdb_id.lower()}_{auth_chain1}{auth_chain2}_{threshold_CA} > monomer_calpha_{pdb_id}_{threshold_CA}"
    subprocess.run(["python2", "interface_contacts_allatom_args.py", f"{pdb_id.lower()}.pdb", auth_chain1, auth_chain2, str(threshold_AA)])
    subprocess.run(["python2", "interface_contacts_calpha_args.py", f"{pdb_id.lower()}.pdb", auth_chain1, auth_chain2, str(threshold_CA)])
    subprocess.run(awk_AA, shell=True, capture_output=True, text=True)
    subprocess.run(awk_CA, shell=True, capture_output=True, text=True)