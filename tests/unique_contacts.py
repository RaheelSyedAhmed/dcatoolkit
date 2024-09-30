from pathlib import Path

def read_contacts(input_filepath: str) -> list[tuple[int, int]]:
    with open(input_filepath, 'r') as fs:
        data = fs.read().splitlines()
    contacts = set()
    for pair in data:
        res1, res2 = pair.split()
        contacts.add((int(res1), int(res2)))
    return sorted(contacts)

for contact_file in Path("tests/pdb_info").glob("monomer*"):
    result = ""
    for pair in read_contacts(str(contact_file)):
        result += str(pair[0]) + "\t" + str(pair[1]) + "\n"
    with open(contact_file, 'w') as fs:
        fs.write(result)