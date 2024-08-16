import numpy as np
from src.dcatoolkit import StructureInformation

struc_2kll = StructureInformation.fetch_pdb("2kll")
print(struc_2kll.get_shift_values('A', 'A'))