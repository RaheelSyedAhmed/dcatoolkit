
from src.dcatoolkit.analytics import MSATools
from src.dcatoolkit.representation import DirectInformationData, StructureInformation, Pairs, ResidueAlignment
import numpy as np

"""
CISD3_MSA = MSATools.load_from_file("examples/files/output_MSA_CISD3")
CISD3_filtered_MSA = MSATools(CISD3_MSA.filter_by_continuous_gaps(35))
CISD3_filtered_MSA.write("examples/outputs/CISD3_filtered_35_MSA.fasta")
"""

struc_6avj = StructureInformation.fetch_pdb("6AVJ")
print(struc_6avj.get_full_sequence('A'))
print(struc_6avj.get_shift_values('A', 'A'))

loaded_pairs = Pairs.load_from_ndarray([(1,2), (3,10)])
print(Pairs.get_pairs(loaded_pairs.pairs, True))