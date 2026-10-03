
from context import MSATools
from context import StructureInformation, Pairs


CISD3_MSA = MSATools.load_from_file("examples/files/output_MSA_CISD3")
CISD3_filtered_MSA = MSATools(CISD3_MSA.filter_by_continuous_gaps(35))
CISD3_filtered_MSA.write("examples/outputs/CISD3_filtered_35_MSA.fasta")


struc_6avj = StructureInformation.fetch_pdb("6AVJ")
print(struc_6avj.get_full_sequence('A'))
# Per-residue mapping from label residue ids to auth residue ids (replaces the old constant get_shift_values offsets).
print(struc_6avj.get_seq_id_mapping('A', seq_to_auth=True))

loaded_pairs = Pairs.load_from_ndarray([(1,2), (3,10)])
print(Pairs.get_pairs(loaded_pairs.pairs, True))