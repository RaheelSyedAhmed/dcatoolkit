from context import StructureInformation

struc_2kll = StructureInformation.fetch_pdb("2kll")
# Starting residue of chain A in label (mmCIF) numbering and in auth (author/RCSB) numbering.
print(struc_2kll.get_start_res_id('A'), struc_2kll.get_start_res_id('A', get_auth_res_ids=True))
# Per-residue mapping from label residue ids to auth residue ids (replaces the old constant get_shift_values offsets).
print(struc_2kll.get_seq_id_mapping('A', seq_to_auth=True))
