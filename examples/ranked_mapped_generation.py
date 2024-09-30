from context import DirectInformationData, ResidueAlignment

# Testing of DirectInformationData loading from two separate files.
beta_trefoil_IDID = DirectInformationData.load_from_dca_output("examples/files/Beta_Trefoil_MSA_1.info")
beta_trefoil_DDID = DirectInformationData.load_from_DI_file("examples/files/Beta_Trefoil_MSA_1.DI")
print(beta_trefoil_DDID.DI_data)
print(beta_trefoil_IDID.DI_data)

# Make sure they load the same
#print(np.all(beta_trefoil_DDID.DI_data == beta_trefoil_IDID.DI_data, where=True))

# Represent Residue Alignment
RA_nkll = ResidueAlignment.load_from_align_file("examples/files/2kll_align")
#print(RA_nkll.reference_mapping) # Analogous to _align_reference.txt

# Attempt finding ranked mapped DIs
print(beta_trefoil_DDID.get_ranked_mapped_pairs(RA_nkll, RA_nkll, True))
DirectInformationData.write_DI_data("examples/outputs/beta_trefoil_trial_rm_DI.DI", beta_trefoil_DDID.get_ranked_mapped_pairs(RA_nkll, RA_nkll, False))
DirectInformationData.write_DI_data("examples/outputs/beta_trefoil_trial_rm.DI", beta_trefoil_DDID.get_ranked_mapped_pairs(RA_nkll, RA_nkll, True))