
from src.dcatoolkit.analytics import MSATools

CISD3_MSA = MSATools.load_from_file("examples/files/output_MSA_CISD3")
CISD3_filtered_MSA = MSATools(CISD3_MSA.filter_by_continuous_gaps(35))
CISD3_filtered_MSA.write("examples/outputs/CISD3_filtered_35_MSA.fasta")