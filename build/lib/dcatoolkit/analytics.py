import re
from collections import Counter
from collections.abc import Callable
from typing import Optional, Union, Literal
import string
import io
import numpy as np
import numpy.typing as npt

class MSATools:
    """
    Tools and interface for encapsulating MSA data and providing functionality for filtering and analysis.

    Parameters
    ----------
    MSA : list of tuple of str, str
        Loaded MSA that is a list of tuples where the first element is the header and the second element is its corresponding sequence.
    """
    def __init__(self, MSA: list[tuple[str, str]]):
        self.MSA = MSA
    
    @staticmethod
    def load_from_file(msa_source: Union[str, io.IOBase]) -> 'MSATools':
        """
        Generates MSATools object from an MSA file in ".afa" format.

        Parameters
        ----------
        msa_source : str or io.IOBase
            Filepath or IOBase of the MSA in ".afa" format that is provided.
        
        Returns
        -------
        MSATools
            An MSATools instance with the appropriate list of (header, sequence) tuples where sequences are simplified and converted to single line format.
        """
        data = ""
        msa_entries: list[tuple[str, str]] = []
        if isinstance(msa_source, str):
            with open(msa_source, 'r') as fs:
                data = fs.read()
        elif isinstance(msa_source, io.BytesIO):
            data = msa_source.getvalue().decode()
        elif isinstance(msa_source, io.StringIO):
            data = msa_source.getvalue()
        else:
            raise Exception("msa_file is not bytesIO, stringIO, or a filepath.")
        split_data = data.split(">")[1:]
        for entry in split_data:
            line_split_entry = entry.split("\n")
            header = line_split_entry[0]
            sequence = "".join(line_split_entry[1:])
            msa_entries.append((">"+header, sequence))
        return MSATools(msa_entries)

    @staticmethod
    def get_sequence_max_cont_gaps(sequence: str) -> int:
        """
        Find maximum number of continuous gaps in a specific sequence.

        Parameters
        ----------
        sequence : str
            Sequence of characters, potentially containing multiple of '-', a gap character.
        
        Returns
        -------
        int
            The maximum number of continuous gaps in a sequence.
        """
        dash_match = re.findall(r"-+", sequence)
        gap_counts = [len(match) for match in dash_match]
        if len(gap_counts) > 0:
            return max(gap_counts)
        else:
            return 0
    
    def gap_frequency(self) -> tuple[dict[int, int], dict[int, float]]:
        """
        Calculates the frequency of maximum continuous gaps throughout the MSA where the key corresponds to the number of continous gaps and the value corresponds to the number of sequences or the cumulative percentage of their sequences.

        Returns
        -------
        tuple of dict of int, int and dict of int, int
            Two element tuple where first element is a frequency count dictionary and the second element is a cumulative percentage of sequences with a specific maximum number of continous gaps.
        """
        max_gap_counts = []
        for header, sequence in self.MSA:
            max_gap_counts.append(MSATools.get_sequence_max_cont_gaps(sequence))
        frequency_count_dict = dict(Counter(max_gap_counts))
        cumul_perc_dict = {}
        cumul_count = 0
        for key in sorted(frequency_count_dict.keys()):
            value = frequency_count_dict[key]
            cumul_count += value
            cumul_perc_dict[key] = cumul_count / len(self.MSA)
        return (frequency_count_dict, cumul_perc_dict)
    
    def filter_by_continuous_gaps(self, max_gaps: Optional[int]=None) -> list[tuple[str, str]]:
        """
        Filter out entries in your MSA by the number of maximum continuous gaps specified unless None is provided. Also, removes .s and lowercase letters from the sequence.

        Parameters
        ----------
        max_gaps : int
            The maximum allowed number of continuous gaps in a sequence

        Returns
        -------
        list of tuple of str, str
            List of entries that are valid in that their sequences' number of maximum continuous gaps is within the threshold supplied as `max_gaps`. 
        """
        table = str.maketrans('', '', string.ascii_lowercase+".")
        if max_gaps is None:
            kept_entries = []
            for header, sequence in self.MSA:
                sequence = sequence.translate(table)
                kept_entries.append((header, sequence))
            return kept_entries
        else:
            kept_entries = []
            for header, sequence in self.MSA:
                sequence = sequence.translate(table)
                if MSATools.get_sequence_max_cont_gaps(sequence) <= max_gaps:
                    kept_entries.append((header, sequence))
        return kept_entries

    def gap_proportion(self, agg_func: Callable[..., float | int]=np.mean, axis: Literal[0, 1] = 0) -> float | int:
        """
        Evaluates gap frequency per alignment position, or column, in the MSA.

        Parameters
        ----------
        agg_func: function = np.mean
            The aggregation function applied to get the expected result, usually a mean, max, or min value, of the gap frequencies present per alignment position in the MSA.
        axis: int or str
            When axis is 0, the aggregation function is applied per column for entries from every row in that column. When axis is 1, the aggregation function is applied per row for entries from every column in that row.
        
        Return
        ------
        float
            A numerical value determined by the aggregation function supplied over the gap frequencies of the alignment positions in the MSA.
        """
        sequence_matrix = self.as_matrix()
        non_alpha_counts = np.sum(~np.char.isalpha(sequence_matrix), axis)
        non_alpha_counts = non_alpha_counts / sequence_matrix.shape[axis]
        return agg_func(non_alpha_counts)
        

    def write(self, destination: Union[str, io.IOBase]) -> None:
        """
        Writes this MSA's headers and sequences to the destination specified.
        
        Parameters
        ----------
        destination : str or io.IOBase
            Filepath or IO to write the MSA supplied to.

        Returns
        -------
        None
        """
        if isinstance(destination, str):
            with open(destination, 'w') as fs:
                for header, sequence in self.MSA:
                    fs.write(header)
                    fs.write("\n")
                    fs.write(sequence)
                    fs.write("\n")
        elif isinstance(destination, io.IOBase):
            for header, sequence in self.MSA:
                destination.write(header)
                destination.write("\n")
                destination.write(sequence)
                destination.write("\n")

    def as_matrix(self) -> npt.NDArray:
        """
        Represents the MSA as a numpy matrix of sequences.

        Returns
        -------
        npt.NDArray
            A matrix of "number of sequences" rows and "number of alignment positions" columns. Each cell is the sequence character for that sequence at that position.
        """
        return np.array([list(seq) for _, seq in self.MSA])

    def __str__(self) -> str:
        """
        Returns the sequences present in the loaded MSA in string format.

        Returns
        -------
        str
            Each sequence in the instance separated with newline characters.
        """
        return "\n".join([seq for _, seq in self.MSA])
    
    def __len__(self):
        """
        Returns the number of sequences, and equivalently, the number of headers in the MSA.

        Returns
        -------
        int
            length of the MSA list of header, sequence tuples.
        """
        return len(self.MSA)