import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dcatoolkit.representation import MMCIFInformation, PDBInformation, StructureInformation

__all__ = ['MMCIFInformation', 'PDBInformation', 'StructureInformation']