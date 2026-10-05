
from importlib.metadata import version as _version

__version__ = _version("dcatoolkit")
from .analytics import MSATools
from .representation import (
    DirectInformationData,
    MMCIFInformation,
    Pairs,
    PDBInformation,
    ResidueAlignment,
    StructureInformation,
)

__all__ = ['DirectInformationData', 'MMCIFInformation', 'MSATools', 'PDBInformation', 'Pairs', 'ResidueAlignment', 'StructureInformation']