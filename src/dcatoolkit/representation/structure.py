from typing import Literal, Union, overload

import biotite.database.rcsb as rcsb
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
import numpy.typing as npt
import pandas as pd
from biotite.sequence import ProteinSequence
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist


class StructureInformation:
    """
    Information regarding a protein structure, obtained from a protein structure file.

    Uses `fetch_pdb()` to pull protein structure information from RCSB. Uses `read_mmCIF_file()` or `read_pdb_file()` to supply a filepath to pull protein structure information from a file.

    See Also
    --------
    MMCIFInformation : Structure information from a PDBx/mmCIF file.
    PDBInformation : Structure information from a PDB file.
    """
    @overload
    @staticmethod
    def fetch_pdb(pdb_id: str, struc_format: Literal["mmcif"], model_num: int=1) -> 'MMCIFInformation':
        ...

    @overload
    @staticmethod
    def fetch_pdb(pdb_id: str, struc_format: Literal["pdb"], model_num: int=1) -> 'PDBInformation':
        ...


    @staticmethod
    def fetch_pdb(pdb_id: str, struc_format: Literal["mmcif", "pdb"]="mmcif", model_num: int=1) -> Union['MMCIFInformation', 'PDBInformation']:
        """
        Fetches a PDB entry from RCSB as a PDBx/mmCIF or PDB file and compiles the information into a StructureInformation instance.

        Parameters
        ----------
        pdb_id : str
            PDB ID to be fetched from the RCSB database.
        struc_format : {"mmcif", "pdb"}, default "mmcif"
            The format of the file to pull from the RCSB database.
        model_num : int, default 1
            The model number to access from the PDB to ensure an AtomArray is returned containing the atom information of the protein structure.

        Returns
        -------
        MMCIFInformation or PDBInformation
            MMCIFInformation if `struc_format` is ``"mmcif"``, PDBInformation if it is ``"pdb"``.

        Raises
        ------
        TypeError
            If the fetched data was not found and None was returned instead.
        ValueError
            If `struc_format` is not ``"mmcif"`` or ``"pdb"``.
        """
        fetched_data = rcsb.fetch(pdb_id, struc_format)
        if fetched_data is None:
            raise TypeError("RCSB fetch failed. Try fetch again.")
        elif struc_format == "mmcif":
            pdbx_file = pdbx.CIFFile.read(fetched_data)
            return MMCIFInformation(pdbx.get_structure(pdbx_file=pdbx_file, model=model_num, use_author_fields=False), pdbx_file, model_num)
        elif struc_format == "pdb":
            pdb_file = pdb.PDBFile.read(fetched_data)
            return PDBInformation(pdb.get_structure(pdb_file=pdb_file, model=model_num), pdb_file=pdb_file, model_num=model_num)
        else:
            raise ValueError(f"struc_format {struc_format} is not valid or currently supported by DCA Toolkit")
    @staticmethod
    def read_mmCIF_file(pdbx_filepath: str, model_num: int=1) -> 'MMCIFInformation':
        """
        Reads a PDBx/mmCIF file from a filepath and compiles the information into an MMCIFInformation instance.

        Parameters
        ----------
        pdbx_filepath : str
            Filepath of the PDBx/mmCIF file to be read.
        model_num : int, default 1
            The model number to access from the PDB to ensure an AtomArray is returned containing the atom information of the protein structure.

        Returns
        -------
        MMCIFInformation
            MMCIFInformation generated from the `biotite.structure.io.pdbx.get_structure()` function using the PDBx file read from `pdbx_filepath`.
        """
        pdbx_file = pdbx.CIFFile.read(pdbx_filepath)
        return MMCIFInformation(pdbx.get_structure(pdbx_file, model=model_num, use_author_fields=False), pdbx_file, model_num)
    
    @staticmethod
    def read_pdb_file(pdb_filepath: str, model_num: int=1) -> 'PDBInformation':
        """
        Reads a PDB file from a filepath and compiles the information into a PDBInformation instance.

        Parameters
        ----------
        pdb_filepath : str
            Filepath of the PDB file to be read.
        model_num : int, default 1
            The model number to access from the PDB to ensure an AtomArray is returned containing the atom information of the protein structure.

        Returns
        -------
        PDBInformation
            PDBInformation generated from the `biotite.structure.io.pdb.get_structure()` function using the PDB file read from `pdb_filepath`.
        """
        pdb_file = pdb.PDBFile.read(pdb_filepath)
        return PDBInformation(pdb.get_structure(pdb_file, model=model_num), pdb_file, model_num)
    
    @staticmethod
    def write_contacts_set(filepath : str, contacts_set : set[tuple[int, int]]) -> None:
        """
        Write the contacts generated from `get_contacts()` or a general set of tuples of pairs, sorted, one tab-separated pair per line.

        Parameters
        ----------
        filepath : str
            Path of file to output `contacts_set` to.
        contacts_set : set of tuple of (int, int)
            Set of tuples of pairs that represent contacts.
        """
        contacts_list = sorted(contacts_set)
        with open(filepath, 'w') as fs:
            for pair in contacts_list:
                fs.write(str(pair[0]) + "\t" + str(pair[1]) + "\n")

class MMCIFInformation(StructureInformation):
    """
    Information regarding a protein structure, obtained from a PDBx/mmCIF protein structure file.

    Parameters
    ----------
    structure : biotite.structure.AtomArray
        Structure obtained from a PDBx/mmCIF file with a specified model number, read with label (not auth) residue and chain ids.
    pdbx_file : biotite.structure.io.pdbx.CIFFile
        A PDBx/mmCIF file that contains generic information and atomic information of the protein structure categorized into mmCIF blocks.
    model_num : int
        The model number to access from the PDB to ensure an AtomArray is returned containing the atom information of the protein structure.

    Attributes
    ----------
    structure : biotite.structure.AtomArray
        The `structure` supplied.
    pdbx_file : biotite.structure.io.pdbx.CIFFile
        The `pdbx_file` supplied.
    model_num : int
        The `model_num` supplied.
    full_sequences : dict of {str : str}
        The full protein sequences, including missing residues, from the PDBx/mmCIF file, keyed by auth chain id. See `_read_full_sequences()`.
    non_missing_sequences : dict of {str : str}
        The protein sequences, without missing residues, built from the non-hetero atoms of `structure`, keyed by label chain id.
    first_block : str
        Name of the first data block of `pdbx_file`.
    unique_chains : numpy.ndarray
        Array of unique ``label_asym_id`` entries of ``ATOM`` records, which corresponds to unique label chain ids.
    chain_auth_dict : dict of {str : str}
        Uses label chain id as a key and provides auth chain id as a value.
    auth_chain_dict : dict of {str : str}
        Uses auth chain id as a key and provides label chain id as a value.
    atom_site_df : pandas.DataFrame
        The full ``atom_site`` category of `pdbx_file` as strings, covering every model and alternate location.
    atom_df : pandas.DataFrame
        The ``ATOM`` rows of `atom_site_df`, with residue ids, atom ids, coordinates, and B-factors converted to numbers.
    """
    def __init__(self, structure, pdbx_file: pdbx.CIFFile, model_num: int):
        self.structure = structure
        self.pdbx_file = pdbx_file
        self.model_num = model_num
        self.full_sequences = self._read_full_sequences(pdbx_file)
        non_hetero_structure = self.structure[~self.structure.hetero]
        self.non_missing_sequences = {str(chain): str(sequence) for (chain, sequence) in list(zip(struc.get_chains(non_hetero_structure), struc.to_sequence(non_hetero_structure)[0]))}
        self._generate_auth_info()

    def _read_full_sequences(self, pdbx_file: pdbx.CIFFile) -> dict[str, str]:
        """
        Full chain sequences from ``entity_poly``, falling back to ``entity_poly_seq`` when the one-letter column is absent (e.g. AlphaFold3 output).

        Parameters
        ----------
        pdbx_file : biotite.structure.io.pdbx.CIFFile
            A PDBx/mmCIF file that contains generic information and atomic information of the protein structure categorized into mmCIF blocks.

        Returns
        -------
        dict of {str : str}
            Dictionary where the key is the auth chain identifier and the value is the full sequence, including missing residues, making up the chain's structure.

        Notes
        -----
        biotite's `get_sequence()` reads ``entity_poly.pdbx_seq_one_letter_code_can``. When that column is missing, sequences are rebuilt from the three-letter residue names in ``entity_poly_seq`` for protein entities only, with unknown residue names becoming ``"X"``.
        """

        try:
            return {chain: str(seq) for chain, seq in pdbx.get_sequence(pdbx_file).items()}
        except KeyError:
            # Some writers (e.g. AlphaFold3) omit entity_poly.pdbx_seq_one_letter_code_can, which get_sequence() needs.
            # Rebuild the same {auth chain id: one-letter sequence} dictionary from entity_poly_seq instead.
            block = pdbx_file.block
            entity_poly = block["entity_poly"]
            # Only protein entities; entity_poly can also list nucleic acids.
            protein_entities = {entity_id for entity_id, entity_type in zip(entity_poly["entity_id"].as_array(str), entity_poly["type"].as_array(str)) if entity_type.startswith("polypeptide")}

            # entity_poly_seq has one row per residue: (entity_id, num, mon_id).
            entity_poly_seq = block["entity_poly_seq"]
            entity_residues: dict[str, dict[int, str]] = {}
            for entity_id, num, residue_name in zip(entity_poly_seq["entity_id"].as_array(str), entity_poly_seq["num"].as_array(int), entity_poly_seq["mon_id"].as_array(str)):
                if entity_id in protein_entities:
                    # Microheterogeneity can list several residues at the same num; keep the first.
                    entity_residues.setdefault(entity_id, {}).setdefault(num, residue_name)

            entity_sequences: dict[str, str] = {}
            for entity_id, residues in entity_residues.items():
                letters = []
                for num in sorted(residues):
                    try:
                        letters.append(ProteinSequence.convert_letter_3to1(residues[num]))
                    except KeyError:
                        # Residue names biotite doesn't know become 'X'.
                        letters.append("X")
                entity_sequences[entity_id] = "".join(letters)

            # pdbx_strand_id lists the auth chain ids of every copy of an entity, comma-separated (e.g. "A,B" for a homodimer).
            full_sequences: dict[str, str] = {}
            for entity_id, strand_ids in zip(entity_poly["entity_id"].as_array(str), entity_poly["pdbx_strand_id"].as_array(str)):
                if entity_id in entity_sequences:
                    for chain in strand_ids.split(","):
                        full_sequences[chain] = entity_sequences[entity_id]
            return full_sequences

    def _generate_auth_info(self) -> None:
        """
        Run as part of the constructor. Generates information needed to access auth information including ``auth_seq_id`` and ``auth_asym_id``, which correspond to alternative residue indices and alternative chain ids.

        Notes
        -----
        Sets the ``first_block``, ``unique_chains``, ``chain_auth_dict``, ``auth_chain_dict``, ``atom_site_df``, and ``atom_df`` attributes. See the class Attributes for details.
        """
        if len(self.pdbx_file.keys()) > 0:
            self.first_block = next(iter(self.pdbx_file))
            atom_site_category = self.pdbx_file[self.first_block].get('atom_site')
            self.chain_auth_dict: dict[str, str] = {}
            self.auth_chain_dict: dict[str, str] = {}
            if atom_site_category:
                categories = ['group_PDB', 'label_seq_id', 'label_asym_id', 'auth_seq_id', 'auth_asym_id', 'pdbx_PDB_model_num']
                atom_site_data = np.column_stack([atom_site_category[category].as_array() for category in categories])
                _, idx = np.unique(atom_site_data, axis=0, return_index=True)
                atom_site_data = atom_site_data[np.sort(idx)]
                atom_data = atom_site_data[atom_site_data[:,0] == "ATOM"]
                self.unique_chains = np.unique(atom_data[:,2])
                for unique_chain in self.unique_chains:
                    unique_entry = atom_data[atom_data[:,2] == unique_chain][0]
                    self.chain_auth_dict[unique_entry[2]] = unique_entry[4]
                    self.auth_chain_dict[unique_entry[4]] = unique_entry[2]
                self.atom_site_df = pd.DataFrame(np.column_stack([atom_site_category[category].as_array() for category in atom_site_category]), columns=atom_site_category.keys())
                type_conversion_dict = {'label_seq_id': 'int64', 'auth_seq_id': 'int64', 'id': 'int64', 'Cartn_x': 'float', 'Cartn_y': 'float','Cartn_z': 'float', 'B_iso_or_equiv': 'float'}
                self.atom_df = self.atom_site_df[self.atom_site_df['group_PDB'] == 'ATOM'].astype(type_conversion_dict)

    def get_start_res_id(self, chain_id: str, get_auth_res_ids: bool=False, auth_chain_id_supplied: bool=False) -> int:
        """
        Gets starting residue id of the specified chain excluding heteroatom group entries.

        Parameters
        ----------
        chain_id : str
            The chain id supplied and selected for from the structure.
        get_auth_res_ids : bool, default False
            If True, return the auth residue id (``auth_seq_id``) instead of the label residue id (``label_seq_id``).
        auth_chain_id_supplied : bool, default False
            If True, `chain_id` is the auth chain id found on the RCSB website.

        Returns
        -------
        int
            The residue id of the first atom in the chain provided.
        """
        if auth_chain_id_supplied:
            chain_df = self.atom_df[self.atom_df['auth_asym_id'] == chain_id]
        else:
            chain_df = self.atom_df[self.atom_df['label_asym_id'] == chain_id]
        if get_auth_res_ids:
            return chain_df['auth_seq_id'].iloc[0]
        else:
            return chain_df['label_seq_id'].iloc[0]

    def get_full_sequence(self, chain_id: str, auth_chain_id_supplied: bool=False) -> str:
        """
        Get the full sequence, including missing residues, of the specified chain from the PDBx/mmCIF file's entity records.

        Parameters
        ----------
        chain_id : str
            Chain id supplied. The full sequence, including missing residues, of this chain will be returned.
        auth_chain_id_supplied : bool, default False
            If True, `chain_id` is the auth chain id found on the RCSB website.

        Returns
        -------
        str
            The full sequence, including missing residues, of the chain specified.
        """
        if auth_chain_id_supplied:
            return str(self.full_sequences[chain_id])
        else:
            return str(self.full_sequences[self.chain_auth_dict[chain_id]])
        
    def get_non_missing_sequence(self, chain_id: str, auth_chain_id_supplied: bool=False) -> str:
        """
        Get the sequence of the specified chain, including only residues present (non-missing) in the structure.

        Parameters
        ----------
        chain_id : str
            Chain id supplied. The sequence of this chain's non-missing residues will be returned.
        auth_chain_id_supplied : bool, default False
            If True, `chain_id` is the auth chain id found on the RCSB website.

        Returns
        -------
        str
            The sequence of the chain specified, with missing residues excluded.
        """
        if auth_chain_id_supplied:
            original_chain_id = self.auth_chain_dict[chain_id]
            return self.non_missing_sequences[original_chain_id]
        else:
            return self.non_missing_sequences[chain_id]
        
    def get_chain_specific_structure(self, ca_only: bool, chain_id: str, remove_hetero=True, auth_chain_id_supplied: bool=False):
        """
        Subsets the ``structure`` attribute to select for chain specific portions of the structure.

        Parameters
        ----------
        ca_only : bool
            If True, the structure will also be subsetted for atom entries where the ``atom_name`` annotation is ``"CA"`` (referring to alpha-carbons).
        chain_id : str
            The name of the chain to be selected for within the structure.
        remove_hetero : bool, default True
            If True, the structure will also be subsetted for atom entries where the ``hetero`` annotation is False, thus removing heteroatoms.
        auth_chain_id_supplied : bool, default False
            If True, `chain_id` is the auth chain id found on the RCSB website.

        Returns
        -------
        biotite.structure.AtomArray
            The atoms of the chain, excluding heteroatoms if `remove_hetero` is True and non-alpha-carbons if `ca_only` is True.
        """
        if auth_chain_id_supplied:
            chain_id = self.auth_chain_dict[chain_id]
        selected_structure = self.structure
        if remove_hetero:
            # Remove hetero atoms via hetero column of structure ndarray
            selected_structure = self.structure[~self.structure.hetero]
        if ca_only:
            # Consider selection of alpha-carbon atoms only
            selected_structure = selected_structure[selected_structure.atom_name == "CA"]
        chain_structure = selected_structure[selected_structure.chain_id == chain_id]
        return chain_structure
    
    def get_chain_site_data(self, ca_only: bool, chain_id: str, remove_hetero=True, auth_chain_id_supplied: bool=False):
        """
        Subsets the ``atom_df`` dataframe to get atom information where the conditions are met.

        Parameters
        ----------
        ca_only : bool
            If True, the dataframe will also be subsetted for atom entries where the ``label_atom_id`` column is ``"CA"`` (referring to alpha-carbons).
        chain_id : str
            The name of the chain to be selected for within the dataframe.
        remove_hetero : bool, default True
            If True, the dataframe will also be subsetted for atom entries where the ``group_PDB`` column is ``"ATOM"`` rather than ``"HETATM"``, thus removing heteroatoms. ``atom_df`` already contains only ``ATOM`` rows, so this has no further effect.
        auth_chain_id_supplied : bool, default False
            If True, `chain_id` is the auth chain id found on the RCSB website.

        Returns
        -------
        pandas.DataFrame
            The rows of ``atom_df`` for the chain that meet the conditions, across all models in the file.
        """
        atom_df = self.atom_df
        if ca_only:
            atom_df = atom_df[atom_df['label_atom_id'] == 'CA']
        if remove_hetero:
            atom_df = atom_df[atom_df['group_PDB'] == 'ATOM']
        if auth_chain_id_supplied:
            return atom_df[atom_df['auth_asym_id'] == chain_id]
        else:
            return atom_df[atom_df['label_asym_id'] == chain_id]
        
    def get_seq_id_mapping(self, chain_id: str, seq_to_auth: bool, auth_chain_id_supplied: bool=False) -> dict[int, int]:
        """
        Gets mapping from auth seq ids to label seq ids or vice-versa.

        Parameters
        ----------
        chain_id : str
            Chain id of the chain addressed for determining residue index mappings.
        seq_to_auth : bool
            If True, the mapping uses the ``label_seq_id`` as a key and the ``auth_seq_id`` as a value. Otherwise, keys and values are switched.
        auth_chain_id_supplied : bool, default False
            If True, `chain_id` is the auth chain id found on the RCSB website.

        Returns
        -------
        dict of {int : int}
            Dictionary with either label seq id or auth seq id as a key and the other as a value. The directionality is dependent on `seq_to_auth`.

        Notes
        -----
        The mapping is built from alpha-carbon (``"CA"``) atoms, so residues without a CA atom are not included.
        """
        chain_df = self.get_chain_site_data(ca_only=True, chain_id=chain_id, remove_hetero=True, auth_chain_id_supplied=auth_chain_id_supplied)
        if seq_to_auth: 
            return dict(zip(chain_df['label_seq_id'], chain_df['auth_seq_id']))
        else:
            return dict(zip(chain_df['auth_seq_id'], chain_df['label_seq_id']))
    
    def get_valid_chain_residues(self, chain_id: str, auth_seq_id: bool=False, auth_chain_id_supplied: bool=False) -> list[tuple[int, str]]:
        """
        Gets valid indexing for residues of a specified chain. This is directly analogous to `get_non_missing_sequence()`, does not contain missing residues, and provides the corresponding indices as well.

        Parameters
        ----------
        chain_id : str
            Chain id of the chain to be selected from the structure. This chain's sequence and corresponding residue indices are what are exclusively selected for.
        auth_seq_id : bool, default False
            If True, the residue ids that are the first element of the tuples in the returned list are auth residue ids (``auth_seq_id``) instead of label residue ids.
        auth_chain_id_supplied : bool, default False
            If True, `chain_id` is the auth chain id found on the RCSB website.
        
        Returns
        -------
        list of tuple of int, str
            A list of residue information in sequential order reflecting the structure. The list consists of tuple elements where each tuple is the residue index and its corresponding one-letter amino acid.
        """
        chain_structure = self.get_chain_specific_structure(ca_only=True, chain_id=chain_id, remove_hetero=True, auth_chain_id_supplied=auth_chain_id_supplied)
        res_ids = chain_structure.res_id.tolist()
        res_names = chain_structure.res_name
        if auth_seq_id:
            seq_id_mapping = self.get_seq_id_mapping(chain_id=chain_id, seq_to_auth=True, auth_chain_id_supplied=auth_chain_id_supplied)
            auth_res_ids = [seq_id_mapping[res_id] for res_id in res_ids]
            return list(zip(auth_res_ids, map(ProteinSequence.convert_letter_3to1, res_names)))
        else:
            return list(zip(res_ids, map(ProteinSequence.convert_letter_3to1, res_names)))

    def generate_dist_matrix(self, ca_only: bool, chain1: str, chain2: str, auth_chain_id_supplied: bool=False):
        """
        Generates distance matrix between two chains in the ``structure`` attribute.

        Parameters
        ----------
        ca_only : bool
            If True, only atoms that have the name ``"CA"`` are selected in the chains the distance matrix is calculated between.
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the second column of residues in the structure.
        auth_chain_id_supplied : bool, default False
            If True, `chain1` and `chain2` are auth chain ids found on the RCSB website.

        Returns
        -------
        tuple of (biotite.structure.AtomArray, biotite.structure.AtomArray, numpy.ndarray)
            Tuple containing the chain 1 structure, the chain 2 structure, and the distance matrix of chain 1 and chain 2's pairwise distances.

        See Also
        --------
        get_contacts : Finds close residue pairs without building the full distance matrix.

        Notes
        -----
        The matrix has one entry per atom pair, so its memory grows with the product of the two chains' atom counts (e.g. about 3.2 GB for two 20,000-atom chains).
        """
        chain1_structure = self.get_chain_specific_structure(ca_only=ca_only, chain_id=chain1, remove_hetero=True, auth_chain_id_supplied=auth_chain_id_supplied)
        chain2_structure = self.get_chain_specific_structure(ca_only=ca_only, chain_id=chain2, remove_hetero=True, auth_chain_id_supplied=auth_chain_id_supplied)
        dist_matrix = cdist(chain1_structure.coord, chain2_structure.coord)
        return (chain1_structure, chain2_structure, dist_matrix)

    def get_min_dist_atom_info(self, pairs: npt.NDArray, chain1: str, chain2: str, auth_chain_id_supplied: bool=False) -> npt.NDArray:
        """
        Generate a ndarray of residue ids and their corresponding atom names such that the distance is the minimum between the initial residues provided.

        Parameters
        ----------
        pairs : numpy.ndarray
            Structured ndarray with the ``residue1`` and ``residue2`` fields, in label residue numbering.
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the second column of residues in the structure.
        auth_chain_id_supplied : bool, default False
            If True, `chain1` and `chain2` are auth chain ids found on the RCSB website.

        Returns
        -------
        min_dist_pairs_atoms_arr : numpy.ndarray
            Structured ndarray that has residue indices, auth residue indices (corresponding to the protein numbering), and atomic names, with ``dtype={'names': ['residue1', 'residue2', 'auth_residue1', 'auth_residue2', 'atom_name1', 'atom_name2'], 'formats': [int, int, int, int, '<U10', '<U10']}``.

        See Also
        --------
        DirectInformationData.get_dist_commands : Uses this output with ``ca_only=False``.
        """
        chain1_structure = self.get_chain_specific_structure(ca_only=False, chain_id=chain1, remove_hetero=True, auth_chain_id_supplied=auth_chain_id_supplied)
        chain2_structure = self.get_chain_specific_structure(ca_only=False, chain_id=chain2, remove_hetero=True, auth_chain_id_supplied=auth_chain_id_supplied)
        # Generate the auth ids of the residues in the pairs ndarray
        seq_mapping_chain1 = self.get_seq_id_mapping(chain_id=chain1, seq_to_auth=True, auth_chain_id_supplied=auth_chain_id_supplied)
        seq_mapping_chain2 = self.get_seq_id_mapping(chain_id=chain2, seq_to_auth=True, auth_chain_id_supplied=auth_chain_id_supplied)
        min_dist_pairs_atoms = []
        for row in pairs:
            # Obtain structure information for chains 1 and 2
            chain1_res1_structure = chain1_structure[chain1_structure.res_id == row['residue1']]
            chain2_res2_structure = chain2_structure[chain2_structure.res_id == row['residue2']]
            
            # Calculate a distance matrix and find the indices of the minimal value in the matrix
            dist_matrix = cdist(chain1_res1_structure.coord, chain2_res2_structure.coord)
            
            ind = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            # Use the indices to access the atom in the atom array and get the correct atom name.
            auth_res_id1 = seq_mapping_chain1[row['residue1']]
            auth_res_id2 = seq_mapping_chain2[row['residue2']]
            min_dist_pairs_atoms.append((row['residue1'], row['residue2'], auth_res_id1, auth_res_id2, chain1_res1_structure[ind[0]].atom_name, chain2_res2_structure[ind[1]].atom_name))
        min_dist_pairs_atoms_arr = np.array(min_dist_pairs_atoms, dtype={'names': ['residue1','residue2','auth_residue1','auth_residue2','atom_name1','atom_name2'], 'formats': [int,int,int,int,'<U10','<U10']})
        return min_dist_pairs_atoms_arr

    def get_contacts(self, ca_only: bool, threshold: float, chain1: str, chain2: str, auth_seq_id: bool=False, auth_chain_id_supplied: bool=False) -> set[tuple[int, int]]:
        """
        Get contacts from the ``structure`` attribute where two residues have a pair of considered atoms within the threshold distance: any atoms, or only their alpha-carbons if `ca_only` is True.

        Parameters
        ----------
        ca_only : bool
            If True, only consider alpha-carbon to alpha-carbon distances.
        threshold : float
            Maximum distance, in Angstroms, between two atoms for their residues to be in contact (inclusive).
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the second column of residues in the structure.
        auth_seq_id : bool, default False
            If True, residues are given as auth residue ids (``auth_seq_id``); otherwise as label residue ids (``label_seq_id``).
        auth_chain_id_supplied : bool, default False
            If True, `chain1` and `chain2` are auth chain ids found on the RCSB website.

        Returns
        -------
        contacts_set : set of tuple of (int, int)
            Set of contacts, as tuples of residue 1 from `chain1` and residue 2 from `chain2` that are within the distance threshold.

        Notes
        -----
        If `chain1` and `chain2` are the same chain, each contact appears once with the lower residue first, and residues are not reported in contact with themselves. Close atoms are found with a KD-tree, so the full distance matrix of `generate_dist_matrix()` is never built.
        """
        # Get chain1 and chain2 structures.
        chain1_structure = self.get_chain_specific_structure(ca_only=ca_only, chain_id=chain1, remove_hetero=True, auth_chain_id_supplied=auth_chain_id_supplied)
        chain2_structure = self.get_chain_specific_structure(ca_only=ca_only, chain_id=chain2, remove_hetero=True, auth_chain_id_supplied=auth_chain_id_supplied)
        # Find the atomic positions where the prior atom is within or equal to the threshold distance of its pair.
        # KD-trees only compare nearby atoms, so the full chain1 x chain2 distance matrix is never built.
        close_pairs = KDTree(chain1_structure.coord).sparse_distance_matrix(KDTree(chain2_structure.coord), threshold, output_type="ndarray")
        # Residue ids of atom positions within threshold distance.
        res1_ids_within_threshold = chain1_structure.res_id[close_pairs["i"]]
        res2_ids_within_threshold = chain2_structure.res_id[close_pairs["j"]]
        
        if chain1 == chain2:
            # Setup indices where res1 is not the same as res2 ever. Eliminates self-contact and mirrored contacts.
            upper_triangle = res1_ids_within_threshold < res2_ids_within_threshold
            res1_ids_within_threshold, res2_ids_within_threshold = res1_ids_within_threshold[upper_triangle], res2_ids_within_threshold[upper_triangle]
        # Sets allow us to store unique contacts only.
        contacts_set = set(zip(res1_ids_within_threshold.tolist(), res2_ids_within_threshold.tolist()))
        if auth_seq_id:
            # Set up maps for label residue id to auth residue id.
            seq_mapping_chain1 = self.get_seq_id_mapping(chain_id=chain1, seq_to_auth=True, auth_chain_id_supplied=auth_chain_id_supplied)
            seq_mapping_chain2 = self.get_seq_id_mapping(chain_id=chain2, seq_to_auth=True, auth_chain_id_supplied=auth_chain_id_supplied)
            # Use maps to have finalized residue ids present.
            contacts_set = {(seq_mapping_chain1[r1], seq_mapping_chain2[r2]) for r1, r2 in contacts_set}
        return contacts_set

class PDBInformation(StructureInformation):
    """
    Information regarding a protein structure, obtained from a PDB format protein structure file.

    Parameters
    ----------
    structure : biotite.structure.AtomArray
        Structure obtained from a PDB file with a specified model number. PDB files use author (auth) residue and chain ids.
    pdb_file : biotite.structure.io.pdb.PDBFile
        PDB file that contains generic information and atomic information of the protein structure.
    model_num : int
        The model number to access from the PDB to ensure an AtomArray is returned containing the atom information of the protein structure.

    Attributes
    ----------
    structure : biotite.structure.AtomArray
        The `structure` supplied.
    pdb_file : biotite.structure.io.pdb.PDBFile
        The `pdb_file` supplied.
    model_num : int
        The `model_num` supplied.
    non_missing_sequences : dict of {str : str}
        The protein sequences, without missing residues, built from the non-hetero atoms of `structure`, keyed by chain id.
    unique_chains : numpy.ndarray
        Array of the chain ids of the non-hetero atoms in `structure`.
    """
    def __init__(self, structure, pdb_file: pdb.PDBFile, model_num: int):
        self.structure = structure
        self.pdb_file = pdb_file
        self.model_num = model_num
        non_hetero_structure = self.structure[~self.structure.hetero]
        self.non_missing_sequences = {str(chain): str(sequence) for (chain, sequence) in list(zip(struc.get_chains(non_hetero_structure), struc.to_sequence(non_hetero_structure)[0]))}
        self.unique_chains = struc.get_chains(non_hetero_structure)

    def get_start_res_id(self, chain_id: str) -> int:
        """
        Gets starting residue id of the specified chain excluding heteroatom group entries.

        Parameters
        ----------
        chain_id : str
            The chain id supplied and selected for from the structure.

        Returns
        -------
        int
            The residue id of the first atom in the chain provided.

        Raises
        ------
        ValueError
            If `chain_id` is not one of the chains in the structure.
        """
        non_hetero_structure = self.structure[~self.structure.hetero]
        if chain_id in self.unique_chains:
            return non_hetero_structure[non_hetero_structure.chain_id == chain_id][0].res_id
        else:
            raise ValueError("Chain supplied not found in structure.")

    def get_non_missing_sequence(self, chain_id: str) -> str:
        """
        Get the sequence of the specified chain, including only residues present (non-missing) in the structure.

        Parameters
        ----------
        chain_id : str
            Chain id supplied. The sequence of this chain's non-missing residues will be returned.

        Returns
        -------
        str
            The sequence of the chain specified, with missing residues excluded.
        """
        return self.non_missing_sequences[chain_id]
    
    def get_chain_specific_structure(self, ca_only: bool, chain_id: str, remove_hetero=True):
        """
        Subsets the ``structure`` attribute to select for chain specific portions of the structure.

        Parameters
        ----------
        ca_only : bool
            If True, the structure will also be subsetted for atom entries where the ``atom_name`` annotation is ``"CA"`` (referring to alpha-carbons).
        chain_id : str
            The name of the chain to be selected for within the structure.
        remove_hetero : bool, default True
            If True, the structure will also be subsetted for atom entries where the ``hetero`` annotation is False, thus removing heteroatoms.

        Returns
        -------
        biotite.structure.AtomArray
            The atoms of the chain, excluding heteroatoms if `remove_hetero` is True and non-alpha-carbons if `ca_only` is True.
        """
        selected_structure = self.structure
        if remove_hetero:
            # Remove hetero atoms via hetero column of structure ndarray
            selected_structure = self.structure[~self.structure.hetero]
        if ca_only:
            # Consider selection of alpha-carbon atoms only
            selected_structure = selected_structure[selected_structure.atom_name == "CA"]
        chain_structure = selected_structure[selected_structure.chain_id == chain_id]
        return chain_structure
    
    def get_valid_chain_residues(self, chain_id: str) -> list[tuple[int, str]]:
        """
        Gets valid indexing for residues of a specified chain. This is directly analogous to `get_non_missing_sequence()`, does not contain missing residues, and provides the corresponding indices as well.

        Parameters
        ----------
        chain_id : str
            Chain id of the chain to be selected from the structure. This chain's sequence and corresponding residue indices are what are exclusively selected for.

        Returns
        -------
        list of tuple of int, str
            A list of residue information in sequential order reflecting the structure. The list consists of tuple elements where each tuple is the residue index and its corresponding one-letter amino acid.
        """
        chain_structure = self.get_chain_specific_structure(ca_only=True, chain_id=chain_id, remove_hetero=True)
        return list(zip(chain_structure.res_id.tolist(), map(ProteinSequence.convert_letter_3to1, chain_structure.res_name)))

    def generate_dist_matrix(self, ca_only: bool, chain1: str, chain2: str):
        """
        Generates distance matrix between two chains in the ``structure`` attribute.

        Parameters
        ----------
        ca_only : bool
            If True, only atoms that have the name ``"CA"`` are selected in the chains the distance matrix is calculated between.
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the second column of residues in the structure.

        Returns
        -------
        tuple of (biotite.structure.AtomArray, biotite.structure.AtomArray, numpy.ndarray)
            Tuple containing the chain 1 structure, the chain 2 structure, and the distance matrix of chain 1 and chain 2's pairwise distances.

        See Also
        --------
        get_contacts : Finds close residue pairs without building the full distance matrix.

        Notes
        -----
        The matrix has one entry per atom pair, so its memory grows with the product of the two chains' atom counts (e.g. about 3.2 GB for two 20,000-atom chains).
        """
        chain1_structure = self.get_chain_specific_structure(ca_only=ca_only, chain_id=chain1, remove_hetero=True)
        chain2_structure = self.get_chain_specific_structure(ca_only=ca_only, chain_id=chain2, remove_hetero=True)
        dist_matrix = cdist(chain1_structure.coord, chain2_structure.coord)
        return (chain1_structure, chain2_structure, dist_matrix)

    def get_min_dist_atom_info(self, pairs: npt.NDArray, chain1: str, chain2: str) -> npt.NDArray:
        """
        Generate a ndarray of residue ids and their corresponding atom names such that the distance is the minimum between the initial residues provided.

        Parameters
        ----------
        pairs : numpy.ndarray
            Structured ndarray with the ``residue1`` and ``residue2`` fields.
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the second column of residues in the structure.

        Returns
        -------
        min_dist_pairs_atoms_arr : numpy.ndarray
            Structured ndarray that has residue indices, auth residue indices, and atomic names, with ``dtype={'names': ['residue1', 'residue2', 'auth_residue1', 'auth_residue2', 'atom_name1', 'atom_name2'], 'formats': [int, int, int, int, '<U10', '<U10']}``. PDB files already use auth numbering, so ``auth_residue1`` and ``auth_residue2`` repeat ``residue1`` and ``residue2``; they are included to match `MMCIFInformation.get_min_dist_atom_info()`.

        See Also
        --------
        DirectInformationData.get_dist_commands : Uses this output with ``ca_only=False``.
        """
        chain1_structure = self.get_chain_specific_structure(ca_only=False, chain_id=chain1, remove_hetero=True)
        chain2_structure = self.get_chain_specific_structure(ca_only=False, chain_id=chain2, remove_hetero=True)
        min_dist_pairs_atoms = []
        for row in pairs:
            # Obtain structure information for chains 1 and 2
            chain1_res1_structure = chain1_structure[chain1_structure.res_id == row['residue1']]
            chain2_res2_structure = chain2_structure[chain2_structure.res_id == row['residue2']]
            
            # Calculate a distance matrix and find the indices of the minimal value in the matrix
            dist_matrix = cdist(chain1_res1_structure.coord, chain2_res2_structure.coord)
            
            ind = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            min_dist_pairs_atoms.append((row['residue1'], row['residue2'], row['residue1'], row['residue2'], chain1_res1_structure[ind[0]].atom_name, chain2_res2_structure[ind[1]].atom_name))
        min_dist_pairs_atoms_arr = np.array(min_dist_pairs_atoms, dtype={'names': ['residue1','residue2','auth_residue1','auth_residue2','atom_name1','atom_name2'], 'formats': [int,int,int,int,'<U10','<U10']})
        return min_dist_pairs_atoms_arr    

    def get_contacts(self, ca_only: bool, threshold: float, chain1: str, chain2: str) -> set[tuple[int, int]]:
        """
        Get contacts from the ``structure`` attribute where two residues have a pair of considered atoms within the threshold distance: any atoms, or only their alpha-carbons if `ca_only` is True.

        Parameters
        ----------
        ca_only : bool
            If True, only consider alpha-carbon to alpha-carbon distances.
        threshold : float
            Maximum distance, in Angstroms, between two atoms for their residues to be in contact (inclusive).
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the second column of residues in the structure.

        Returns
        -------
        contacts_set : set of tuple of (int, int)
            Set of contacts, as tuples of residue 1 from `chain1` and residue 2 from `chain2` that are within the distance threshold, in the PDB file's (auth) residue numbering.

        Notes
        -----
        If `chain1` and `chain2` are the same chain, each contact appears once with the lower residue first, and residues are not reported in contact with themselves. Close atoms are found with a KD-tree, so the full distance matrix of `generate_dist_matrix()` is never built.
        """
        # Get chain1 and chain2 structures.
        chain1_structure = self.get_chain_specific_structure(ca_only=ca_only, chain_id=chain1, remove_hetero=True)
        chain2_structure = self.get_chain_specific_structure(ca_only=ca_only, chain_id=chain2, remove_hetero=True)
        # Find the atomic positions where the prior atom is within or equal to the threshold distance of its pair.
        close_pairs = KDTree(chain1_structure.coord).sparse_distance_matrix(KDTree(chain2_structure.coord), threshold, output_type="ndarray")
        # Residue ids of atom positions within threshold distance.
        res1_ids_within_threshold = chain1_structure.res_id[close_pairs["i"]]
        res2_ids_within_threshold = chain2_structure.res_id[close_pairs["j"]]

        if chain1 == chain2:
            # Setup indices where res1 is not the same as res2 ever. Eliminates self-contact and mirrored contacts.
            upper_triangle = res1_ids_within_threshold < res2_ids_within_threshold
            res1_ids_within_threshold, res2_ids_within_threshold = res1_ids_within_threshold[upper_triangle], res2_ids_within_threshold[upper_triangle]
        # Sets allow us to store unique contacts only.
        return set(zip(res1_ids_within_threshold.tolist(), res2_ids_within_threshold.tolist()))