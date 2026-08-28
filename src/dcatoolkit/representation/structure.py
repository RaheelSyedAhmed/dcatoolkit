import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb
import biotite.database.rcsb as rcsb
from biotite.sequence import ProteinSequence

from typing import Optional, Union, Literal, overload
import numpy.typing as npt

class StructureInformation:
    """
    Information regarding a protein structure, obtained from a protein structure file.

    Uses fetch_pdb() to pull protein structure information from RCSB. Uses read_x_file() to supply a filepath to pull protein structure information from a file.
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
        Fetches PDB as mmCIF file from RCSB and compiles the information into a StructureInformation instance.

        Parameters
        ----------
        pdb_id : str
            PDB ID to be fetched from the RCSB database.
        struc_format : str
            The format of the file to pull from the RCSB database.
        model_num : int
            The model number to access from the PDB to ensure an AtomArray is returned containing the atom information of the protein structure.
            
        Returns
        -------
        StructureInformation
            StructureInformation generated from pdbx.get_structure() function using the pdbx file fetched from RCSB.
        
        Raises
        ------
        TypeError
            Fetched data was not found and returned None instead.
        ValueError
            Structure format may be invalid (not PDBx/mmCIF or PDB).
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
        Reads PDB mmCIF file from filepath and compiles the information into a CIFInformation instance.

        Parameters
        ----------
        pdbx_filepath : str
            Filepath of the PDB mmCIF file to be read.
        model_num : int
            The model number to access from the PDB to ensure an AtomArray is returned containing the atom information of the protein structure.

        Returns
        -------
        CIFInformation
            CIFInformation generated from pdbx.get_structure() function using the PDBx file read from the pdbx_filepath.
        """
        pdbx_file = pdbx.CIFFile.read(pdbx_filepath)
        return MMCIFInformation(pdbx.get_structure(pdbx_file, model=model_num, use_author_fields=False), pdbx_file, model_num)
    
    @staticmethod
    def read_pdb_file(pdb_filepath: str, model_num: int=1) -> 'PDBInformation':
        """
        Reads PDB file from filepath and compiles the information into a PDBInformation instance.

        Parameters
        ----------
        pdb_filepath : str
            Filepath of the PDB mmCIF file to be read.
        model_num : int
            The model number to access from the PDB to ensure an AtomArray is returned containing the atom information of the protein structure.

        Returns
        -------
        PDBInformation
            PDBInformation generated from pdb.get_structure() function using the PDB file read from the pdb_filepath.
        """
        pdb_file = pdb.PDBFile.read(pdb_filepath)
        return PDBInformation(pdb.get_structure(pdb_file, model=model_num), pdb_file, model_num)
    
    @staticmethod
    def write_contacts_set(filepath : str, contacts_set : set[tuple[int, int]]) -> None:
        """
        Write the contacts generated from get_contacts or general set of tuples of pairs.

        Parameters
        ----------
        filepath : str
            Path of file to output contacts_set to.
        contacts_set : set of tuple of int, int
            Set of tuples of pairs that represent contacts.
        
        Returns
        -------
        None
        """
        contacts_list = list(sorted(contacts_set))
        with open(filepath, 'w') as fs:
            for pair in contacts_list:
                fs.write(str(pair[0]) + "\t" + str(pair[1]) + "\n")

class MMCIFInformation(StructureInformation):
    """
    Information regarding a protein structure, obtained from a protein structure file.

    Parameters
    ----------
    structure : biotite.structure.AtomArray
        Structure obtained from an RCSB entry with a provided pdbx/mmcif file with a specified model number.
    pdbx_file : biotite.io.pdbx.CIFFile
        mmCIF file that contains generic information and atomic information of the protein structure categorized into mmCIF blocks.
    model_num : int
        The model number to access from the PDB to ensure an AtomArray is returned containing the atom information of the protein structure.

    Attributes
    ----------
    self.full_sequences : dict of str, biotite.sequence.ProteinSequence
        The full protein sequences from the pdbx file used to generate the structure stored in a dictionary where auth_chain_id is the key and the ProteinSequence object is the value.
    self.non_missing_sequences : dict of str, biotite.sequence.ProteinSequence
        The protein sequences, without missing residues, compiled in the structure of the StructureInformation instance stored in a dictionary where chain_id is the key and the sequence string is the value.
    self.atom_data : numpy.ndarray, optional
        Entries in the format 'ATOM', residue index, chain ID, auth residue index, auth chain ID, model number
    self.het_atom_data : numpy.ndarray, optional
        Array of entries in the format 'HETATM', residue index, chain ID, auth residue index, auth chain ID, model number
    self.unique_chains : numpy.ndarray, optional
        Array of unique asym_id entries which corresponds to unique chain IDs.
    self.chain_auth_dict : dict of str, str, optional
        Uses chain id as a key and provides auth chain id as a value.
    self.auth_chain_dict : dict of str, str, optional
        Uses auth chain id as a key and provides original/label chain id as a value. 
    self.res_auth_dict : dict of str, tuple of int, int or optional 
        Uses chain id as a key and an array of residue index and auth residue index as a value.
    """
    def __init__(self, structure, pdbx_file: pdbx.CIFFile, model_num: int):
        self.structure = structure
        self.pdbx_file = pdbx_file
        self.model_num = model_num
        self.full_sequences = pdbx.get_sequence(pdbx_file)
        non_hetero_structure = self.structure[self.structure.hetero == False]
        self.non_missing_sequences = {str(chain): str(sequence) for (chain, sequence) in list(zip(struc.get_chains(non_hetero_structure), struc.to_sequence(non_hetero_structure)[0]))}
        self._generate_auth_info()

    def _generate_auth_info(self) -> None:
        """
        Ran as part of constructor function. Generates information needed to access auth information including auth_seq_id and auth_asym_id, which correspond to alternative chain ids and alternative residue indices.
        
        Note
        ----
        See attributes for details.
        
        Returns
        -------
        None
        """
        if len(self.pdbx_file.keys()) > 0:
            self.first_block = list(self.pdbx_file)[0]
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
                self.atom_site_df = pd.DataFrame(np.column_stack([atom_site_category[category].as_array() for category in atom_site_category.keys()]), columns=atom_site_category.keys())
                type_conversion_dict = {'label_seq_id': 'int64', 'auth_seq_id': 'int64', 'id': 'int64', 'Cartn_x': 'float', 'Cartn_y': 'float','Cartn_z': 'float', 'B_iso_or_equiv': 'float'}
                self.atom_df = self.atom_site_df[self.atom_site_df['group_PDB'] == 'ATOM'].astype(type_conversion_dict)

    def get_start_res_id(self, chain_id: str, get_auth_res_ids: bool=False, auth_chain_id_supplied: bool=False) -> int:
        """
        Gets starting residue id of the specified chain excluding heteroatom group entries.

        Parameters
        ----------
        chain_id : str
            The chain id supplied and selected for from the structure.
        get_auth_res_ids : bool
            True if you want alt_ids for residues indices, False if cif residue indexing is needed.
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.            

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
            return chain_df['auth_seq_id'][0]
        else:
            return chain_df['label_seq_id'][0]

    def get_full_sequence(self, chain_id: str, auth_chain_id_supplied: bool=False) -> str:
        """
        Get full sequence, including missing residues, from the specified chain off of RCSB.

        Parameters
        ----------
        chain_id : str
            Chain id supplied. The full sequence, including missing residues, of this chain will be returned.
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.

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
        Get sequence, including only non-missing residues, from the specified chain off of RCSB.

        Parameters
        ----------
        chain_id : str
            Chain id supplied. The full sequence, including only non-missing residues, of this chain will be returned.
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.

        Returns
        -------
        str
            The full sequence, including only non-missing residues, of the chain specified.
        """
        if auth_chain_id_supplied:
            original_chain_id = self.auth_chain_dict[chain_id]
            return self.non_missing_sequences[original_chain_id]
        else:
            return self.non_missing_sequences[chain_id]
        
    def get_chain_specific_structure(self, ca_only: bool, chain_id: str, remove_hetero=True, auth_chain_id_supplied: bool=False):
        """
        Subsets structure attribute to select for chain specific portions of the structure.

        Parameters
        ----------
        ca_only : bool
            If true, the structure will also be subsetted for atom entries where the atom_name annotation is "CA" (referring to alpha-carbons)
        chain_id : str
            The name of the chain to be selected for within the structure.
        remove_hetero : bool, default=True
            If true, the structure will also be subsetted for atom entries where the hetero annotation is False, thus removing heteroatoms.
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.

        Returns
        -------
        tuple of biotite.structure.AtomArray, biotite.structure.AtomArray
            Two AtomArrays that refer to atoms in the first chain and second chain, respectively without accounting for the presence of heteroatoms if `remove_hetero` is True.
        """
        if auth_chain_id_supplied:
            chain_id = self.auth_chain_dict[chain_id]
        selected_structure = self.structure
        if remove_hetero:
            # Remove hetero atoms via hetero column of structure ndarray
            selected_structure = self.structure[self.structure.hetero == False]
        if ca_only:
            # Consider selection of alpha-carbon atoms only
            selected_structure = selected_structure[selected_structure.atom_name == "CA"]
        chain_structure = selected_structure[selected_structure.chain_id == chain_id]
        return chain_structure
    
    def get_chain_site_data(self, ca_only: bool, chain_id: str, remove_hetero=True, auth_chain_id_supplied: bool=False):
        """
        Subsets the atom_site dataframe to get atom information where the conditions are met.

        Parameters
        ----------
        ca_only : bool
            If true, the dataframe will also be subsetted for atom entries where the label_atom_id annotation is "CA" (referring to alpha-carbons)
        chain_id : str
            The name of the chain to be selected for within the dataframe.
        remove_hetero : bool, default=True
            If true, the dataframe will also be subsetted for atom entries where the group_PDB annotation is ATOM rather than HETATM, thus removing heteroatoms.
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.
        """
        atom_df = self.atom_df.copy()
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
            If True, this indicates the mapping uses the label_seq_id as a key and the auth_seq_id as a value. Otherwise, keys and values are switched.
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.

        Returns
        -------
        dict of int, int
            Dictionary with either label seq id or auth seq id as a key and the other as a value. The directionality is dependent on seq_to_auth.
        """
        chain_df = self.get_chain_site_data(ca_only=True, chain_id=chain_id, remove_hetero=True, auth_chain_id_supplied=auth_chain_id_supplied)
        if seq_to_auth: 
            return dict(zip(chain_df['label_seq_id'], chain_df['auth_seq_id']))
        else:
            return dict(zip(chain_df['auth_seq_id'], chain_df['label_seq_id']))
    
    def get_valid_chain_residues(self, chain_id: str, auth_seq_id: bool=False, auth_chain_id_supplied: bool=False) -> list[tuple[int, str]]:
        """
        Gets valid indexing for residues of a specified chain. This is directly analogous to get_non_missing_sequence, does not contain missing residues, and provides the corresponding indices as well.

        Parameters
        ----------
        chain_id : str
            Chain id of the chain to be selected from the structure. This chain's sequence and corresponding residue indices are what are exclusively selected for.
        auth_seq_id: bool
            If True, the seq_ids that are the first element of the tuples in the returned list are auth_seq_ids. 
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.
        
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
            return list(zip(auth_res_ids, map(lambda symbol: ProteinSequence.convert_letter_3to1(symbol), res_names)))
        else:
            return list(zip(res_ids, map(lambda symbol: ProteinSequence.convert_letter_3to1(symbol), res_names)))

    def generate_dist_matrix(self, ca_only: bool, chain1: str, chain2: str, auth_chain_id_supplied: bool=False):
        """
        Generates distance matrix between two chains in the structure attribute.

        Parameters
        ----------
        ca_only : bool
            If True, only atoms that have the name "CA" are selected in the chains the distance matrix is calculated between.
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the first column of residues in the structure.
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.

        Returns
        -------
        tuple of biotite.structure.AtomArray, biotite.structure.AtomArray, numpy.ndarray
            Tuple containing the chain 1 structure, the chain 2 structure, and the distance matrix of chain 1 and chain 2's pairwise distances.
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
            Pairs structured ndarray with "residue1" and "residue2" columns.
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the second column of residues in the structure.
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.
            
        Returns
        -------
        min_dist_pairs_atoms_arr : numpy.ndarray
            Structured ndarray that has residue indices, auth residue indices (corresponding to the protein numbering), and atomic names in the format {'names': ['residue1','residue2','auth_residue1','auth_residue2','atom_name1','atom_name2'], 'formats': [int,int,int,int,'<U10','<U10']}
        """
        chain1_structure = self.get_chain_specific_structure(ca_only=False, chain_id=chain1, remove_hetero=True, auth_chain_id_supplied=auth_chain_id_supplied)
        chain2_structure = self.get_chain_specific_structure(ca_only=False, chain_id=chain2, remove_hetero=True, auth_chain_id_supplied=auth_chain_id_supplied)
        min_dist_pairs_atoms = []
        for row in pairs:
            # Obtain structure information for chains 1 and 2
            chain1_res1_structure = chain1_structure[chain1_structure.res_id == row['residue1']]
            chain2_res2_structure = chain2_structure[chain2_structure.res_id == row['residue2']]
            
            # Calculate a distance matrix and find the indices of the minimal value in the matrix
            dist_matrix = cdist(chain1_res1_structure.coord, chain2_res2_structure.coord)
            
            ind = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            # Use the indices to access the atom in the atom array and get the correct atom name.
            # Generate the auth ids of the residues in the pairs ndarray
            seq_mapping_chain1 = self.get_seq_id_mapping(chain_id=chain1, seq_to_auth=True, auth_chain_id_supplied=auth_chain_id_supplied)
            seq_mapping_chain2 = self.get_seq_id_mapping(chain_id=chain2, seq_to_auth=True, auth_chain_id_supplied=auth_chain_id_supplied)
            auth_res_id1 = seq_mapping_chain1[row['residue1']]
            auth_res_id2 = seq_mapping_chain2[row['residue2']]
            min_dist_pairs_atoms.append((row['residue1'], row['residue2'], auth_res_id1, auth_res_id2, chain1_res1_structure[ind[0]].atom_name, chain2_res2_structure[ind[1]].atom_name))
        min_dist_pairs_atoms_arr = np.array(min_dist_pairs_atoms, dtype={'names': ['residue1','residue2','auth_residue1','auth_residue2','atom_name1','atom_name2'], 'formats': [int,int,int,int,'<U10','<U10']})
        return min_dist_pairs_atoms_arr

    def get_contacts(self, ca_only: bool, threshold: float, chain1: str, chain2: str, auth_seq_id: bool=False, auth_chain_id_supplied: bool=False) -> set[tuple[int, int]]:
        """
        Get contacts from the structure attribute where the distance between two residues is less than the threshold.

        Parameters
        ----------
        ca_only : bool
            If true, only consider alpha-carbon to alpha-carbon distances. 
        threshold : float
            Maximum distance to consider between two atoms.
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the second column of residues in the structure.
        auth_seq_id : bool
            True if you want auth_seq_ids for residues indices, False if cif residue indexing is needed.
        auth_chain_id_supplied : bool
            If True, the chain_id supplied is the auth chain id found on the RCSB website.

        Returns
        -------
        contacts_set : set of tuple of ints
            Set of contacts, tuples with "residue1" and "residue2" from the structure that are within the distance threshold.
        """
        chain1_structure, chain2_structure, dist_matrix = self.generate_dist_matrix(ca_only, chain1, chain2, auth_chain_id_supplied=auth_chain_id_supplied)
        seq_mapping_chain1 = self.get_seq_id_mapping(chain_id=chain1, seq_to_auth=True, auth_chain_id_supplied=auth_chain_id_supplied)
        seq_mapping_chain2 = self.get_seq_id_mapping(chain_id=chain2, seq_to_auth=True, auth_chain_id_supplied=auth_chain_id_supplied)
        thresh_ind = np.argwhere(dist_matrix <= threshold)
        contacts_set = set()
        for indices in thresh_ind:
            chain1_atom = chain1_structure[indices[0]]
            chain2_atom = chain2_structure[indices[1]]
            res1 = chain1_atom.res_id
            res2 = chain2_atom.res_id
            if not(chain1==chain2 and res1 >= res2):
                if auth_seq_id:
                    contacts_set.add((seq_mapping_chain1[res1], seq_mapping_chain2[res2]))
                else:
                    contacts_set.add((res1, res2))
        return contacts_set

class PDBInformation(StructureInformation):
    """
    Information regarding a protein structure, obtained from a protein structure file.

    Parameters
    ----------
    structure : biotite.structure.AtomArray
        Structure obtained from an RCSB entry with a provided pdbx/mmcif file with a specified model number.
    pdb_file : biotite.io.pdb.PDBFile
        mmCIF file that contains generic information and atomic information of the protein structure categorized into mmCIF blocks.
    model_num : int
        The model number to access from the PDB to ensure an AtomArray is returned containing the atom information of the protein structure.

    Attributes
    ----------
    self.non_missing_sequences : dict of str, biotite.sequence.ProteinSequence
        The protein sequences, without missing residues, compiled in the structure of the StructureInformation instance stored in a dictionary where chain_id is the key and the sequence string is the value.
    
    """
    def __init__(self, structure, pdb_file: pdb.PDBFile, model_num: int):
        self.structure = structure
        self.pdb_file = pdb_file
        self.model_num = model_num
        non_hetero_structure = self.structure[self.structure.hetero == False]
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
        """
        non_hetero_structure = self.structure[self.structure.hetero == False]
        if chain_id in self.unique_chains:
            return non_hetero_structure[non_hetero_structure.chain_id == chain_id][0].res_id
        else:
            raise ValueError("Chain supplied not found in structure.")

    def get_non_missing_sequence(self, chain_id: str) -> str:
        """
        Get sequence, including only non-missing residues, from the specified chain.

        Parameters
        ----------
        chain_id : str
            Chain id supplied. The full sequence, including only non-missing residues, of this chain will be returned.

        Returns
        -------
        str
            The full sequence, including only non-missing residues, of the chain specified.
        """
        return self.non_missing_sequences[chain_id]
    
    def get_chain_specific_structure(self, ca_only: bool, chain_id: str, remove_hetero=True):
        """
        Subsets structure attribute to select for chain specific portions of the structure.

        Parameters
        ----------
        ca_only : bool
            If true, the structure will also be subsetted for atom entries where the atom_name annotation is "CA" (referring to alpha-carbons)
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the second column of residues in the structure.
        remove_hetero : bool, default=True
            If true, the structure will also be subsetted for atom entries where the hetero annotation is False, thus removing heteroatoms.

        Returns
        -------
        tuple of biotite.structure.AtomArray, biotite.structure.AtomArray
            Two AtomArrays that refer to atoms in the first chain and second chain, respectively without accounting for the presence of heteroatoms if `remove_hetero` is True.
        """
        selected_structure = self.structure
        if remove_hetero:
            # Remove hetero atoms via hetero column of structure ndarray
            selected_structure = self.structure[self.structure.hetero == False]
        if ca_only:
            # Consider selection of alpha-carbon atoms only
            selected_structure = selected_structure[selected_structure.atom_name == "CA"]
        chain_structure = selected_structure[selected_structure.chain_id == chain_id]
        return chain_structure
    
    def get_valid_chain_residues(self, chain_id: str) -> list[tuple[int, str]]:
        """
        Gets valid indexing for residues of a specified chain. This is directly analogous to get_non_missing_sequence, does not contain missing residues, and provides the corresponding indices as well.

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
        return list(zip(chain_structure.res_id.tolist(), map(lambda symbol: ProteinSequence.convert_letter_3to1(symbol), chain_structure.res_name)))

    def generate_dist_matrix(self, ca_only: bool, chain1: str, chain2: str):
        """
        Generates distance matrix between two chains in the structure attribute.

        Parameters
        ----------
        ca_only : bool
            If True, only atoms that have the name "CA" are selected in the chains the distance matrix is calculated between.
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the first column of residues in the structure.

        Returns
        -------
        tuple of biotite.structure.AtomArray, biotite.structure.AtomArray, numpy.ndarray
            Tuple containing the chain 1 structure, the chain 2 structure, and the distance matrix of chain 1 and chain 2's pairwise distances.
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
            Pairs structured ndarray with "residue1" and "residue2" columns.
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the second column of residues in the structure.
            
        Returns
        -------
        min_dist_pairs_atoms_arr : numpy.ndarray
            Structured ndarray that has residue indices, auth residue indices (corresponding to the protein numbering), and atomic names in the format {'names': ['residue1','residue2','auth_residue1','auth_residue2','atom_name1','atom_name2'], 'formats': [int,int,int,int,'<U10','<U10']}
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
        Get contacts from the structure attribute where the distance between two residues is less than the threshold.

        Parameters
        ----------
        ca_only : bool
            If true, only consider alpha-carbon to alpha-carbon distances. 
        threshold : float
            Maximum distance to consider between two atoms.
        chain1 : str
            Chain id corresponding to the first column of residues in the structure.
        chain2 : str
            Chain id corresponding to the second column of residues in the structure.

        Returns
        -------
        contacts_set : set of tuple of ints
            Set of contacts, tuples with "residue1" and "residue2" from the structure that are within the distance threshold.
        """
        
        chain1_structure, chain2_structure, dist_matrix = self.generate_dist_matrix(ca_only, chain1, chain2)
        thresh_ind = np.argwhere(dist_matrix <= threshold)
        contacts_set = set()
        for indices in thresh_ind:
            chain1_atom = chain1_structure[indices[0]]
            chain2_atom = chain2_structure[indices[1]]
            res1 = chain1_atom.res_id
            res2 = chain2_atom.res_id
            if not(chain1==chain2 and res1 >= res2):
                contacts_set.add((res1, res2))
        return contacts_set