import os
import pickle
import torch
import numpy as np
import pandas as pd
import networkx as nx
import collections
import math
import re
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, Draw
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import Data, InMemoryDataset, Batch
from itertools import repeat, product, chain
from openbabel import openbabel, pybel

# Constants
NUM_ATOM_FEATURES = 4
NUM_BOND_FEATURES = 2
MAX_ATTEMPTS_3D = 3
FILE_PATH = '/work/yunruili/2dNMR_project_GNN/data_nmr_alignment/nmrshiftdb2.nmredata.sd'
GRAPH_OUTPUT_DIR = '/work/yunruili/2dNMR_project_GNN/data_nmr_alignment/graph3d/'
DB_ID_PATTERN = r'DB_ID=(\d+)'

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_atom_partial_charge': (-1.00000, 1.00000),
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def mol_to_graph_data_obj_simple(mol):
    """
    Converts an RDKit Mol object to a PyTorch Geometric Data object.
    
    Args:
        mol (rdkit.Mol): RDKit Mol object.

    Returns:
        torch_geometric.data.Data: Data object with graph representation of the molecule.
    """
    # atoms
    AllChem.ComputeGasteigerCharges(mol)
    num_atom_features = 4   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = \
        [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
        [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())] + \
        [allowable_features['possible_hybridization_list'].index(atom.GetHybridization())]

        atom_features_list.append(atom_feature)
        #print(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # Get atom positions (coordinates)
    # Add position features
    positions = mol.GetConformer().GetPositions()
    pos = torch.tensor(positions, dtype=torch.float)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

    return data

def read_sdf_file(file_path):
    """
    Reads molecules from an SDF file.

    Args:
        file_path (str): Path to the SDF file.

    Returns:
        list: List of RDKit Mol objects.
    """
    supplier = Chem.SDMolSupplier(file_path)
    molecules = []
    for mol in supplier:
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                molecules.append(mol)
            except Exception as e:
                print(f"Sanitization error for molecule: {e}")
    return molecules


def generate_3d_graphs(molecules):
    """
    Generates 3D graphs from a list of molecules and saves them as pickle files.
    Args:
        molecules (list): List of RDKit Mol objects.
    """
    for mol in molecules:
        if mol:
            properties_dict = mol.GetPropsAsDict()
            rdkit_molblock = Chem.MolToMolBlock(mol)
            mol = pybel.readstring("mol", rdkit_molblock)
            mol.make3D()
            ff = pybel._forcefields["mmff94"]  # Universal Force Field
            ff.Setup(mol.OBMol)
            ff.SteepestDescent(100)  # You can adjust the number of steps
            # for i, atom in enumerate(mol.atoms):
            #     index = i + 1  # Adding 1 to make the index start from 1
            #     atom_type = atom.type
            #     coords = atom.coords
            #     print(f"Atom {index}: Type {atom_type}, Coordinates {coords}")
            ob_molblock = mol.write("mol")
            # with open('obabel.mol2', 'w') as ob_file:
            #     ob_file.write(ob_molblock)
            mol = Chem.MolFromMolBlock(ob_molblock)
            mol = Chem.AddHs(mol)

            # # Generate 3D coordinates (optional)
            for attempt in range(MAX_ATTEMPTS_3D):
                params = AllChem.ETKDGv3()
                params.randomSeed = attempt  # Use attempt number as seed for reproducibility
                if AllChem.EmbedMolecule(mol, params) == 0:
                    AllChem.MMFFOptimizeMolecule(mol)  # Optimize the conformation
                    # Check if the molecule is not planar by inspecting the z-coordinates
                    conf = mol.GetConformer()
                    z_coords = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
                    if any(z != 0 for z in z_coords):
                        success = True
                        break

            if not success:
                print(f"Failed to generate 3D coordinates for molecule. Skipping.")


            # Chem.AllChem.EmbedMolecule(mol)
            # Chem.AllChem.MMFFOptimizeMolecule(mol)
            # atom_types = [atom.GetAtomicNum() for atom in rdkit_mol.GetAtoms()]
            # coords = rdkit_mol.GetConformer().GetPositions()
            # print(".............")
            # # Print atom index and coordinates
            # for atom_idx, (coord, atom_type) in enumerate(zip(coords, atom_types)):
            #     x, y, z = coord
            #     print(f"Atom {atom_idx + 1}: Index {atom_idx}, Coordinates ({x:.4f}, {y:.4f}, {z:.4f}), Atom Type {atom_type}")
            # mol2_block = Chem.MolToMolBlock(rdkit_mol)
            # with open('rdkit.mol2', "w") as mol2_file:
            #     mol2_file.write(mol2_block)
            # break
            
            if 'NMREDATA_ID' in properties_dict:
                id = re.search(DB_ID_PATTERN, str(properties_dict['NMREDATA_ID'])).group(1).zfill(9)
                try:
                    data = mol_to_graph_data_obj_simple(mol)
                    graph_filename = os.path.join(GRAPH_OUTPUT_DIR, f'{id}.pickle')
                    # print(graph_filename)
                    with open(graph_filename, 'wb') as f:
                        pickle.dump(data, f)
                except Exception as e:
                    print(f"Error processing molecule {id}: {e}")

molecules = read_sdf_file(FILE_PATH)
generate_3d_graphs(molecules)
