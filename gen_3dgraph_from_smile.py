import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import numpy as np
import os
import pandas as pd
import pickle
import argparse

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)), ### changed
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


def smiles_to_pyg_graph_3d(smiles, max_attempts=5):
    # Convert the SMILES string to an RDKit molecule
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES string")

    # Add hydrogens
    molecule = Chem.AddHs(molecule)

    # Generate 3D coordinates
    # Attempt to generate 3D coordinates up to max_attempts times
    for attempt in range(max_attempts):
        if AllChem.EmbedMolecule(molecule, AllChem.ETKDG()) == 0:
            break
    AllChem.UFFOptimizeMolecule(molecule)

    # Get atom features (atomic number, chirality, hybridization)
    atom_features_list = []
    for atom in molecule.GetAtoms():
        atom_feature = \
        [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
        [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())] + \
        [allowable_features['possible_hybridization_list'].index(atom.GetHybridization())]

        atom_features_list.append(atom_feature)
        #print(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # Get atom positions (coordinates)
    conformer = molecule.GetConformer()
    pos = torch.tensor([list(conformer.GetAtomPosition(i)) for i in range(molecule.GetNumAtoms())], dtype=torch.float)

    # Get edge indices
    num_bond_features = 2
    if len(molecule.GetBonds()) > 0:
        edge_list = []
        edge_feature_list = []
        for bond in molecule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                    bond.GetBondType())] + [allowable_features['possible_bond_dirs'].index(
                    bond.GetBondDir())]
            edge_list.append((i, j))
            edge_feature_list.append(edge_feature)
            edge_list.append((j, i))  # add both directions for undirected graph
            edge_feature_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_feature_list), dtype=torch.long)

    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    # Create PyG data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

    return data

def main(df):
    """
    Convert SMILES representations to molecular graphs and save them as pickle files.

    Parameters:
        csv (str): Path to the CSV file containing SMILES representations.
    """
    csv_arr = csv.split("/")[-1].split(".")
    folder_path = 'graph_3d'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    smiles = df['SMILES'].tolist()
    filenames = df['File_name'].tolist()
    
    id_saved = []

    for id, smile in enumerate(smiles):

        name = filenames[id].split('.')[0]
        try:
            # Convert the molecule to a Data object
            data = smiles_to_pyg_graph_3d(smile)

            # Generate the filename based on the property value
            graph_filename = os.path.join(folder_path, f'{name}.pickle')

            # Save the Data object using pickle
            with open(graph_filename, 'wb') as f:
                pickle.dump(data, f)
                id_saved.append(id)

        except AttributeError as e:
            print(f"AttributeError processing molecule {name}: {e}")
            # Print molecule details if needed, e.g., mol.GetPropsAsDict()
        except Exception as e:
            print(f"Error processing molecule {name}: {type(e).__name__}: {e}")
    
    df_saved = df[df.index.isin(id_saved)].reset_index()
    return df_saved

if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Generate 3D graph object from SMILEs representation.')

    # Add command-line arguments
    parser.add_argument('--csv', help='Name of the input csv file', default='nmr_smile_solvent_web_sat_combined2.csv')

    # smile = 'CCO'
    # mol = Chem.MolFromSmiles(smile)
    # data = mol_to_graph_data_obj_simple(mol)
    # print(data.x)

    # Parse the command-line arguments
    args = parser.parse_args()
    csv = args.csv
    df = pd.read_csv(csv)
    df_saved = main(df)
    # if len(df_saved) != len(df):
    #     csv_arr = csv.split("/")[-1].split(".")
        # df_saved.to_csv(csv_arr[0]+"_3d.csv")

# # Example usage
# smiles = "CCO"  # Ethanol
# pyg_graph = smiles_to_pyg_graph_3d(smiles)

# print(pyg_graph.pos)
# print(pyg_graph.x)
