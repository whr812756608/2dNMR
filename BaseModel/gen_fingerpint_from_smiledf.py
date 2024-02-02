import os
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import torch
import pandas as pd

# Function to convert SMILES to RDKit molecule object
def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol

# Function to calculate fingerprint and return torch.tensor
def calculate_fingerprint(mol):
    if mol is not None:
        # Generate Morgan fingerprint (circular fingerprint) with radius 2
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        # Convert RDKit fingerprint to torch.tensor
        fingerprint_tensor = torch.tensor(list(fingerprint), dtype=torch.float32)
        return fingerprint_tensor
    else:
        return None

# Path to the directory containing SMILES files
# smiles_directory = 'smiles'

# Create 'fingerprint' directory if it doesn't exist
output_directory = 'fingerprint'
os.makedirs(output_directory, exist_ok=True)

# Loop through smiles in df
df = pd.read_csv('nmr_smile_solvent_web_sat_combined2.csv')
smiles = df['SMILES'].tolist()
filenames = df['File_name'].tolist()

for i, smile in enumerate(smiles):
    name = filenames[i].split('.')[0]
    # Convert SMILES to RDKit molecule object
    mol = smiles_to_mol(smile)

    # Check if the molecule is None
    if mol is None:
        print(f"Skipping molecule {i}, {smile} because it is None.")
        continue

    try:
        # Calculate fingerprint
        fingerprint = calculate_fingerprint(mol)

        # Save fingerprint as pickle file in the 'fingerprint' directory
        output_pickle_file = os.path.join(output_directory, f'{name}.pkl')
        with open(output_pickle_file, 'wb') as pickle_file:
            pickle.dump(fingerprint, pickle_file)

        # print(f"Fingerprint for {filename} saved as {output_pickle_file}")
    
    except Exception as e:
        print(f"Error processing {i}, {smile}: {str(e)}")

