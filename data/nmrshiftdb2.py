import numpy as np
import os
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures
    
# the dataset can be downloaded from
# https://nmrshiftdb.nmr.uni-koeln.de/portal/js_pane/P-Help
# nmrshiftdb2withsignals.sd


def get_atom_shifts(mol):
    
    molprops = mol.GetPropsAsDict()
    
    atom_shifts = {}
    for key in molprops.keys():
    
        if key.startswith('Spectrum 13C'):
            
            for shift in molprops[key].split('|')[:-1]:
            
                [shift_val, _, shift_idx] = shift.split(';')
                shift_val, shift_idx = float(shift_val), int(shift_idx)
            
                if shift_idx not in atom_shifts: atom_shifts[shift_idx] = []
                atom_shifts[shift_idx].append(shift_val)

    return atom_shifts


def add_mol(mol_dict, mol):

    def _DA(mol):

        D_list, A_list = [], []
        for feat in chem_feature_factory.GetFeaturesForMol(mol):
            if feat.GetFamily() == 'Donor': D_list.append(feat.GetAtomIds()[0])
            if feat.GetFamily() == 'Acceptor': A_list.append(feat.GetAtomIds()[0])
        
        return D_list, A_list

    def _chirality(atom):

        if atom.HasProp('Chirality'):
            #assert atom.GetProp('Chirality') in ['Tet_CW', 'Tet_CCW']
            c_list = [(atom.GetProp('Chirality') == 'Tet_CW'), (atom.GetProp('Chirality') == 'Tet_CCW')] 
        else:
            c_list = [0, 0]

        return c_list

    def _stereochemistry(bond):

        if bond.HasProp('Stereochemistry'):
            #assert bond.GetProp('Stereochemistry') in ['Bond_Cis', 'Bond_Trans']
            s_list = [(bond.GetProp('Stereochemistry') == 'Bond_Cis'), (bond.GetProp('Stereochemistry') == 'Bond_Trans')] 
        else:
            s_list = [0, 0]

        return s_list    
    
    # Count the number of allowed atoms
    allowed_atoms = {'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl'}
    atom_count = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in allowed_atoms:
            atom_count += 1
    
    if atom_count > 64:
        return mol_dict  # Skip this molecule if it has more than 64 allowed atoms
    

    n_node = mol.GetNumAtoms()
    n_edge = mol.GetNumBonds() * 2

    D_list, A_list = _DA(mol)
    rings = mol.GetRingInfo().AtomRings()
    atom_fea1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
    atom_fea2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-2]
    atom_fea5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs(includeNeighbors = True)) for a in mol.GetAtoms()]][:,:-1]
    atom_fea6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea7 = np.array([[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
    atom_fea8 = np.array([_chirality(a) for a in mol.GetAtoms()], dtype = bool)
    atom_fea9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
    atom_fea10 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
    
    node_attr = np.concatenate([atom_fea1, atom_fea2, atom_fea3, atom_fea4, atom_fea5, atom_fea6, atom_fea7, atom_fea8, atom_fea9, atom_fea10], 1)
    # node_attr = np.concatenate([atom_fea1, atom_fea4, atom_fea8], 1)

    shift = np.array([atom.GetDoubleProp('shift') for atom in mol.GetAtoms()])
    mask = np.array([atom.GetBoolProp('mask') for atom in mol.GetAtoms()])

    mol_dict['n_node'].append(n_node)
    mol_dict['n_edge'].append(n_edge)
    mol_dict['node_attr'].append(node_attr)

    mol_dict['shift'].append(shift)
    mol_dict['mask'].append(mask)
    mol_dict['smi'].append(Chem.MolToSmiles(mol))
    
    if n_edge > 0:

        bond_fea1 = np.eye(len(bond_list), dtype = bool)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
        bond_fea2 = np.array([_stereochemistry(b) for b in mol.GetBonds()], dtype = bool)
        bond_fea3 = [[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()]   
        
        edge_attr = np.concatenate([bond_fea1, bond_fea2, bond_fea3], 1)
        # edge_attr = np.concatenate([bond_fea1], 1)
        edge_attr = np.vstack([edge_attr, edge_attr])
        
        bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype = int)
        src = np.hstack([bond_loc[:,0], bond_loc[:,1]])
        dst = np.hstack([bond_loc[:,1], bond_loc[:,0]])
        
        mol_dict['edge_attr'].append(edge_attr)
        mol_dict['src'].append(src)
        mol_dict['dst'].append(dst)
    
    return mol_dict


molsuppl = Chem.SDMolSupplier('nmrshiftdb2withsignals.sd', removeHs = False)

# added 'H', 'Ga', 'Pt' for atom list
atom_list = ['Li','B','C','N','O','F','Na','Mg','Al','Si','P','S','Cl','K','Ti','Zn','Ge','As','Se','Br','Pd','Ag','Sn','Sb','Te','I','Hg','Tl','Pb','Bi', 'H', 'Ga', 'Pt']
charge_list = [1, 2, 3, -1, -2, -3, 0]
degree_list = [1, 2, 3, 4, 5, 6, 8, 0] # add 8
valence_list = [1, 2, 3, 4, 5, 6, 8, 0] # add 8
hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']
hydrogen_list = [1, 2, 3, 4, 0]
ringsize_list = [3, 4, 5, 6, 7, 8]

bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'DATIVE'] # added DATIVE

rdBase.DisableLog('rdApp.error') 
rdBase.DisableLog('rdApp.warning')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))

mol_dict = {'n_node': [],
            'n_edge': [],
            'node_attr': [],
            'edge_attr': [],
            'src': [],
            'dst': [],
            'shift': [],
            'mask': [],
            'smi': []}
                 
for i, mol in enumerate(molsuppl):

    try:
        ### added. convert to smiles then convert back. try to preserve atom index
        # Create a copy of the molecule where each atom is mapped to its index
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())

        # Convert the molecule to SMILES, including atom mapping
        smiles_with_mapping = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=False, allHsExplicit=True)

        # Convert back to a molecule from the mapped SMILES
        mol2 = Chem.MolFromSmiles(smiles_with_mapping)

        # Reorder the atoms in the molecule using the atom map numbers
        mol2 = Chem.RenumberAtoms(mol2, [atom.GetAtomMapNum() for atom in mol2.GetAtoms()])

        for atom in mol2.GetAtoms():
            atom.SetAtomMapNum(0)
        ###
    except:
        continue
    
    try:
        Chem.SanitizeMol(mol2)
        si = Chem.FindPotentialStereo(mol2)
        for element in si:
            if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified':
                mol2.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
            elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified':
                mol2.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
        assert '.' not in Chem.MolToSmiles(mol)
    except:
        continue

    atom_shifts = get_atom_shifts(mol) #we can only use mol here, because the smile converted mol2 will not have all the shifts
    if len(atom_shifts) == 0: 
        continue
    for j, atom in enumerate(mol2.GetAtoms()):
        if j in atom_shifts:
            atom.SetDoubleProp('shift', np.median(atom_shifts[j]))
            atom.SetBoolProp('mask', 1)
        else:
            atom.SetDoubleProp('shift', 0)
            atom.SetBoolProp('mask', 0)

    # mol = Chem.RemoveHs(mol, implicitOnly=False, updateExplicitCount=False, sanitize=True)
    
    mol_dict = add_mol(mol_dict, mol2)

    

    if (i+1) % 1000 == 0: 
        print('%d/%d processed' %(i+1, len(molsuppl)))
        print(len(mol_dict['n_node']))

print('%d/%d processed' %(i+1, len(molsuppl)))   

mol_dict['n_node'] = np.array(mol_dict['n_node']).astype(int)
mol_dict['n_edge'] = np.array(mol_dict['n_edge']).astype(int)
mol_dict['node_attr'] = np.vstack(mol_dict['node_attr']).astype(bool)
mol_dict['edge_attr'] = np.vstack(mol_dict['edge_attr']).astype(bool)
mol_dict['src'] = np.hstack(mol_dict['src']).astype(int)
mol_dict['dst'] = np.hstack(mol_dict['dst']).astype(int)
mol_dict['shift'] = np.hstack(mol_dict['shift'])
mol_dict['mask'] = np.hstack(mol_dict['mask']).astype(bool)
mol_dict['smi'] = np.array(mol_dict['smi'])

for key in mol_dict.keys(): print(key, mol_dict[key].shape, mol_dict[key].dtype)
    
np.savez_compressed('./data/dataset_graph_C_2.npz', data = [mol_dict])
