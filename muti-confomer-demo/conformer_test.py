import torch
import numpy as np
import pandas as pd
import pickle
import os
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS
from torch_geometric.data import Data, Batch
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Import your model classes
from GraphModel.GNN_2d_hsolvent import GNNNodeEncoder, NodeEncodeInterface

# Feature definitions
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_atom_partial_charge': (-1.00000, 1.00000),
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def create_atom_mapping_by_environment(input_mol, pickle_mol):
    """
    Create atom mapping between input molecule and pickle molecule based on atom environments.
    """
    print("    Creating atom mapping based on chemical environments...")

    # First try canonical SMILES comparison
    input_canonical = Chem.MolToSmiles(input_mol, canonical=True)
    pickle_canonical = Chem.MolToSmiles(pickle_mol, canonical=True)

    print(f"    Input canonical SMILES:  {input_canonical}")
    print(f"    Pickle canonical SMILES: {pickle_canonical}")

    if input_canonical == pickle_canonical:
        print("    Canonical SMILES match - molecules are identical")

        # Try MCS-based alignment first
        try:
            mcs = rdFMCS.FindMCS([input_mol, pickle_mol],
                                 atomCompare=rdFMCS.AtomCompare.CompareElements,
                                 bondCompare=rdFMCS.BondCompare.CompareOrder,
                                 matchValences=True,
                                 ringMatchesRingOnly=True,
                                 completeRingsOnly=False,
                                 timeout=30)

            if not mcs.canceled and mcs.numAtoms > 0:
                mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
                if mcs_mol is not None:
                    matches1 = input_mol.GetSubstructMatches(mcs_mol, useChirality=False)
                    matches2 = pickle_mol.GetSubstructMatches(mcs_mol, useChirality=False)

                    if matches1 and matches2:
                        match1 = matches1[0]
                        match2 = matches2[0]

                        atom_mapping = {}
                        for i, (atom1_idx, atom2_idx) in enumerate(zip(match1, match2)):
                            atom_mapping[atom1_idx] = atom2_idx

                        print(f"    MCS mapping: {len(atom_mapping)} atoms")
                        return atom_mapping
        except Exception as e:
            print(f"    WARNING: MCS failed: {e}")

        # Fallback: try to match by atom environments
        print("    Trying environment-based matching...")

        # Generate atom invariants
        input_invariants = Chem.CanonicalRankAtoms(input_mol, breakTies=True)
        pickle_invariants = Chem.CanonicalRankAtoms(pickle_mol, breakTies=True)

        # Create mapping based on invariants
        input_inv_to_idx = {inv: idx for idx, inv in enumerate(input_invariants)}
        pickle_inv_to_idx = {inv: idx for idx, inv in enumerate(pickle_invariants)}

        atom_mapping = {}
        for inv in input_inv_to_idx:
            if inv in pickle_inv_to_idx:
                input_idx = input_inv_to_idx[inv]
                pickle_idx = pickle_inv_to_idx[inv]
                atom_mapping[input_idx] = pickle_idx

        if len(atom_mapping) == input_mol.GetNumAtoms():
            print(f"    Complete environment mapping: {len(atom_mapping)} atoms")
            return atom_mapping
        else:
            print(f"    WARNING: Incomplete mapping: {len(atom_mapping)}/{input_mol.GetNumAtoms()} atoms")
            return atom_mapping if len(atom_mapping) > 0 else None
    else:
        print("    ERROR: Canonical SMILES don't match - molecules may be different!")
        return None


def load_conformers_from_pickle(pickle_dir: str, target_smiles: str, max_conformers: int = 20) -> Tuple[
    List[Chem.Mol], List[float]]:
    """
    Load conformers and Boltzmann weights from pickle file
    """
    pickle_filename = f"{target_smiles}.pickle"
    pickle_path = os.path.join(pickle_dir, pickle_filename)

    if not os.path.exists(pickle_path):
        print(f"    Pickle file not found: {pickle_filename}")
        return [], []

    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        print(f"    Loaded pickle file: {pickle_filename}")

        if 'conformers' not in data:
            print("    ERROR: No conformers found in pickle data")
            return [], []

        rd_mols = []
        boltzmann_weights = []

        conformers_to_process = data['conformers'][:max_conformers]
        print(f"    Processing {len(conformers_to_process)} conformers (max: {max_conformers})")

        for i, conf in enumerate(conformers_to_process):
            rd_mol = conf.get('rd_mol')
            boltzmann_weight = conf.get('boltzmannweight')

            if rd_mol is not None and boltzmann_weight is not None:
                rd_mols.append(rd_mol)
                boltzmann_weights.append(boltzmann_weight)

        print(f"    Valid conformers: {len(rd_mols)}/{len(conformers_to_process)}")
        return rd_mols, boltzmann_weights

    except Exception as e:
        print(f"    ERROR: Error loading pickle: {e}")
        return [], []


class ConformerEnhancedSMILESEvaluator:
    """
    SMILES evaluator with conformer integration for better 3D coordinates
    """

    def __init__(self, model_path: str, csv_file: str = None, annotation_path: str = None,
                 pickle_dir: str = None):
        """
        Initialize evaluator with conformer support

        Args:
            model_path: Path to trained model
            csv_file: CSV file with molecule data
            annotation_path: Path to annotation files with ground truth
            pickle_dir: Directory containing conformer pickle files
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.csv_file = csv_file
        self.annotation_path = annotation_path
        self.pickle_dir = pickle_dir

        # Load molecule database if provided
        self.molecule_db = None
        if csv_file and os.path.exists(csv_file):
            self.molecule_db = pd.read_csv(csv_file)
            print(f"Loaded molecule database: {len(self.molecule_db)} entries")

        if pickle_dir:
            print(f"Conformer directory: {pickle_dir}")

    def _load_model(self, model_path: str):
        """Load the trained model"""
        nodeEncoder = GNNNodeEncoder(
            num_layer=5,
            emb_dim=512,
            JK="last",
            gnn_type='gin',
            aggr='add'
        )

        model = NodeEncodeInterface(
            nodeEncoder,
            hidden_channels=512,
            c_out_hidden=[128, 64],
            h_out_hidden=[128, 64],
            solvent_emb_dim=32,
            h_out_channels=2,
            use_solvent=True
        )

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        print(model)
        model.eval()
        return model

    def smiles_to_graph_basic(self, smiles: str, solvent_class: int = 8) -> Data:
        """Convert SMILES to graph data using basic 3D generation"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")

            mol = Chem.AddHs(mol)

            # Generate 3D coordinates
            success = False
            for attempt in range(5):
                params = AllChem.ETKDGv3()
                params.randomSeed = attempt

                if AllChem.EmbedMolecule(mol, params) == 0:
                    AllChem.MMFFOptimizeMolecule(mol)
                    conf = mol.GetConformer()
                    z_coords = [conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())]
                    if any(abs(z) > 0.1 for z in z_coords):
                        success = True
                        break

            if not success:
                print(f"    WARNING: Poor 3D coordinates generated")

            return self._mol_to_graph_data(mol, solvent_class, "basic_3d")

        except Exception as e:
            print(f"    ERROR: Error in basic graph conversion: {e}")
            return None

    def smiles_to_graph_conformer(self, input_smiles: str, pickle_smiles: str,
                                  conformer_mol: Chem.Mol, solvent_class: int = 8,
                                  conformer_idx: int = None) -> Data:
        """Convert SMILES to graph using conformer coordinates"""
        try:
            input_mol = Chem.MolFromSmiles(input_smiles)
            if input_mol is None:
                raise ValueError(f"Invalid input SMILES: {input_smiles}")

            input_mol = Chem.AddHs(input_mol)

            # Create atom mapping
            atom_mapping = create_atom_mapping_by_environment(input_mol, conformer_mol)
            if atom_mapping is None:
                print(f"    ERROR: Could not create atom mapping for conformer {conformer_idx}")
                return None

            # Get conformer coordinates and reorder them
            if conformer_mol.GetNumConformers() > 0:
                pickle_positions = conformer_mol.GetConformer(0).GetPositions()
                reordered_positions = np.zeros((input_mol.GetNumAtoms(), 3))

                for input_idx in range(input_mol.GetNumAtoms()):
                    if input_idx in atom_mapping:
                        pickle_idx = atom_mapping[input_idx]
                        reordered_positions[input_idx] = pickle_positions[pickle_idx]
                    else:
                        reordered_positions[input_idx] = [0.0, 0.0, 0.0]

                # Set the reordered coordinates to input molecule
                conf = Chem.Conformer(input_mol.GetNumAtoms())
                for i in range(input_mol.GetNumAtoms()):
                    conf.SetAtomPosition(i, reordered_positions[i])
                input_mol.AddConformer(conf, assignId=True)

            return self._mol_to_graph_data(input_mol, solvent_class, f"conformer_{conformer_idx}")

        except Exception as e:
            print(f"    ERROR: Error in conformer graph conversion: {e}")
            return None

    def _mol_to_graph_data(self, mol: Chem.Mol, solvent_class: int, coord_type: str) -> Data:
        """Convert RDKit mol to graph data with detailed printing"""
        try:
            # Compute charges
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                pass

            # Print molecule structure
            print(f"    COORDINATES ({coord_type}):")
            print(f"    {'Atom':<4} {'Symbol':<6} {'X (A)':<10} {'Y (A)':<10} {'Z (A)':<10}")
            print(f"    {'-' * 4} {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10}")

            # Node features
            atom_features_list = []
            for atom_idx, atom in enumerate(mol.GetAtoms()):
                try:
                    atomic_num = atom.GetAtomicNum()
                    chiral_tag = atom.GetChiralTag()
                    hybridization = atom.GetHybridization()
                    symbol = atom.GetSymbol()

                    atomic_num_idx = allowable_features['possible_atomic_num_list'].index(atomic_num)
                    chirality_idx = allowable_features['possible_chirality_list'].index(chiral_tag)
                    hybridization_idx = allowable_features['possible_hybridization_list'].index(hybridization)

                    atom_feature = [atomic_num_idx, chirality_idx, hybridization_idx]
                    atom_features_list.append(atom_feature)

                except ValueError:
                    atom_feature = [5, 0, 6]  # Default
                    atom_features_list.append(atom_feature)

            x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

            # 3D positions
            try:
                if mol.GetNumConformers() > 0:
                    positions = mol.GetConformer().GetPositions()
                    pos = torch.tensor(positions, dtype=torch.float)

                    # Print coordinates
                    for i in range(len(positions)):
                        atom = mol.GetAtomWithIdx(i)
                        symbol = atom.GetSymbol()
                        x_coord, y_coord, z_coord = positions[i]
                        print(f"    {i:<4} {symbol:<6} {x_coord:<10.4f} {y_coord:<10.4f} {z_coord:<10.4f}")
                else:
                    pos = torch.zeros((mol.GetNumAtoms(), 3), dtype=torch.float)
                    print(f"    WARNING: No conformer - using zero coordinates")
            except:
                pos = torch.zeros((mol.GetNumAtoms(), 3), dtype=torch.float)
                print(f"    ERROR: Error getting coordinates - using zeros")

            # Edge features
            edges_list = []
            edge_features_list = []

            for bond in mol.GetBonds():
                try:
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()

                    bond_type_idx = allowable_features['possible_bonds'].index(bond.GetBondType())
                    bond_dir_idx = allowable_features['possible_bond_dirs'].index(bond.GetBondDir())

                    edge_feature = [bond_type_idx, bond_dir_idx]
                    edges_list.extend([(i, j), (j, i)])
                    edge_features_list.extend([edge_feature, edge_feature])
                except ValueError:
                    continue

            if edges_list:
                edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
                edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 2), dtype=torch.long)

            # Create data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
            data.has_c = True
            data.has_h = True
            data.solvent_class = torch.tensor([solvent_class])

            print(f"    Graph: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
            return data

        except Exception as e:
            print(f"    ERROR: Error creating graph data: {e}")
            return None

    def predict_nmr_single(self, smiles: str, solvent_class: int = 8, use_conformers: bool = True,
                           max_conformers: int = 10) -> Dict:
        """
        Predict NMR spectrum from SMILES with optional conformer weighting

        Args:
            smiles: Input SMILES string
            solvent_class: Solvent class
            use_conformers: Whether to use conformer weighting if available
            max_conformers: Maximum number of conformers to use
        """
        print(f"\nPREDICTING NMR FOR: {smiles}")
        print(f"Solvent class: {solvent_class}")
        print(f"Use conformers: {use_conformers}")
        print("-" * 60)

        # Check if conformers are available
        conformer_available = False
        canonical_smiles = None

        if use_conformers and self.pickle_dir:
            try:
                mol = Chem.MolFromSmiles(smiles)
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

                # Try to find conformers
                rd_mols, boltzmann_weights = load_conformers_from_pickle(
                    self.pickle_dir, canonical_smiles, max_conformers
                )

                if rd_mols and boltzmann_weights:
                    conformer_available = True
                    print(f"Found {len(rd_mols)} conformers for canonical SMILES: {canonical_smiles}")
                else:
                    print(f"No conformers found for: {canonical_smiles}")

            except Exception as e:
                print(f"WARNING: Error checking conformers: {e}")

        # Predict using conformers if available
        if conformer_available:
            return self._predict_with_conformers(smiles, canonical_smiles, rd_mols,
                                                 boltzmann_weights, solvent_class)
        else:
            return self._predict_basic(smiles, solvent_class)

    def _predict_basic(self, smiles: str, solvent_class: int) -> Dict:
        """Predict using basic 3D generation"""
        print("Using basic 3D coordinate generation...")

        try:
            graph_data = self.smiles_to_graph_basic(smiles, solvent_class)
            if graph_data is None:
                return {"error": "Failed to convert SMILES to graph"}

            batch_data = Batch.from_data_list([graph_data])
            batch_data = batch_data.to(self.device)

            with torch.no_grad():
                model_output = self.model(batch_data)

                if isinstance(model_output, tuple) and len(model_output) == 2:
                    predictions, ch_idx = model_output
                    c_shifts, h_shifts = predictions

                    c_shifts = c_shifts.cpu().numpy() * 200
                    h_shifts = h_shifts.cpu().numpy() * 10
                    h_shifts_mean = np.mean(h_shifts, axis=1) if h_shifts.ndim > 1 else h_shifts

                    result = {
                        'method': 'basic_3d',
                        'c_shifts': c_shifts.flatten(),
                        'h_shifts': h_shifts_mean,
                        'ch_idx': ch_idx,
                        'num_pairs': len(c_shifts)
                    }

                    print(f"Basic prediction: {result['num_pairs']} C-H pairs")
                    return result
                else:
                    return {"error": "Unexpected model output"}

        except Exception as e:
            return {"error": f"Basic prediction failed: {str(e)}"}

    def _predict_with_conformers(self, input_smiles: str, pickle_smiles: str,
                                 rd_mols: List[Chem.Mol], boltzmann_weights: List[float],
                                 solvent_class: int) -> Dict:
        """Predict using conformer-weighted ensemble"""
        print("Using conformer-weighted prediction...")

        try:
            # Normalize weights
            boltzmann_weights = np.array(boltzmann_weights)
            boltzmann_weights = boltzmann_weights / np.sum(boltzmann_weights)

            conformer_predictions = []

            for i, (conformer_mol, weight) in enumerate(zip(rd_mols, boltzmann_weights)):
                print(f"\n    Conformer {i + 1}/{len(rd_mols)} (weight: {weight:.4f})")

                print(f"Conformer {conformer_mol}")

                graph_data = self.smiles_to_graph_conformer(
                    input_smiles, pickle_smiles, conformer_mol, solvent_class, i
                )

                print(f"graph_data_pos for {conformer_mol}: {graph_data.pos}")

                batch_data = Batch.from_data_list([graph_data])
                batch_data = batch_data.to(self.device)

                print(f"batch_data, {batch_data} ")

                with torch.no_grad():
                    model_output = self.model(batch_data)



                    if isinstance(model_output, tuple) and len(model_output) == 2:
                        predictions, ch_idx = model_output
                        c_shifts, h_shifts = predictions

                        c_shifts = c_shifts.cpu().numpy() * 200
                        h_shifts = h_shifts.cpu().numpy() * 10
                        h_shifts_mean = np.mean(h_shifts, axis=1) if h_shifts.ndim > 1 else h_shifts

                        conformer_predictions.append({
                            'c_shifts': c_shifts.flatten(),
                            'h_shifts': h_shifts_mean,
                            'weight': weight,
                            'conformer_idx': i
                        })

                        # Print individual conformer predictions
                        print(f"    CONFORMER {i + 1} PREDICTIONS:")
                        for j, (c, h) in enumerate(zip(c_shifts.flatten(), h_shifts_mean)):
                            print(f"      Peak {j + 1}: 13C={c:.5f} ppm, 1H={h:.5f} ppm")


            if not conformer_predictions:
                print("ERROR: All conformer predictions failed, falling back to basic method")
                return self._predict_basic(input_smiles, solvent_class)

            # Compute weighted average
            print(f"\nComputing weighted ensemble from {len(conformer_predictions)} conformers...")

            num_pairs = conformer_predictions[0]['c_shifts'].shape[0]
            weighted_c_shifts = np.zeros(num_pairs)
            weighted_h_shifts = np.zeros(num_pairs)
            total_weight = 0

            for pred in conformer_predictions:
                weight = pred['weight']
                weighted_c_shifts += pred['c_shifts'] * weight
                weighted_h_shifts += pred['h_shifts'] * weight
                total_weight += weight

            if total_weight > 0:
                weighted_c_shifts /= total_weight
                weighted_h_shifts /= total_weight

            result = {
                'method': 'conformer_weighted',
                'c_shifts': weighted_c_shifts,
                'h_shifts': weighted_h_shifts,
                'num_pairs': len(weighted_c_shifts),
                'num_conformers_used': len(conformer_predictions),
                'total_conformers': len(rd_mols),
                'total_weight': total_weight,
                'individual_predictions': conformer_predictions
            }

            print(f"Conformer-weighted prediction: {result['num_pairs']} C-H pairs")
            print(f"   Used {result['num_conformers_used']}/{result['total_conformers']} conformers")

            return result

        except Exception as e:
            print(f"ERROR: Conformer prediction failed: {e}")
            return self._predict_basic(input_smiles, solvent_class)

    def load_ground_truth(self, filename: str) -> Dict:
        """Load ground truth annotations"""
        try:
            if self.annotation_path and os.path.exists(self.annotation_path):
                annotation_file = os.path.join(self.annotation_path, f'{filename}.csv')
                if os.path.exists(annotation_file):
                    gt_data = pd.read_csv(annotation_file)
                    gt_data = gt_data.sort_values(by='c_idx')

                    return {
                        'c_shifts': gt_data['C'].values,
                        'h_shifts': gt_data[['H 1', 'H 2']].values,
                        'h_shifts_mean': np.mean(gt_data[['H 1', 'H 2']].values, axis=1),
                        'num_pairs': len(gt_data)
                    }
        except Exception as e:
            print(f"WARNING: Could not load ground truth for {filename}: {e}")

        return None

    def get_molecule_info(self, identifier: str) -> Tuple[str, str, int]:
        """Get molecule info from database"""
        if self.molecule_db is None:
            return identifier, identifier, 8

        # Try to find by filename first
        if not identifier.endswith('.csv'):
            identifier_csv = f'{identifier}.csv'
        else:
            identifier_csv = identifier
            identifier = identifier[:-4]

        row = self.molecule_db[self.molecule_db['File_name'] == identifier_csv]

        if not row.empty:
            return (identifier,
                    row['SMILES'].iloc[0],
                    row['solvent_class'].iloc[0] if 'solvent_class' in row.columns else 8)

        # If not found by filename, try as SMILES
        row = self.molecule_db[self.molecule_db['SMILES'] == identifier]
        if not row.empty:
            filename = row['File_name'].iloc[0].replace('.csv', '')
            return (filename,
                    identifier,
                    row['solvent_class'].iloc[0] if 'solvent_class' in row.columns else 8)

        return identifier, identifier, 0

    def evaluate_molecules(self, molecule_list: List[str], use_conformers: bool = True,
                           max_conformers: int = 10, save_results: bool = True) -> List[Dict]:
        """
        Evaluate multiple molecules with optional conformer enhancement
        """
        results = []

        print(f"EVALUATING {len(molecule_list)} MOLECULES")
        print(f"Use conformers: {use_conformers}")
        print(f"Max conformers: {max_conformers}")
        print("=" * 80)

        for i, mol_id in enumerate(molecule_list):
            print(f"\nMolecule {i + 1}/{len(molecule_list)}: {mol_id}")
            print("-" * 60)

            # Get molecule information
            filename, smiles, solvent_class = self.get_molecule_info(mol_id)

            print(f"Filename: {filename}")
            print(f"SMILES: {smiles}")
            print(f"Solvent class: {solvent_class}")

            # Make prediction
            prediction = self.predict_nmr_single(smiles, solvent_class, use_conformers, max_conformers)

            if 'error' in prediction:
                print(f"ERROR: Prediction failed: {prediction['error']}")
                results.append({
                    'filename': filename,
                    'smiles': smiles,
                    'solvent_class': solvent_class,
                    'prediction': None,
                    'ground_truth': None,
                    'error': prediction['error']
                })
                continue

            # Load ground truth
            ground_truth = self.load_ground_truth(filename)

            # Calculate errors if ground truth available
            errors = None
            if ground_truth:
                try:
                    if len(prediction['c_shifts']) == len(ground_truth['c_shifts']):
                        c_mae = np.mean(np.abs(prediction['c_shifts'] - ground_truth['c_shifts']))
                        h_mae = np.mean(np.abs(prediction['h_shifts'] - ground_truth['h_shifts_mean']))
                        errors = {'c_mae': c_mae, 'h_mae': h_mae}

                        print(f"Prediction successful ({prediction['method']}): {prediction['num_pairs']} C-H pairs")
                        print(f"Errors: C={c_mae:.3f} ppm, H={h_mae:.3f} ppm")
                    else:
                        print(
                            f"WARNING: Size mismatch: pred={len(prediction['c_shifts'])}, gt={len(ground_truth['c_shifts'])}")
                except Exception as e:
                    print(f"ERROR: Error calculation failed: {e}")
            else:
                print(f"Prediction successful ({prediction['method']}): {prediction['num_pairs']} C-H pairs")
                print("No ground truth available")

            # Print chemical shifts
            print(f"\nPREDICTED C-H PAIRS ({prediction.get('method', 'unknown')}):")
            for j, (c, h) in enumerate(zip(prediction['c_shifts'], prediction['h_shifts'])):
                print(f"   Peak {j + 1}: 13C={c:.5f} ppm, 1H={h:.5f} ppm")

            if ground_truth:
                print(f"\nGROUND TRUTH C-H PAIRS:")
                for j, (c, h) in enumerate(zip(ground_truth['c_shifts'], ground_truth['h_shifts_mean'])):
                    print(f"   Peak {j + 1}: 13C={c:.5f} ppm, 1H={h:.5f} ppm")

            # Store results
            result = {
                'filename': filename,
                'smiles': smiles,
                'solvent_class': solvent_class,
                'prediction': prediction,
                'ground_truth': ground_truth,
                'errors': errors
            }
            results.append(result)

        # Save results
        if save_results:
            self.save_evaluation_results(results)

        # Print summary
        self.print_summary(results)

        return results

    def save_evaluation_results(self, results: List[Dict]):
        """Save evaluation results to files"""
        try:
            # Create summary DataFrame
            summary_data = []
            for result in results:
                if 'error' in result:
                    summary_data.append({
                        'filename': result['filename'],
                        'smiles': result['smiles'][:50] + '...' if len(result['smiles']) > 50 else result['smiles'],
                        'solvent_class': result['solvent_class'],
                        'method': 'FAILED',
                        'status': 'FAILED',
                        'error': result['error'],
                        'num_pairs': None,
                        'num_conformers': None,
                        'c_mae': None,
                        'h_mae': None
                    })
                else:
                    pred = result['prediction']
                    summary_data.append({
                        'filename': result['filename'],
                        'smiles': result['smiles'][:50] + '...' if len(result['smiles']) > 50 else result['smiles'],
                        'solvent_class': result['solvent_class'],
                        'method': pred.get('method', 'unknown'),
                        'status': 'SUCCESS',
                        'error': None,
                        'num_pairs': pred['num_pairs'],
                        'num_conformers': pred.get('num_conformers_used', 'N/A'),
                        'c_mae': result['errors']['c_mae'] if result['errors'] else None,
                        'h_mae': result['errors']['h_mae'] if result['errors'] else None
                    })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv('conformer_enhanced_evaluation_summary.csv', index=False)
            print(f"\nResults saved to: conformer_enhanced_evaluation_summary.csv")

            # Save detailed results as pickle
            with open('conformer_enhanced_evaluation_detailed.pkl', 'wb') as f:
                pickle.dump(results, f)
            print(f"Detailed results saved to: conformer_enhanced_evaluation_detailed.pkl")

        except Exception as e:
            print(f"WARNING: Failed to save results: {e}")

    def print_summary(self, results: List[Dict]):
        """Print evaluation summary"""
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        with_gt = [r for r in successful if r['ground_truth'] is not None]

        # Count by method
        basic_results = [r for r in successful if r['prediction'].get('method') == 'basic_3d']
        conformer_results = [r for r in successful if r['prediction'].get('method') == 'conformer_weighted']

        print(f"\nEVALUATION SUMMARY")
        print("=" * 50)
        print(f"Total molecules: {len(results)}")
        print(f"Successful predictions: {len(successful)}")
        print(f"  - Basic 3D method: {len(basic_results)}")
        print(f"  - Conformer-weighted: {len(conformer_results)}")
        print(f"Failed predictions: {len(failed)}")
        print(f"With ground truth: {len(with_gt)}")

        if conformer_results:
            total_conformers_used = sum([r['prediction'].get('num_conformers_used', 0) for r in conformer_results])
            avg_conformers = total_conformers_used / len(conformer_results) if conformer_results else 0
            print(f"Average conformers used: {avg_conformers:.1f}")

        if with_gt:
            # Overall errors
            c_errors = [r['errors']['c_mae'] for r in with_gt if r['errors']]
            h_errors = [r['errors']['h_mae'] for r in with_gt if r['errors']]

            if c_errors and h_errors:
                print(f"\nOVERALL ERROR STATISTICS")
                print("-" * 30)
                print(f"Carbon MAE:   {np.mean(c_errors):.3f} ± {np.std(c_errors):.3f} ppm")
                print(f"Hydrogen MAE: {np.mean(h_errors):.3f} ± {np.std(h_errors):.3f} ppm")

            # Method comparison
            basic_with_gt = [r for r in with_gt if r['prediction'].get('method') == 'basic_3d']
            conformer_with_gt = [r for r in with_gt if r['prediction'].get('method') == 'conformer_weighted']

            if basic_with_gt and conformer_with_gt:
                basic_c_errors = [r['errors']['c_mae'] for r in basic_with_gt if r['errors']]
                basic_h_errors = [r['errors']['h_mae'] for r in basic_with_gt if r['errors']]
                conf_c_errors = [r['errors']['c_mae'] for r in conformer_with_gt if r['errors']]
                conf_h_errors = [r['errors']['h_mae'] for r in conformer_with_gt if r['errors']]

                print(f"\nMETHOD COMPARISON")
                print("-" * 30)
                if basic_c_errors and basic_h_errors:
                    print(f"Basic 3D method:")
                    print(f"  Carbon MAE:   {np.mean(basic_c_errors):.3f} ± {np.std(basic_c_errors):.3f} ppm")
                    print(f"  Hydrogen MAE: {np.mean(basic_h_errors):.3f} ± {np.std(basic_h_errors):.3f} ppm")

                if conf_c_errors and conf_h_errors:
                    print(f"Conformer-weighted:")
                    print(f"  Carbon MAE:   {np.mean(conf_c_errors):.3f} ± {np.std(conf_c_errors):.3f} ppm")
                    print(f"  Hydrogen MAE: {np.mean(conf_h_errors):.3f} ± {np.std(conf_h_errors):.3f} ppm")

        if failed:
            print(f"\nFAILED MOLECULES:")
            for result in failed:
                print(f"   {result['filename']}: {result['error']}")

    def compare_methods(self, smiles: str, solvent_class: int = 8, max_conformers: int = 10) -> Dict:
        """
        Compare basic and conformer-weighted predictions for a single molecule
        """
        print(f"\nMETHOD COMPARISON FOR: {smiles}")
        print("=" * 60)

        results = {}

        # Basic prediction
        print("\n1. BASIC 3D PREDICTION:")
        basic_result = self.predict_nmr_single(smiles, solvent_class, use_conformers=False)
        results['basic'] = basic_result

        # Conformer-weighted prediction
        print("\n2. CONFORMER-WEIGHTED PREDICTION:")
        conformer_result = self.predict_nmr_single(smiles, solvent_class, use_conformers=True,
                                                   max_conformers=max_conformers)
        results['conformer'] = conformer_result

        # Compare results
        print("\nCOMPARISON:")
        print("-" * 30)

        if 'error' not in basic_result and 'error' not in conformer_result:
            print(f"Basic method: {basic_result['num_pairs']} C-H pairs")
            print(f"Conformer method: {conformer_result['num_pairs']} C-H pairs")

            if conformer_result.get('method') == 'conformer_weighted':
                print(
                    f"Conformers used: {conformer_result['num_conformers_used']}/{conformer_result['total_conformers']}")

            # Show differences
            if basic_result['num_pairs'] == conformer_result['num_pairs']:
                c_diff = np.abs(basic_result['c_shifts'] - conformer_result['c_shifts'])
                h_diff = np.abs(basic_result['h_shifts'] - conformer_result['h_shifts'])

                print(f"\nAverage differences:")
                print(f"  Carbon: {np.mean(c_diff):.3f} ppm (max: {np.max(c_diff):.3f})")
                print(f"  Hydrogen: {np.mean(h_diff):.3f} ppm (max: {np.max(h_diff):.3f})")

        return results


def demo_conformer_enhanced_evaluation():
    """Demonstrate conformer-enhanced evaluation"""

    # Configuration
    model_path = 'ckpt/model_2dnmr.pt'
    csv_file = './data_csv/2dnmr/all_files_testdata_solvent_mw.csv'
    annotation_path = './test_data_annotation'
    pickle_dir = "./muti-confomer-demo"  # Your conformer directory

    # Test molecules
    test_molecules = [
        #'CC(C)CC#C',  # O
       #'CSC[C@H](N)C(=O)O',  # Different SMILES ordering (same molecule)
        'ON=C1CCCC1',      # From database (if you want to test with filename)
        # Add more molecules as needed
    ]

    try:
        # Initialize evaluator with conformer support
        evaluator = ConformerEnhancedSMILESEvaluator(
            model_path=model_path,
            csv_file=csv_file,
            annotation_path=annotation_path,
            pickle_dir=pickle_dir
        )

        print("DEMO: Conformer-Enhanced SMILES Evaluation")
        print("=" * 60)

        # Run evaluation with conformers
        print("\nEVALUATION WITH CONFORMER ENHANCEMENT:")
        results_with_conformers = evaluator.evaluate_molecules(
            test_molecules,
            use_conformers=True,
            max_conformers=40
        )

        # Run evaluation without conformers for comparison
        print("\nEVALUATION WITH BASIC 3D GENERATION:")
        results_basic = evaluator.evaluate_molecules(
            test_molecules,
            use_conformers=False
        )

        print(f"\nEvaluation complete!")
        print("Check the output files for detailed results:")
        print("  - conformer_enhanced_evaluation_summary.csv")
        print("  - conformer_enhanced_evaluation_detailed.pkl")

    except Exception as e:
        print(f"ERROR: Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_conformer_enhanced_evaluation()