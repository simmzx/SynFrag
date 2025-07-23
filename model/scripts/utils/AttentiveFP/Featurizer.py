from rdkit.Chem import MolFromSmiles
from rdkit import Chem
import numpy as np


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"Input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    return [x == s for s in allowable_set] if x in allowable_set else [False] * len(allowable_set)


def atom_to_id(atom):
    return atom.GetAtomicNum()


def atom_features(atom, bool_id_feat=False, explicit_H=False, use_chirality=True):
    results = one_of_k_encoding_unk(atom.GetSymbol(), ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'other']) + \
              one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2, 'other'
              ]) + \
              [atom.GetIsAromatic()]

    if not explicit_H:
        results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    if use_chirality:
        try:
            results += one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except KeyError:
            results += [False, False] + [atom.HasProp('_ChiralityPossible')]

    return np.array([atom_to_id(atom)]) if bool_id_feat else np.array(results)


def bond_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]

    if use_chirality:
        bond_feats += one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])

    return np.array(bond_feats)


def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    return len(atom_features(m.GetAtoms()[0]))


def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))
