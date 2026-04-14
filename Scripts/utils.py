import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from chembl_structure_pipeline import standardizer
from chembl_structure_pipeline import exclude_flag

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def is_valid_structure(smiles: str) -> bool:
    """Check if a SMILES string represents a valid, non-excluded chemical structure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    exclude = exclude_flag.exclude_flag(mol)
    return not exclude

def standardize_and_canonicalize(smiles: str) -> str:
    """Standardize and canonicalize a SMILES string using ChEMBL pipeline."""
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        m_no_salts = standardizer.get_parent_mol(molecule) # Remove salts
        to_standardize = m_no_salts[0]
        std_mol = standardizer.standardize_mol(to_standardize)
        canonical_smiles = Chem.MolToSmiles(std_mol)
        return canonical_smiles
    except Exception as e:
        logging.warning(f"Error processing SMILES: {smiles}\n{e}")
        return None

def computeMorganFP(mol, depth: int = 2, nBits: int = 1024):
    """Compute Morgan fingerprint for a molecule as a NumPy array."""
    try:
        gen = _morgan_gen if (depth == 2 and nBits == 1024) else \
              rdFingerprintGenerator.GetMorganGenerator(radius=depth, fpSize=nBits)
        return gen.GetFingerprintAsNumPy(mol).astype(np.float32)
    except:
        return None