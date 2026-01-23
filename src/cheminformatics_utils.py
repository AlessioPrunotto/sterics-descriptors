import numpy as np
from typing import List, Dict
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors


def process_molecule(
    mol: Chem.Mol, patterns: Dict[str, Chem.Mol]
) -> tuple[bool, str | None]:
    """Evaluate if a molecule contains exactly one amine group
    and if so, of which type.

    Args:
        mol (Chem.Mol): RDKit molecule to evaluate.
        patterns (Dict[str, Chem.Mol]): Dictionary of amine patterns to match.

    Returns:
        tuple[bool, str | None]: Tuple indicating if exactly one amine is
        present and its type.
    """    
    # Find all matches for each type
    matches = {
        label: len(mol.GetSubstructMatches(patt)) for label, patt in patterns.items()
    }
    total_amines = sum(matches.values())

    # Logic: Keep only if exactly 1 amine is present
    if total_amines == 1:
        # Determine which one it was
        amine_type = next(label for label, count in matches.items() if count == 1)
        return True, amine_type
    else:
        return False, None


def calc_descriptors(smiles_list: List[str]) -> Dict[str, List[float]]:
    """Calculate molecular descriptors for a list of SMILES strings.

    Args:
        smiles_list (List[str]): List of SMILES strings representing molecules.

    Returns:
        Dict[str, List[float]]: Dictionary with descriptor names as keys and
        lists of descriptor values.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    return {
        "logP": [Descriptors.MolLogP(m) for m in mols],  # type: ignore
        "MW": [Descriptors.MolWt(m) for m in mols],  # type: ignore
        "HDonors": [Descriptors.NumHDonors(m) for m in mols],  # type: ignore
        "HAcceptors": [Descriptors.NumHAcceptors(m) for m in mols],  # type: ignore
    }


def calculate_tanimoto_similarity(fps: List[Chem.rdchem.Mol]) -> np.ndarray:
    """Compute the Tanimoto similarity matrix for a list of fingerprints.

    Args:
        fps (List[Chem.rdchem.Mol]): List of RDKit molecule fingerprints.

    Returns:
        np.ndarray: 2D array representing the Tanimoto similarity matrix.
    """    
    n = len(fps)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = DataStructs.TanimotoSimilarity(fps[i], fps[j])
    return similarity_matrix
