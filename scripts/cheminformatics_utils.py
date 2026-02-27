import numpy as np
from typing import List, Dict, Tuple, Union
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors


def process_molecule(
    mol: Chem.Mol, patterns: Dict[str, Chem.Mol]
) -> Tuple[bool, Union[str, None]]:
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
        "logP": [Descriptors.MolLogP(m) for m in mols],
        "MW": [Descriptors.MolWt(m) for m in mols],
        "HDonors": [Descriptors.NumHDonors(m) for m in mols],
        "HAcceptors": [Descriptors.NumHAcceptors(m) for m in mols],
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


def generate_conformers(mol: Chem.Mol, num_conformers: int = 10) -> Union[Chem.Mol, None]:
    """Generate 3D conformers for a given molecule.

    Args:
        mol (Chem.Mol): RDKit molecule for which to generate conformers.
        num_conformers (int): Number of conformers to generate.

    Returns:
        Union[Chem.Mol, None]: Molecule with generated conformers or None if
          optimization fails.
    """
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers)
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol)
        return mol
    except ValueError:
        return None


def write_xyz(mol, conf_id, file_path):
    """Write the 3D coordinates of a molecule's conformer to an XYZ file.

    Args:
        mol (Chem.Mol): RDKit molecule containing the conformer.
        conf_id (int): ID of the conformer to write.
        file_path (str): Path to the output XYZ file.
    """
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    conf = mol.GetConformer(conf_id)
    n_atoms = mol.GetNumAtoms()
    with open(file_path, "w") as f:
        f.write(f"{n_atoms}\n\n")
        for i in range(n_atoms):
            pos = conf.GetAtomPosition(i)
            f.write(f"{atoms[i]} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
