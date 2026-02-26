import pandas as pd
import requests
from typing import List


def extract_smiles_and_ids(molecule_list: List[dict]) -> pd.DataFrame:
    """
    Extract SMILES and ChEMBL IDs from a list of molecule dicts.
    Args:
        molecule_list (list): List of molecule dicts from ChEMBL API.
    Returns:
        pd.DataFrame: DataFrame with columns 'smiles' and 'chembl_id'.
    """
    smiles = []
    chembl_ids = []
    for mol in molecule_list:
        if (
            "molecule_structures" in mol
            and mol["molecule_structures"] is not None
        ):
            smi = mol["molecule_structures"]["canonical_smiles"]
            smiles.append(smi)
            chembl_ids.append(mol["molecule_chembl_id"])
    return pd.DataFrame({"smiles": smiles, "chembl_id": chembl_ids})


def fetch_chembl_molecules(
    desired_total: int = 10000,
    limit: int = 1000
) -> List[dict]:
    """
    Fetch molecules from ChEMBL API.
    Args:
        desired_total (int): Total number of molecules to fetch.
        limit (int): Number of molecules per request (max 1000 for ChEMBL API).
    Returns:
        List of molecule dicts.
    """
    url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
    all_molecules = []
    for offset in range(0, desired_total, limit):
        params = {"limit": limit, "offset": offset}
        r = requests.get(url, params=params).json()
        all_molecules.extend(r["molecules"])
    return all_molecules