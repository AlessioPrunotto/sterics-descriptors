import numpy as np

# A simple van der Waals radii dictionary (can be expanded)
VDW_RADII = {
    "H": 1.2,
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "P": 1.8,
    "S": 1.8,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
    # Metals and others would need to be added
}


def parse_xyz(filename):
    """Parse an XYZ file and return a list of atoms with their symbols and
    coordinates.

    Args:
        filename (str): Path to the XYZ file.

    Returns:
        list: A list of tuples, each containing an atom symbol and its
          coordinates as a NumPy array.
    """
    atoms = []
    with open(filename) as f:
        lines = f.readlines()[2:]  # skip first 2 header lines
        for line in lines:
            parts = line.split()
            symbol = parts[0]
            x, y, z = map(float, parts[1:4])
            atoms.append((symbol, np.array([x, y, z])))
    return atoms


def compute_vbur(xyz_file, central_index, sphere_radius=3.5, grid_density=0.2):
    """Compute the buried volume around a central atom in a molecule.

    Args:
        xyz_file (str): Path to the XYZ file.
        central_index (int): Index of the central atom in the XYZ file.
        sphere_radius (float, optional): Radius of the sphere around the
          central atom. Defaults to 3.5.
        grid_density (float, optional): Density of the grid points within the
          sphere. Defaults to 0.2.

    Returns:
        float: The percentage of the sphere volume that is buried by
          surrounding atoms.
    """
    atoms = parse_xyz(xyz_file)
    central_atom, center = atoms[central_index]

    # Generate a grid of points inside sphere
    r = sphere_radius
    grid = np.arange(-r, r, grid_density)
    points = np.array(
        [
            (x, y, z)
            for x in grid
            for y in grid
            for z in grid
            if x**2 + y**2 + z**2 <= r**2
        ]
    )
    points = points + center  # shift to central atom position

    buried = 0
    total = len(points)

    for p in points:
        for sym, pos in atoms:
            if np.allclose(pos, center):
                continue  # skip central atom
            vdw = VDW_RADII.get(sym, 1.7)  # default if unknown
            if np.linalg.norm(p - pos) <= vdw:
                buried += 1
                break  # point is buried by this atom

    vbur_percent = (buried / total) * 100
    return vbur_percent