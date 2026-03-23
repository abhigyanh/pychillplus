# chillplus.py
### IMPORTS ###
import os
import argparse
from tqdm import tqdm

import numpy as np
from scipy.special import sph_harm_y

import pandas as pd
try:
    import tables
except ImportError:
    print('pytables is required for reading and writing HDF5 files for this analysis.')

import MDAnalysis as mda
from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch
from MDAnalysis.lib.mdamath import triclinic_vectors
### IMPORTS ###



### HELPER AND COMPUTATION FUNCTIONS ###
def minimum_image(vector, box):
    """
    Calculates displacement vector j - i using Minimum Image Convention.
    Works for orthogonal boxes. box = [lx, ly, lz, 90, 90, 90]
    """
    H = triclinic_vectors(box)
    cartesian = vector
    fractional = np.linalg.solve(H, cartesian)
    fractional -= np.round(fractional)
    answer = np.dot(H, fractional)
    return answer

def q_i_lm(atom_i, neighbours, l, m):
    q_i = 0+0j
    """Calculate q_lm(i) for atom_i given its neighbours."""
    for atom_j in neighbours:
        displacement = atom_j.position - atom_i.position
        # Minimum image convention
        r_ij = minimum_image(displacement, box=u.dimensions)
        r = r_ij/np.linalg.norm(r_ij)
        theta = np.arccos(r[2])  # polar angle
        phi = np.arctan2(r[1], r[0])  # azimuthal angle
        q_i += sph_harm_y(l,m,theta,phi)
    return q_i/4

def calculate_qi_all(OW_atomgroup, ns, l=3):
    qilm_values = np.zeros((len(OW_atomgroup), 2*l+1), dtype=complex)
    for i_atom,atom_i in enumerate(OW_atomgroup):
        # Build neighbor list tree
        neighbors = ns.search(atom_i, 3.5, level='A')
        neighbors = [atom_j for atom_j in neighbors if atom_j.index != atom_i.index]  # Exclude self
        for m in range(-l,l+1):
            qilm = q_i_lm(atom_i, neighbors, l=l, m=m)
            qilm_values[i_atom, m+l] = qilm
    return qilm_values

def C_ij(q_i, q_j, l=3):
    numerator = np.sum(q_i*np.conj(q_j))
    denom1 = np.sqrt(np.sum(q_i*np.conj(q_i)))
    denom2 = np.sqrt(np.sum(q_j*np.conj(q_j)))
    return numerator / (denom1 * denom2)
### HELPER AND COMPUTATION FUNCTIONS ###



### MAIN ###
if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-j','--jobs', type=int)
    parser.add_argument('-s','--top', type=str, help='MDAnalysis-readable topology file', required=True)
    parser.add_argument('-f','--traj', type=str, help='MDAnalysis-readable trajectory file', required=True)
    parser.add_argument('-o','--output', type=str, default='chill-plus.csv', help='Output filename in CSV format')
    parser.add_argument('--select', type=str, default='name OW', help='Selection string (default: "name OW")')
    parser.add_argument('-l', type=int, default=3, help='Spherical harmonic l-value (default: 3)')
    args = parser.parse_args()

    # File existence check
    if not os.path.exists(args.top):
        raise FileNotFoundError(f"{args.top} not found.")
    if not os.path.exists(args.traj):
        raise FileNotFoundError(f"{args.traj} not found.")

    # Other parameters
    l = args.l

    # Define universe and select atomgroup
    u = mda.Universe(args.top, args.traj)
    # Selection of atoms using criteria
    OW_atomgroup = u.select_atoms(args.select)
    # Create a mapping from global atom indices to local positions in OW_atomgroup
    # this will allow fast lookups later when iterating over neighbours.
    ow_index_map = {atom.index: idx for idx, atom in enumerate(OW_atomgroup)}
    n_frames = u.trajectory.n_frames
    n_atoms = len(OW_atomgroup)

    # 3-index data array
    # indices: frame-idx, atom-idx, type-value:coordination,eclipsed,staggered
    full_data = np.zeros((n_frames,n_atoms,3))

    # helper for processing a single frame; this will run in worker processes
    def process_frame(frameidx: int) -> np.ndarray:
        """Return an (n_atoms,3) array of coordination/eclipsed/staggered for frameidx."""
        # create fresh universe in each worker to avoid pickling issues
        u_local = mda.Universe(args.top, args.traj)
        OW_local = u_local.select_atoms(args.select)
        ow_map_local = {atom.index: idx for idx, atom in enumerate(OW_local)}
        u_local.trajectory[frameidx]
        ns_local = AtomNeighborSearch(OW_local, box=u_local.dimensions)
        qilm_local = calculate_qi_all(OW_local, ns_local, l=l)
        data = np.zeros((len(OW_local), 3))
        for i_atom, atom_i in enumerate(OW_local):
            neighs = ns_local.search(atom_i, 3.5, level='A')
            neighs = [a for a in neighs if a.index != atom_i.index]
            coord = len(neighs)
            eclipsed = 0
            staggered = 0
            for atom_j in neighs:
                ni = ow_map_local.get(atom_j.index)
                if ni is None:
                    continue
                c_ij = C_ij(qilm_local[i_atom], qilm_local[ni], l=l)
                real_part = np.real(c_ij)
                imag_part = np.imag(c_ij)
                if imag_part > 1e-3:
                    raise Warning(f"C_ij has a significant non-zero imaginary part: {imag_part}")
                c_ij = real_part
                if -0.35 <= c_ij <= 0.25:
                    eclipsed += 1
                elif c_ij <= -0.8:
                    staggered += 1
            data[i_atom, :] = [coord, eclipsed, staggered]
        return data

    # process frames in parallel
    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(process_frame, idx) for idx in range(n_frames)]
        for fut in tqdm(as_completed(futures), total=n_frames, desc='Frame'):
            results.append(fut.result())
    # stack results back into full_data
    full_data = np.stack(results, axis=0)

    # Dataframe conversion
    # Define labels matching your array shape
    frame_labels = [f'frame-{i}' for i in range(n_frames)]
    item_labels = [f'atom-{j}' for j in range(n_atoms)]
    count_labels = ['coordination','eclipsed','staggered']

    # convert
    df = pd.DataFrame(
        full_data.reshape(-1, 3),  # (n_frames*n_items, 3)
        index=pd.MultiIndex.from_product([frame_labels, item_labels], 
                                    names=['frame', 'atom']),
        columns=count_labels
    )
    print("Output dataframe overview:")
    print(df)

    # Save
    df.to_csv(args.output)
    print(f"""--------------------
    Output files saved to {args.output}. 
    Pandas example for reading the output data for further analysis:

    df_loaded = pd.read_csv('{args.output}', index_col=['frame', 'atom'])
    # To get the metric (coordination/eclipsed/staggered) of atom-i from frame-f:
    df_loaded.loc[('frame-f', 'atom-i'), 'coordination']  # or 'eclipsed' / 'staggered'
    --------------------""")
