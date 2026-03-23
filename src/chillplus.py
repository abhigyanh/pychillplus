# chillplus.py
### IMPORTS ###
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from numba import njit, prange
import numpy as np
from scipy.special import sph_harm_y

import pandas as pd

import mdtraj as md
### IMPORTS ###



### HELPER AND COMPUTATION FUNCTIONS ###
@njit(cache=True)
def minimum_image(vector: np.ndarray, H: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    """
    Minimum image convention for orthogonal or triclinic boxes.
    H     : (3,3) matrix of box vectors (rows = a, b, c)  [nm or Å, consistent]
    H_inv : precomputed np.linalg.inv(H)
    Passing H/H_inv avoids recomputing per call and makes the function pure.
    """
    fractional = H_inv @ vector
    fractional -= np.round(fractional)
    return H @ fractional

@njit(cache=True)
def C_ij(q_i: np.ndarray, q_j: np.ndarray) -> complex:
    """Orientational correlation between two qlm vectors. l is implicit in array length."""
    numerator = np.sum(q_i * np.conj(q_j))
    denom1    = np.sqrt(np.sum(q_i * np.conj(q_i)))
    denom2    = np.sqrt(np.sum(q_j * np.conj(q_j)))
    return numerator / (denom1 * denom2)

def _compute_angles(positions_i: np.ndarray,
                    positions_j: np.ndarray,
                    H: np.ndarray,
                    H_inv: np.ndarray):
    """
    Vectorised (theta, phi) for all i→j displacement vectors.
    positions_i : (N, 3)
    positions_j : (N, 3)   — already matched/expanded to pair with i
    Returns theta (N,), phi (N,)
    """
    disp = positions_j - positions_i                        # (N, 3)
    # apply MIC row-wise — keep as numpy loop over rows for numba compat
    mic = np.array([minimum_image(disp[k], H, H_inv) for k in range(len(disp))])
    norms = np.linalg.norm(mic, axis=1, keepdims=True)
    r_hat = mic / norms
    theta = np.arccos(np.clip(r_hat[:, 2], -1.0, 1.0))
    phi   = np.arctan2(r_hat[:, 1], r_hat[:, 0])
    return theta, phi

def calculate_qi_all(positions: np.ndarray,
                     neigh_dict: dict,
                     box_vectors,         # md.Trajectory.unitcell_vectors[frame] (3,3) nm
                     l: int = 3) -> np.ndarray:
    """
    MDTraj-native replacement for calculate_qi_all.

    Parameters
    ----------
    positions  : (n_atoms, 3) float32/64  — frame_sel.xyz[0], in nm
    neigh_dict : {local_idx: [local_neighbour_idxs]}  from md.compute_neighbors
    box_vectors: (3, 3) array in nm  — frame.unitcell_vectors[0]
    l          : bond-order parameter degree

    Returns
    -------
    qilm_values : (n_atoms, 2l+1) complex128
    """
    H     = box_vectors.T.astype(np.float32)
    H_inv = np.linalg.inv(H).astype(np.float32)
    n_atoms   = len(positions)
    n_m       = 2 * l + 1
    qilm_values = np.zeros((n_atoms, n_m), dtype=np.complex128)

    for i_atom in range(n_atoms):
        neighs = neigh_dict.get(i_atom, [])
        if len(neighs) == 0:
            continue
        pos_i   = positions[i_atom]                              # (3,)
        pos_j   = positions[neighs]                              # (K, 3)
        n_neigh = len(neighs)

        theta, phi = _compute_angles(
            np.tile(pos_i, (n_neigh, 1)),
            pos_j, H, H_inv
        )

        for mi, m in enumerate(range(-l, l + 1)):
            # scipy 1.17+ convention: (l,m,theta,phi)
            ylm = sph_harm_y(l,m,theta,phi)
            qilm_values[i_atom, mi] = np.sum(ylm) / 4.0

    return qilm_values
### HELPER AND COMPUTATION FUNCTIONS ###

# ── Module-level worker — must be defined here for pickling ─────────────────
def process_frame(
    frameidx: int,
    traj: str,
    top: str,
    atom_indices: np.ndarray,   # ← indices, not a string
    l: int) -> tuple[int, np.ndarray]:
    """Load a single frame, compute qi and C_ij, return (frameidx, (n_atoms, 3))."""

    frame     = md.load_frame(traj, frameidx, top=top)
    frame_sel = frame.atom_slice(atom_indices)
    n_sel     = frame_sel.n_atoms

    # compute_neighbors returns List[ndarray] of length n_frames.
    # Each ndarray is the flat set of haystack indices within cutoff of ANY query atom.
    # To get per-atom neighbours, query one atom at a time.
    neigh_dict = {}
    for i in range(n_sel):
        matches = md.compute_neighbors(
            frame_sel,
            cutoff=0.35,
            query_indices=np.array([i]),
            haystack_indices=np.arange(n_sel),
            periodic=True,
        )[0]  # 1D array of matching haystack indices
        neigh_dict[i] = [int(j) for j in matches if j != i]

        positions   = frame_sel.xyz[0]
        box_vectors = frame_sel.unitcell_vectors[0]

    qilm_local = calculate_qi_all(positions, neigh_dict, box_vectors, l=l)

    data = np.zeros((n_sel, 3))
    for i_atom in range(n_sel):
        neighs   = neigh_dict[i_atom]
        coord    = len(neighs)
        eclipsed = staggered = 0
        for j_atom in neighs:
            c_ij_val  = C_ij(qilm_local[i_atom], qilm_local[j_atom])
            real_part = np.real(c_ij_val)
            imag_part = np.imag(c_ij_val)
            if abs(imag_part) > 1e-3:
                raise ValueError(f"C_ij has significant imaginary part: {imag_part}")
            if   -0.35 <= real_part <= 0.25:  eclipsed  += 1
            elif  real_part <= -0.8:           staggered += 1
        data[i_atom] = [coord, eclipsed, staggered]

    return frameidx, data
# ── Module-level worker — must be defined here for pickling ─────────────────





### MAIN ###
def main():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jobs', type=int)
    parser.add_argument('-s', '--top',  type=str, required=True,  help='MDTraj-readable topology file')
    parser.add_argument('-f', '--traj', type=str, required=True,  help='MDTraj-readable trajectory file')
    parser.add_argument('-o', '--output', type=str, default='chill-plus', help='Output file basename (<o>.frame1234.csv)')
    parser.add_argument('--select', type=str, default='name OW', help='Selection string (default: "name OW")')
    parser.add_argument('-l', type=int, default=3, help='Spherical harmonic l-value (default: 3)')
    args = parser.parse_args()

    # File existence check
    if not os.path.exists(args.top):
        raise FileNotFoundError(f"{args.top} not found.")
    if not os.path.exists(args.traj):
        raise FileNotFoundError(f"{args.traj} not found.")

    l = args.l

    # Load topology to resolve atom indices once; avoid holding full traj in memory
    top          = md.load_topology(args.top)
    atom_indices = top.select(args.select)   # numpy int array
    n_atoms      = len(atom_indices)

    # Need n_frames without loading all coordinates — peek with iterload
    n_frames = sum(1 for _ in md.iterload(args.traj, top=args.top, chunk=1))

    # Initialize full output array
    full_data = np.zeros((n_frames, n_atoms, 3))

    # ── Parallel dispatch — pass everything explicitly, no closures ──────────
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(process_frame, i, args.traj, args.top, atom_indices, l): i
            for i in range(n_frames)
        }
        for fut in tqdm(as_completed(futures), total=n_frames, desc="Processing frames"):
            fidx, frame_data = fut.result()
            full_data[fidx] = frame_data

    # ── Dataframe conversion and output ─────────────────────────────────────────
    frame_labels = [f'frame-{i}' for i in range(n_frames)]
    atom_labels  = [f'atom-{j}'  for j in range(n_atoms)]
    count_labels = ['coordination', 'eclipsed', 'staggered']

    df = pd.DataFrame(
        full_data.reshape(-1, 3),
        index=pd.MultiIndex.from_product([frame_labels, atom_labels],
                                         names=['frame', 'atom']),
        columns=count_labels,
    )
    print("Output dataframe overview:")
    print(df)

    df.to_csv(f'{args.output}.csv')
    print(f"""--------------------
    Output(s) saved to {args.output}.csv
    Pandas example for reading the output:

    df_loaded = pd.read_csv('{args.output}.csv', index_col=['frame', 'atom'])
    # Metric for atom-i from frame-f:
    df_loaded.loc[('frame-f', 'atom-i'), 'coordination']  # or 'eclipsed'/'staggered'
    --------------------""")
### MAIN ###

if __name__ == '__main__':
    main()