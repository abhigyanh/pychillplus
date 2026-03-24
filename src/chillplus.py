# chillplus.py
### IMPORTS ###
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from numba import njit
import numpy as np
from scipy.special import sph_harm_y

import MDAnalysis as mda
from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch
from MDAnalysis.lib.mdamath import triclinic_vectors
### IMPORTS ###



### HELPER AND COMPUTATION FUNCTIONS ###
@njit(cache=True)
def minimum_image(vector: np.ndarray, H: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    """
    Minimum image convention for orthogonal or triclinic boxes.
    H     : (3,3) matrix of box vectors (columns = a, b, c)  [Å, consistent with MDA]
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
    Vectorised (theta, phi) for all i→j displacement vectors under MIC.
    positions_i : (N, 3)
    positions_j : (N, 3)  — already matched/expanded to pair with i
    Returns theta (N,), phi (N,)
    """
    disp = positions_j - positions_i                        # (N, 3)
    mic  = np.array([minimum_image(disp[k], H, H_inv) for k in range(len(disp))])
    norms = np.linalg.norm(mic, axis=1, keepdims=True)
    r_hat = mic / norms
    theta = np.arccos(np.clip(r_hat[:, 2], -1.0, 1.0))
    phi   = np.arctan2(r_hat[:, 1], r_hat[:, 0])
    return theta, phi

def calculate_qi_all(positions: np.ndarray,
                     neigh_dict: dict,
                     H: np.ndarray,
                     H_inv: np.ndarray,
                     l: int = 3) -> np.ndarray:
    """
    Compute per-atom q_lm vectors for all atoms.

    Parameters
    ----------
    positions : (n_atoms, 3) float64  — atom positions in Å
    neigh_dict: {local_idx: [local_neighbour_idxs]}
    H         : (3,3) box matrix (columns = box vectors) in Å
    H_inv     : np.linalg.inv(H)
    l         : bond-order parameter degree

    Returns
    -------
    qilm_values : (n_atoms, 2l+1) complex128
    """
    n_atoms     = len(positions)
    n_m         = 2 * l + 1
    qilm_values = np.zeros((n_atoms, n_m), dtype=np.complex128)

    for i_atom in range(n_atoms):
        neighs = neigh_dict.get(i_atom, [])
        if len(neighs) == 0:
            continue
        pos_i   = positions[i_atom]           # (3,)
        pos_j   = positions[neighs]            # (K, 3)
        n_neigh = len(neighs)

        theta, phi = _compute_angles(
            np.tile(pos_i, (n_neigh, 1)),
            pos_j, H, H_inv
        )

        for mi, m in enumerate(range(-l, l + 1)):
            ylm = sph_harm_y(l, m, theta, phi)
            qilm_values[i_atom, mi] = np.sum(ylm) / 4.0

    return qilm_values
### HELPER AND COMPUTATION FUNCTIONS ###



# ── Module-level worker — must be defined at module scope for pickling ───────
def process_frame(
    frameidx:   int,
    top:        str,
    traj:       str,
    select_str: str,   # MDAnalysis selection string — dynamic, evaluated per worker
    l:          int,
) -> tuple[int, float, np.ndarray]:
    """
    Load a single frame via MDAnalysis, compute q_lm and C_ij, and return
    (frameidx, time_ps, data) where data is (n_sel, 4):
        col 0 — global atom number (MDA atom.index)
        col 1 — coordination number
        col 2 — eclipsed neighbour count   (-0.35 ≤ C_ij ≤ 0.25)
        col 3 — staggered neighbour count  (C_ij ≤ -0.80)
    n_sel may vary between frames for dynamic selections.
    """
    # Fresh Universe per worker avoids pickling / thread-safety issues
    u_local  = mda.Universe(top, traj)
    # Dynamic selection evaluated fresh at the target frame
    u_local.trajectory[frameidx]
    ag       = u_local.select_atoms(select_str)
    n_sel    = len(ag)
    time_ps  = u_local.trajectory.time          # ps, from the frame timestamp

    # ── Box matrix (Å) ──────────────────────────────────────────────────────
    # triclinic_vectors returns the (3,3) upper-triangular H matrix
    # whose *columns* are the box vectors (MDA convention).
    H     = triclinic_vectors(u_local.dimensions).astype(np.float64)
    H_inv = np.linalg.inv(H)

    # ── Neighbour search (3.5 Å cutoff) ─────────────────────────────────────
    ns         = AtomNeighborSearch(ag, box=u_local.dimensions)
    positions  = ag.positions.astype(np.float64)             # (n_sel, 3) in Å

    # Build {local_idx: [local_neighbour_idxs]} using the AtomGroup's own index map
    global_to_local = {atom.index: i for i, atom in enumerate(ag)}
    neigh_dict: dict[int, list[int]] = {}
    for i_local, atom_i in enumerate(ag):
        raw_neighbours = ns.search(atom_i, 3.5, level='A')
        neigh_dict[i_local] = [
            global_to_local[a.index]
            for a in raw_neighbours
            if a.index != atom_i.index and a.index in global_to_local
        ]

    # ── Compute q_lm for all atoms ───────────────────────────────────────────
    qilm_local = calculate_qi_all(positions, neigh_dict, H, H_inv, l=l)

    # ── Classify neighbour pairs ─────────────────────────────────────────────
    # 4 columns: global atom index | coordination | eclipsed | staggered
    data = np.zeros((n_sel, 4))
    for i_atom, atom_i in enumerate(ag):
        neighs   = neigh_dict[i_atom]
        coord    = len(neighs)
        eclipsed = staggered = 0
        for j_atom in neighs:
            c_val     = C_ij(qilm_local[i_atom], qilm_local[j_atom])
            real_part = float(np.real(c_val))
            imag_part = float(np.imag(c_val))
            if abs(imag_part) > 1e-3:
                raise ValueError(f"C_ij has significant imaginary part: {imag_part:.6f}")
            if   -0.35 <= real_part <= 0.25:  eclipsed  += 1
            elif  real_part <= -0.8:           staggered += 1
        data[i_atom] = [atom_i.index, coord, eclipsed, staggered]

    return frameidx, time_ps, data
# ── Module-level worker ──────────────────────────────────────────────────────



### MAIN ###
def main():
    parser = argparse.ArgumentParser(
        description='CHILL+ bond-order analysis using MDAnalysis and numba-accelerated math.'
    )
    parser.add_argument('-j', '--jobs',   type=int,
                        help='Number of parallel worker processes (default: all CPUs)')
    parser.add_argument('-s', '--top',    type=str, required=True,
                        help='MDAnalysis-readable topology file')
    parser.add_argument('-f', '--traj',   type=str, required=True,
                        help='MDAnalysis-readable trajectory file')
    parser.add_argument('-o', '--output', type=str, default='chill-plus',
                        help='Output file basename; one CSV per frame: <basename>.frame-NNNN.csv')
    parser.add_argument('--select',       type=str, default='name OW',
                        help='MDAnalysis atom selection string (default: "name OW")')
    parser.add_argument('-l',             type=int, default=3,
                        help='Spherical harmonic l-value (default: 3)')
    args = parser.parse_args()

    # ── Sanity checks ────────────────────────────────────────────────────────
    if not os.path.exists(args.top):
        raise FileNotFoundError(f"Topology not found: {args.top}")
    if not os.path.exists(args.traj):
        raise FileNotFoundError(f"Trajectory not found: {args.traj}")

    l = args.l

    # ── Probe trajectory for frame count only ───────────────────────────────
    # Atom count is intentionally NOT fixed here: dynamic selections can yield
    # a different n_sel per frame; each worker reports its own.
    u        = mda.Universe(args.top, args.traj)
    n_frames = u.trajectory.n_frames
    del u

    print(f"Working dir : {os.getcwd()}")
    print(f"Topology    : {args.top}")
    print(f"Trajectory  : {args.traj}")
    print(f"Selection   : '{args.select}'")
    print(f"Frames      : {n_frames}")
    print(f"l           : {l}")
    print(f"Output base : {args.output}.frame-NNNN.csv")
    print()

    # ── Parallel dispatch — write each frame to disk as soon as it arrives ───
    saved_files = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(process_frame, i, args.top, args.traj, args.select, l): i
            for i in range(n_frames)
        }
        for fut in tqdm(as_completed(futures), total=n_frames, desc='Processing frames'):
            fidx, time_ps, frame_data = fut.result()
            fname  = f'{args.output}.frame-{fidx:05d}.csv'
            header = f'frame={fidx} time={time_ps:.3f}ps \nAt.Num,Coordination,Eclipsed,Staggered'
            np.savetxt(fname, frame_data, delimiter=',', header=header,
                       comments='# ', fmt=['%d', '%d', '%d', '%d'])
            saved_files.append(fname)

    # ── Summary ───────────────────────────────────────────────────────────────
    saved_files.sort()
    print(f'\n{n_frames} CSV file(s) written:')
    print(f"""
    --------------------
    NumPy example for reading a single frame's output:

    data = np.loadtxt('{args.output}.frame-00000.csv',
                    delimiter=',', skiprows=2, dtype=int)
    # columns: At.Num | Coordination | Eclipsed | Staggered
    --------------------
    """)
### MAIN ###

if __name__ == '__main__':
    main()