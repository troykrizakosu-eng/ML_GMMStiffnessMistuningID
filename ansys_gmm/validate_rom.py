"""
Validates assemble_rom's cyclic reduction + blade CMS + real-Fourier-matrix
ROM assembly against a brute-force direct assembly of the same toy sector,
since a real ANSYS .full file isn't available to test against in
development.

Toy sector (local node ids used directly as "node numbers", 1 scalar DOF
each, dof_ref = [(node, 0), ...]):
  0: L    cyclic-low boundary node
  1: Id1  disk interior mass 1
  2: Id2  disk interior mass 2 (two interior masses, not one, so the
          disk's cyclic eigenvectors don't degenerate into a real vector
          times a single complex phase -- see the note in main() below)
  3: R    cyclic-high boundary node
  4: C    contact node (disk/blade interface)
  5: B    blade mass

A second, blade-suppressed "disk-only" copy of the same sector (nodes
0-4 only) is also built, matching the two-model workflow assemble_rom
expects when disk_only is provided.

Run this file directly (`python3 -m ansys_gmm.validate_rom` from the repo
root) to print both results. See the VALIDATION STATUS note in rom.py's
module docstring for what's confirmed vs. still open:
  - the mistuning delta matrices (KD_delta_full/KB_delta_full), built from
    the disk-only vs. combined model via subtraction, are exactly correct
    (checked against the known spring constants below).
  - the full disk+blade TUNED spectrum does not yet match brute force
    exactly -- this is the known open issue.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

from . import rom as rom_mod

N = 8
k1, k2, k3, k4, k5, k6 = 5.0e5, 4.3e5, 3.7e5, 3.0e5, 4.0e5, 2.0e5
m_L, m_1, m_2, m_R, m_C, m_B = 0.05, 0.8, 0.9, 0.05, 0.02, 0.3


def _add_spring(K, i, j, k):
    K[i, i] += k
    K[j, j] += k
    K[i, j] -= k
    K[j, i] -= k


def _build():
    K_local = np.zeros((6, 6))
    M_local = np.diag([m_L, m_1, m_2, m_R, m_C, m_B])
    _add_spring(K_local, 0, 1, k1)  # L-Id1
    _add_spring(K_local, 1, 2, k2)  # Id1-Id2
    _add_spring(K_local, 2, 3, k3)  # Id2-R
    _add_spring(K_local, 2, 4, k4)  # Id2-C
    _add_spring(K_local, 4, 5, k5)  # C-B
    K_local[5, 5] += k6             # B-ground
    return K_local, M_local


def brute_force_frequencies(K_local, M_local, n_sectors):
    """Assemble the full N-sector wheel directly (boundary nodes shared
    between adjacent sectors) and return its natural frequencies (Hz)."""
    n_boundary = n_sectors
    n_per = 4  # Id1, Id2, C, B per sector
    n_global = n_boundary + n_sectors * n_per

    def boundary(n):
        return n % n_sectors

    def Id1(n):
        return n_boundary + n_per * n

    def Id2(n):
        return n_boundary + n_per * n + 1

    def C(n):
        return n_boundary + n_per * n + 2

    def B(n):
        return n_boundary + n_per * n + 3

    K_glob = np.zeros((n_global, n_global))
    M_glob = np.zeros((n_global, n_global))
    for n in range(n_sectors):
        l2g = {0: boundary(n), 1: Id1(n), 2: Id2(n), 3: boundary(n + 1), 4: C(n), 5: B(n)}
        for a in range(6):
            M_glob[l2g[a], l2g[a]] += M_local[a, a]
            for b in range(6):
                K_glob[l2g[a], l2g[b]] += K_local[a, b]
    w2, _ = eigh(K_glob, M_glob)
    return np.sort(np.sqrt(np.clip(w2, 0, None))) / 2 / np.pi


def main():
    K_local, M_local = _build()
    K_sector, M_sector = sp.csc_matrix(K_local), sp.csc_matrix(M_local)
    dof_ref = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
    node_components = dict(cyclic_low=[0], cyclic_high=[3], disk=[1, 2],
                            contact=[4], blade=[5])

    # Genuinely separate disk-only matrix (no C-B spring at all), not a
    # slice of K_local -- slicing off the blade row still leaves the C-B
    # spring's contribution baked into contact node C's own diagonal entry
    # (K_local[4,4] = k4 + k5), exactly the K_cc^D vs K_cc^D+K_cc^B
    # distinction this disk_only argument exists to avoid.
    K_disk = np.zeros((5, 5))
    M_disk = np.diag([m_L, m_1, m_2, m_R, m_C])
    _add_spring(K_disk, 0, 1, k1)
    _add_spring(K_disk, 1, 2, k2)
    _add_spring(K_disk, 2, 3, k3)
    _add_spring(K_disk, 2, 4, k4)
    K_disk_sp, M_disk_sp = sp.csc_matrix(K_disk), sp.csc_matrix(M_disk)
    dof_ref_disk = dof_ref[:5]

    rom = rom_mod.assemble_rom(K_sector, M_sector, dof_ref, node_components,
                                n_sectors=N, n_disk_modes=1, n_blade_modes=1,
                                disk_only=(K_disk_sp, M_disk_sp, dof_ref_disk))
    w2, _ = eigh(rom["K_ROM"], rom["M_ROM"])
    f_rom = np.sort(np.sqrt(np.clip(w2, 0, None))) / 2 / np.pi
    f_bf = brute_force_frequencies(K_local, M_local, N)

    print("ROM natural frequencies (Hz):\n", f_rom)
    print("\nBrute-force full-wheel natural frequencies, lowest",
          len(f_rom), "(Hz):\n", f_bf[:len(f_rom)])
    print("\nmax abs diff (Hz):", np.max(np.abs(f_rom - f_bf[:len(f_rom)])))
    print("(NOT yet an exact match -- see VALIDATION STATUS in rom.py)")

    print("\n--- mistuning delta matrices (should be exact) ---")
    print("KD_delta_full disk-disk block (expect [[k1+k2, -k2],[-k2, k2+k3+k4]]):")
    print(rom["KD_delta_full"][:2, :2])
    print("KD_delta_full contact self-term (expect k4):", rom["KD_delta_full"][2, 2])
    print("KB_delta_full (expect Kcc^B=k5, Kcb=Kbc=-k5, Kbb=k5+k6):")
    print(rom["KB_delta_full"][2:, 2:])


if __name__ == "__main__":
    main()
