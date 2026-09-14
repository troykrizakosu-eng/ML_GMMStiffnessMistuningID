"""
Validates assemble_rom's cyclic reduction + blade CMS + ROM assembly
against a brute-force direct assembly of the same toy sector, since a real
ANSYS .full file isn't available to test against in development.

Toy sector (local node ids used directly as "node numbers", 1 scalar DOF
each, dof_ref = [(node, 0), ...]):
  0: L   cyclic-low boundary node
  1: Id  disk interior mass
  2: R   cyclic-high boundary node
  3: C   contact node (disk/blade interface)
  4: B   blade mass

A second, blade-suppressed "disk-only" copy of the same sector (nodes
0-3 only) is also built, matching the two-model workflow assemble_rom
expects when disk_only is provided.

Run this file directly (`python3 -m ansys_gmm.validate_rom` from the repo
root) to print both results. See the VALIDATION STATUS note in rom.py's
module docstring for what's confirmed vs. still open:
  - the disk-alone case (no blade) matches brute force exactly.
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

N = 6
# L-Id and Id-R deliberately unequal -- equal values create a coincidental
# exact decoupling at the Nyquist nodal diameter for this tiny topology
# (an artifact of the toy model, not the reduction algorithm).
k1a, k1b, k2, k3, k4 = 5.0e5, 4.3e5, 3.0e5, 4.0e5, 2.0e5
m_L, m_Id, m_R, m_C, m_B = 0.05, 1.0, 0.05, 0.02, 0.3


def _add_spring(K, i, j, k):
    K[i, i] += k
    K[j, j] += k
    K[i, j] -= k
    K[j, i] -= k


def _build():
    K_local = np.zeros((5, 5))
    M_local = np.diag([m_L, m_Id, m_R, m_C, m_B])
    _add_spring(K_local, 0, 1, k1a)   # L-Id
    _add_spring(K_local, 1, 2, k1b)   # Id-R
    _add_spring(K_local, 1, 3, k2)    # Id-C
    _add_spring(K_local, 3, 4, k3)    # C-B
    K_local[4, 4] += k4               # B-ground
    return K_local, M_local


def brute_force_frequencies(K_local, M_local, n_sectors):
    """Assemble the full N-sector wheel directly (boundary nodes shared
    between adjacent sectors) and return its natural frequencies (Hz)."""
    n_boundary = n_sectors
    n_per = 3  # Id, C, B per sector
    n_global = n_boundary + n_sectors * n_per

    def boundary(n):
        return n % n_sectors

    def Id(n):
        return n_boundary + n_per * n

    def C(n):
        return n_boundary + n_per * n + 1

    def B(n):
        return n_boundary + n_per * n + 2

    K_glob = np.zeros((n_global, n_global))
    M_glob = np.zeros((n_global, n_global))
    for n in range(n_sectors):
        l2g = {0: boundary(n), 1: Id(n), 2: boundary(n + 1), 3: C(n), 4: B(n)}
        for a in range(5):
            M_glob[l2g[a], l2g[a]] += M_local[a, a]
            for b in range(5):
                K_glob[l2g[a], l2g[b]] += K_local[a, b]
    w2, _ = eigh(K_glob, M_glob)
    return np.sort(np.sqrt(np.clip(w2, 0, None))) / 2 / np.pi


def main():
    K_local, M_local = _build()
    K_sector, M_sector = sp.csc_matrix(K_local), sp.csc_matrix(M_local)
    dof_ref = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
    node_components = dict(cyclic_low=[0], cyclic_high=[2], disk=[1],
                            contact=[3], blade=[4])

    # NOTE: this must be a genuinely separate disk-only matrix (no C-B
    # spring at all), not a slice of K_local -- slicing off the blade row
    # still leaves the C-B spring's contribution baked into contact node
    # C's own diagonal entry (K_local[3,3] = k2 + k3), which is exactly
    # the K_cc^D vs K_cc^D+K_cc^B distinction this disk_only argument
    # exists to avoid.
    K_disk = np.zeros((4, 4))
    M_disk = np.diag([m_L, m_Id, m_R, m_C])
    _add_spring(K_disk, 0, 1, k1a)
    _add_spring(K_disk, 1, 2, k1b)
    _add_spring(K_disk, 1, 3, k2)
    K_disk_sp = sp.csc_matrix(K_disk)
    M_disk_sp = sp.csc_matrix(M_disk)
    dof_ref_disk = dof_ref[:4]

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

    print("\n--- mistuning delta matrices (should be exact) ---")
    print("KD_delta_full (expect Kdd=1.23e6, Kdc=Kcd=-3e5, Kcc^D=3e5):")
    print(rom["KD_delta_full"])
    print("KB_delta_full (expect Kcc^B=4e5, Kcb=Kbc=-4e5, Kbb=6e5):")
    print(rom["KB_delta_full"])


if __name__ == "__main__":
    main()
