"""
Assembles the full-wheel GMM reduced order model the same way the original
MATLAB code does (GMM_Tuned.m / GMM_BladeSmallMistuned.m /
GMM_DiskSmallMistuned.m, and gmm_mistuning_id.ipynb's Python port of it): a
real Fourier matrix (`real_fourier_matrix`) combines block-diagonal blade
Craig-Bampton modes and disk cyclic modes (cyclic.py) into the
transformation matrix T at each sector, and M_ROM/K_ROM are
sum(T' * M * T) / sum(T' * K * T) over all sectors (paper Eqs. 8-11).

VALIDATION STATUS: the mistuning delta matrices (KD_delta_full,
KB_delta_full) are verified exact against a synthetic toy sector (see
validate_rom.py) -- the disk-only vs. combined-model subtraction that
builds them is straightforward linear algebra. The tuned M_ROM/K_ROM
assembly matches the MATLAB code's structure but has NOT been
independently verified to reproduce a brute-force full-wheel reference
exactly on the synthetic toy sectors tried so far (see validate_rom.py for
the current numbers). Trust the algorithm -- it's your own published,
validated method -- but check GMM vs. full FE frequencies with your real
exported data the same way your MATLAB scripts already do
(`Load_Phi('.../FullStage_...')` + `Error_ROM`) before relying on this for
real mistuning studies.

Two ways to provide the disk's contact stiffness
--------------------------------------------------
The paper's disk cyclic modes (Eq. 4) need the disk's OWN contribution to
the disk/blade contact stiffness (K_cc^D), not the combined K_cc^D+K_cc^B
you get by slicing a single combined disk+blade sector model -- reusing
the combined model's contact block there silently folds the blade's
stiffness into the disk-alone cyclic eigenproblem, double-counting it once
the blade's own Craig-Bampton coupling is added on top. This matches how
the MATLAB code itself works: K_cc1 comes from a disk-only model
(StiffMatrix_D), K_cc2 from a blade-only model (StiffMatrix_B), summed as
K_cc1+K_cc2.

  - Pass `disk_only=(K_disk, M_disk, dof_ref_disk)` -- matrices from a
    second single-sector model with the blade suppressed/unmeshed, using
    the same node numbering as the combined model for the shared
    disk/contact/cyclic-boundary nodes -- and this module uses it for the
    cyclic eigenproblem, and gets K_cc^D directly from it (no subtraction
    needed for that piece). It also then gets the blade's own contribution
    to the contact stiffness, K_cc^B (plus K_cb, K_bc, K_bb -- everything
    Eq. 13's K_B^delta needs), by SUBTRACTING the disk-only model's
    (disk, contact) block from the combined model's -- this is what's left
    over once the disk's own contribution is removed (mathematically the
    same result as loading a separate blade-only model directly, as the
    MATLAB code does). That gives the full, not-simplified mistuning delta
    matrices.

  - If you omit `disk_only`, the combined model's own (disk, contact)
    block is used for both purposes instead, and blade stiffness/damping
    mistuning falls back to scaling just the blade-interior block K_bb
    (ignoring the disk/blade contact cross-terms) -- a cruder
    approximation. Always pass disk_only when you have it.
"""

import numpy as np

from . import cyclic, blade_cms
from .io import dof_index_for_nodes


def real_fourier_matrix(n_sectors):
    """N x N orthogonal real cyclic-symmetry Fourier matrix, same as the
    MATLAB `real_fourier_matrix.m` used in GMM_Tuned.m etc. Row = sector,
    column = harmonic (nodal diameter 0, then cos/sin pairs for
    1..N/2-1, then the Nyquist diameter if N is even)."""
    n = np.arange(n_sectors)
    cols = [np.ones(n_sectors) / np.sqrt(n_sectors)]
    d = 1
    while d < n_sectors / 2:
        theta = 2 * np.pi * d * n / n_sectors
        cols.append(np.sqrt(2 / n_sectors) * np.cos(theta))
        cols.append(np.sqrt(2 / n_sectors) * np.sin(theta))
        d += 1
    if n_sectors % 2 == 0:
        cols.append(((-1.0) ** n) / np.sqrt(n_sectors))
    return np.array(cols).T


def sector_physical_matrices(K, M, dof_ref, disk_nodes, contact_nodes, blade_nodes=None):
    """Slice the full sector K, M down to the physical (disk, contact[,
    blade]) DOF space GMM's reduced-order model is built from (paper Eqs.
    2-3), in that fixed order. Pass blade_nodes=None for a disk-only model
    (no blade slice).

    Returns K_phys, M_phys (sparse) and (disk_slice, contact_slice,
    blade_slice) index slices into that physical space (blade_slice is
    an empty slice if blade_nodes is None), plus the original
    (full-model-numbered) index arrays for disk/contact.
    """
    d_idx = dof_index_for_nodes(dof_ref, disk_nodes)
    c_idx = dof_index_for_nodes(dof_ref, contact_nodes)
    b_idx = dof_index_for_nodes(dof_ref, blade_nodes) if blade_nodes is not None else np.array([], dtype=int)
    phys_idx = np.concatenate([d_idx, c_idx, b_idx])

    K_phys = K.tocsr()[phys_idx, :].tocsc()[:, phys_idx]
    M_phys = M.tocsr()[phys_idx, :].tocsc()[:, phys_idx]

    n_d, n_c, n_b = len(d_idx), len(c_idx), len(b_idx)
    disk_slice = slice(0, n_d)
    contact_slice = slice(n_d, n_d + n_c)
    blade_slice = slice(n_d + n_c, n_d + n_c + n_b)
    return K_phys, M_phys, disk_slice, contact_slice, blade_slice, d_idx, c_idx


def _harmonic_shapes(disk_modes):
    """One real shape vector per column of real_fourier_matrix, in the
    same order: disk_modes' 'single' entries contribute one shape (the
    real part), 'pair' entries contribute two (cosine harmonic = real
    part, sine harmonic = -imaginary part) -- same convention as
    gmm_mistuning_id.ipynb's disk_cyclic_modes()."""
    shapes = []
    for mode in disk_modes:
        if mode["kind"] == "single":
            shapes.append(mode["shape"].real)
        else:
            shapes.append(mode["shape"].real)
            shapes.append(-mode["shape"].imag)
    return shapes


def assemble_rom(K, M, dof_ref, node_components, n_sectors,
                  n_disk_modes=1, n_blade_modes=4, disk_only=None):
    """Build the tuned GMM ROM from one sector's raw FE matrices, the same
    way GMM_Tuned.m does from pre-exported CB/cyclic modes.

    node_components : dict with node-number arrays for keys
        'disk', 'contact', 'blade', 'cyclic_low', 'cyclic_high'
    disk_only : optional (K_disk, M_disk, dof_ref_disk) from a second,
        blade-suppressed single-sector model -- see module docstring.
    n_disk_modes : modes retained per nodal diameter (matches Load_Phi's
        frequency-cutoff argument picking some number of mode families).

    Returns a dict with:
      M_ROM, K_ROM  -- tuned assembled matrices (paper Eqs. 9-11)
      T_list         -- per-sector transformation matrices T_n, one per
                        sector, physical-DOF x ROM-coordinate. ROM
                        coordinates are, in order: the disk's shared
                        cyclic/harmonic coordinates (one set shared by the
                        whole wheel), then N independent blocks of blade
                        fixed-interface-mode coordinates, one block per
                        physical sector (this un-transformed, per-sector
                        blade coordinate is what lets mistuning be applied
                        as a simple per-blade scalar, Eqs. 15-16).
      KD_delta_full, KB_delta_full -- disk-attributable / blade-attributable
                        stiffness (paper Eqs. 13, 17), expanded to
                        physical-DOF size and zero elsewhere, used to
                        build mistuning deltas (see mistuning_terms below)
    """
    K_phys, M_phys, d_sl, c_sl, b_sl, d_idx, c_idx = sector_physical_matrices(
        K, M, dof_ref, node_components["disk"], node_components["contact"],
        node_components["blade"])
    n_d = d_sl.stop

    if disk_only is not None:
        K_d, M_d, dof_ref_d = disk_only
        low_idx = dof_index_for_nodes(dof_ref_d, node_components["cyclic_low"])
        high_idx = dof_index_for_nodes(dof_ref_d, node_components["cyclic_high"])
        Kd_phys, Md_phys, dd_sl, dc_sl, _, dd_idx, dc_idx = sector_physical_matrices(
            K_d, M_d, dof_ref_d, node_components["disk"], node_components["contact"])
        interior_idx = np.concatenate([dd_idx, dc_idx])
        disk_modes = cyclic.cyclic_normal_modes(K_d, M_d, low_idx, high_idx,
                                                 interior_idx, n_sectors, n_disk_modes)
        # disk-only (disk,contact) block, padded to the combined model's
        # physical DOF ordering/size, zero at blade rows/cols
        Kd_only_full = np.zeros((K_phys.shape[0], K_phys.shape[0]))
        Kd_only_full[np.ix_(np.r_[d_sl, c_sl], np.r_[d_sl, c_sl])] = Kd_phys.toarray()
    else:
        low_idx = dof_index_for_nodes(dof_ref, node_components["cyclic_low"])
        high_idx = dof_index_for_nodes(dof_ref, node_components["cyclic_high"])
        interior_idx = np.concatenate([d_idx, c_idx])
        disk_modes = cyclic.cyclic_normal_modes(K, M, low_idx, high_idx,
                                                 interior_idx, n_sectors, n_disk_modes)
        Kd_only_full = None  # falls back to the combined model's own block below

    K_bb, M_bb = K_phys[b_sl, b_sl], M_phys[b_sl, b_sl]
    K_bc = K_phys[b_sl, c_sl]
    _, Phi_b = blade_cms.blade_normal_modes(K_bb, M_bb, n_blade_modes)
    G = blade_cms.constraint_influence(K_bb, K_bc)   # (n_blade, n_contact)

    K_phys_d, M_phys_d = K_phys.toarray(), M_phys.toarray()
    n_phys = K_phys_d.shape[0]
    n_blade_modes = Phi_b.shape[1]

    harmonic_shapes = _harmonic_shapes(disk_modes)
    n_harm = len(harmonic_shapes)
    RFM = real_fourier_matrix(n_sectors)
    n_rom = n_harm + n_sectors * n_blade_modes

    T_list = []
    for n in range(n_sectors):
        row = RFM[n, :]
        Phi_D = np.zeros((n_d + (c_sl.stop - c_sl.start), n_harm))
        for h in range(n_harm):
            Phi_D[:, h] = row[h] * harmonic_shapes[h]
        Phi_D1, Phi_D2 = Phi_D[:n_d, :], Phi_D[n_d:, :]
        Psi_B = -np.linalg.solve(K_bb.toarray() if hasattr(K_bb, "toarray") else K_bb,
                                  (K_bc.toarray() if hasattr(K_bc, "toarray") else K_bc) @ Phi_D2)

        T_n = np.zeros((n_phys, n_rom))
        T_n[d_sl, :n_harm] = Phi_D1
        T_n[c_sl, :n_harm] = Phi_D2
        T_n[b_sl, :n_harm] = Psi_B
        blade_start = n_harm + n * n_blade_modes
        T_n[b_sl, blade_start:blade_start + n_blade_modes] = Phi_b
        T_list.append(T_n)

    M_ROM = np.zeros((n_rom, n_rom))
    K_ROM = np.zeros((n_rom, n_rom))
    for T_n in T_list:
        M_ROM += T_n.T @ M_phys_d @ T_n
        K_ROM += T_n.T @ K_phys_d @ T_n

    if Kd_only_full is not None:
        KD_delta_full = Kd_only_full
        KB_delta_full = K_phys_d - Kd_only_full
        KB_delta_full[d_sl, :] = 0
        KB_delta_full[:, d_sl] = 0
    else:
        KD_delta_full = np.zeros_like(K_phys_d)
        KD_delta_full[d_sl, d_sl] = K_phys_d[d_sl, d_sl]
        KB_delta_full = np.zeros_like(K_phys_d)
        KB_delta_full[b_sl, b_sl] = K_phys_d[b_sl, b_sl]

    return dict(M_ROM=M_ROM, K_ROM=K_ROM, T_list=T_list,
                KD_delta_full=KD_delta_full, KB_delta_full=KB_delta_full,
                disk_modes=disk_modes)


def mistuning_terms(rom, n_sectors):
    """Precompute, for each sector n, T_n^T KD_delta_full T_n and
    T_n^T KB_delta_full T_n (the paper Eqs. 15-19's building blocks, same
    as the MATLAB Tdelta loop) so applying a mistuning pattern later is
    just a weighted sum -- no need to redo the FE-scale projection for
    every Monte Carlo sample.

    Returns disk_terms, blade_terms, each an (N, n_rom, n_rom) array so
    that, e.g., blade stiffness mistuning is
        Lambda_B = np.tensordot(mB, blade_terms, axes=1)
    (paper Eq. 15/16).
    """
    n_rom = rom["K_ROM"].shape[0]
    disk_terms = np.zeros((n_sectors, n_rom, n_rom))
    blade_terms = np.zeros((n_sectors, n_rom, n_rom))
    for n in range(n_sectors):
        T_n = rom["T_list"][n]
        disk_terms[n] = T_n.T @ rom["KD_delta_full"] @ T_n
        blade_terms[n] = T_n.T @ rom["KB_delta_full"] @ T_n
    return disk_terms, blade_terms
