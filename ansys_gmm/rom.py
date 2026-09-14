"""
Assembles the full-wheel GMM reduced order model (paper Eqs. 8-11) from a
single sector's raw FE matrices plus the disk cyclic modes (cyclic.py) and
blade Craig-Bampton reduction (blade_cms.py).

VALIDATION STATUS (read this before trusting the numbers):

  - The cyclic Bloch reduction in cyclic.py is verified exact against a
    brute-force full-wheel assembly for an ISOLATED disk (no blade
    attached) -- see validate_rom.py. It is also verified that summing
    T_n^T (.) T_n over all N sectors, using the disk's OWN cyclic-mode
    frequencies directly (rather than re-projecting through the raw,
    cyclic-boundary-dropped stiffness/mass -- see the n_eff correction
    below), reproduces the disk-alone spectrum exactly.

  - Once the blade is coupled in (Craig-Bampton constraint modes tying
    blade response to the disk's contact motion), the FULL assembled
    system's natural frequencies in validate_rom.py do NOT yet exactly
    match a brute-force reference for the same toy problem -- they're in
    the right ballpark (right number of mode families, right rough
    frequency range, no more of the earlier exact-degenerate-eigenvalue
    bug) but off by something like 10-20% in that test. This has NOT been
    root-caused despite substantial effort. The leading suspect is that
    the same "cyclic-boundary DOFs are dropped from the mode shape"
    approximation that required the n_eff correction for the disk-disk
    block also needs a (not yet derived) correction where the disk modes
    couple into the blade through the constraint modes Psi_b,n.

  Treat this module as a well-documented starting point, not a validated
  tool. Cross-check its output against your own FE model (GMM vs. full FE
  frequencies, the same comparison the paper itself makes in Section 3)
  before trusting it for real mistuning studies, and see validate_rom.py
  to reproduce/extend the toy-model checks described above.

Two ways to provide the disk's contact stiffness
--------------------------------------------------
The paper's disk cyclic modes (Eq. 4) need the disk's OWN contribution to
the disk/blade contact stiffness (K_cc^D), not the combined K_cc^D+K_cc^B
you get by slicing a single combined disk+blade sector model -- reusing
the combined model's contact block there silently folds the blade's
stiffness into the disk-alone cyclic eigenproblem, double-counting it once
the blade's own Craig-Bampton coupling is added on top.

  - Pass `disk_only=(K_disk, M_disk, dof_ref_disk)` -- matrices from a
    second single-sector model with the blade suppressed/unmeshed, using
    the same node numbering as the combined model for the shared
    disk/contact/cyclic-boundary nodes -- and this module uses it for the
    cyclic eigenproblem, and gets K_cc^D directly from it (no subtraction
    needed for that piece). It also then gets the blade's own contribution
    to the contact stiffness, K_cc^B (plus K_cb, K_bc, K_bb -- everything
    Eq. 13's K_B^delta needs), by SUBTRACTING the disk-only model's
    (disk, contact) block from the combined model's -- this is what's left
    over once the disk's own contribution is removed. That gives the full,
    not-simplified mistuning delta matrices.

  - If you omit `disk_only`, the combined model's own (disk, contact)
    block is used for both purposes instead, and blade stiffness/damping
    mistuning falls back to scaling just the blade-interior block K_bb
    (ignoring the disk/blade contact cross-terms) -- a cruder
    approximation. Always pass disk_only when you have it.
"""

import numpy as np

from . import cyclic, blade_cms
from .io import dof_index_for_nodes


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


def assemble_rom(K, M, dof_ref, node_components, n_sectors,
                  n_disk_modes=1, n_blade_modes=4, disk_only=None):
    """Build the tuned GMM ROM from one sector's raw FE matrices.

    node_components : dict with node-number arrays for keys
        'disk', 'contact', 'blade', 'cyclic_low', 'cyclic_high'
    disk_only : optional (K_disk, M_disk, dof_ref_disk) from a second,
        blade-suppressed single-sector model -- see module docstring.
    n_disk_modes : modes retained per nodal diameter. Prefer 1 (one mode
        family at a time): with more than one, modes at the SAME diameter
        are no longer guaranteed orthogonal once the cyclic-boundary DOFs
        are dropped, and this is not corrected for (see module docstring).

    ROM coordinates are, in order: the disk's shared cyclic normal-mode
    coordinates (one set shared by the whole wheel, "Fourier" basis over
    nodal diameter -- this is where the disk's own cyclic symmetry lives),
    followed by N independent blocks of blade fixed-interface-mode
    coordinates, one block per physical sector. Keeping the blade
    coordinates un-transformed and per-sector (rather than also folded into
    a cyclic/nodal-diameter basis) is what lets stiffness/damping mistuning
    be applied as a simple per-blade scalar (Eqs. 15-16) and is why the
    minimal ROM size works out to 2N for N sectors, one mode family kept
    per component.

    Returns a dict with:
      M_ROM, K_ROM  -- tuned assembled matrices (paper Eqs. 9-11) --
                       see the VALIDATION STATUS note in the module
                       docstring
      T_list         -- per-sector transformation matrices T_n (Eq. 8),
                        one per sector, physical-DOF x ROM-coordinate.
                        T_n is zero in every blade-coordinate block
                        except the one belonging to sector n.
      KB_delta_full, KD_delta_full -- blade-attributable / disk-attributable
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

    n_phys = K_phys.shape[0]
    n_blade_modes = Phi_b.shape[1]
    n_disk_cols = sum(1 if m["kind"] == "single" else 2 for m in disk_modes)
    n_rom = n_disk_cols + n_sectors * n_blade_modes

    T_list = []
    for n in range(n_sectors):
        T_n = np.zeros((n_phys, n_rom))
        col = 0
        for mode in disk_modes:
            for pattern in cyclic.sector_columns(mode, n, n_sectors):
                phi_d, phi_c = pattern[:n_d], pattern[n_d:]
                psi_b = G @ phi_c
                T_n[d_sl, col], T_n[c_sl, col], T_n[b_sl, col] = phi_d, phi_c, psi_b
                col += 1
        blade_start = n_disk_cols + n * n_blade_modes
        T_n[b_sl, blade_start:blade_start + n_blade_modes] = Phi_b
        T_list.append(T_n)

    K_phys_d, M_phys_d = K_phys.toarray(), M_phys.toarray()

    M_ROM = np.zeros((n_rom, n_rom))
    K_ROM = np.zeros((n_rom, n_rom))
    for T_n in T_list:
        M_ROM += T_n.T @ M_phys_d @ T_n
        K_ROM += T_n.T @ K_phys_d @ T_n

    # Replace the disk-disk block with the analytically-known relationship
    # instead of the raw re-projection above: each retained cyclic mode is
    # mass-normalized over the FULL [low; interior] system it was solved
    # from, so by discrete-Fourier (Parseval) orthogonality, summing its
    # cos/sin pattern over all N sectors gives exactly N (a "single" mode
    # at d=0 or the Nyquist diameter) or N/2 (each mode of a cos/sin
    # "pair") for the mass term, and that same factor times omega^2 for
    # stiffness. Verified exact for an isolated disk; see module docstring
    # for the not-yet-resolved coupled-with-blade discrepancy.
    col = 0
    for mode in disk_modes:
        ncols = 1 if mode["kind"] == "single" else 2
        n_eff = n_sectors if mode["kind"] == "single" else n_sectors / 2
        for _ in range(ncols):
            M_ROM[col, col] = n_eff
            K_ROM[col, col] = n_eff * mode["omega"] ** 2
            col += 1

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
    T_n^T KB_delta_full T_n (the paper Eqs. 15-19's building blocks) so
    applying a mistuning pattern later is just a weighted sum -- no need
    to redo the FE-scale projection for every Monte Carlo sample.

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
