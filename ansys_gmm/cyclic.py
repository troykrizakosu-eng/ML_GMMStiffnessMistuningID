"""
Cyclic-symmetric modal reduction of the disk (+contact) portion of a single
sector, following the classic "real cyclic symmetry" method (e.g. Thomas
1979) that GMM (paper Eq. 4) builds on.

The disk sector's DOFs are split into three groups:
  L (cyclic_low)   -- the sector's low-theta boundary face
  R (cyclic_high)  -- the sector's high-theta boundary face, node-for-node
                       matched to L (same order)
  I (interior)     -- everything else being reduced here: disk-interior
                       DOFs and the contact DOFs

For a traveling-wave solution with nodal diameter d (0 <= d <= N/2), the
Bloch condition u_R = u_L * exp(i*2*pi*d/N) eliminates R, leaving a small
complex Hermitian generalized eigenvalue problem in [u_L; u_I]. Solving it
per d gives the disk's cyclic normal modes and their frequencies (Fig. 2 of
the paper) -- the same modes GMM_Tuned.m loads directly from an ANSYS
CYCLIC modal solve (Load_Phi) rather than computing here, since raw FE
matrices are what this package works from instead. rom.py combines a
mode's real and imaginary parts (real part = cosine harmonic, -imaginary
part = sine harmonic) via real_fourier_matrix the same way GMM_Tuned.m's
`kron(RFM(i,:), I) * DPhiBLKD` does.

Everything here works on sparse matrices and only ever forms a dense array
for the (small) reduced [L;I] problem at a single nodal diameter -- the
full-size sector matrices (which can be huge for a real FE sector) are only
ever sliced, never densified.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def _slice(A, rows, cols):
    """Sparse row/col slice that returns a sparse csr matrix."""
    return A.tocsr()[rows, :].tocsc()[:, cols].tocsr()


def _reduced_KM(K, M, L, I, R, lam):
    """Complex-Hermitian sparse reduced K(d), M(d) in [u_L; u_I] coordinates
    for Bloch phase factor lam = exp(i*2*pi*d/N). See module docstring."""
    def reduce_one(A):
        A_LL, A_LI, A_LR = _slice(A, L, L), _slice(A, L, I), _slice(A, L, R)
        A_IL, A_II, A_IR = _slice(A, I, L), _slice(A, I, I), _slice(A, I, R)
        A_RL, A_RI, A_RR = _slice(A, R, L), _slice(A, R, I), _slice(A, R, R)

        top_left = A_LL + lam * A_LR + np.conj(lam) * A_RL + A_RR
        top_right = A_LI + np.conj(lam) * A_RI
        bot_left = A_IL + lam * A_IR
        bot_right = A_II
        return sp.bmat([[top_left, top_right], [bot_left, bot_right]], format="csc")

    return reduce_one(K), reduce_one(M)


def cyclic_normal_modes(K, M, low_idx, high_idx, interior_idx, n_sectors, n_modes):
    """Solve the disk's cyclic normal modes for every nodal diameter.

    K, M          : full sector matrices (scipy sparse)
    low_idx, high_idx, interior_idx : row/column indices (see module docstring)
    n_sectors     : N, number of sectors in the full wheel
    n_modes       : number of modes to keep per nodal diameter

    Returns a list of "modes", each a dict with keys:
      d        -- nodal diameter
      kind     -- 'single' (d = 0 or Nyquist, one real coordinate) or
                  'pair'   (0 < d < N/2, two real coordinates: cos, sin)
      omega    -- natural frequency (rad/s)
      shape    -- complex eigenvector over `interior_idx`, mass-normalized
                  (this is [Phi_d,n ; Phi_c,n] evaluated at sector n=0)
    """
    L, R, I = np.asarray(low_idx), np.asarray(high_idx), np.asarray(interior_idx)
    if len(L) != len(R):
        raise ValueError("cyclic_low and cyclic_high must have the same length "
                          "(matched node pairs, same order)")
    n_LI = len(L) + len(I)

    modes = []
    n_diam = n_sectors // 2
    for d in range(0, n_diam + 1):
        is_nyquist = (n_sectors % 2 == 0) and (d == n_diam)
        kind = "single" if (d == 0 or is_nyquist) else "pair"
        lam = np.exp(1j * 2 * np.pi * d / n_sectors)

        Kd, Md = _reduced_KM(K, M, L, I, R, lam)

        k = min(n_modes, n_LI - 2)
        # small reduced systems: eigsh needs k < n; fall back to dense eigh
        if k < n_modes or n_LI < 200:
            from scipy.linalg import eigh
            w2, phi = eigh(Kd.toarray(), Md.toarray())
            w2, phi = w2[:n_modes], phi[:, :n_modes]
        else:
            w2, phi = eigsh(Kd, k=n_modes, M=Md, sigma=0, which="LM")
            order = np.argsort(w2)
            w2, phi = w2[order], phi[:, order]

        w2 = np.clip(np.real(w2), 0, None)
        for r in range(phi.shape[1]):
            shape_LI = phi[:, r]
            shape_I = shape_LI[len(L):]  # drop the L part, keep interior only
            # fix arbitrary complex phase so the pattern is well-defined
            k_max = np.argmax(np.abs(shape_I))
            shape_I = shape_I * np.exp(-1j * np.angle(shape_I[k_max]))
            modes.append(dict(d=d, kind=kind, omega=np.sqrt(w2[r]), shape=shape_I))
    return modes
