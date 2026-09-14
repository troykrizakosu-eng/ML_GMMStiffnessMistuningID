"""
Standard (non-cyclic) Craig-Bampton reduction of the blade portion of a
single sector: fixed-interface normal modes (paper Eq. 7, Phi_b) and the
static constraint-mode influence matrix used to build Psi_b,n (Eq. 6).

Unlike the disk, the blade doesn't touch the cyclic boundary, so this is
the textbook Craig-Bampton step -- interior DOFs fixed at the contact
interface for the normal modes, then a static solve for how a unit contact
displacement pattern deforms the blade interior.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, splu


def blade_normal_modes(K_bb, M_bb, n_modes):
    """Fixed-interface normal modes Phi_b (contact DOFs held at zero).
    Same for every sector (the blade model doesn't depend on n).

    Returns omega (n_modes,), Phi_b (n_blade_dof, n_modes), mass-normalized.
    """
    n = K_bb.shape[0]
    if n < 200 or n_modes >= n - 2:
        from scipy.linalg import eigh
        K_dense = K_bb.toarray() if sp.issparse(K_bb) else np.asarray(K_bb)
        M_dense = M_bb.toarray() if sp.issparse(M_bb) else np.asarray(M_bb)
        w2, phi = eigh(K_dense, M_dense)
        w2, phi = w2[:n_modes], phi[:, :n_modes]
    else:
        w2, phi = eigsh(K_bb, k=n_modes, M=M_bb, sigma=0, which="LM")
        order = np.argsort(w2)
        w2, phi = w2[order], phi[:, order]
    return np.sqrt(np.clip(w2, 0, None)), phi


def constraint_influence(K_bb, K_bc):
    """Static constraint-mode influence matrix G = -K_bb^-1 K_bc, so that a
    contact-side displacement pattern phi_c produces blade-interior response
    Psi_b = G @ phi_c (paper Eq. 6).

    Returns G (n_blade_dof, n_contact_dof) as a dense array (it's used as a
    dense projection later, but the factorization/solve itself is sparse).
    """
    if sp.issparse(K_bb):
        lu = splu(K_bb.tocsc())
        K_bc = K_bc.toarray() if sp.issparse(K_bc) else np.asarray(K_bc)
        G = -lu.solve(K_bc)
    else:
        G = -np.linalg.solve(np.asarray(K_bb), np.asarray(K_bc))
    return G
