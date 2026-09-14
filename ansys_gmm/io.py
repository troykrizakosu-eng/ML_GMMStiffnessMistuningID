"""
Loaders for FE data exported from ANSYS MAPDL.

Two files are needed per single-sector model:

1. A ``.full`` file with the sector's assembled mass and stiffness matrices.
   This is written automatically by MAPDL during a normal static or modal
   solve (``ANTYPE,MODAL`` / ``ANTYPE,STATIC`` + ``SOLVE``); just point this
   loader at the resulting ``.full`` file (rename/copy it if MAPDL deletes it
   after solving, e.g. by adding ``KEEP,YES`` before ``SOLVE``, or by not
   letting MAPDL clean up scratch files).

2. Plain-text node lists, one node number per line, for each of these named
   groups (APDL example below):
     - "disk"        interior disk nodes
     - "blade"       interior blade nodes
     - "contact"     the shared disk/blade interface nodes
     - "cyclic_low"  the sector's low-theta cyclic-symmetry face
     - "cyclic_high" the sector's high-theta cyclic-symmetry face

   ``cyclic_low`` and ``cyclic_high`` must list matched node pairs in the
   SAME order (node i in one file sits at the same (r, z) as node i in the
   other, just rotated by one sector angle) -- this is how ANSYS meshes
   cyclic-symmetric sectors by default (low/high edges built from the same
   node pattern). Example APDL to write one of these files for a component
   named BLADE already selected with CM,BLADE,NODE::

       CMSEL,S,BLADE
       *GET,ncnt,NODE,,COUNT
       *DIM,nlist,ARRAY,ncnt
       *VGET,nlist(1),NODE,,NLIST
       *CFOPEN,blade_nodes,txt
       *VWRITE,nlist(1)
       (F12.0)
       *CFCLOS
       ALLSEL
"""

import numpy as np
from ansys.mapdl.reader.full import FullFile


def load_full(path):
    """Read a MAPDL .full file.

    Returns
    -------
    dof_ref : (n, 2) int array -- (node number, dof direction 0=x,1=y,2=z)
              for every row/column of K and M.
    K, M    : (n, n) scipy sparse matrices
    """
    full = FullFile(path)
    dof_ref, K, M = full.load_km(as_sparse=True, sort=True)
    return dof_ref, K, M


def load_node_list(path):
    """Read a plain-text file of node numbers, one per line."""
    return np.loadtxt(path, dtype=np.int64).reshape(-1)


def dof_index_for_nodes(dof_ref, node_ids):
    """Row/column indices in K, M belonging to the given node numbers
    (all DOF directions of each node)."""
    node_ids = set(int(n) for n in np.atleast_1d(node_ids))
    return np.array([i for i, (node, _dof) in enumerate(dof_ref) if node in node_ids])
