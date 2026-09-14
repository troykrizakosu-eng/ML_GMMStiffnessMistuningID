# ansys_gmm: loading real FE data from ANSYS MAPDL

Builds a GMM-style reduced order model directly from raw ANSYS single-sector
FE matrices, instead of the hand-built lumped 2-DOF-per-sector model in
`gmm_mistuning_id.ipynb`. **Read the VALIDATION STATUS note at the top of
`rom.py` before trusting its output** -- the mistuning delta matrices are
verified exact, the tuned ROM's natural frequencies are not yet.

## What to export from MAPDL

Two single-sector models, sharing the same node numbering for their shared
disk/contact/cyclic-boundary nodes:

1. **Combined model** -- disk + blade + contact all meshed together, one
   sector, with cyclic symmetry (matched low/high theta boundary faces).
   Run a normal modal or static analysis (`ANTYPE,MODAL` or `,STATIC` +
   `SOLVE`) -- MAPDL writes a `.full` file with the assembled mass and
   stiffness matrices automatically.

2. **Disk-only model** -- the exact same disk mesh and cyclic boundary,
   with the blade suppressed or unmeshed (contact nodes present, blade
   absent). Same node numbers for every shared node. Also produces a
   `.full` file.

   This second model exists because the paper's disk cyclic modal analysis
   (Eq. 4) needs the disk's own contribution to the disk/blade contact
   stiffness (`K_cc^D`), not the combined model's `K_cc^D + K_cc^B` --
   reusing the combined model there double-counts the blade's stiffness.
   `rom.assemble_rom` also uses the difference between the two models
   (combined minus disk-only) to get the blade's exact contribution
   (`K_cc^B`, `K_cb`, `K_bc`, `K_bb`) for the mistuning delta matrices,
   instead of the cruder "just scale K_bb" approximation you'd need with
   only one model.

For both models, also export plain-text node lists (one node number per
line) for these named groups -- see `io.py`'s docstring for the APDL
snippet that writes one:

  - `disk`        interior disk nodes
  - `contact`     the shared disk/blade interface nodes
  - `blade`       interior blade nodes (combined model only)
  - `cyclic_low`  the sector's low-theta cyclic-symmetry face
  - `cyclic_high` the sector's high-theta face, listed in the SAME node
                   order as `cyclic_low` (matched pairs)

## Usage

```python
from ansys_gmm.io import load_full, load_node_list
from ansys_gmm.rom import assemble_rom, mistuning_terms

dof_ref, K, M = load_full("combined_sector.full")
dof_ref_disk, K_disk, M_disk = load_full("disk_only_sector.full")

node_components = {
    "disk": load_node_list("disk_nodes.txt"),
    "contact": load_node_list("contact_nodes.txt"),
    "blade": load_node_list("blade_nodes.txt"),
    "cyclic_low": load_node_list("cyclic_low_nodes.txt"),
    "cyclic_high": load_node_list("cyclic_high_nodes.txt"),
}

rom = assemble_rom(K, M, dof_ref, node_components, n_sectors=24,
                    n_disk_modes=1, n_blade_modes=4,
                    disk_only=(K_disk, M_disk, dof_ref_disk))

disk_terms, blade_terms = mistuning_terms(rom, n_sectors=24)
# Lambda_B = np.tensordot(mB, blade_terms, axes=1)   -- Eq. 15/16
# Lambda_D = np.tensordot(mD, disk_terms, axes=1)    -- Eq. 17-19
# From here, rom["M_ROM"], rom["K_ROM"] + Lambda_B/Lambda_D plug into the
# same complex-stiffness forced-response solve as gmm_mistuning_id.ipynb
# (Eq. 21), once the open validation issue below is resolved.
```

## Validation status (see `rom.py` for full detail)

Run `python3 -m ansys_gmm.validate_rom` from the repo root. Since no real
ANSYS `.full` file is available in this environment, it checks against a
hand-built tiny synthetic "sector" (a handful of masses and springs) and a
brute-force direct assembly of the full N-sector wheel:

- The disk's cyclic reduction (`cyclic.py`) is exact for an isolated disk
  (no blade).
- `KD_delta_full` / `KB_delta_full` (the mistuning building blocks) are
  exactly correct -- checked against the toy model's known spring
  constants.
- The full disk+blade **tuned** ROM frequencies do not yet exactly match
  brute force (right ballpark, not exact) -- an open issue, most likely in
  how the disk's cyclic modes couple into the blade through the
  constraint modes once the cyclic-boundary DOFs are dropped from the
  mode shape. Cross-check against your own real FE model (GMM vs. full FE
  frequencies -- the same comparison the paper makes in Section 3) before
  relying on this for real mistuning studies.
