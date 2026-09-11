# ML Stiffness/Damping Mistuning Identification (GMM-based)

Single script (`gmm_mistuning_id.py`) that:

1. Builds a reduced order bladed-disk model (2 DOF per sector: one disk DOF,
   one blade DOF, `N` sectors) with stiffness and damping mistuning added the
   same way GMM does in Krizak & D'Souza, *"A generalized model of mistuning
   for bladed disks,"* J. Sound Vib. 606 (2025) 119003 — a per-sector delta
   added to the blade's own stiffness (Eqs. 15-16), a per-sector delta added
   to the disk's own stiffness (Eqs. 17-19), and a structural damping delta
   matrix (Eq. 22), solved in the complex-stiffness form of Eq. 21.

   The real GMM ROM in the paper is built from finite element Craig-Bampton
   component matrices, which aren't available here, so this uses the
   classic 2-DOF-per-sector lumped model instead (which is also the minimal
   ROM size GMM itself reduces to, `2N`). Swap `build_K`/`build_C` for
   matrices exported from an actual GMM run and the rest of the script
   (dataset generation + ML) works unchanged.

2. Generates training data by running many random mistuning patterns through
   the model and recording the forced blade response (engine order 6
   traveling-wave excitation, swept over frequency) as the ML input feature.

3. Trains a neural network (`MLPRegressor`) to map the response back to the
   mistuning values, for three separate cases:
   - stiffness mistuning only (damping tuned)
   - damping mistuning only (stiffness tuned)
   - both together

Run it with:

```
pip install -r requirements.txt
python3 gmm_mistuning_id.py
```

Outputs: `results_tuned_response.png` (sanity check of the tuned forced
response) and `results_<case>.png` (true vs. predicted mistuning scatter
plots with R2/RMSE) for each of the three cases.
