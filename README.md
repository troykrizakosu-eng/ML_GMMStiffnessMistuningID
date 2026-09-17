# ML Stiffness/Damping Mistuning Identification (GMM-based)

Single notebook (`gmm_mistuning_id.ipynb`) that:

1. Builds the GMM reduced order model using the same reduction technique as
   the original MATLAB code (`GMM_Tuned.m`, `GMM_BladeSmallMistuned.m`,
   `GMM_DiskSmallMistuned.m`): a real Fourier matrix combines block-diagonal
   blade Craig-Bampton modes and disk cyclic modes into the transformation
   matrix `T` at each sector, `M_ROM`/`K_ROM` are assembled as
   `sum(T' * M * T)` / `sum(T' * K * T)` over all sectors, and stiffness
   mistuning is added via a `Tdelta` matrix per sector weighted by that
   sector's mistuning value, matching Krizak & D'Souza, *"A generalized
   model of mistuning for bladed disks,"* J. Sound Vib. 606 (2025) 119003.

   Real data (`StiffMatrix`/`MassMatrix` for the combined sector,
   `StiffMatrix_D`/`MassMatrix_D` disk-only, blade CB modes, disk cyclic
   modes) normally comes from ANSYS exports, exactly like the MATLAB
   scripts load them. Since those files aren't available in this repo, the
   notebook builds small synthetic stand-ins with the same variable names
   so it runs end-to-end -- swap the "sector data" cell for your real
   loaded matrices and everything downstream (reduction, mistuning, ML) is
   unchanged. This demonstrates the pipeline runs; it doesn't independently
   re-verify the GMM method itself (that's your published, validated
   method) -- check GMM vs. full FE frequencies with your real data the
   same way your MATLAB scripts already do (`Load_Phi('.../FullStage_...')`
   + `Error_ROM`).

2. Adds structural damping mistuning the same way (Eq. 21-22): tuned
   baseline damping plus a per-sector delta using the same blade/disk
   stiffness-delta building blocks.

3. Generates training data by running many random mistuning patterns through
   the model and recording the forced blade-tip response (engine-order
   traveling-wave excitation, swept over frequency near the first mode
   family) as the ML input feature.

4. Trains a neural network (`MLPRegressor`) to map the response back to the
   mistuning values, for three separate cases:
   - stiffness mistuning only (damping tuned)
   - damping mistuning only (stiffness tuned)
   - both together

Run it with:

```
pip install -r requirements.txt jupyter
jupyter notebook gmm_mistuning_id.ipynb
```

(or run non-interactively with `jupyter nbconvert --to notebook --execute --inplace gmm_mistuning_id.ipynb`)

Outputs (also saved to disk when the notebook is run): `results_tuned_response.png`
(sanity check of the tuned forced response) and `results_<case>.png` (true
vs. predicted mistuning scatter plots with R2/RMSE) for each of the three
cases. The committed notebook already has these plots and R2/RMSE values
embedded in its cell outputs, so you can read the results without re-running
it.

## Loading real FE data from ANSYS MAPDL (`.full` files directly)

`ansys_gmm/` is a separate, earlier exploration of building the ROM
directly from raw ANSYS `.full` matrix files (rather than from
pre-exported CB/cyclic mode shapes as the notebook above does). **Its
tuned-ROM natural frequencies are not fully validated** -- see
`ansys_gmm/README.md`. The notebook above (matching your actual MATLAB
workflow) is the more faithful and currently better-supported approach;
treat `ansys_gmm/` as reference infrastructure for a `.full`-file-based
loader if you want one later.
