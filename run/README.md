# run/ — SLURM launcher scripts for the mapy postprocessing chain

Thin `sbatch` wrappers that submit the heavy `src/` scripts as batch jobs. Each script
documents its own usage in its header. Logs go to `mapy/plot_logs/`.

| Launcher | Drives | Purpose |
|---|---|---|
| `rplot_pmap.sh` | `src/pmap_slc_lid.py` | slice + virtual-lidar animation |
| `rcube_pmap.sh` | `src/cube_lid.py` | 3D λ₂ cube multiview animation |
| `rtrack_pmap.sh` | `src/cube_track.py` | vortex-feature tracking animation |
| `rprof_pmap.sh` | `src/cube_track_profile.py` | vertical-profile vortex tracking |
| `rdyn_pmap.sh` | `src/dyn_overview.py` | 9-panel dynamics-overview animation |
| `rbuild_background3d.sh` | `src/envs-and-dems/build_background_3d.py` | ERA5+JAWARA → PMAP nest input files |
| `rera5_process.sh` | `src/envs-and-dems/download_era5_ml.py process` | ERA5 model-level → height interpolation |
| `rnest_pmap.sh` | `pmaputils.preprocessing.nesting` (external pmap-utils package) | big-brother cube → little-brother BC input files |

## Porting to another HPC system (e.g. JUPITER)

The scripts were written for Levante (DKRZ). Site-specific values are concentrated at the
top of each script and in the `#SBATCH` block:

- `USER` and the hardcoded path roots (`/work/bd0620/$USER/...`, `/home/b/$USER/...`,
  `/scratch/b/$USER/...` in `rnest_pmap.sh`)
- `PLOT_VENV` / `VENV` / `NEST_VENV` — the postprocessing python venv
  (xarray, matplotlib, cmcrameri, pyvista, trackpy, ...)
- `#SBATCH --partition=compute`, `--account=bd0620`, and the memory/CPU/time requests

All internal paths are absolute, so the launchers can be invoked from any directory, e.g.
`bash mapy/run/rplot_pmap.sh over <sim> notest` from the work-dir root.
