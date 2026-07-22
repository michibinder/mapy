# mapy

Visualization and postprocessing software for NetCDF output of the numerical flow solvers
PMAP and EULAG and corresponding measurement data (lidar, AMTM, ...).

## Layout

- `src/` — analysis and plotting scripts (Python)
- `run/` — SLURM launcher scripts that submit the heavy `src/` jobs on HPC systems;
  see [run/README.md](run/README.md) for the launcher ↔ script mapping and the
  site-specific settings to adapt when porting to another machine
- `data/` — input data, figures and animation output (not tracked in git)
- `plot_logs/` — SLURM job logs of the launchers (not tracked in git)

## Main scripts (`src/`)

**PMAP animations** (frame-parallel renders, submitted via `run/`):
- `pmap_slc_lid.py` — slice + virtual-lidar animation from `slices_{x,y,z}.nc`
  (region/preset tables at the top)
- `cube_lid.py` — 3D λ₂-isosurface multiview + virtual-lidar T′ curtain from a
  `cube_<N>.nc` sub-volume (PyVista headless render)
- `cube_track.py` / `cube_track_profile.py` — vortex-feature tracking in horizontal cube
  slices (trackpy, Geach et al. 2020) and its vertical-profile variant
- `dyn_overview.py` — 9-panel dynamics overview for ERA5-nested runs

**One-shot diagnostics** (run directly):
- `spectral_zslice.py` — scale decomposition + PSD of a horizontal slice
- `lidar_obs_model_compare.py`, `lidar_spectra_stats.py`, `lidar_event_detect.py` —
  CORAL lidar observation vs. model comparisons and event statistics

**Environments & DEMs** (`src/envs-and-dems/`):
- `build_background_1d.py` / `build_background_3d.py` — 1D ambient column and
  transient 3D ambient/BC input files (ERA5 + JAWARA + CORAL) for PMAP runs
- `download_era5_ml.py` + `era5_processor.py` — ERA5 model-level download (CDS) and
  interpolation to geometric height
- `extract_jawara_region.py` — regional cutouts from the global JAWARA files
- `download_copdem.py`, `build_topo_nest.py`, `build_topo_darwin800.py`,
  `dem_patagonia.ipynb` — Copernicus GLO-90 DEM download and model topographies

All plots use `src/latex_default.mplstyle`; shared helpers live in `src/plt_helper.py`
and `src/cmaps.py`.
