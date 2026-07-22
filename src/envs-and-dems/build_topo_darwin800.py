"""Build the 800 m full-orography topo for the darwin nest inside the sam domain.

Companion of build_topo_nest.py (imports its block-max binning): same GLO-90
projection pipeline, but on the 800 m little-brother grid of the ERA5-driven
nesting chain (sam 4 km -> darwin 800 m). Full orography: no IFS fill, no
northern cutoff, no edge taper (the lateral absorbers relax toward the
big-brother cube fields, which themselves contain terrain-forced flow).

Grid: nx=1020, ny=764, dx=800 m, centred on the cube_0 centre (the darwin
400 m domain centre). Output coordinates are SAM-FRAME km (origin = CORAL,
like the big-brother run), matching the cube's xcr/ycr, i.e.
x in [-521.6, 293.6] km, y in [-337.2, 273.2] km.

    /home/b/b309199/venvs/post-venv/bin/python build_topo_darwin800.py
"""

import numpy as np
import scipy.ndimage
import xarray as xr

from build_topo_nest import block_max_from_mosaic

CORAL_PROJ_XY_M = (214000.0, -315000.0)
CENTER_SAM_XY_M = (-114000.0, -32000.0)

CONFIG = {
    "mosaic": "/work/bd0620/b309199/mapy/data/dems/cop90m_ssa.tif",
    "proj4": "+proj=lcc +lat_0=-51 +lon_0=-71 +lat_1=-49 +lat_2=-53 +x_0=0 +y_0=0 "
             "+ellps=aust_SA +units=m +no_defs +type=crs",
    "center_xy_m": (
        CORAL_PROJ_XY_M[0] + CENTER_SAM_XY_M[0],
        CORAL_PROJ_XY_M[1] + CENTER_SAM_XY_M[1],
    ),
    "dx": 800.0,
    "nx": 1020,
    "ny": 764,
    "smooth_cutoff_dx": 2.0,
    "chunk_rows": 1000,
    "out": "/work/bd0620/b309199/mapy/data/pmap-topos/darwin_0800m_1020x764_fullorog.nc",
}


def main():
    cfg = CONFIG
    dx_km = cfg["dx"] / 1000.0
    z = block_max_from_mosaic(cfg)

    sigma = cfg["smooth_cutoff_dx"] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    z = scipy.ndimage.gaussian_filter(z, sigma)

    x_sam = CENTER_SAM_XY_M[0] / 1000.0 + (np.arange(cfg["nx"]) - (cfg["nx"] - 1) / 2.0) * dx_km
    y_sam = CENTER_SAM_XY_M[1] / 1000.0 + (np.arange(cfg["ny"]) - (cfg["ny"] - 1) / 2.0) * dx_km
    ds = xr.Dataset(
        {"orog": (("x", "y"), z.astype(np.float64))},
        coords={"x": x_sam, "y": y_sam},
        attrs={
            "proj4": cfg["proj4"],
            "center_x_m": cfg["center_xy_m"][0],
            "center_y_m": cfg["center_xy_m"][1],
            "coordinate_frame": "sam-local km (origin = CORAL proj (214000, -315000) m)",
            "dx_m": cfg["dx"],
            "source": cfg["mosaic"],
            "coarsen": "block-max",
            "taper_inset_km": "None (full orography for cube-driven nesting)",
            "smoothing": f"gaussian, cutoff {cfg['smooth_cutoff_dx']}*dx",
        },
    )
    ds.to_netcdf(cfg["out"])
    ix, iy = np.unravel_index(np.argmax(z), z.shape)
    print(
        f"[out ] {cfg['out']}  orog {z.shape}, max {z.max():.0f} m "
        f"at x={x_sam[ix]:.1f} km, y={y_sam[iy]:.1f} km"
    )


if __name__ == "__main__":
    main()
