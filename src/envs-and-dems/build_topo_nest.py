"""Build a PMAP orography file for the large nested (sam) domain.

Script counterpart of dem_patagonia.ipynb for the ERA5-driven nest: reads the GLO-90
mosaic (download_copdem.py), projects every DEM pixel into the model's Lambert grid
(same projection as all darwin/patagonia topos), coarsens by block maximum
(peak-preserving), tapers the terrain to zero inside the lateral absorber band
(5th-order polynomial, as in the notebook), and applies the 2*dx Gaussian
differentiability filter. Output: `orog` with dims (x, y) and PMAP-local km coords,
ready for pmap.geometry.orography.from_file.

    /home/b/b309199/venvs/post-venv/bin/python build_topo_nest.py
"""

import numpy as np
import pyproj
import rasterio
import scipy.ndimage
import xarray as xr

CONFIG = {
    "mosaic": "/work/bd0620/b309199/mapy/data/dems/cop90m_sam_33-71S_115-20W_9s.tif",
    "proj4": "+proj=lcc +lat_0=-51 +lon_0=-71 +lat_1=-49 +lat_2=-53 +x_0=0 +y_0=0 "
             "+ellps=aust_SA +units=m +no_defs +type=crs",
    "center_xy_m": (214000.0, -315000.0),
    "dx": 4000.0,
    "nx": 1100,
    "ny": 900,
    "taper_inset_km": None,
    "taper_width_km": 50.0,
    "smooth_cutoff_dx": 2.0,
    "chunk_rows": 1000,
    "out": "/work/bd0620/b309199/mapy/data/pmap-topos/sam_4000m_1100x900_fullorog.nc",
}


def tf(zeta):
    """5th-order polynomial transition (0 -> 1), as in dem_patagonia.ipynb."""
    return zeta * zeta * zeta * (10.0 - 15.0 * zeta + 6.0 * zeta * zeta)


def block_max_from_mosaic(cfg):
    """Project all DEM pixels to the model grid and take the per-cell maximum."""
    proj = pyproj.Proj(cfg["proj4"])
    cx, cy = cfg["center_xy_m"]
    dx, nx, ny = cfg["dx"], cfg["nx"], cfg["ny"]
    x0 = cx - (nx - 1) / 2.0 * dx - dx / 2.0
    y0 = cy - (ny - 1) / 2.0 * dx - dx / 2.0

    zmax = np.zeros((nx, ny), dtype=np.float32)
    with rasterio.open(cfg["mosaic"]) as src:
        t = src.transform
        lon = t.c + t.a * (np.arange(src.width) + 0.5)
        for r0 in range(0, src.height, cfg["chunk_rows"]):
            r1 = min(r0 + cfg["chunk_rows"], src.height)
            z = src.read(1, window=((r0, r1), (0, src.width)))
            lat = t.f + t.e * (np.arange(r0, r1) + 0.5)
            lon2, lat2 = np.meshgrid(lon, lat)
            px, py = proj(lon2, lat2)
            ix = np.floor((px - x0) / dx).astype(np.int64)
            iy = np.floor((py - y0) / dx).astype(np.int64)
            m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (z > 0)
            np.maximum.at(zmax, (ix[m], iy[m]), z[m])
            print(f"[bin ] rows {r0}:{r1} done", flush=True)
    return zmax


def taper_profile(n, dx_km, inset_km, width_km):
    """1 in the interior, 5th-order transition to 0 within `inset` of each edge."""
    coord = np.arange(n) * dx_km
    lo, hi = inset_km, coord[-1] - inset_km
    w = width_km
    prof = np.ones(n)
    prof[coord <= lo - w] = 0.0
    prof[coord >= hi + w] = 0.0
    rising = (coord > lo - w) & (coord < lo + w)
    falling = (coord > hi - w) & (coord < hi + w)
    prof[rising] = tf((coord[rising] - (lo - w)) / (2 * w))
    prof[falling] = 1.0 - tf((coord[falling] - (hi - w)) / (2 * w))
    return prof


def main():
    cfg = CONFIG
    dx_km = cfg["dx"] / 1000.0
    z = block_max_from_mosaic(cfg)

    if cfg["taper_inset_km"] is not None:
        mod_x = taper_profile(cfg["nx"], dx_km, cfg["taper_inset_km"], cfg["taper_width_km"])
        mod_y = taper_profile(cfg["ny"], dx_km, cfg["taper_inset_km"], cfg["taper_width_km"])
        z = z * mod_x[:, None] * mod_y[None, :]

    sigma = cfg["smooth_cutoff_dx"] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    z = scipy.ndimage.gaussian_filter(z, sigma)

    x_local = (np.arange(cfg["nx"]) - (cfg["nx"] - 1) / 2.0) * dx_km
    y_local = (np.arange(cfg["ny"]) - (cfg["ny"] - 1) / 2.0) * dx_km
    ds = xr.Dataset(
        {"orog": (("x", "y"), z.astype(np.float64))},
        coords={"x": x_local, "y": y_local},
        attrs={
            "proj4": cfg["proj4"],
            "center_x_m": cfg["center_xy_m"][0],
            "center_y_m": cfg["center_xy_m"][1],
            "dx_m": cfg["dx"],
            "source": cfg["mosaic"],
            "coarsen": "block-max",
            "taper_inset_km": str(cfg["taper_inset_km"]),
            "smoothing": f"gaussian, cutoff {cfg['smooth_cutoff_dx']}*dx",
        },
    )
    ds.to_netcdf(cfg["out"])
    print(f"[out ] {cfg['out']}  orog {z.shape}, max {z.max():.0f} m")


if __name__ == "__main__":
    main()
