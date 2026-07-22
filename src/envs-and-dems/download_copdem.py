"""Download and mosaic Copernicus GLO-90 DEM tiles for a lat/lon box.

Fetches the 1x1 degree tiles from the AWS open-data bucket (no credentials needed),
caches them in CONFIG['tile_dir'], and merges everything inside CONFIG bounds onto a
uniform grid of CONFIG['out_res_arcsec'] (3 = native; coarser keeps the merge cheap for
very large boxes — at a 4-5 km model grid, 9 arcsec ~ 280 m is still >10x oversampled).
The bucket's tileList.txt (cached once) says which tiles exist; everything else is
ocean and comes out as 0 m. Tiles poleward of 50 deg have coarser native longitude
spacing and are resampled onto the common grid by the merge (bilinear).

Rerunning with a different box reuses all cached tiles and only fetches new ones,
so subsetting/enlarging later is cheap.

    /home/b/b309199/venvs/post-venv/bin/python download_copdem.py
"""

import os
import urllib.error
import urllib.request

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge

CONFIG = {
    "north": -33.0,
    "south": -71.0,
    "west": -115.0,
    "east": -20.5,
    "out_res_arcsec": 9.0,
    "tile_dir": "/work/bd0620/b309199/mapy/data/dems/cop90m_tiles",
    "out_tif": "/work/bd0620/b309199/mapy/data/dems/cop90m_sam_33-71S_115-20W_9s.tif",
}

BASE_URL = "https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com"


def tile_name(lat_sw, lon_sw):
    """Bucket tile name from the SW-corner integer lat/lon of a 1x1 degree tile."""
    ns = f"N{lat_sw:02d}" if lat_sw >= 0 else f"S{-lat_sw:02d}"
    ew = f"E{lon_sw:03d}" if lon_sw >= 0 else f"W{-lon_sw:03d}"
    return f"Copernicus_DSM_COG_30_{ns}_00_{ew}_00_DEM"


def load_tile_list(cfg):
    """Return the set of existing tile names (cached download of tileList.txt)."""
    cache = os.path.join(cfg["tile_dir"], "tileList.txt")
    if not os.path.exists(cache):
        urllib.request.urlretrieve(f"{BASE_URL}/tileList.txt", cache)
    with open(cache) as f:
        return {line.strip() for line in f if line.strip()}


def fetch_tiles(cfg):
    """Download all existing tiles intersecting the CONFIG box; return local tif paths."""
    os.makedirs(cfg["tile_dir"], exist_ok=True)
    existing = load_tile_list(cfg)
    tifs = []
    n_ocean = n_cached = n_new = 0
    for lat in range(int(np.floor(cfg["south"])), int(np.ceil(cfg["north"]))):
        for lon in range(int(np.floor(cfg["west"])), int(np.ceil(cfg["east"]))):
            name = tile_name(lat, lon)
            if name not in existing:
                n_ocean += 1
                continue
            local = os.path.join(cfg["tile_dir"], f"{name}.tif")
            if os.path.exists(local):
                tifs.append(local)
                n_cached += 1
                continue
            url = f"{BASE_URL}/{name}/{name}.tif"
            try:
                urllib.request.urlretrieve(url, local + ".part")
                os.replace(local + ".part", local)
                tifs.append(local)
                n_new += 1
                print(f"[get ] {name}", flush=True)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    n_ocean += 1
                else:
                    raise
    print(f"[info] tiles: {n_new} downloaded, {n_cached} cached, {n_ocean} ocean/absent")
    return tifs


def build_mosaic(tifs, cfg):
    """Merge the tiles onto a uniform out_res_arcsec grid over the CONFIG bounds."""
    res = cfg["out_res_arcsec"] / 3600.0
    srcs = [rasterio.open(t) for t in tifs]
    bounds = (cfg["west"], cfg["south"], cfg["east"], cfg["north"])
    arr, transform = rio_merge(
        srcs, bounds=bounds, res=(res, res), resampling=Resampling.bilinear
    )
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": arr.shape[1],
        "width": arr.shape[2],
        "crs": srcs[0].crs,
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
        "bigtiff": "if_safer",
    }
    for s in srcs:
        s.close()
    with rasterio.open(cfg["out_tif"], "w", **profile) as dst:
        dst.write(arr[0].astype("float32"), 1)
    print(f"[out ] {cfg['out_tif']}  shape {arr.shape[1:]},  bounds {bounds},  res {cfg['out_res_arcsec']}\"")


def main():
    tifs = fetch_tiles(CONFIG)
    if not tifs:
        raise SystemExit("no land tiles in the requested box")
    build_mosaic(tifs, CONFIG)


if __name__ == "__main__":
    main()
