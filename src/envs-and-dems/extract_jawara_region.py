"""Extract a regional JAWARA cutout on a uniform altitude grid from the raw global files.

Script counterpart of data/jawara/jawara_processing.ipynb (cells 2-3): merges the monthly
global Z/T/U/V files (124 pressure levels, ~2.8 deg, hourly), subsets time + region, adds
th = t*(P0/p)^kappa, and interpolates every column from pressure levels onto a uniform
altitude grid using the geopotential height z. Output matches the jawara_patagonia_*.nc
layout (time, level, latitude, longitude; vars u, v, p, t, th).

Differences to the notebook (both intentional):
- p is interpolated in ln(p) (exponential profile; linear-in-z p was biased),
- z is converted geopotential -> geometric height before gridding (level = GEOMETRIC
  altitude), consistent with the ERA5 -ml-int convention used by build_background_3d.py.

    /home/b/b309199/venvs/post-venv/bin/python extract_jawara_region.py
"""
import numpy as np
import xarray as xr

CONFIG = {
    "raw_dir": "/work/bd0620/b309199/data/jawara/raw",
    "yymm": "1806",
    "time_frame": ("2018-06-16", "2018-06-18"),
    "lat_slice": (-30.0, -74.0),
    "lon_slice": (242.0, 342.0),
    "z_top_m": 140000.0,
    "dz_m": 400.0,
    "geometric_height": True,
    "out": "/work/bd0620/b309199/mapy/data/jawara/jawara_sam_180616.nc",
}

G, RD, CP, P0, RE = 9.80616, 287.05, 1004.0, 1.0e5, 6371229.0
KAPPA = RD / CP


def load_raw(cfg):
    parts = [xr.open_dataset(f"{cfg['raw_dir']}/{v}{cfg['yymm']}.nc") for v in "ZTUV"]
    ds = parts[0]
    for p in parts[1:]:
        ds = ds.merge(p)
    ds = ds.sel(
        time=slice(*cfg["time_frame"]),
        latitude=slice(*cfg["lat_slice"]),
        longitude=slice(*cfg["lon_slice"]),
    )
    p_pa = ds["level"].values * 100.0
    ds["th"] = ds["t"] * (P0 / p_pa[np.newaxis, :, np.newaxis, np.newaxis]) ** KAPPA
    return ds, p_pa


def main():
    cfg = CONFIG
    ds, p_pa = load_raw(cfg)
    nt, nlev = ds.sizes["time"], ds.sizes["level"]
    nlat, nlon = ds.sizes["latitude"], ds.sizes["longitude"]
    print(f"raw cutout: {nt} times, {nlat} lat x {nlon} lon, {nlev} p-levels")

    z_new = np.arange(0.0, cfg["z_top_m"], cfg["dz_m"])
    z_raw = ds["z"].values
    if cfg["geometric_height"]:
        z_raw = RE * z_raw / (RE - z_raw)

    col0 = z_raw[0, :, nlat // 2, 0]
    fin = np.isfinite(col0)
    ascending = bool(np.all(np.diff(col0[fin]) > 0))
    sl = slice(None) if ascending else slice(None, None, -1)
    lnp_full = np.log(p_pa)[sl]

    src = {v: ds[v].values for v in ("t", "u", "v", "th")}
    out = {v: np.empty((nt, len(z_new), nlat, nlon), dtype=np.float32) for v in ("t", "u", "v", "th", "p")}
    for it in range(nt):
        for j in range(nlat):
            for i in range(nlon):
                zp_full = z_raw[it, sl, j, i]
                valid = np.isfinite(zp_full)
                zp = zp_full[valid]
                lnp = lnp_full[valid]
                for v in ("t", "u", "v", "th"):
                    out[v][it, :, j, i] = np.interp(z_new, zp, src[v][it, sl, j, i][valid])
                slope_l = (lnp[1] - lnp[0]) / (zp[1] - zp[0])
                slope_h = (lnp[-1] - lnp[-2]) / (zp[-1] - zp[-2])
                lp = np.interp(z_new, zp, lnp)
                lp[z_new < zp[0]] = lnp[0] + slope_l * (z_new[z_new < zp[0]] - zp[0])
                lp[z_new > zp[-1]] = lnp[-1] + slope_h * (z_new[z_new > zp[-1]] - zp[-1])
                out["p"][it, :, j, i] = np.exp(lp)

    res = xr.Dataset(
        {v: (("time", "level", "latitude", "longitude"), out[v]) for v in out},
        coords={
            "time": ds["time"].values,
            "level": z_new,
            "latitude": ds["latitude"].values,
            "longitude": ds["longitude"].values,
        },
        attrs={
            "source": f"{cfg['raw_dir']}/[ZTUV]{cfg['yymm']}.nc",
            "level_type": "geometric altitude [m]" if cfg["geometric_height"] else "geopotential height [m]",
            "note": "p interpolated in ln(p); th = t*(1e5/p)^kappa on native levels",
        },
    )
    res.to_netcdf(cfg["out"])
    print(f"-> {cfg['out']}")
    print(res)


if __name__ == "__main__":
    main()
