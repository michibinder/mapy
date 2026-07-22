"""ERA5 model-level processing for PMAP input preparation.

Adapted copy of ma-lidar-visualizations/src/era5_processor.py (DLR, LGPL-3+).
Reconstructs pressure and geopotential on the 137 ERA5 model levels following the
ECMWF recipe (compute_z_level), converts to geometric height, and interpolates the
fields to a uniform geometric-height grid (prepare_ml_int).

Changes vs the original:
- no T21 / tprime branch (the 3D ambient fields keep the resolved waves)
- q (specific humidity) kept in the interpolated output
- vertical interpolation defaults to geometric height (original: geopotential height)
- target grid configurable via z_top/dz arguments
- old and new CDS dim/coord names handled ('level'/'model_level', 'time'/'valid_time')
- era5-ml-coeff.csv resolved relative to this file
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

g = 9.80665
Rd = 287.06
Re = 6371229
FILE_ML_COEFF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "era5-ml-coeff.csv")


def _normalize(ds):
    """Rename new-style CDS dims/coords to 'time'/'level' and order the dims."""
    ren = {}
    if "model_level" in ds.dims:
        ren["model_level"] = "level"
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ren["valid_time"] = "time"
    ds = ds.rename(ren)
    ds = ds.drop_vars([v for v in ("expver", "number") if v in ds.variables])
    return ds.transpose("time", "level", "latitude", "longitude")


def compute_z_level(ds, p_half):
    """Compute geopotential at all full levels from t/q/lnsp (ECMWF recipe).

    https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height
    """
    z_h = ds["z"][:, 0, :, :].copy()
    ds["t_moist"] = ds["t"] * (1.0 + 0.609133 * ds["q"])

    for i in range(136, 0, -1):
        ph_levplusone = p_half[:, i + 1, :, :].values
        ph_lev = p_half[:, i, :, :].values

        dlog_p = np.log(ph_levplusone / ph_lev)
        alpha = 1.0 - ((ph_lev / (ph_levplusone - ph_lev)) * dlog_p)

        t_level = ds["t_moist"][:, i, :, :] * Rd
        ds["z"][:, i, :, :] = z_h + (t_level * alpha)
        z_h = z_h + (t_level * dlog_p)

    i = 0
    alpha = np.log(2)
    t_level = ds["t_moist"][:, i, :, :] * Rd
    ds["z"][:, i, :, :] = z_h + (t_level * alpha)

    ds["geop_height"] = ds["z"] / g
    return ds


def interp_ds_vertically(ds, z_new, alt_var, vars):
    """Interpolate model-level fields column-wise to the uniform height grid z_new."""
    for var in vars:
        shape = np.shape(ds[vars[0]].values)
        data = np.zeros((shape[0], len(z_new), shape[2], shape[3]))
        for t in range(0, shape[0]):
            for lat in range(0, shape[2]):
                for lon in range(0, shape[3]):
                    data[t, :, lat, lon] = np.interp(
                        z_new,
                        ds[alt_var].values[t, ::-1, lat, lon],
                        ds[var].values[t, ::-1, lat, lon],
                    )
        if var == vars[0]:
            ds_new = xr.Dataset(
                {var: (["time", "level", "latitude", "longitude"], data, ds[var].attrs)},
                coords={
                    "time": ds["time"],
                    "level": z_new,
                    "latitude": ds["latitude"],
                    "longitude": ds["longitude"],
                },
                attrs=ds.attrs,
            )
        else:
            ds_new[var] = (["time", "level", "latitude", "longitude"], data, ds[var].attrs)
    return ds_new


def _prepare_one(ds, ml_coeff, z_new, vars, alt_var):
    """Reconstruct p/geometric height and interpolate one in-memory time block."""
    lnsp = ds["lnsp"][:, 0, :, :].drop_vars("level").expand_dims(dim={"level": 138}, axis=1)
    dims = np.shape(lnsp)

    a = xr.DataArray(ml_coeff["a [Pa]"]).rename({"dim_0": "level"})
    a = a.expand_dims(dim={"time": dims[0], "latitude": dims[2], "longitude": dims[3]}, axis=[0, 2, 3])
    b = xr.DataArray(ml_coeff["b"]).rename({"dim_0": "level"})
    b = b.expand_dims(dim={"time": dims[0], "latitude": dims[2], "longitude": dims[3]}, axis=[0, 2, 3])

    p_half = a + b * np.exp(lnsp.values)

    ds = compute_z_level(ds, p_half)
    ds["geom_height"] = Re * ds["geop_height"] / (Re - ds["geop_height"])

    i = np.arange(0, 137)
    ds["p"] = ds["t"].copy()
    ds["p"][:, i, :, :] = (p_half[:, i + 1, :, :].values + p_half[:, i, :, :].values) / 2

    vars = [v for v in vars if v in ds]
    ds = interp_ds_vertically(ds, z_new, alt_var, vars)

    for var_name in vars:
        ds[var_name] = ds[var_name].astype("float32")
    return ds


def prepare_ml_int(
    file_ml,
    file_ml_p,
    file_ml_int,
    z_top=80000.0,
    dz=400.0,
    vars=("t", "p", "u", "v", "q"),
    alt_var="geom_height",
    time_chunk=None,
):
    """Build the height-interpolated dataset from the raw model-level downloads.

    file_ml    : netcdf with t/u/v/q on model levels 1-137
    file_ml_p  : netcdf with z/lnsp on level 1
    time_chunk : process this many timestamps per block, writing one temporary part each
                 and concatenating at the end (None = all at once, the original behaviour).
                 Both the ECMWF pressure/geopotential reconstruction and the vertical
                 interpolation are per column and per timestamp, so blocking over time is
                 exact -- it only bounds peak memory. That matters: the float64
                 (time, 138, lat, lon) half-level intermediates and the float64
                 (time, nz, lat, lon) interpolation buffers scale with the request length,
                 so a multi-day window at dz=100 m needs >100 GB in one go.
    """
    ml_coeff = pd.read_csv(FILE_ML_COEFF)
    nz = int(round(z_top / dz)) + 1
    z_new = np.linspace(0.0, z_top, nz)

    parts = []
    with xr.open_dataset(file_ml) as ds_raw, xr.open_dataset(file_ml_p) as dsp_raw:
        ds_all = _normalize(ds_raw.merge(dsp_raw))
        nt = ds_all.sizes["time"]
        step = nt if time_chunk is None else int(time_chunk)

        if step >= nt:
            _prepare_one(ds_all.load(), ml_coeff, z_new, vars, alt_var).to_netcdf(file_ml_int)
            return file_ml_int

        for i0 in range(0, nt, step):
            part = f"{file_ml_int}.t{i0:04d}.tmp.nc"
            print(f"[proc] times {i0}..{min(i0 + step, nt) - 1} of {nt - 1}", flush=True)
            block = ds_all.isel(time=slice(i0, i0 + step)).load()
            _prepare_one(block, ml_coeff, z_new, vars, alt_var).to_netcdf(part)
            del block
            parts.append(part)

    with xr.open_mfdataset(parts, combine="by_coords") as ds_int:
        ds_int.to_netcdf(file_ml_int)
    for part in parts:
        os.remove(part)
    return file_ml_int
