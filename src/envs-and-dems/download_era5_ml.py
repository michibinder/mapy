"""Download ERA5 model-level data for a region and time window (nested-PMAP input).

Retrieves the full 137-level analysis (t, u, v, q = param 130/131/132/133) plus the
level-1 surface fields (z, lnsp = param 129/152) from the CDS MARS endpoint
'reanalysis-era5-complete', one request per calendar date, concatenates the parts,
and (optionally) builds the height-interpolated file via era5_processor.prepare_ml_int.

Requires a CDS API token in ~/.cdsapirc (https://cds.climate.copernicus.eu/how-to-api)
and the post-venv python (cdsapi installed there):

    cd /work/bd0620/b309199/mapy/src/envs-and-dems
    /home/b/b309199/venvs/post-venv/bin/python download_era5_ml.py [download|process|all]

The stage argument (default 'all') splits the two very different resource profiles, which
matters for windows longer than a few hours: 'download' is network-bound and needs almost
no memory (run it in the background on the login node, MARS queue waits are long), while
'process' is the memory/CPU-heavy height interpolation (submit it to a compute node via
rera5_process.sh). Both stages skip work whose output already exists, so 'all' after a
finished download just processes.

Output files in CONFIG['outdir']:
    <name>-ml.nc      raw t/u/v/q on 137 model levels
    <name>-ml-p.nc    raw z/lnsp on level 1 (surface geopotential, log surface pressure)
    <name>-ml-int.nc  t/p/u/v/q interpolated to a uniform geometric-height grid
"""

import datetime as dt
import os
import sys

import cdsapi
import xarray as xr

import era5_processor

# sam3d_180616: the 2018-06-16/17 case on the same sam envelope as sam3d_240718 (the
# footprint of the 4400 x 3600 km CORAL-centred nest -- do not narrow it, it is derived
# from the projected rectangle's corners, not from a lat/lon box).
# The window runs one hour PAST the two requested days (-> 06-18T00:00, 49 timestamps) so
# that a later full 48-h run can still be forced at its last boundary step; PMAP needs
# input_<nstep> at model_endtime.
CONFIG = {
    "name": "sam3d_180616",
    "outdir": "/work/bd0620/b309199/data/era5-data",
    "area": "-33/-114.5/-70.6/-20.7",
    "grid": "0.25/0.25",
    "start": "2018-06-16T00:00",
    "end": "2018-06-18T00:00",
    "params_ml": "130/131/132/133",
    "process": True,
    "z_top_m": 80000.0,
    "dz_m": 100.0,
    "proc_chunk_h": 6,
    "keep_raw": True,
}

LEVELS_ML = "/".join(str(n) for n in range(1, 138))


def hours_by_date(start, end):
    """Group the hourly timestamps of [start, end] by calendar date for MARS requests."""
    t0 = dt.datetime.fromisoformat(start)
    t1 = dt.datetime.fromisoformat(end)
    groups = {}
    t = t0
    while t <= t1:
        groups.setdefault(t.strftime("%Y-%m-%d"), []).append(t.strftime("%H:%M:%S"))
        t += dt.timedelta(hours=1)
    return groups


def retrieve_ml(client, date, times, levelist, param, target):
    """Run one MARS model-level request for a single date; skip if target exists."""
    if os.path.exists(target):
        print(f"[skip] {target}")
        return
    print(f"[mars] {target}  ({date} {times[0]}..{times[-1]}, param {param})")
    client.retrieve(
        "reanalysis-era5-complete",
        {
            "date": date,
            "levelist": levelist,
            "levtype": "ml",
            "param": param,
            "stream": "oper",
            "time": "/".join(times),
            "type": "an",
            "area": CONFIG["area"],
            "grid": CONFIG["grid"],
            "format": "netcdf",
            "resol": "av",
        },
        target,
    )


def concat_parts(parts, target):
    """Concatenate per-date part files along time into one netcdf."""
    if os.path.exists(target):
        print(f"[skip] {target}")
        return
    def _drop_aux(ds):
        return ds.drop_vars([v for v in ("expver", "number") if v in ds.variables])
    with xr.open_mfdataset(parts, combine="by_coords", preprocess=_drop_aux) as ds:
        ds.sortby("valid_time" if "valid_time" in ds.coords else "time").to_netcdf(target)
    print(f"[out ] {target}")


def main(stage="all"):
    cfg = CONFIG
    os.makedirs(cfg["outdir"], exist_ok=True)
    base = os.path.join(cfg["outdir"], cfg["name"])
    file_ml = f"{base}-ml.nc"
    file_ml_p = f"{base}-ml-p.nc"
    file_ml_int = f"{base}-ml-int.nc"

    groups = hours_by_date(cfg["start"], cfg["end"])
    parts_ml, parts_p = [], []
    for date in groups:
        tag = date.replace("-", "")
        parts_ml.append(f"{base}-ml-{tag}.part.nc")
        parts_p.append(f"{base}-ml-p-{tag}.part.nc")

    if stage in ("download", "all"):
        client = cdsapi.Client()
        for (date, times), part_ml, part_p in zip(groups.items(), parts_ml, parts_p):
            retrieve_ml(client, date, times, LEVELS_ML, cfg["params_ml"], part_ml)
            retrieve_ml(client, date, times, "1", "129/152", part_p)

        concat_parts(parts_ml, file_ml)
        concat_parts(parts_p, file_ml_p)
        print(f"[done] download stage complete ({len(groups)} dates)")

    if stage in ("process", "all") and cfg["process"]:
        print(f"[proc] interpolating model levels -> {file_ml_int}", flush=True)
        era5_processor.prepare_ml_int(
            file_ml,
            file_ml_p,
            file_ml_int,
            z_top=cfg["z_top_m"],
            dz=cfg["dz_m"],
            time_chunk=cfg["proc_chunk_h"],
        )
        print(f"[out ] {file_ml_int}")

        if not cfg["keep_raw"]:
            for f in parts_ml + parts_p:
                if os.path.exists(f):
                    os.remove(f)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg not in ("download", "process", "all"):
        raise SystemExit(f"usage: {sys.argv[0]} [download|process|all]  (got {arg!r})")
    main(arg)
