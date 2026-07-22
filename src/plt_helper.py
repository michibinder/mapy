################################################################################
# Copyright 2023 German Aerospace Center                                       #
################################################################################
# This is free software you can redistribute/modify under the terms of the     #
# GNU Lesser General Public License 3 or later: http://www.gnu.org/licenses    #
################################################################################

import os
import glob
import re
import matplotlib.dates as mdates
import time
from datetime import datetime
import imageio.v2 as imageio

import numpy as np
import xarray as xr
import scipy
import pywt

import yaml

# from plteulag.graveyard.waveletFunctions import wave_signif, wavelet
# from statistics import linear_regression

# import logging
# import warnings
# warnings.simplefilter("ignore", RuntimeWarning)
# warnings.filterwarnings('ignore', category=UserWarning, module='imageio_ffmpeg')
# logging.getLogger('imageio_ffmpeg').setLevel(logging.ERROR)

"""Config"""
pbar_interval = 5 # %

mother = 'MORLET'
lag1 = 0.72  # lag-1 autocorrelation for red noise background
sig_lev = 1


def vorticity_xz(u, w, dx, dz):
    """
    Compute vorticity for a velocity field in the x-z plane.
    
    Parameters:
        u (2D numpy array): Velocity in the x-direction (shape: Nz, Nx)
        w (2D numpy array): Velocity in the z-direction (shape: Nz, Nx)
        dx (float): Grid spacing in the x-direction
        dz (float): Grid spacing in the z-direction
        
    Returns:
        omega_y (2D numpy array): Vorticity in the y-direction (shape: Nz, Nx)
    """
    # Compute derivatives using central differences
    du_dz = np.gradient(u, dz, axis=0)  # ∂u/∂z
    dw_dx = np.gradient(w, dx, axis=1)  # ∂w/∂x
    
    # Compute vorticity (omega_y = ∂w/∂x - ∂u/∂z)
    omega_y = dw_dx - du_dz
    
    return omega_y


def get_scalar_meta(ds, name, default=None):
    if name in getattr(ds, "attrs", {}):
        return ds.attrs[name]
    if hasattr(ds, "variables") and name in ds.variables:
        value = ds[name].values
        try:
            return np.asarray(value).item()
        except ValueError:
            return value
    if default is not None:
        return default
    raise KeyError(f"Could not find scalar metadata '{name}'")


def get_eulag_t_and_tenv(th, the, p, ppe, cap, pref00):
    cap = float(np.asarray(cap).item())
    pref00 = float(np.asarray(pref00).item())
    thloc = np.asarray(the) + np.asarray(th)
    ploc = np.asarray(ppe) + np.asarray(p)
    tloc = thloc * (ploc / pref00) ** cap
    tenv = np.asarray(the) * (np.asarray(ppe) / pref00) ** cap
    return tloc, tenv

def wavelet_1D_decomp(data, dx, dim=0):
    """1D Wavelet analysis along given dim of 2D array"""

    shape = np.shape(data)
    lambdax = np.zeros(shape)
    variance = np.std(data, ddof=1) ** 2

    # pad = 1    # pad the time series with zeroes (recommended)
    dj = 0.0625  # this will do 4 sub-octaves per octave
    s0 = 2 * dx  # this says start at a scale of 6 months
    j1 = 5 / dj  # this says do 7 powers-of-two with dj sub-octaves each

    n = shape[0]
    m = shape[1]
    if dim == 0:
        n0 = n
        n1 = m
    else:
        n0 = m 
        n1 = n
    for y in range(0,n1):
        if dim == 0:
            narray = data[y,:]
        else:
            narray = data[:,y]
        wave, period, scale, coi = wavelet(narray, dx, dj=dj, s0=s0, J1=j1, mother=mother)
        power = (np.abs(wave)) ** 2  # / variance # wavelet power spectrum normalized by variance

        # - Significance levels - #
        signif = wave_signif(([variance]), dt=dx, sigtest=0, scale=scale, lag1=lag1, mother=mother)
        # expand signif --> (J+1)x(N) array
        sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
        sig95 = power / sig95  # where ratio > 1, power is significant

        # jvec holds indices of maximum power for each x-location
        # indices refer to certain wavelengths
        jvec = np.argmax(power,axis=0)
        xvec = np.arange(0,len(jvec),1)

        if dim == 0:
            lambdax[y,:] = np.where(sig95[jvec,xvec] > sig_lev, period[jvec], np.nan)
        else:
            lambdax[:,y] = np.where(sig95[jvec,xvec] > sig_lev, period[jvec], np.nan)
        # lambdax[y,:] = np.where(power[jvec,xvec] > E_THRESHOLD, period[jvec], np.nan) # E_THRESHOLD = 25
    return lambdax


def detect_model_output(fpath):
    if os.path.exists(os.path.join(fpath, "grd.nc")):
        return "eulag"
    if os.path.exists(os.path.join(fpath, "config.yml")) and os.path.exists(os.path.join(fpath, "slices_y.nc")):
        return "pmap"
    raise FileNotFoundError(f"Could not detect supported model output in {fpath}")


def _normalize_z_slices(z_slices):
    if isinstance(z_slices, (int, np.integer)):
        return [int(z_slices)]
    return [int(z) for z in z_slices]


def _get_pmap_reference_path(fpath):
    """Resolve restart folders like `sim_r1` back to the original `sim` folder."""
    normalized = os.path.normpath(os.fspath(fpath))
    dirname, basename = os.path.split(normalized)
    match = re.match(r"^(?P<base>.+)_r[1-5]$", basename)
    if not match:
        return normalized

    reference_path = os.path.join(dirname, match.group("base"))
    if not os.path.isdir(reference_path):
        raise FileNotFoundError(
            f"Could not find reference simulation folder '{reference_path}' for restart folder '{normalized}'."
        )
    return reference_path


def get_slice_inventory(fpath):
    model = detect_model_output(fpath)
    if model == "eulag":
        return {
            "model": model,
            "x": len(sorted(glob.glob(os.path.join(fpath, "yzslc_*")))),
            "y": len(sorted(glob.glob(os.path.join(fpath, "xzslc_*")))),
            "z": len(sorted(glob.glob(os.path.join(fpath, "xyslc_*")))),
        }

    with xr.open_dataset(os.path.join(fpath, "slices_x.nc"), decode_times=False, chunks={}) as dsx:
        x_count = int(dsx.sizes.get("x", 1))
    with xr.open_dataset(os.path.join(fpath, "slices_y.nc"), decode_times=False, chunks={}) as dsy:
        y_count = int(dsy.sizes.get("y", 1))
    with xr.open_dataset(os.path.join(fpath, "slices_z.nc"), decode_times=False, chunks={}) as dsz:
        z_count = int(dsz.sizes.get("z", 1))
    return {"model": model, "x": x_count, "y": y_count, "z": z_count}


def sanitize_slice_request(fpath, slices):
    inventory = get_slice_inventory(fpath)

    def clamp(index, count):
        if count <= 0:
            raise ValueError(f"No slice available for count={count}")
        return max(0, min(int(index), count - 1))

    z_slices = _normalize_z_slices(slices.get("z", [0]))
    sanitized = {
        "x": clamp(slices.get("x", 0), inventory["x"]),
        "y": clamp(slices.get("y", 0), inventory["y"]),
        "z": [clamp(z, inventory["z"]) for z in z_slices],
    }
    return sanitized


def _load_pmap_cfg(fpath):
    with open(os.path.join(fpath, "config.yml"), "r") as stream:
        cfg = yaml.safe_load(stream)

    cfg["dx"] = (cfg["xmax"] - cfg["xmin"]) / (cfg["nx"]-1)
    cfg["dy"] = (cfg["ymax"] - cfg["ymin"]) / (cfg["ny"]-1)
    cfg["dz"] = (cfg["zmax"] - cfg["zmin"]) / (cfg["nz"]-1)

    cfg["constants"]["R"] = cfg["constants"]["Ravo"] * cfg["constants"]["Rbol"]
    cfg["constants"]["Rd"] = 1000 * cfg["constants"]["R"] / cfg["constants"]["Rmd"]
    cfg["constants"]["cpd"] = (
        cfg["constants"]["gamma"] / (cfg["constants"]["gamma"] - 1.0) * cfg["constants"]["Rd"]
    )
    cfg["constants"]["Rd_cpd"] = cfg["constants"]["Rd"] / cfg["constants"]["cpd"]
    cfg["constants"]["cap"] = cfg["constants"]["Rd"] / cfg["constants"]["cpd"]
    return cfg


def _rename_pmap_vars(ds):
    rename_map = {}
    for old, new in {
        "density": "rho",
        "theta_total": "th",
        "uvelx": "u",
        "uvely": "v",
        "uvelz": "w",
        "uvelx_ambient": "ue",
        "uvely_ambient": "ve",
        "uvelz_ambient": "we",
    }.items():
        if old in ds:
            rename_map[old] = new
    return ds.rename(rename_map) if rename_map else ds


def _maybe_hours(time_da):
    values = np.asarray(time_da.values)
    if np.issubdtype(values.dtype, np.number):
        return time_da / 3600
    return time_da


def _broadcast_scalar_field(reference, value):
    return xr.full_like(reference, value)


def _augment_pmap_fields(ds, ds0, cfg):
    ds = _rename_pmap_vars(ds)
    ds0 = _rename_pmap_vars(ds0)

    p0 = cfg["constants"]["p0"]
    cpd_over_rd = cfg["constants"]["cpd"] / cfg["constants"]["Rd"]

    ds["p"] = p0 * ds["exner_total"] ** cpd_over_rd
    ds0["p"] = p0 * ds0["exner_total"] ** cpd_over_rd
    ds["pprime"] = ds["p"] - ds0["p"].broadcast_like(ds["p"])
    ds["thprime"] = ds["th"] - ds0["th"].broadcast_like(ds["th"])
    ds["rhoprime"] = ds["rho"] - ds0["rho"].broadcast_like(ds["rho"])
    ds["the"] = ds0["th"].broadcast_like(ds["th"])
    ds["ppe"] = ds0["p"].broadcast_like(ds["p"])
    ds["rh0"] = ds0["rho"].broadcast_like(ds["rho"])

    # ambient wind: prefer the model output (uvelx/uvely_ambient, time-dependent for
    # transient_wind runs); fall back to the first time step / the config value
    if "ue" in ds:
        pass
    elif "u" in ds0:
        ds["ue"] = ds0["u"].broadcast_like(ds["u"])
    else:
        ds["ue"] = _broadcast_scalar_field(ds["th"], cfg["ambient_fields"]["velocity_x"])

    if "ve" in ds or "v" not in ds:
        pass
    elif "v" in ds0:
        ds["ve"] = ds0["v"].broadcast_like(ds["v"])
    else:
        ds["ve"] = _broadcast_scalar_field(ds["v"], cfg["ambient_fields"]["velocity_y"])

    ds["t"] = ds["th"] * (ds["p"] / p0) ** cfg["constants"]["Rd_cpd"]
    return ds


def _to_canonical_dims(ds, dimorder="txyz"):
    """Reorder a loaded slice to the internal canonical order (time, z, y, x) the plotting
    expects, so it is correct for either stored order (new "txyz" or legacy "tzyx"). Spatial
    dims are reordered by name when present (robust to the file's stored order); if they are
    not named x/y/z they are first named from the declared ``dimorder``."""
    if not any(d in ds.dims for d in ("x", "y", "z")):
        spatial = [c for c in dimorder if c in "xyz"]
        body = [d for d in ds.dims if d != "time"]
        if len(spatial) == len(body):
            ds = ds.rename(dict(zip(body, spatial)))
    order = [d for d in ("time", "z", "y", "x") if d in ds.dims]
    return ds.transpose(*order) if order else ds


def _prepare_pmap_xz_dataset(ds, y_index, dimorder="txyz"):
    ds = _to_canonical_dims(ds, dimorder)
    ds = ds.squeeze(drop=True)
    ds = ds.assign_coords({"time": _maybe_hours(ds["time"]), "x": ds["x"] / 1000, "z": ds["z"] / 1000})
    ds["zcr"] = ds["zcr"] / 1000
    ds["xcr"] = xr.DataArray(
        np.broadcast_to(ds["x"].values, (ds.sizes["z"], ds.sizes["x"])),
        dims=("z", "x"),
        coords={"z": ds["z"], "x": ds["x"]},
    )
    ypos = float(ds["y"].values) / 1000 if "y" in ds.coords else float(y_index)
    ds = ds.assign_coords({"j": int(y_index), "ypos": ypos})
    ds.attrs["j"] = int(y_index)
    ds.attrs["ypos"] = ypos
    return ds


def _prepare_pmap_yz_dataset(ds, x_index, dimorder="txyz"):
    ds = _to_canonical_dims(ds, dimorder)
    ds = ds.squeeze(drop=True)
    ds = ds.assign_coords({"time": _maybe_hours(ds["time"]), "y": ds["y"] / 1000, "z": ds["z"] / 1000})
    ds["zcr"] = ds["zcr"] / 1000
    ds["ycr"] = xr.DataArray(
        np.broadcast_to(ds["y"].values, (ds.sizes["z"], ds.sizes["y"])),
        dims=("z", "y"),
        coords={"z": ds["z"], "y": ds["y"]},
    )
    xpos = float(ds["x"].values) / 1000 if "x" in ds.coords else float(x_index)
    ds = ds.assign_coords({"i": int(x_index), "xpos": xpos})
    ds.attrs["i"] = int(x_index)
    ds.attrs["xpos"] = xpos
    return ds


def _prepare_pmap_xy_dataset(ds, topo_ds, k_index, dimorder="txyz"):
    ds = _to_canonical_dims(ds, dimorder)
    topo_ds = _to_canonical_dims(topo_ds, dimorder)
    ds = ds.squeeze(drop=True)
    ds = ds.assign_coords({"time": _maybe_hours(ds["time"]), "x": ds["x"] / 1000, "y": ds["y"] / 1000, "z": ds["z"] / 1000})
    ds["zcr"] = ds["zcr"] / 1000

    x_mesh, y_mesh = np.meshgrid(ds["x"].values, ds["y"].values)
    ds["xcr"] = xr.DataArray(x_mesh, dims=("y", "x"), coords={"y": ds["y"], "x": ds["x"]})
    ds["ycr"] = xr.DataArray(y_mesh, dims=("y", "x"), coords={"y": ds["y"], "x": ds["x"]})
    ds["zcrtopo"] = topo_ds["zcr"].squeeze(drop=True) * 1000
    # print("Topo Max:", np.max(topo_ds["zcr"].values))
    zpos = float(ds["z"].values)
    dz00 = 0.4 # km
    # ds = ds.assign_coords({"k": int(zpos/dz00), "zpos": zpos})
    ds.attrs["k"] = int(zpos/dz00)
    ds.attrs["zpos"] = zpos
    return ds


def _split_pmap_xy_slices(dsxy, topo_ds, slice_indices, dimorder="txyz"):
    if "z" not in dsxy.dims:
        dsxy = dsxy.expand_dims({"z": [dsxy["z"].item()]})

    dsxy_list = []
    for pos, k_index in enumerate(slice_indices):
        dsxy_list.append(_prepare_pmap_xy_dataset(dsxy.isel(z=pos), topo_ds, k_index, dimorder))
    return dsxy_list


def preprocess_pmap_xzyz(fpath, slices={"x": 0, "y": 0, "z": [0]}, dimorder="txyz"):
    slices = sanitize_slice_request(fpath, slices)
    cfg = _load_pmap_cfg(fpath)

    dsy = xr.open_dataset(os.path.join(fpath, "slices_y.nc"), decode_times=False, chunks={}).isel(y=slices["y"])
    dsy0 = xr.open_dataset(os.path.join(fpath, "slices_y.nc"), decode_times=False, chunks={}).isel(y=slices["y"], time=0)
    dsxz = _prepare_pmap_xz_dataset(_augment_pmap_fields(dsy, dsy0, cfg), slices["y"], dimorder)

    dsx = xr.open_dataset(os.path.join(fpath, "slices_x.nc"), decode_times=False, chunks={}).isel(x=slices["x"])
    dsx0 = xr.open_dataset(os.path.join(fpath, "slices_x.nc"), decode_times=False, chunks={}).isel(x=slices["x"], time=0)
    dsyz = _prepare_pmap_yz_dataset(_augment_pmap_fields(dsx, dsx0, cfg), slices["x"], dimorder)

    return cfg, dsxz, dsyz


def _match_reference_coords(ds_ref, ds_data):
    """Relabel a reference dataset's dimension coordinates to the data's, by index, where the
    sizes match. The reference run of a restart can label shared dims differently than the data
    — notably the vertical: a restart writes the stretched level heights as ``z`` while the base
    run wrote a uniform computational ``z`` — even though the model grid is identical (``zcr``
    matches by index). Without this, the coordinate-aware subtraction in ``_augment_pmap_fields``
    aligns on the mismatched labels and zeroes the perturbations (blank T')."""
    shared = {
        d: ds_data[d]
        for d in ds_ref.dims
        if d in ds_ref.coords and d in ds_data.coords and ds_ref.sizes[d] == ds_data.sizes[d]
    }
    return ds_ref.assign_coords(shared)


_AMBIENT_CACHE = {}


def _load_pmap_ambient(reference_fpath):
    """Load the t=0 ambient as a 1D vertical profile (theta/exner/density vs height ``zcr``) from
    a single column of the reference run's ``data_0.nc`` — the true initial state on the model
    grid (and the natural source once the ambient becomes a 3D global field for nested runs). The
    tiny profile is cached to ``<ref>/_ambient_profile.npz`` so the one HDF5 read happens once and
    the animation's many worker processes only load the small npz — never opening ``data_0.nc``
    concurrently, which serializes/hangs under HDF5 file locking on Lustre. Returns a dict of 1D
    arrays (``zcr`` ascending) or ``None`` if no source, so callers fall back to the legacy
    ``slices.isel(0)`` reference. A profile (not full planes) also keeps the reference decoupled
    from the first stored slice/cube frame (which need not be at t=0). Cached in memory per
    reference run (the ambient is time-invariant)."""
    if reference_fpath not in _AMBIENT_CACHE:
        npz = os.path.join(reference_fpath, "_ambient_profile.npz")
        data0 = os.path.join(reference_fpath, "data_0.nc")
        keys = ("zcr", "theta_total", "exner_total", "density")
        profile = None
        if os.path.isfile(npz) and (not os.path.isfile(data0) or os.path.getmtime(npz) >= os.path.getmtime(data0)):
            with np.load(npz) as z:
                profile = {k: z[k] for k in keys}
        elif os.path.isfile(data0):
            ds = xr.open_dataset(data0, decode_times=False)
            if "time" in ds.dims:
                ds = ds.isel(time=0)
            col = ds[list(keys)].isel(x=0, y=0).load()
            ds.close()
            order = np.argsort(np.asarray(col["zcr"].values))
            profile = {k: np.asarray(col[k].values)[order] for k in keys}
            try:
                np.savez(npz, **profile)
            except OSError:
                pass
        _AMBIENT_CACHE[reference_fpath] = profile
    return _AMBIENT_CACHE[reference_fpath]


_AMBIENT_DS0_CACHE = {}


def _ambient_for(profile, target):
    """Evaluate the 1D ambient profile at a slice's physical heights (``target['zcr']``) to build
    the reference ``ds0`` on the slice grid — works uniformly for xz/yz/xy without strided
    full-plane reads, and exactly equals the (horizontally uniform) 1D ambient. A future 3D
    ambient would replace this with a per-column vertical interpolation. Carries only the slice's
    dimension coordinates, so the (numpy) result is reusable for every time step."""
    zt = np.asarray(target["zcr"].values)
    coords = {d: target[d] for d in target["zcr"].dims if d in target.coords}
    ds0 = xr.Dataset()
    for v in ("theta_total", "exner_total", "density"):
        amb = np.interp(zt.ravel(), profile["zcr"], profile[v]).reshape(zt.shape)
        ds0[v] = xr.DataArray(amb, dims=target["zcr"].dims, coords=coords)
    return ds0


def _ambient_ds0_key(reference_fpath, fname, sel):
    (axis, idx), = sel.items()
    return (reference_fpath, fname, axis, tuple(np.atleast_1d(idx).tolist()))


def precompute_ambient(fpath, slices):
    """Compute and cache the ambient reference ``ds0`` for the xz/yz/xy slices ONCE — call this in
    the parent process before the render forks its workers. ``data_0.nc`` (via the profile npz) is
    read a single time, the per-slice references are built into ``_AMBIENT_DS0_CACHE`` as numpy,
    and every forked worker then merely subtracts the inherited reference (no per-worker file read
    or dask compute, which otherwise oversubscribes a many-core node). Returns False (leaving the
    legacy per-frame reference in place) when no ambient source is available."""
    reference_fpath = _get_pmap_reference_path(fpath)
    ambient = _load_pmap_ambient(reference_fpath)
    if ambient is None:
        return False
    for fname, sel in (("slices_y.nc", {"y": slices["y"]}),
                       ("slices_x.nc", {"x": slices["x"]}),
                       ("slices_z.nc", {"z": slices["z"]})):
        # use a context manager so no slice file handle is left open across the worker fork
        with xr.open_dataset(os.path.join(fpath, fname), decode_times=False) as ds:
            _reference_state(ambient, ds.isel(time=0, **sel), reference_fpath, fname, sel)
    return True


def _reference_state(ambient, target, reference_fpath, fname, sel):
    """Build the t=0 reference ``ds0`` for ``_augment_pmap_fields`` from the ambient profile
    (data_0.nc), caching it per slice so it is computed once (in the parent, before the render
    parallelizes) and every worker only subtracts it; falls back to the legacy first frame of the
    reference slices file when no ambient is available."""
    if ambient is None:
        legacy = xr.open_dataset(os.path.join(reference_fpath, fname), decode_times=False, chunks={}).isel(time=0, **sel)
        return _match_reference_coords(legacy, target)
    key = _ambient_ds0_key(reference_fpath, fname, sel)
    if key not in _AMBIENT_DS0_CACHE:
        _AMBIENT_DS0_CACHE[key] = _ambient_for(ambient, target).load()
    return _AMBIENT_DS0_CACHE[key]


def preprocess_pmap_tstep(fpath, t, slices={"x": 0, "y": 0, "z": [0]}, dimorder="txyz"):
    slices = sanitize_slice_request(fpath, slices)
    cfg = _load_pmap_cfg(fpath)
    reference_fpath = _get_pmap_reference_path(fpath)
    ambient = _load_pmap_ambient(reference_fpath)

    dsy = xr.open_dataset(os.path.join(fpath, "slices_y.nc"), decode_times=False, chunks={}).isel(time=t, y=slices["y"])
    dsy0 = _reference_state(ambient, dsy, reference_fpath, "slices_y.nc", {"y": slices["y"]})
    dsxz = _prepare_pmap_xz_dataset(_augment_pmap_fields(dsy, dsy0, cfg), slices["y"], dimorder)

    dsx = xr.open_dataset(os.path.join(fpath, "slices_x.nc"), decode_times=False, chunks={}).isel(time=t, x=slices["x"])
    dsx0 = _reference_state(ambient, dsx, reference_fpath, "slices_x.nc", {"x": slices["x"]})
    dsyz = _prepare_pmap_yz_dataset(_augment_pmap_fields(dsx, dsx0, cfg), slices["x"], dimorder)

    # z slices can be made more efficient without the list since all in one file
    # need option to select certain indices
    dsz = xr.open_dataset(os.path.join(fpath, "slices_z.nc"), decode_times=False, chunks={}).isel(time=t, z=slices["z"])
    dsz0 = _reference_state(ambient, dsz, reference_fpath, "slices_z.nc", {"z": slices["z"]})
    topo_raw = xr.open_dataset(os.path.join(reference_fpath, "slices_z.nc"), decode_times=False, chunks={}).isel(time=0, z=0)
    dsxy = _augment_pmap_fields(dsz, dsz0, cfg)
    topo_aug = _augment_pmap_fields(topo_raw, topo_raw, cfg)
    topo = _prepare_pmap_xy_dataset(topo_aug, topo_aug, 0, dimorder)
    ds_xyslices = _split_pmap_xy_slices(dsxy, topo, slices["z"], dimorder)

    return cfg, dsxz, dsyz, ds_xyslices


def preprocess_pmap(fpath, t=-1, slices={"x": 0, "y": 0, "z": [0]}):
    with open(os.path.join(fpath, "config.yml"), 'r') as stream:
        cfg = yaml.safe_load(stream)
        ## TODO: check dx
        cfg['dx'] = (cfg['xmax'] - cfg['xmin']) / cfg['nx']
        cfg['dy'] = (cfg['ymax'] - cfg['ymin']) / cfg['ny']
        cfg['dz'] = (cfg['zmax'] - cfg['zmin']) / cfg['nz']
        
        cfg['constants']['R'] = cfg['constants']['Ravo']*cfg['constants']['Rbol']
        cfg['constants']['Rd'] = 1000 * cfg['constants']['R'] / cfg['constants']['Rmd']
        cfg['constants']['cpd'] = cfg['constants']['gamma'] / (cfg['constants']['gamma'] - 1.0) * cfg['constants']['Rd']
        cfg['constants']['Rd_cpd'] = cfg['constants']['Rd'] / cfg['constants']['cpd']
        
        # TODO
        cfg['constants']['cap'] = cfg['constants']['Rd'] / cfg['constants']['cpd']
        
        # ds.attrs['cp']=3.5*ds.rg # Earth
        # ds.attrs['cap']=ds.rg/ds.cp
        # ds.attrs['cap']=ds.rg/ds.cp
        
        # cfg['ambient_fields']['velocity_x']
        # cfg['define_orography']['args']
        # cfg['constants']['fcoriolis0']
        # cfg['constants']['angle0']

    ## XZ
    # TODO: Add slice index if needed
    if t is None: 
        dsxz   = xr.open_dataset(os.path.join(fpath, f"slices_y.nc"), decode_times=False, chunks={}).isel(y=slices['y'])
    else:
        dsxz   = xr.open_dataset(os.path.join(fpath, f"slices_y.nc"), chunks={}).isel(time=t, y=slices['y'])
    dsxz0  = xr.open_dataset(os.path.join(fpath, f"slices_y.nc"), chunks={}).isel(time=0, y=slices['y'])
    # print(dsxz)
    try:
        dsxz  = dsxz.rename({"density":"rho", "theta_total":"th", "uvelx": "u", "uvely": "v", "uvelz": "w"})
    except:
        dsxz  = dsxz.rename({"density":"rho", "theta_total":"th"})
    dsxz0 = dsxz0.rename({"density":"rho", "theta_total":"th"})

    dsxz['p'] =  cfg['constants']['p0'] * dsxz["exner_total"] ** (cfg['constants']['cpd'] / cfg['constants']['Rd']) # (cp / Rd)
    dsxz0['p'] =  cfg['constants']['p0'] * dsxz0["exner_total"] ** (cfg['constants']['cpd'] / cfg['constants']['Rd']) # (cp / Rd)
    dsxz['pprime']=dsxz['p'] - dsxz0['p'].values
    dsxz['thprime']=dsxz['th'] - dsxz0['th'].values
    dsxz['rhoprime']=dsxz['rho'] - dsxz0['rho'].values

    # ds['dzdx_surf'] = ds['zcr'][0,:,:]
    # ds['dzdx_surf'].values = np.gradient(ds['zcr'][0,:,:].values, cfg['dx'], axis=1)

    ## Brunt-Vaisala frequency N and temperature t
    # N  = (g/th * np.gradient(th, vres))**(1/2)
    dsxz['N'] = (cfg['constants']['gravity0']/dsxz['th'] * np.gradient(dsxz['th'], cfg['dz'], axis=0))**(1/2)
    dsxz['t'] = dsxz['th']*(dsxz['p']/cfg['constants']['p0'])**cfg['constants']['Rd_cpd']

    ## YZ
    
    ## XY
    if t is None:
        dsxy  = xr.open_dataset(os.path.join(fpath, f"slices_z.nc"), decode_times=False, chunks={}).isel(z=slices['z'])
    else:
        dsxy  = xr.open_dataset(os.path.join(fpath, f"slices_z.nc"), chunks={}).isel(time=t, z=slices['z'])
    dsxy0 = xr.open_dataset(os.path.join(fpath, f"slices_z.nc"), chunks={}).isel(time=0, z=slices['z'])
    try:
        dsxy  = dsxy.rename({"density":"rho", "theta_total":"th", "uvelx": "u", "uvely": "v", "uvelz": "w"})
    except:
        dsxy  = dsxy.rename({"density":"rho", "theta_total":"th"})
    dsxy0 = dsxy0.rename({"density":"rho", "theta_total":"th"})

    dsxy['p'] =  cfg['constants']['p0'] * dsxy["exner_total"] ** (cfg['constants']['cpd'] / cfg['constants']['Rd']) # (cp / Rd)
    dsxy0['p'] =  cfg['constants']['p0'] * dsxy0["exner_total"] ** (cfg['constants']['cpd'] / cfg['constants']['Rd']) # (cp / Rd)
    dsxy['pprime']   = dsxy['p'] - dsxy0['p'].values
    dsxy['thprime']  = dsxy['th'] - dsxy0['th'].values
    dsxy['rhoprime'] = dsxy['rho'] - dsxy0['rho'].values
    dsxy['t']  = dsxy['th']*(dsxy['p']/cfg['constants']['p0'])**cfg['constants']['Rd_cpd']

    return dsxz, dsxy, cfg


def preprocess_pmap_3D(sim, folder, t=4):
    with open(os.path.join(folder,sim, "config.yml"), 'r') as stream:
        cfg = yaml.safe_load(stream)
        ## TODO: check dx
        cfg['dx'] = (cfg['xmax'] - cfg['xmin']) / cfg['nx']
        cfg['dy'] = (cfg['ymax'] - cfg['ymin']) / cfg['ny']
        cfg['dz'] = (cfg['zmax'] - cfg['zmin']) / cfg['nz']
        
        cfg['constants']['R'] = cfg['constants']['Ravo']*cfg['constants']['Rbol']
        cfg['constants']['Rd'] = 1000 * cfg['constants']['R'] / cfg['constants']['Rmd']
        cfg['constants']['cpd'] = cfg['constants']['gamma'] / (cfg['constants']['gamma'] - 1.0) * cfg['constants']['Rd']
        cfg['constants']['Rd_cpd'] = cfg['constants']['Rd'] / cfg['constants']['cpd']

        # cfg['ambient_fields']['velocity_x']
        # cfg['define_orography']['args']
        # cfg['constants']['fcoriolis0']
        # cfg['constants']['angle0']

    ds  = xr.open_dataset(os.path.join(folder,sim, f"data_{t}.nc"))
    ds0 = xr.open_dataset(os.path.join(folder,sim, f"data_0.nc"))
    ds['pprime']=ds['pressure'] - ds0['pressure'].values
    # ds['pprime']=ds['pressure'] - cfg['constants']['p0']
    ds['thprime']=ds['theta_total'] - ds0['theta_total'].values
    ds['rhoprime']=ds['density'] - ds0['density'].values

    # ds['dzdx_surf'] = ds['zcr'][0,:,:]
    # ds['dzdx_surf'].values = np.gradient(ds['zcr'][0,:,:].values, cfg['dx'], axis=1)

    ## Brunt-Vaisala N
    # N  = (g/th * np.gradient(th, vres))**(1/2)
    ds['N'] = (cfg['constants']['gravity0']/ds['theta_total'] * np.gradient(ds['theta_total'], cfg['dz'], axis=1))**(1/2)
    
    ## Temperature
    ds['t_total'] = ds['theta_total']*(ds['pressure']/cfg['constants']['p0'])**cfg['constants']['Rd_cpd']

    return ds, cfg


def preprocess_eulag_tapes(fpath, load_ds=False):
    """Process EULAG output"""
    grid_path  = os.path.join(fpath, "grd.nc") # --> ds
    tapes_path = os.path.join(fpath, "tapes.nc")
    if load_ds:
        ds = xr.load_dataset(grid_path)
        ds_full = xr.load_dataset(tapes_path)
    else:
        ds = xr.open_dataset(grid_path)
        ds_full = xr.open_dataset(tapes_path)
    if ds.ibcx == 1:
        # Shift to centered domain
        ds['xcr'] = ds['xcr'] - np.max(ds['xcr']/2)
        ds['ycr'] = ds['ycr'] - np.max(ds['ycr']/2)
    ds = ds.assign_coords({'xcr':ds['xcr'], 'ycr':ds['ycr'], 'zcr':ds_full['ELEVATION']})
    
    ## Sim parameters
    ds.attrs['bv'] = ds.attrs['bv'].round(3)
    ds.attrs['cp']=3.5*ds.rg # Earth
    ds.attrs['cap']=ds.rg/ds.cp
    if "itopo" not in ds.attrs:
        ds.attrs['itopo'] = 1
    
    ds_full = ds_full.rename({"p":"pprime", "th":"thprime"})
    ds = ds.merge(ds_full)
        
    return ds


def preprocess_eulag_tstep(fpath, t, slices={"x": 0, "y": 0, "z": [0]}):
    """Process EULAG output"""
    slices = sanitize_slice_request(fpath, slices)
    grid_path  = os.path.join(fpath, "grd.nc") # --> ds
    ds = xr.open_dataset(grid_path)
    if ds.ibcx == 1:
        # Shift to centered domain
        ds['xcr'] = ds['xcr'] - np.max(ds['xcr']/2)
        ds['ycr'] = ds['ycr'] - np.max(ds['ycr']/2)
    ds = ds.assign_coords({'xcr':ds['xcr']/1000, 'ycr':ds['ycr']/1000, 'zcr':ds['zcr']/1000})
    
    ## Sim parameters
    ds.attrs['bv'] = ds.attrs['bv'].round(3)
    ds.attrs['cp']=3.5*ds.rg # Earth
    ds.attrs['cap']=ds.rg/ds.cp
    if "itopo" not in ds.attrs:
        ds.attrs['itopo'] = 1

    """Slice outputs"""
    # dsxy['zcr'] = dsxy['zcr'] / 1000
    xzslices = sorted(glob.glob(os.path.join(fpath, "xzslc_*")))
    yzslices = sorted(glob.glob(os.path.join(fpath, "yzslc_*")))
    xyslices = sorted(glob.glob(os.path.join(fpath, "xyslc_*")))
    ds_xyslices = []

    # XY-slices
    for slc_ind in slices['z']:
        slc = xyslices[slc_ind]
        ds_slc = xr.open_dataset(slc, chunks={}).isel(t=t)
        ds_slc['zcr'] = ds_slc['zcr'] / 1000 #zkm
        ds_slc.attrs['k'] = int(slc.split("/")[-1][-8:-3])
        ds_slc.attrs['zpos'] = ds_slc.k * ds.dz00/1000
        ds_slc = ds_slc.assign_coords(xcr = ds['xcr'])
        ds_slc = ds_slc.assign_coords(ycr = ds['ycr'])
        ds_slc = ds_slc.rename({"p":"pprime", "th":"thprime"})
        ds_xyslices.append(ds_slc)
        
    # XZ-slice
    slc = xzslices[slices['y']]
    ds_slc = xr.open_dataset(slc, chunks={}).isel(t=t)
    ds_slc['zcr'] = ds_slc['zcr'] / 1000 #km
    ds_slc.attrs['j'] = int(slc.split("/")[-1][-8:-3])
    ds_slc.attrs['ypos'] = (ds_slc.j - ds.ny/2) * ds.dy00/1000
    # ds.attrs['pref00'] = ds_slc['pr0'].max()
    
    # time = ds_slc.t * ds.nlid * ds.dt00/3600
    ds_slc = ds_slc.assign_coords({'xcr': ds['xcr'], 'zcr':ds_slc['zcr']})
    # ds_slc = ds_slc.assign_coords({'time': time, 'xcr': ds['xcr'], 'zcr':ds_slc['zcr']})

    ds_slc = ds_slc.assign_attrs(ds.attrs)
    ds_slc["theta_total"] = ds_slc["the"] + ds_slc["th"]
    ds_slc["pressure"] = ds_slc["ppe"] + ds_slc["p"]
    ds_slc = ds_slc.rename({"p":"pprime", "pressure":"p", "th":"thprime", "theta_total":"th"})
    dsxz = ds_slc

    # YZ-slice
    slc = yzslices[slices['x']]
    ds_slc = xr.open_dataset(slc, chunks={}).isel(t=t)
    ds_slc['zcr'] = ds_slc['zcr'] / 1000 #km
    ds_slc.attrs['i'] = int(slc.split("/")[-1][-8:-3])
    if ds.ibcx == 0:
        ds_slc.attrs['xpos'] = (ds_slc.i - ds.nx/2)  * ds.dx00/1000
    else:
        ds_slc.attrs['xpos'] = ds_slc.i  * ds.dx00/1000

    ds_slc = ds_slc.assign_coords({'ycr': ds['ycr'], 'zcr':ds_slc['zcr']})
    ds_slc = ds_slc.rename({"p":"pprime", "th":"thprime"})
    dsyz = ds_slc
    
    return ds, dsxz, dsyz, ds_xyslices


def preprocess_eulag_xzyz(fpath, slices={"x": 0, "y": 0, "z": [0]}):
    """Process EULAG output"""
    slices = sanitize_slice_request(fpath, slices)
    grid_path  = os.path.join(fpath, "grd.nc") # --> ds
    ds = xr.open_dataset(grid_path)
    if ds.ibcx == 1:
        # Shift to centered domain
        ds['xcr'] = ds['xcr'] - np.max(ds['xcr']/2)
        ds['ycr'] = ds['ycr'] - np.max(ds['ycr']/2)
    ds = ds.assign_coords({'xcr':ds['xcr']/1000, 'ycr':ds['ycr']/1000, 'zcr':ds['zcr']/1000})
    
    ## Sim parameters
    ds.attrs['bv'] = ds.attrs['bv'].round(3)
    ds.attrs['cp']=3.5*ds.rg # Earth
    ds.attrs['cap']=ds.rg/ds.cp
    if "itopo" not in ds.attrs:
        ds.attrs['itopo'] = 1

    """Slice outputs"""
    xzslices = sorted(glob.glob(os.path.join(fpath, "xzslc_*")))
    yzslices = sorted(glob.glob(os.path.join(fpath, "yzslc_*")))

    # XZ-slice
    slc = xzslices[slices['y']]
    ds_slc = xr.open_dataset(slc)
    ds_slc['zcr'] = ds_slc['zcr'] / 1000 #km
    ds_slc.attrs['j'] = int(slc.split("/")[-1][-8:-3])
    ds_slc.attrs['ypos'] = (ds_slc.j - ds.ny/2) * ds.dy00/1000
    # ds.attrs['pref00'] = ds_slc['pr0'].max()
    
    # ds_slc = ds_slc.assign_coords({ds_slc.zcr:ds_slc.z})
    time = ds_slc.t * ds.nlid * ds.dt00/3600
    # ds_slc['time'] = ds_slc['time'].expand_dims({'z':ds_lid.z}, axis=1)

    ds_slc = ds_slc.assign_coords({'time': time, 'xcr': ds['xcr'], 'zcr':ds_slc['zcr']})
    ds_slc = ds_slc.assign_attrs(ds.attrs)
    ds_slc = ds_slc.rename({"p":"pprime", "th":"thprime"})
    dsxz = ds_slc
    
    # YZ-slice
    slc = yzslices[slices['x']]
    ds_slc = xr.open_dataset(slc)
    ds_slc['zcr'] = ds_slc['zcr'] / 1000 #km
    ds_slc.attrs['i'] = int(slc.split("/")[-1][-8:-3])
    if ds.ibcx == 0:
        ds_slc.attrs['xpos'] = (ds_slc.i - ds.nx/2)  * ds.dx00/1000
    else:
        ds_slc.attrs['xpos'] = ds_slc.i  * ds.dx00/1000

    ds_slc = ds_slc.assign_coords({'time': time, 'ycr': ds['ycr'], 'zcr':ds_slc['zcr']})
    ds_slc = ds_slc.rename({"p":"pprime", "th":"thprime"})
    dsyz = ds_slc

    return dsxz, dsyz


def preprocess_eulag_xy_merged(fpath, slices={"x": 0, "y": 0, "z": [0]}, t=None):
    slices = sanitize_slice_request(fpath, slices)
    grid_path = os.path.join(fpath, "grd.nc")
    ds = xr.open_dataset(grid_path)
    if ds.ibcx == 1:
        ds["xcr"] = ds["xcr"] - np.max(ds["xcr"] / 2)
        ds["ycr"] = ds["ycr"] - np.max(ds["ycr"] / 2)
    ds = ds.assign_coords({"xcr": ds["xcr"] / 1000, "ycr": ds["ycr"] / 1000, "zcr": ds["zcr"] / 1000})

    xyslices = sorted(glob.glob(os.path.join(fpath, "xyslc_*")))
    merged = []
    for slc_ind in slices["z"]:
        ds_slc = xr.open_dataset(xyslices[slc_ind], chunks={})
        if t is not None:
            ds_slc = ds_slc.isel(t=t)
        ds_slc["zcr"] = ds_slc["zcr"] / 1000
        ds_slc = ds_slc.assign_coords({"xcr": ds["xcr"], "ycr": ds["ycr"]})
        ds_slc = ds_slc.rename({"p": "pprime", "th": "thprime"})
        ds_slc = ds_slc.expand_dims({"zslc": [int(slc_ind)]})
        ds_slc = ds_slc.assign_coords({"k": ("zslc", [int(slc_ind)]), "zpos": ("zslc", [int(slc_ind) * ds.dz00 / 1000])})
        merged.append(ds_slc)
    return xr.concat(merged, dim="zslc")


def preprocess_eulag_output(fpath, slices={"x": 0, "y": 0, "z": [0]}, load_ds=False):
    """Process EULAG output"""
    slices = sanitize_slice_request(fpath, slices)
    grid_path  = os.path.join(fpath, "grd.nc") # --> ds
    if load_ds:
        ds = xr.load_dataset(grid_path)
    else:
        ds = xr.open_dataset(grid_path)
    if ds.ibcx == 1:
        # Shift to centered domain
        ds['xcr'] = ds['xcr'] - np.max(ds['xcr']/2)
        ds['ycr'] = ds['ycr'] - np.max(ds['ycr']/2)
    ds = ds.assign_coords({'xcr':ds['xcr']/1000, 'ycr':ds['ycr']/1000, 'zcr':ds['zcr']/1000})
    
    ## Sim parameters
    ds.attrs['bv'] = ds.attrs['bv'].round(3)
    ds.attrs['cp']=3.5*ds.rg # Earth
    ds.attrs['cap']=ds.rg/ds.cp
    if "itopo" not in ds.attrs:
        ds.attrs['itopo'] = 1

    """Slice outputs"""
    # dsxy['zcr'] = dsxy['zcr'] / 1000
    xzslices = sorted(glob.glob(os.path.join(fpath, "xzslc_*")))
    yzslices = sorted(glob.glob(os.path.join(fpath, "yzslc_*")))
    xyslices = sorted(glob.glob(os.path.join(fpath, "xyslc_*")))
    ds_xyslices = []

    # XZ-slice
    slc = xzslices[slices['y']]
    if load_ds:
        ds_slc = xr.load_dataset(slc)
    else:
        ds_slc = xr.open_dataset(slc)
    ds_slc['zcr'] = ds_slc['zcr'] / 1000 #km
    ds_slc.attrs['j'] = int(slc.split("/")[-1][-8:-3])
    ds_slc.attrs['ypos'] = (ds_slc.j - ds.ny/2) * ds.dy00/1000
    # ds.attrs['pref00'] = ds_slc['pr0'].max()
    
    # ds_slc = ds_slc.assign_coords({ds_slc.zcr:ds_slc.z})
    time = ds_slc.t * ds.nlid * ds.dt00/3600
    # ds_slc['time'] = ds_slc['time'].expand_dims({'z':ds_lid.z}, axis=1)

    ds_slc = ds_slc.assign_coords({'time': time, 'xcr': ds['xcr'], 'zcr':ds_slc['zcr']})
    ds_slc = ds_slc.assign_attrs(ds.attrs)
    dsxz = ds_slc

    # YZ-slice
    slc = yzslices[slices['x']]
    if load_ds:
        ds_slc = xr.load_dataset(slc)
    else:
        ds_slc = xr.open_dataset(slc)
    ds_slc['zcr'] = ds_slc['zcr'] / 1000 #km
    ds_slc.attrs['i'] = int(slc.split("/")[-1][-8:-3])
    if ds.ibcx == 0:
        ds_slc.attrs['xpos'] = (ds_slc.i - ds.nx/2)  * ds.dx00/1000
    else:
        ds_slc.attrs['xpos'] = ds_slc.i  * ds.dx00/1000

    ds_slc = ds_slc.assign_coords({'time': time, 'ycr': ds['ycr'], 'zcr':ds_slc['zcr']})
    dsyz = ds_slc

    # XY-slices
    for slc_ind in slices['z']:
        slc = xyslices[slc_ind]
        if load_ds:
            ds_slc = xr.load_dataset(slc)
        else:
            ds_slc = xr.open_dataset(slc)
        ds_slc['zcr'] = ds_slc['zcr'] / 1000 #km
        ds_slc.attrs['k'] = int(slc.split("/")[-1][-8:-3])
        ds_slc.attrs['zpos'] = ds_slc.k * ds.dz00/1000
        ds_slc = ds_slc.assign_coords(xcr = ds['xcr'])
        ds_slc = ds_slc.assign_coords(ycr = ds['ycr'])
        ds_xyslices.append(ds_slc)
    
    """Lidar outputs with high temporal resolution"""
    # lid_colors = ["purple", "forestgreen"]
    # lidars = sorted(glob.glob(os.path.join(fpath, "lid_*")))
    # ds_lidars = []
    # for i, lid_file in enumerate(lidars):
    #     if load_ds:
    #         ds_lid = xr.load_dataset(lid_file)
    #     else:
    #         ds_lid = xr.open_dataset(lid_file)
    #     ds_lid['time'] = ds_lid.t * ds.nlid * ds.dt00/3600
    #     ds_lid['time'] = ds_lid['time'].expand_dims({'z':ds_lid.z}, axis=1)
    #     ds_lid['zcr'] = ds_lid['zcr']/1000
        
    #     loc_str = lid_file.split("/")[-1][:-3]
    #     ds_lid.attrs['i'] = int(str(loc_str)[4:9])
    #     ds_lid.attrs['j'] = int(str(loc_str)[-5:])
    #     if ds.ibcx == 0:
    #         ds_lid.attrs['xpos'] = (ds_lid.i - ds.nx/2)  * ds.dx00/1000
    #     else:
    #         ds_lid.attrs['xpos'] = ds_lid.i * ds.dx00/1000
    #     ds_lid.attrs['ypos'] = (ds_lid.j - ds.ny/2) * ds.dy00/1000
    #     ds_lid.attrs['color'] = lid_colors[i]
    #     ds_lidars.append(ds_lid)

    # return ds, dsxz, dsyz, ds_xyslices, ds_lidars
    return ds, dsxz, dsyz, ds_xyslices


def preprocess_model_xzyz(fpath, slices={"x": 0, "y": 0, "z": [0]}):
    model = detect_model_output(fpath)
    slices = sanitize_slice_request(fpath, slices)
    if model == "pmap":
        cfg, dsxz, dsyz = preprocess_pmap_xzyz(fpath, slices=slices)
        return model, cfg, dsxz, dsyz
    grid_path = os.path.join(fpath, "grd.nc")
    ds = xr.open_dataset(grid_path)
    if ds.ibcx == 1:
        ds["xcr"] = ds["xcr"] - np.max(ds["xcr"] / 2)
        ds["ycr"] = ds["ycr"] - np.max(ds["ycr"] / 2)
    ds = ds.assign_coords({"xcr": ds["xcr"] / 1000, "ycr": ds["ycr"] / 1000, "zcr": ds["zcr"] / 1000})
    ds.attrs["bv"] = ds.attrs["bv"].round(3)
    ds.attrs["cp"] = 3.5 * ds.rg
    ds.attrs["cap"] = ds.rg / ds.cp
    if "itopo" not in ds.attrs:
        ds.attrs["itopo"] = 1
    dsxz, dsyz = preprocess_eulag_xzyz(fpath, slices=slices)
    return model, ds, dsxz, dsyz


def preprocess_model_tstep(fpath, t, slices={"x": 0, "y": 0, "z": [0]}):
    model = detect_model_output(fpath)
    slices = sanitize_slice_request(fpath, slices)
    if model == "pmap":
        cfg, dsxz, dsyz, ds_xyslices = preprocess_pmap_tstep(fpath, t=t, slices=slices)
        return model, cfg, dsxz, dsyz, ds_xyslices
    ds, dsxz, dsyz, ds_xyslices = preprocess_eulag_tstep(fpath, t=t, slices=slices)
    return model, ds, dsxz, dsyz, ds_xyslices


def timelab_format_func(value, tick_number):
    dt = mdates.num2date(value)
    if dt.hour == 0:
        return "{}\n{}".format(dt.strftime("%Y-%b-%d"), dt.strftime("%H"))
    else:
        return dt.strftime("%H")


def major_formatter_lon(x, pos):
    """Using western coordinates"""
    return "%.f°W" % abs(x)
    ##return "%.f°E" % abs(x)


def major_formatter_lat(x, pos):
    return "%.f°S" % abs(x)


def create_animation(image_folder, animation_name, fps=10):
    """Create animation (mp4) from pngs"""

    filenames        = sorted(os.listdir(image_folder))
    macro_block_size = 16

    # Increase the probesize to give FFmpeg more data to estimate the rate
    # writer_options = {'ffmpeg_params': ['-probesize', '100M']}  # Increase probesize to 5MB
    # writer_options = {'ffmpeg_params': ['-probesize', '5000000', '-analyzeduration', '5000000']}

    with imageio.get_writer(os.path.join(image_folder, animation_name), fps=fps) as writer: # duration=1000*1/fps
        for filename in filenames:
            if filename.endswith(".png"):
                image = imageio.imread(os.path.join(image_folder, filename))
                image = resize_to_macro_block(image, macro_block_size)
                writer.append_data(image)

    # imageio.mimsave(image_folder + "/era5_sequence.gif", images, duration=1/fps, palettesize=256/2)  # loop=0, quantizer="nq", palettesize=256
    print("[i]  MP4 Video created successfully!")


def resize_to_macro_block(image, macro_block_size):
    """Function to make image dimensions divisible by macro block size"""
    height, width = image.shape[:2]
    new_height = (height + macro_block_size - 1) // macro_block_size * macro_block_size
    new_width = (width + macro_block_size - 1) // macro_block_size * macro_block_size
    if (new_height != height) or (new_width != width):
        image = np.pad(image, ((0, new_height - height), (0, new_width - width), (0, 0)), 'constant')
    return image


def show_progress(progress_counter, lock, stime, total_tasks):
    with lock:
        progress_counter.value += 1
        if total_tasks <= 100/pbar_interval:
            print(f"[p]  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Number of tasks below progress bar limit.")
        else:
            if (progress_counter.value % (total_tasks // (100/pbar_interval))) == 0 or progress_counter.value == total_tasks or progress_counter.value == 1:
                progress = progress_counter.value / total_tasks
                elapsed = time.time() - stime
                eta = (elapsed / progress) * (1 - progress)

                # Convert elapsed and ETA to hours, minutes, and seconds
                elapsed_hrs, elapsed_rem = divmod(elapsed, 3600)
                elapsed_min, elapsed_sec = divmod(elapsed_rem, 60)
                eta_hrs, eta_rem = divmod(eta, 3600)
                eta_min, eta_sec = divmod(eta_rem, 60)

                # Progress bar
                total_hashtags = int(100/pbar_interval)
                hashtag_str = "#" * int(np.ceil(progress * total_hashtags))
                minus_str = "-" * int((1 - progress) * total_hashtags)

                print(f"[p]  |{hashtag_str}{minus_str}| Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Progress: {progress*100:05.2f}% - Elapsed: {int(elapsed_hrs):02d}:{int(elapsed_min):02d}:{int(elapsed_sec):02d} - ETA: {int(eta_hrs):02d}:{int(eta_min):02d}:{int(eta_sec):02d} (hh:mm:ss)", flush=True)


def get_colormap_bins_and_labels(max_level=64,linear=False):
    """
    Get different bin and label settings for varying colormap ranges.
    The bins are used for norm and later within pcolormesh / contourf / ...

    Good matplotlib colormaps:
    - 'turbo' better than 'jet'?
    - Plasma
    - Spectral
    - RdYlBu_r
    - RdBu_r -> perturbations of simulations
    
    cmap = plt.get_cmap('RdBu_r')

    """
    # - Logarithmic - #
    CLEV_256 = [-512,-256,-128,-64,-32,-16,-8,-4,4,8,16,32,64,128,256,512]
    CLEV_256_LABELS = [-512,-128,-32,-8,8,32,128,512]
    
    CLEV_64 = [-64,-32,-16,-8,-4,-2,-1,-0.5,-0.25,0.25,0.5,1,2,4,8,16,32,64]
    CLEV_64_LABELS = [-64,-16,-4,-1,1,4,16,64]

    CLEV_32 = [-32,-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16,32]
    CLEV_32_LABELS = [-16,-4,-1,1,4,16]
    ##  CLEV_32 = [-32,-16,-8,-4,-2,-1,-0.5,-0.25,-0.125,0.125,0.25,0.5,1,2,4,8,16,32]
    # CLEV_32 = [-32,-16,-8,-4,-2,-1,-0.5,-0.25,0.25,0.5,1,2,4,8,16,32]
    # CLEV_32_LABELS = [-32,-8,-2,-0.5,0.5,2,8,32]

    CLEV_16 = [-16,-8,-4,-2,-1,-0.5,-0.25,-0.125,-0.0625,0.0625,0.125,0.25,0.5,1,2,4,8,16]
    CLEV_16_LABELS = [-16,-4,-1,-0.25,0.25,1,4,16]
    #CLEV_16 = [-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16]
    #CLEV_16_LABELS = [-16,-4,-1,1,4,16]

    CLEV_8 = [-8,-4,-2,-1,-0.5,-0.25,-0.125,-0.0625,0.0625,0.125,0.25,0.5,1,2,4,8]
    CLEV_8_LABELS = [-4,-1,-0.25,-0.0625,0.0625,0.25,1,4]

    CLEV_2 = [-2,-1,-0.5,-0.25,-0.125,-0.0625,-0.03125,-0.015625,0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2]
    CLEV_2_LABELS = [-1,-0.25,-0.0625,0.0625,0.25,1]

    CLEV_08 = np.array(CLEV_8) / 10
    CLEV_08_LABELS = np.array(CLEV_8_LABELS) / 10

    CLEV_02 = np.array(CLEV_2) / 10
    CLEV_02_LABELS = np.array(CLEV_2_LABELS) / 10

    # CLEV_100 = np.array([-100,-50,-20,-10,-7,-5,-3,-1,1,3,5,7,10,20,50,100])
    # CLEV_100_LABELS = np.array([-50,-10,-5,-1,1,5,10,50])

    # CLEV_30 = [-30,-10,-7,-5,-3,-2,-1,-0.5,0.5,1,2,3,5,7,10,30]
    # CLEV_30_LABELS = [-10,-5,-2,-0.5,0.5,2,5,10]

    # CLEV_20 = [-20.,-7.,-5.,-3.,-2.,-1.,-0.7,-0.5,-0.2,-0.1,0.1,0.2,0.5,0.7,1.,2.,3.,5.,7.,20.]

    # CLEV_10 = np.array([-10.,-5.,-2.,-1.,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,1.,2.,5.,10.])
    # CLEV_10_LABELS = np.array([-5.,-1.,-0.5,-0.1,0.1,0.5,1.,5.])

    # CLEV_5 = [-5.,-2.,-1.,-0.7,-0.5,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.5,0.7,1.,2.,5.]
    # CLEV_5_LABELS = [-2.,-0.7,-0.3,-0.1,0.1,0.3,0.7,2.]

    # CLEV_2 = [-2.,-1.,-0.5,-0.3,-0.2,-0.1,-0.07,-0.05,0.05,0.07,0.1,0.2,0.3,0.5,1.,2.]
    # CLEV_2_LABELS = [-1.,-0.3,-0.1,-0.05,0.05,0.1,0.3,1.]

    # CLEV_05 = [-0.5,-0.3,-0.1,-0.05,-0.03,-0.01,-0.005,-0.002,0.002,0.005,0.01,0.03,0.05,0.1,0.3,0.5]
    # CLEV_05_LABELS = [-0.3,-0.05,-0.01,-0.002,0.002,0.01,0.05,0.3]

    # CLEV_01 = CLEV_100/1000
    # CLEV_01_LABELS = CLEV_100_LABELS/1000

    # - Linear - #
    CLEV_22 = [-22,-18,-14,-10,-6,-2,2,6,10,14,18,22]
    CLEV_11 = [-11,-9,-7,-5,-3,-1,1,3,5,7,9,11]
    CLEV_11_LABELS = [-9,-5,-1,1,5,9]

    if linear:
        if max_level==64:
            CLEV = CLEV_64
            CLEV_LABELS = CLEV_64_LABELS
    else:    
        if max_level==64:
            CLEV = CLEV_64
            CLEV_LABELS = CLEV_64_LABELS
        elif max_level==256:
            CLEV = CLEV_256
            CLEV_LABELS = CLEV_256_LABELS
        elif max_level==32:
            CLEV = CLEV_32
            CLEV_LABELS = CLEV_32_LABELS
        elif max_level==16:
            CLEV = CLEV_16
            CLEV_LABELS = CLEV_16_LABELS
        elif max_level==8:
            CLEV = CLEV_8
            CLEV_LABELS = CLEV_8_LABELS
        elif max_level==2:
            CLEV = CLEV_2
            CLEV_LABELS = CLEV_2_LABELS
        elif max_level==0.8:
            CLEV = CLEV_08
            CLEV_LABELS = CLEV_08_LABELS
        elif max_level==0.2:
            CLEV = CLEV_02
            CLEV_LABELS = CLEV_02_LABELS

    return CLEV, CLEV_LABELS
