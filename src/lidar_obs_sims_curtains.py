#!/usr/bin/env python
"""Six-panel CORAL vs virtual-lidar curtain comparison for the 18 July 2024 case.

Columns (left to right):
  a/d : CORAL measurement (``data/coral/20240718-2136_T2Z900.nc``)
  b/e : 800 m nested run, virtual lidar at CORAL (slices_x, y=0; SAM frame -> CORAL = origin).
        The panel fills itself with whatever the (still running) simulation has written;
        while the slices file is missing the panel is left empty with a note.
  c/f : 4 km big-brother run (darwin_dear_moist), virtual lidar at CORAL (slices_x, y=0)

Rows: absolute temperature (top) and the temperature perturbation T' from a vertical
Butterworth filter with 20 km cutoff (bottom, same filter as lidar_processor.calculate_primes),
so only the wave part remains.

All panels share one UTC time axis spanning the full measurement (model curtains cover only
their simulated hours; model time 0 = 2024-07-18T20:00 UTC for both runs). The two colorbars
(temperature, perturbation) sit on the right of their row at 0.8x row height.

Run on JUPITER (site-aware paths also resolve on Levante):
    module load Stages/2026 GCC Python/3.13.5 netCDF
    cd .../mapy/src && .../venvs/post-venv/bin/python lidar_obs_sims_curtains.py
"""

import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import cmaps
from filter import butterworth_filter

HERE = Path(__file__).resolve().parent
plt.style.use(HERE / "latex_default.mplstyle")

ON_JUPITER = Path("/e/project1/gwturb").is_dir()
WORK = Path("/e/project1/gwturb/binder5") if ON_JUPITER else Path("/work/bd0620/b309199")
SCRATCH = Path("/e/scratch/gwturb/binder5") if ON_JUPITER else Path("/scratch/b/b309199")

OBS_FILE = WORK / "mapy/data/coral/20240718-2136_T2Z900.nc"
MODEL_T0_UTC = dt.datetime(2024, 7, 18, 20, 0)   # model time 0 of BOTH runs (bb and nest)
SIMS = (   # (panel label, slices file with the x=0 yz-slice through CORAL)
    ("PMAP 800 m nest", SCRATCH / "darwin_240718_800m_nest/slices_x.nc"),
    ("PMAP 4 km", SCRATCH / "darwin_dear_moist/slices_x.nc"),
)

VERT_CUTOFF_KM = 20.0        # vertical Butterworth cutoff wavelength
BW_ORDER = 5
DZ_UNIFORM_M = 100.0         # model profiles go on this uniform grid = CORAL's native 100 m
ZLIM_KM = (25.0, 80.0)

TEMP_CLIM = (190.0, 280.0)
TEMP_CMAP = "turbo"
PERT_CLIM = (-20.0, 20.0)
PERT_CMAP = cmaps.get_wave_cmap()
CBAR_SHRINK = 0.8            # colorbar length as fraction of its row height

XLIM_UTC = (MODEL_T0_UTC, dt.datetime(2024, 7, 19, 4, 0))   # None -> that end of the measurement

# panel-corner label placement (same convention as lidar_obs_model_compare.py)
XLBL, YPP = 0.03, 0.92
XPP = 1.0 - XLBL
LABEL_BOX_ROUND = {"boxstyle": "round", "lw": 0.67, "facecolor": "white", "edgecolor": "black"}
LABEL_BOX_CIRCLE = {"boxstyle": "circle", "lw": 0.67, "facecolor": "white", "edgecolor": "black"}

FIGSIZE = (15.0, 9.0)
DPI = 150
OUT = WORK / "mapy/data/figures/coral_obs_sims_curtains_240718.png"


def vertical_primes(temp, dz_m):
    """T' of a (time, z) field: vertical Butterworth, cutoff VERT_CUTOFF_KM (km-domain units)."""
    tprime, _ = butterworth_filter(temp, cutoff=1.0 / VERT_CUTOFF_KM,
                                   fs=1.0 / (dz_m / 1000.0), order=BW_ORDER, mode="both")
    return tprime


def load_obs():
    """CORAL absolute temperature: (times UTC, z m, T(time, z)) with the 0-fill set to NaN."""
    ds = xr.open_dataset(OBS_FILE, decode_times=False)
    unix_s = float(ds["time_offset"].values[0]) + np.asarray(ds["time"].values, float) / 1000.0
    z = (np.asarray(ds["altitude"].values, float)
         + float(ds["altitude_offset"].values[0]) + float(ds["station_height"].values[0]))
    temp = np.asarray(ds["temperature"].values, float)
    ds.close()
    temp = np.where(temp == 0.0, np.nan, temp)
    times = np.array([dt.datetime(1970, 1, 1) + dt.timedelta(seconds=float(s)) for s in unix_s])
    return times, z, temp


def load_virtual_lidar(slices_path):
    """Virtual lidar at CORAL (x~0 slice, y=0 column) of a run's slices_x file.

    Returns (times UTC, z m uniform, T(time, z)) with T interpolated from the terrain-following
    zcr onto a uniform DZ_UNIFORM_M grid (equidistant, as the Butterworth filter requires).
    Half-written trailing records of an aborted run (zeros) come back as NaN.
    """
    ds = xr.open_dataset(slices_path, decode_times=False)
    iy = int(np.argmin(np.abs(np.asarray(ds["y"].values, float))))
    col = dict(x=0, y=iy)
    print(f"{slices_path.parent.name}: virtual lidar at x = {float(ds['x'][0]) / 1000.0:g} km, "
          f"y = {float(ds['y'][iy]) / 1000.0:g} km (SAM frame, CORAL = origin)")
    temp = (ds["theta_total"].isel(**col) * ds["exner_total"].isel(**col))
    temp = temp.transpose("time", "z").values
    zcr = ds["zcr"].isel(**col)
    zcr = zcr.transpose("time", "z").values if "time" in zcr.dims else \
        np.broadcast_to(np.asarray(zcr.values, float), temp.shape)
    seconds = np.asarray(ds["time"].values, float)
    ds.close()

    written = np.isfinite(seconds)                            # a running sim preallocates records
    seconds, temp, zcr = seconds[written], temp[written], zcr[written]   # (NaN time = not written)
    temp = np.where(temp < 50.0, np.nan, temp)                # zero-filled records -> NaN
    zu = np.arange(0.0, np.nanmax(zcr) + 1e-6, DZ_UNIFORM_M)
    temp_u = np.full((temp.shape[0], zu.size), np.nan)
    for it in range(temp.shape[0]):
        good = np.isfinite(temp[it]) & np.isfinite(zcr[it])
        if good.sum() >= 10:
            temp_u[it] = np.interp(zu, zcr[it][good], temp[it, good])
    times = np.array([MODEL_T0_UTC + dt.timedelta(seconds=float(s)) for s in seconds])
    keep = np.isfinite(temp_u).any(axis=1)
    return times[keep], zu, temp_u[keep]


def draw_pair(ax_temp, ax_pert, times, z, temp, dz_m):
    """Temperature curtain (top) + vertical-BW T' curtain (bottom) for one source."""
    tprime = vertical_primes(temp, dz_m)
    z_km = z / 1000.0
    zsel = (z_km >= ZLIM_KM[0] - 1.0) & (z_km <= ZLIM_KM[1] + 1.0)
    x = mdates.date2num(times)
    ax_temp.pcolormesh(x, z_km[zsel], temp[:, zsel].T, cmap=TEMP_CMAP,
                       vmin=TEMP_CLIM[0], vmax=TEMP_CLIM[1], shading="nearest", rasterized=True)
    ax_pert.pcolormesh(x, z_km[zsel], tprime[:, zsel].T, cmap=PERT_CMAP,
                       vmin=PERT_CLIM[0], vmax=PERT_CLIM[1], shading="nearest", rasterized=True)


def main():
    o_times, o_z, o_temp = load_obs()

    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, sharex=True, sharey=True,
                             constrained_layout=True)

    draw_pair(axes[0, 0], axes[1, 0], o_times, o_z, o_temp,
              float(np.median(np.diff(o_z))))
    obs_label = f"CORAL ({o_times[0]:%d %b %Y})"
    col_labels = [obs_label]

    for j, (label, path) in enumerate(SIMS, start=1):
        col_labels.append(label)
        if not path.exists():
            axes[0, j].text(0.5, 0.5, f"{label}\n(simulation running)", transform=axes[0, j].transAxes,
                            ha="center", va="center", color="grey")
            continue
        m_times, m_z, m_temp = load_virtual_lidar(path)
        if m_times.size == 0:
            axes[0, j].text(0.5, 0.5, f"{label}\n(no data yet)", transform=axes[0, j].transAxes,
                            ha="center", va="center", color="grey")
            continue
        draw_pair(axes[0, j], axes[1, j], m_times, m_z, m_temp, DZ_UNIFORM_M)
        print(f"{label}: {m_times[0]:%H:%M} - {m_times[-1]:%H:%M} UTC "
              f"({(m_times[-1] - m_times[0]).total_seconds() / 3600.0:.1f} h simulated)")

    # shared UTC axis: measurement extent unless overridden (models show what they simulated)
    xlim = (XLIM_UTC[0] or o_times[0], XLIM_UTC[1] or o_times[-1])
    axes[0, 0].set_xlim(mdates.date2num(xlim[0]), mdates.date2num(xlim[1]))
    axes[0, 0].set_ylim(*ZLIM_KM)
    for ax in axes[1]:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
        ax.set_xlabel(f"time (UTC), {xlim[0]:%d}-{xlim[1]:%d %b %Y}")
    for ax in axes[:, 0]:
        ax.set_ylabel("altitude / km")
    for ax in axes.ravel():
        ax.grid(True)        # dotted grey, from latex_default.mplstyle

    for j, label in enumerate(col_labels):
        axes[0, j].text(XLBL, YPP, label, transform=axes[0, j].transAxes,
                        weight="bold", bbox=LABEL_BOX_ROUND, zorder=6)
        axes[1, j].text(XLBL, YPP, rf"$T'$ ($\lambda_z <$ {VERT_CUTOFF_KM:.0f} km)",
                        transform=axes[1, j].transAxes, bbox=LABEL_BOX_ROUND, zorder=6)
    for label, ax in zip("abcdef", axes.ravel()):
        ax.text(XPP, YPP, label, transform=ax.transAxes, ha="right", weight="bold",
                bbox=LABEL_BOX_CIRCLE, zorder=6)

    cbar_t = fig.colorbar(ScalarMappable(Normalize(*TEMP_CLIM), TEMP_CMAP),
                          ax=axes[0, :].tolist(), location="right",
                          shrink=CBAR_SHRINK, pad=0.015, extend="both")
    cbar_t.set_label("temperature / K")
    cbar_p = fig.colorbar(ScalarMappable(Normalize(*PERT_CLIM), PERT_CMAP),
                          ax=axes[1, :].tolist(), location="right",
                          shrink=CBAR_SHRINK, pad=0.015, extend="both")
    cbar_p.set_label(r"$T'$ / K")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=DPI, facecolor="w")
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
