"""Compare CORAL virtual-lidar temperature observations against PMAP model output.

Builds a six-panel figure (two columns x three rows) per model cube. The left column is the
CORAL measurement (``data/coral/20240718-2136_T2Z900.nc``) over a one-hour window; the right
column is the PMAP model cube sampled at a virtual-lidar location, the same diagnostic as
panels (b)/(c) of ``cube_lid.py`` but reorganised and switched to absolute temperature.

Per column, top to bottom:
  curtain  : absolute temperature, altitude vs time (pcolormesh)
  profiles : absolute-temperature time series, one line per altitude
  spectra  : normalised PSD of those per-altitude series, period in minutes

Two figures are produced (see ``CASES``): cube_0 sampled over Mount Darwin, and cube_1
(the Rio Grande / CORAL region) sampled at the column of strongest wave perturbation.

The profile altitudes, the highlighted level and the curtain altitude range are kept
identical to ``cube_lid.py`` so the two columns are directly comparable.
"""

import sys
import time
import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import signal
from scipy.ndimage import uniform_filter1d, maximum_filter1d, minimum_filter1d
from scipy.interpolate import RegularGridInterpolator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from cmcrameri import cm

plt.style.use("/work/bd0620/b309199/mapy/src/latex_default.mplstyle")

MODEL_SIM = "darwin_240718_400m_r1"
MODEL_ROOT = Path("/scratch/b/b309199")
OBS_FILE = Path("/work/bd0620/b309199/mapy/data/coral/20240718-2136_T2Z900.nc")
OUTPUT_DIR = Path("/work/bd0620/b309199/mapy/data/figures")

# One figure per entry. lidar_xy = fixed (x, y) column, or None to auto-pick the best-matching column:
# a persistent warm phase in search_zband carrying a fast (~3-5 min) oscillation (see find_lidar_column).
CASES = (
    {"cube": 0, "label": "PMAP (Mt Darwin)", "lidar_xy": (12600.0, -89800.0), "search_zband": (55000.0, 60000.0),
     "out": "lidar_obs_model_compare_darwin.png"},  # pinned to the preferred 57 km column (set None to auto-pick)
    {"cube": 1, "label": "PMAP (Rio Grande)", "lidar_xy": None, "search_zband": (56000.0, 60000.0),
     "out": "lidar_obs_model_compare_coral.png"},
)

THETA_NAME = "theta_total"
EXNER_NAME = "exner_total"

LIDAR_PROFILE_ALTITUDES = tuple(55000.0 + 1000.0 * k for k in range(8))   # 55-62 km, 1 km (on the 200 m grid)
CURTAIN_ZLIM_KM = (51.0, 67.0)

MODEL_WINDOW_DURATION_S = 3600.0   # model window = the LAST hour of available cube data (anchored to the
                                   # last time step, going 1 h back), matching the 1 h obs window
PTP_WINDOW_MIN = 4.0               # sliding window [min] for the max single swing (peak-to-peak). Short, so
                                   # the FAST ~3-5 min obs-like oscillations win over slow ~12 min drifts;
                                   # used for BOTH the lidar-column selection and the profile annotation.
FAST_OSC_PCT = 85.0                # lidar-column pick: require a genuine fast (~3-5 min) oscillation (fast
                                   # swing in the top (100-this)% of columns), THEN take the WARMEST of
                                   # those -> a warm persistent phase like the obs, carrying a fast swing.
OBS_WINDOW_UTC = ("2024-07-19T00:00", "2024-07-19T01:00")

# Column search for the None lidar_xy case: strongest |T'| over this band and (strided) time.
SEARCH_ZBAND_M = (min(LIDAR_PROFILE_ALTITUDES), max(LIDAR_PROFILE_ALTITUDES))
SEARCH_TIME_STRIDE = 3

# Match the model to CORAL's sampling before analysis. REGRID puts the model on CORAL's exact
# 30 s / 100 m grid first, then applies the same 2-min / 900-m running average, so the two share
# grid AND effective resolution; without REGRID the running average is done on the native grid.
MATCH_LIDAR_SAMPLING = True
MATCH_LIDAR_REGRID = True
LIDAR_TIME_RES_S = 120.0
LIDAR_VERT_RES_M = 900.0
LIDAR_GRID_TIME_S = 30.0
LIDAR_GRID_VERT_M = 100.0

TEMP_CLIM = (195.0, 270.0)
TEMP_CMAP = "turbo"
TEMP_CBAR_TICK_STEP = 15.0

ALTITUDE_CMAP = cm.batlow
PROFILE_LW = 0.9

PERIOD_XLIM_MIN = (2.0, 60.0)
PERIOD_XTICKS_MIN = (2, 3, 5, 10, 20, 30, 60)
PERIOD_FREQ_TICKS_MHZ = (0.5, 1, 2, 4, 8)
FREQ_XLIM_MHZ = (1.0e3 / (20.0 * 60.0), 6.0)  # cut at period 20 min (0.83 mHz) .. ~2.8 min (6 mHz)
# "period_log": log period (min) primary, frequency (mHz) twin on top.
# "freq_linear": linear frequency (mHz) primary, period (min) twin on top — long periods crowd
# into the low-frequency left edge (de-emphasised). Output filename gets a "_freqlin" suffix.
SPECTRUM_XAXIS = "freq_linear"
PSD_DETREND = "linear"

# Panel-corner label placement (matches pmap_slc_lid.py / cube_lid.py). XLBL is the inset of
# the descriptive top-left label from the LEFT spine (drawn with the default ha='left'). The
# panel-letter label sits top-right with ha='right', so its x is measured from the RIGHT spine:
# set XPP = 1 - XLBL so both corners keep the same visual margin by default. Keep XPP tied to
# XLBL (do not hardcode a number) so changing one inset moves both corners symmetrically.
XLBL = 0.04
YPP = 0.93
XPP = 1.0 - XLBL
LABEL_BOX_ROUND = {"boxstyle": "round", "lw": 0.67, "facecolor": "white", "edgecolor": "black"}
LABEL_BOX_CIRCLE = {"boxstyle": "circle", "lw": 0.67, "facecolor": "white", "edgecolor": "black"}

FIGSIZE = (13.0, 11.0)
DPI = 150


def _hms_from_seconds(total_seconds, _pos=None):
    total_seconds = max(0, int(round(float(total_seconds))))
    return f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}"


def _period_min_to_freq_mhz(period_min):
    period_min = np.asarray(period_min, dtype=float)
    with np.errstate(divide="ignore"):
        return 1.0e3 / (period_min * 60.0)


def _freq_mhz_to_period_min(freq_mhz):
    freq_mhz = np.asarray(freq_mhz, dtype=float)
    with np.errstate(divide="ignore"):
        return 1.0e3 / (freq_mhz * 60.0)


def _configure_spectrum_xaxis(ax, freq_xlim=None):
    """Set up the primary + twin x-axes of a spectrum panel per SPECTRUM_XAXIS.

    freq_xlim overrides the linear-frequency range (mHz) in the freq_linear mode.
    """
    freq_xlim = FREQ_XLIM_MHZ if freq_xlim is None else freq_xlim
    if SPECTRUM_XAXIS == "freq_linear":
        ax.set_xscale("linear")
        ax.set_xlim(freq_xlim[1], freq_xlim[0])
        ax.set_xlabel("frequency / mHz")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        # period twin lives in the (finite) frequency coordinate so f=0 does not map to period=inf;
        # its ticks are placed at the frequencies of round period values and relabelled as periods.
        sec = ax.twiny()
        sec.set_xlim(ax.get_xlim())
        p_ticks = [p for p in PERIOD_XTICKS_MIN
                   if freq_xlim[0] <= _period_min_to_freq_mhz(p) <= freq_xlim[1]]
        sec.set_xticks([_period_min_to_freq_mhz(p) for p in p_ticks])
        sec.set_xticklabels([str(p) for p in p_ticks])
        sec.xaxis.set_minor_locator(mticker.NullLocator())
        sec.set_xlabel("period / min")
    else:
        ax.set_xscale("log")
        ax.set_xlim(*PERIOD_XLIM_MIN)
        ax.set_xlabel("period / min")
        ax.xaxis.set_major_locator(mticker.FixedLocator(PERIOD_XTICKS_MIN))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(p) for p in PERIOD_XTICKS_MIN]))
        sec = ax.secondary_xaxis("top", functions=(_period_min_to_freq_mhz, _freq_mhz_to_period_min))
        sec.set_xlabel("frequency / mHz")
        sec.xaxis.set_major_locator(mticker.FixedLocator(PERIOD_FREQ_TICKS_MHZ))
        sec.xaxis.set_minor_locator(mticker.NullLocator())
        sec.xaxis.set_major_formatter(mticker.FixedFormatter([str(f) for f in PERIOD_FREQ_TICKS_MHZ]))


def find_lidar_column(cube_index, zband=SEARCH_ZBAND_M):
    """Pick the virtual-lidar column that best matches the CORAL-obs picture.

    Priorities (in order): (1) a PERSISTENT WARM PHASE in the band (56-60 km) — gate to columns whose
    warmest band-level time-mean T is in the top (100-WARM_PHASE_PCT)%; (2) a FAST oscillation like
    the obs (~3-5 min) — scored by the max temperature range within a short PTP_WINDOW_MIN sliding
    window, which gives a fast ~3 min swing its full amplitude but truncates a slow ~12 min drift;
    (3) amplitude — maximised as the tie-breaker among warm columns. Window = last hour of cube data.
    """
    path = MODEL_ROOT / MODEL_SIM / f"cube_{cube_index}.nc"
    ds = xr.open_dataset(path, decode_times=False)
    x = np.asarray(ds["x"].values, dtype=float)
    y = np.asarray(ds["y"].values, dtype=float)
    z = np.asarray(ds["z"].values, dtype=float)
    kband = np.where((z >= zband[0]) & (z <= zband[1]))[0]
    times = np.asarray(ds["time"].values, dtype=float)
    win = np.where(times >= (float(np.nanmax(times)) - MODEL_WINDOW_DURATION_S))[0]

    T = (ds[THETA_NAME].isel(time=win, z=kband) * ds[EXNER_NAME].isel(time=win, z=kband))
    T = T.transpose("x", "y", "z", "time").values          # (nx, ny, nband, nwin)
    ds.close()

    nwin = max(3, int(round(PTP_WINDOW_MIN / 60.0 * T.shape[3])))       # samples in PTP_WINDOW_MIN (1 h series)
    fast_band = (maximum_filter1d(T, nwin, axis=3, mode="nearest")
                 - minimum_filter1d(T, nwin, axis=3, mode="nearest")).max(axis=3)   # (nx, ny, nband)
    mean_T = T.mean(axis=3)                                # (nx, ny, nband): time-mean per level
    warm = mean_T.max(axis=2)                              # (nx, ny): warmest band-level mean T (warm phase)
    fast_col = fast_band.max(axis=2)                       # (nx, ny): fast swing amplitude

    # Among columns that HAVE a strong fast (~3-5 min) oscillation, pick the WARMEST -> the warm persistent
    # phase matches the obs (~250-255 K) and it still carries a fast swing. (Maximising the swing instead
    # gives a colder column; the surviving amplitude is the resolution-limited residual, expected to rise
    # at 200 m.)
    gate = fast_col >= np.percentile(fast_col, FAST_OSC_PCT)
    ix, iy = np.unravel_index(np.argmax(np.where(gate, warm, -np.inf)), warm.shape)
    z_warm = float(z[kband[int(mean_T[ix, iy].argmax())]])
    return (float(x[ix]), float(y[iy])), float(fast_col[ix, iy]), z_warm, float(warm[ix, iy])


def load_model_lidar(cube_index, lidar_xy):
    """Absolute temperature at a virtual-lidar column of the model cube.

    Returns the raw cube time coordinate (seconds, used as a clock for the model-time axis), the
    height grid (m) and a (time, z) temperature field, restricted to the run after
    ``MODEL_TIME_OFFSET_MIN``.
    """
    path = MODEL_ROOT / MODEL_SIM / f"cube_{cube_index}.nc"
    ds = xr.open_dataset(path, decode_times=False)

    x = np.asarray(ds["x"].values, dtype=float)
    y = np.asarray(ds["y"].values, dtype=float)
    ix = int(np.argmin(np.abs(x - lidar_xy[0])))
    iy = int(np.argmin(np.abs(y - lidar_xy[1])))

    theta = ds[THETA_NAME].isel(x=ix, y=iy)
    exner = ds[EXNER_NAME].isel(x=ix, y=iy)
    temperature = (theta * exner).transpose("time", "z").values

    times = np.asarray(ds["time"].values, dtype=float)
    z = np.asarray(ds["z"].values, dtype=float)
    ds.close()

    keep = times >= (float(np.nanmax(times)) - MODEL_WINDOW_DURATION_S)
    return times[keep], z, temperature[keep]


def load_obs_lidar():
    """Absolute temperature from the CORAL file over ``OBS_WINDOW_UTC``.

    Returns the profile times as python datetimes (UTC), the height grid (m above the station) and
    a (time, z) temperature field with the file's 0-fill set to NaN.
    """
    ds = xr.open_dataset(OBS_FILE, decode_times=False)
    time_offset = float(ds["time_offset"].values[0])
    station_height = float(ds["station_height"].values[0])
    altitude_offset = float(ds["altitude_offset"].values[0])

    unix_s = time_offset + np.asarray(ds["time"].values, dtype=float) / 1000.0
    z = np.asarray(ds["altitude"].values, dtype=float) + altitude_offset + station_height

    temperature = np.asarray(ds["temperature"].values, dtype=float)
    temperature = np.where(temperature == 0.0, np.nan, temperature)
    ds.close()

    epoch = dt.datetime(1970, 1, 1)
    w0 = (dt.datetime.fromisoformat(OBS_WINDOW_UTC[0]) - epoch).total_seconds()
    w1 = (dt.datetime.fromisoformat(OBS_WINDOW_UTC[1]) - epoch).total_seconds()
    keep = (unix_s >= w0) & (unix_s <= w1)

    times = np.array([epoch + dt.timedelta(seconds=float(s)) for s in unix_s[keep]])
    return times, z, temperature[keep]


def altitude_indices(z):
    return [int(np.argmin(np.abs(z - a))) for a in LIDAR_PROFILE_ALTITUDES]


def compute_psd(series, dt_seconds):
    """Variance-normalised one-sided PSD of a (possibly gappy) temperature time series.

    The mean and a linear trend are removed and a Hann window applied before the FFT, so the
    spectrum reflects the oscillations rather than the ~230 K background. Each spectrum is then
    divided by its own integral, so it integrates to one over frequency (Parseval: the integral
    of a PSD equals the signal variance, hence "variance normalisation"). That compares the
    *shape* / dominant period across altitudes independently of the oscillation amplitude —
    unlike peak-normalisation, which pins every curve to 1 at its own maximum and so hides both
    amplitude and how sharply peaked the spectrum is. Frequency is kept in cycles-per-minute so
    the density stays O(1-30) and pairs naturally with the period (min) axis.
    (Caveat: on the log-period x-axis the eye integrates over d(ln f), not df, so "equal area =
    equal variance" is exact only on a linear-frequency axis; the ranking of peaks still holds.)
    """
    y = np.asarray(series, dtype=float)
    idx = np.arange(y.size)
    good = np.isfinite(y)
    if good.sum() < 8:
        return np.array([]), np.array([])
    y = np.interp(idx, idx[good], y[good])

    y = signal.detrend(y, type=PSD_DETREND)
    window = signal.windows.hann(y.size)
    spectrum = np.fft.rfft(y * window)
    freq_cpm = np.fft.rfftfreq(y.size, d=dt_seconds) * 60.0

    df_cpm = 60.0 / (y.size * dt_seconds)
    nonzero = freq_cpm > 0
    psd = np.abs(spectrum[nonzero]) ** 2
    period_min = 1.0 / freq_cpm[nonzero]
    area = psd.sum() * df_cpm
    if area > 0:
        psd = psd / area
    return period_min, psd


def resample_like_lidar(temperature, z, dt_seconds):
    """Box-car average a model (time, z) field over CORAL's effective resolution, on its own grid.

    Running average only (used when MATCH_LIDAR_REGRID is False): keeps the native cube grid but
    makes each point a 2-min / 900-m running mean.
    """
    dz = float(np.median(np.diff(z)))
    n_time = max(1, int(round(LIDAR_TIME_RES_S / dt_seconds)))
    n_vert = max(1, int(round(LIDAR_VERT_RES_M / dz)))
    smoothed = uniform_filter1d(temperature, size=n_time, axis=0, mode="nearest")
    smoothed = uniform_filter1d(smoothed, size=n_vert, axis=1, mode="nearest")
    return smoothed


def match_lidar_grid_and_sampling(temperature, times, z):
    """Put a model (time, z) column on CORAL's exact 30 s / 100 m grid, then apply the same
    2-min / 900-m running-average oversampling.

    This matches the model to the observation in BOTH grid spacing and effective resolution (not
    just resolution): the model is bilinearly interpolated onto the CORAL grid and then box-car
    averaged over 4 samples in time (2 min) x 9 samples in height (900 m), reproducing CORAL's
    ~4x-in-time / ~9x-in-vertical oversampling. Returns (temperature, times, z) on the new grid.

    Cost: the regrid + running average act on a ~O(200x190) array (a few ms); run time is dominated
    by reading the cube column from disk, so this step adds essentially nothing.
    """
    target_t = np.arange(times[0], times[-1] + 1e-6, LIDAR_GRID_TIME_S)
    z_lo = np.ceil(z.min() / LIDAR_GRID_VERT_M) * LIDAR_GRID_VERT_M
    z_hi = np.floor(z.max() / LIDAR_GRID_VERT_M) * LIDAR_GRID_VERT_M
    target_z = np.arange(z_lo, z_hi + 1e-6, LIDAR_GRID_VERT_M)

    interp = RegularGridInterpolator((times, z), temperature, bounds_error=False, fill_value=None)
    grid_t, grid_z = np.meshgrid(target_t, target_z, indexing="ij")
    grid = interp((grid_t, grid_z))

    n_time = max(1, int(round(LIDAR_TIME_RES_S / LIDAR_GRID_TIME_S)))
    n_vert = max(1, int(round(LIDAR_VERT_RES_M / LIDAR_GRID_VERT_M)))
    grid = uniform_filter1d(grid, size=n_time, axis=0, mode="nearest")
    grid = uniform_filter1d(grid, size=n_vert, axis=1, mode="nearest")
    return grid, target_t, target_z


def draw_column(fig, axes, time_numeric, z, temperature, line_colors,
                dt_seconds, time_formatter, time_label, corner_label):
    """Render the curtain / profiles / spectra trio for one data source."""
    ax_curtain, ax_lines, ax_psd = axes
    z_km = z / 1000.0
    z_sel = (z_km >= CURTAIN_ZLIM_KM[0]) & (z_km <= CURTAIN_ZLIM_KM[1])

    pcm = ax_curtain.pcolormesh(
        time_numeric, z_km[z_sel], temperature[:, z_sel].T,
        cmap=TEMP_CMAP, vmin=TEMP_CLIM[0], vmax=TEMP_CLIM[1], shading="nearest",
    )
    ax_curtain.set_ylim(*CURTAIN_ZLIM_KM)
    ax_curtain.set_ylabel("altitude / km")
    ax_curtain.text(XLBL, YPP, corner_label, transform=ax_curtain.transAxes,
                    weight="bold", bbox=LABEL_BOX_ROUND, zorder=6)
    ax_curtain.grid(True, alpha=0.2)

    if SPECTRUM_XAXIS == "period_log":
        period_lo, period_hi = PERIOD_XLIM_MIN
    else:
        period_lo = _freq_mhz_to_period_min(FREQ_XLIM_MHZ[1])
        period_hi = _freq_mhz_to_period_min(FREQ_XLIM_MHZ[0])

    psd_visible_max = 0.0
    for iz, color in zip(altitude_indices(z), line_colors):
        ax_lines.plot(time_numeric, temperature[:, iz], color=color, lw=PROFILE_LW, zorder=2)
        ax_curtain.axhline(z[iz] / 1000.0, ls="--", lw=0.7, color=color, alpha=0.9)

        period_min, psd = compute_psd(temperature[:, iz], dt_seconds)
        if period_min.size:
            psd_x = period_min if SPECTRUM_XAXIS == "period_log" else _period_min_to_freq_mhz(period_min)
            ax_psd.plot(psd_x, psd, color=color, lw=PROFILE_LW, zorder=2)
            visible = (period_min >= period_lo) & (period_min <= period_hi)
            if visible.any():
                psd_visible_max = max(psd_visible_max, float(psd[visible].max()))

    ax_lines.set_ylabel("temperature / K")
    ax_lines.grid(True, alpha=0.25)
    ax_lines.xaxis.set_major_formatter(time_formatter)
    ax_lines.set_xlabel(time_label)
    plt.setp(ax_curtain.get_xticklabels(), visible=False)

    ax_psd.set_ylabel("normalised PSD")
    ax_psd.grid(True, which="both", alpha=0.25)
    _configure_spectrum_xaxis(ax_psd)
    if psd_visible_max > 0:
        ax_psd.set_ylim(0, 1.05 * psd_visible_max)
    else:
        ax_psd.set_ylim(bottom=0)

    return pcm


def annotate_peak_to_peak(ax, ax_psd, time_numeric, z, temperature, line_colors, dt_min, n_mark=2):
    """Mark the n_mark largest distinct peak-to-peak T swings and label each amplitude + timing.

    For each swing: peak (v) and trough (^) markers sit at their actual times, a vertical double-arrow
    spans the swing, and the label carries dT_pp, the altitude, and dt (the peak->trough time in min).
    A dashed line in the PSD panel (ax_psd) marks the corresponding wave PERIOD = 2 x dt (the peak->trough
    time is a half period), coloured to match the profile line the swing belongs to. `dt_min` is the
    per-sample spacing in minutes. Successive swings are forced to be temporally distinct (the time
    neighbourhood of an accepted swing is masked before searching for the next).
    """
    izs = altitude_indices(z)
    n = temperature.shape[0]
    nwin = max(3, int(round(PTP_WINDOW_MIN / 60.0 * n)))   # ~PTP_WINDOW_MIN sliding window over the 1 h series
    nw = n - nwin + 1
    pp = np.full((len(izs), nw), -np.inf)                  # windowed peak-to-peak per (altitude, window start)
    ext = np.zeros((len(izs), nw, 2), int)                 # (imax, imin) sample indices per window
    for jz, iz in enumerate(izs):
        s_ = temperature[:, iz]
        for i0 in range(nw):
            seg = s_[i0:i0 + nwin]
            if np.isfinite(seg).sum() < 2:
                continue
            pp[jz, i0] = float(np.nanmax(seg) - np.nanmin(seg))
            ext[jz, i0] = (i0 + int(np.nanargmax(seg)), i0 + int(np.nanargmin(seg)))

    used = np.zeros(n, bool)                               # time samples already claimed by a marked swing
    out = []
    for _ in range(n_mark):
        m = pp.copy()
        for i0 in range(nw):
            if used[i0:i0 + nwin].any():
                m[:, i0] = -np.inf
        if not np.isfinite(m).any():
            break
        jz, i0 = np.unravel_index(int(np.argmax(m)), m.shape)
        iz = izs[jz]
        color = line_colors[jz]
        imax, imin = int(ext[jz, i0, 0]), int(ext[jz, i0, 1])
        ptp = float(pp[jz, i0])
        dt = abs(imax - imin) * dt_min                     # peak->trough time [min]
        period = 2.0 * dt                                  # full wave period [min]
        s = temperature[:, iz]
        ax.plot(time_numeric, s, color=color, lw=1.6, zorder=4)
        ax.plot(time_numeric[imax], s[imax], "v", color=color, mec="k", mew=0.6, ms=8, zorder=6)
        ax.plot(time_numeric[imin], s[imin], "^", color=color, mec="k", mew=0.6, ms=8, zorder=6)
        ax.annotate("", xy=(time_numeric[imax], s[imax]), xytext=(time_numeric[imax], s[imin]),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.1), zorder=6)
        # label BELOW the trough (min) so it clears the other profile lines; box edge = the line colour
        ax.annotate(f"$\\Delta T_{{pp}}$ = {ptp:.0f} K @ {z[iz] / 1000.0:.0f} km\n"
                    f"$\\Delta t$ = {dt:.1f} min",
                    xy=(time_numeric[imin], s[imin]), xytext=(0, -9), textcoords="offset points",
                    va="top", ha="center", fontsize=7.5, zorder=7,
                    bbox=dict(boxstyle="round", fc="white", ec=color, lw=1.0, alpha=0.9))
        # dashed line in the PSD panel at the corresponding period (= 2 x peak->trough time)
        ax_psd.axvline(_period_min_to_freq_mhz(period), ls="--", lw=1.2, color=color, alpha=0.9, zorder=2)
        lo, hi = max(0, min(imax, imin) - nwin), min(n, max(imax, imin) + nwin)
        used[lo:hi] = True
        out.append((ptp, z[iz] / 1000.0, dt, period))
    return out


def add_panel_labels(axes_grid):
    # axes_grid order is [top-left, top-right, mid-left, mid-right, bottom-left, bottom-right];
    # label row-major (a b / c d / e f) so the top-right panel is 'b'.
    for label, ax in zip("abcdef", axes_grid):
        ax.text(XPP, YPP, label, transform=ax.transAxes, ha="right", weight="bold",
                bbox=LABEL_BOX_CIRCLE, zorder=6)


def build_figure(cube_index, lidar_xy, model_label):
    t0 = time.time()
    m_times, m_z, m_temp = load_model_lidar(cube_index, lidar_xy)
    t_load = time.time() - t0
    o_times, o_z, o_temp = load_obs_lidar()

    t1 = time.time()
    if MATCH_LIDAR_SAMPLING and MATCH_LIDAR_REGRID:
        m_temp, m_times, m_z = match_lidar_grid_and_sampling(m_temp, m_times, m_z)
    elif MATCH_LIDAR_SAMPLING:
        m_temp = resample_like_lidar(m_temp, m_z, float(np.median(np.diff(m_times))))
    t_match = time.time() - t1
    print(f"  model load {t_load:.2f} s | grid-match+oversample {t_match * 1e3:.1f} ms")

    m_x = m_times
    o_x = mdates.date2num(o_times)
    m_dt = float(np.median(np.diff(m_times)))
    o_dt = float(np.median(np.diff([t.timestamp() for t in o_times])))

    alt_min, alt_max = min(LIDAR_PROFILE_ALTITUDES), max(LIDAR_PROFILE_ALTITUDES)
    alt_norm = Normalize(vmin=alt_min / 1000.0, vmax=alt_max / 1000.0)
    line_colors = [ALTITUDE_CMAP(alt_norm(a / 1000.0)) for a in LIDAR_PROFILE_ALTITUDES]

    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    axL = [fig.add_subplot(gs[0, 0]), None, None]
    axL[1] = fig.add_subplot(gs[1, 0], sharex=axL[0])
    axL[2] = fig.add_subplot(gs[2, 0])
    axR = [fig.add_subplot(gs[0, 1]), None, None]
    axR[1] = fig.add_subplot(gs[1, 1], sharex=axR[0])
    axR[2] = fig.add_subplot(gs[2, 1])

    obs_fmt = mdates.DateFormatter("%H:%M")
    model_fmt = mticker.FuncFormatter(_hms_from_seconds)

    obs_label = "CORAL (" + dt.datetime.fromisoformat(OBS_WINDOW_UTC[0]).strftime("%d %b %Y") + ")"
    draw_column(fig, axL, o_x, o_z, o_temp, line_colors, o_dt,
                obs_fmt, "time (UTC)", obs_label)
    pcm = draw_column(fig, axR, m_x, m_z, m_temp, line_colors, m_dt,
                      model_fmt, "model time", model_label)

    line_lims = [axL[1].get_ylim(), axR[1].get_ylim()]
    shared_line_ylim = (min(l[0] for l in line_lims), max(l[1] for l in line_lims))
    axL[1].set_ylim(*shared_line_ylim)
    axR[1].set_ylim(*shared_line_ylim)

    o_res = annotate_peak_to_peak(axL[1], axL[2], o_x, o_z, o_temp, line_colors, o_dt / 60.0)
    m_res = annotate_peak_to_peak(axR[1], axR[2], m_x, m_z, m_temp, line_colors, m_dt / 60.0)
    for tag, res in (("obs", o_res), ("model", m_res)):
        for k, (ptp, alt, dt_pt, period) in enumerate(res, 1):
            print(f"  {tag} swing {k}: dT_pp {ptp:.0f} K @ {alt:.0f} km | dt {dt_pt:.1f} min -> period {period:.1f} min")

    # colorbar WIDTH = length / aspect. The altitude bar spans 2 rows (shrink 0.7) vs the temperature
    # bar's 1 row (shrink 0.9), i.e. ~1.56x longer, so its aspect is scaled by the same factor to give
    # both the SAME width.
    cbar_t = fig.colorbar(pcm, ax=[axL[0], axR[0]], location="right", shrink=0.9, aspect=30,
                          pad=0.015, extend="both",
                          ticks=np.arange(TEMP_CLIM[0], TEMP_CLIM[1] + 1, TEMP_CBAR_TICK_STEP))
    cbar_t.set_label("temperature / K")

    sm = ScalarMappable(norm=alt_norm, cmap=ALTITUDE_CMAP)
    cbar_a = fig.colorbar(sm, ax=[axL[1], axR[1], axL[2], axR[2]], location="right",
                          shrink=0.7, aspect=47, pad=0.015,
                          ticks=[a / 1000.0 for a in LIDAR_PROFILE_ALTITUDES])
    cbar_a.set_label("profile altitude / km")
    cbar_a.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    add_panel_labels([axL[0], axR[0], axL[1], axR[1], axL[2], axR[2]])

    return fig


def main():
    # usage: lidar_obs_model_compare.py [SIM] [raw|matched]
    #   raw     -> analyse the model on its native cube grid (no lidar smoothing) -> full amplitude
    #   matched -> smooth/regrid the model to CORAL's 2 min / 900 m resolution (default; obs-comparable)
    global MODEL_SIM, MATCH_LIDAR_SAMPLING
    if len(sys.argv) > 1:
        MODEL_SIM = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2].lower() in ("raw", "native", "unmatched"):
        MATCH_LIDAR_SAMPLING = False
    mode = "raw" if not MATCH_LIDAR_SAMPLING else "matched"
    print(f"model sim: {MODEL_SIM}  |  model analysis: {mode} "
          f"({'native cube grid' if mode == 'raw' else 'smoothed to CORAL 2 min/900 m'})")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        t0 = time.time()
        lidar_xy = case["lidar_xy"]
        print(f"cube {case['cube']} -> {case['out']}")
        if lidar_xy is None:
            lidar_xy, ptp, z_warm, warm_T = find_lidar_column(case["cube"], case["search_zband"])
            print(f"  warmest fast-oscillation column at (x, y) = ({lidar_xy[0]:.0f}, {lidar_xy[1]:.0f}) m, "
                  f"warm-phase {warm_T:.0f} K @ {z_warm / 1e3:.1f} km, raw fast dT_pp = {ptp:.1f} K  "
                  f"[search {time.time() - t0:.1f} s]")

        model_label = case["label"] + (" - raw" if mode == "raw" else "")
        fig = build_figure(case["cube"], lidar_xy, model_label)
        suffix = ("" if SPECTRUM_XAXIS == "period_log" else "_freqlin") + ("_rawmodel" if mode == "raw" else "")
        out = OUTPUT_DIR / f"{MODEL_SIM}_{case['out'].replace('.png', f'{suffix}.png')}"
        fig.savefig(out, dpi=DPI, facecolor="w", bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}  [{time.time() - t0:.1f} s total]")


if __name__ == "__main__":
    main()
