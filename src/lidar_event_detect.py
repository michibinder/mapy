"""Vortex-tube overpass detection in (virtual-)lidar T' curtains -- cube tuning/validation draft.

A vortex tube advected over the lidar produces a large T' swing within a short time window
(peak->trough ~5-8 min for a ~5-10 km structure drifting at ~20-30 m/s, see the cube_track results)
that is also vertically confined to a few km -- unlike the mountain-wave background whose vertical
wavelengths are tens of km. Detection combines the two constraints:

  (1) AMPLITUDE: the windowed peak-to-peak amplitude P(t, z) = max - min of T' within a sliding
      ``EVENT_WINDOW_MIN`` window must reach ``AMP_THRESHOLD_K`` = 25 K (candidate seed). NB the
      windowed max-min of T' equals the plain TEMPERATURE DIFFERENCE within the window up to the
      negligible drift of the background over 8 min -- so the criterion transfers to raw
      measured T(t, z) directly;
  (2) TEMPORAL confinement ("fast"): the peak->trough time of the swing must fall inside
      ``DT_PT_RANGE_MIN``. The time is min(same-level peak->trough, time between the max and the
      min of the 2D window x +/-``SEARCH_HALF_Z_KM`` box) -- the trough of a passing tube may sit
      on a neighbouring level. Slow drifts of the persistent warm breaking layer fill the whole
      window (peak/trough at the window edges) and are rejected here;
  (3) VERTICAL confinement ("confined"): the half-amplitude vertical extent (FWHM) of the WARM
      core anomaly at the overpass time must not exceed ``MAX_VERT_FWHM_KM`` (else wave phase,
      not tube). The event core is always the temperature MAXIMUM of the window (tubes are
      warm-cored, cf. the l2seg picture) -- cold minima are never circled.

Candidate seeds are extracted greedily from P(t, z) (largest first, suppressing the +/-window x
+/-``SUPPRESS_Z_KM`` neighbourhood of each accepted seed -- same philosophy as the distinct-swing
extraction in ``lidar_spectra_stats.py``), so one tube is counted once per column and chained
events in a long-lived layer stay separable. Events passing BOTH confinements are the VORTEX
candidates; rejected candidates are kept in the table/figures (flags) for threshold tuning.
Per event: core time & altitude, swing amplitude, peak->trough time (apparent half-period; 2x it
estimates the lidar-apparent period, cf. the l2seg 2d/v prediction) and the vertical FWHM (the
measurement-side tube-size estimate to validate against the l2seg peak-trough sizes).

Run on a PMAP cube it processes an ensemble of stride-sampled columns exactly like
``lidar_spectra_stats.py`` (obs-like 2-min/900-m smoothing, T' = T - horizontal cube mean per (z, t))
and writes ONE combined figure + the event table per cube:
  * ``<SIM>_lidar_events_cube<N>.png`` -- (a) example ABSOLUTE-temperature curtain (the virtual
    lidar view; column pinned per cube via ``EXAMPLE_XY_KM``, else the strongest-event column)
    with events outlined, over (b-e) ensemble histograms + FWHM-vs-swing-time scatter
  * ``<SIM>_lidar_events_cube<N>.csv``  -- the full event table
The detection core ``detect_events(tprime, times, z)`` is measurement-agnostic: the planned CORAL
extension only needs an obs loader (anomaly from the vertical-Butterworth background as in
``lidar_obs_model_compare.py`` for the FWHM; the amplitude/period criteria work on raw T) feeding
the same function.

Usage:  post-venv python lidar_event_detect.py [SIM] [AMP_THRESHOLD_K]   (cube ensemble mode)
        post-venv python lidar_event_detect.py coral [AMP_THRESHOLD_K]   (full CORAL night)
"""

import csv
import datetime
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage import (maximum_filter, maximum_filter1d, minimum_filter, minimum_filter1d,
                           uniform_filter1d)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import lidar_obs_model_compare as base

plt.style.use("/work/bd0620/b309199/mapy/src/latex_default.mplstyle")

# detection-mode keyword anywhere on the CLI (default "level") -- three DISTINCT approaches:
#   level = ONE altitude level: per-level windowed peak-trough of raw T (no background, no box;
#           the period is the same-level peak->trough time only) (alias "runmean")
#   bwf   = vertical high-pass: max-min of the Butterworth T' over the 10-min x 1-km box
#           (alias "tprime")
#   box   = RAW absolute T: max-min over the 10-min x 1-km box (the ~2-3 K stratification
#           across 1 km is accepted)
# the positional args are parsed from the remainder: MODE/SIM, AMP, HALF_Z, NIGHT.
_MODE_KEYWORDS = {"level": "level", "runmean": "level", "bwf": "bwf", "tprime": "bwf",
                  "box": "box"}
DETECT_MODE = next((_MODE_KEYWORDS[a] for a in sys.argv[1:] if a in _MODE_KEYWORDS), "level")
ARGS = [a for a in sys.argv[1:] if a not in _MODE_KEYWORDS]

MODEL_SIM = ARGS[0] if len(ARGS) > 0 else "darwin_240718_400m_coralT_ifs_wcoast"
MODEL_ROOT = base.MODEL_ROOT
OUTPUT_DIR = base.OUTPUT_DIR
THETA_NAME = base.THETA_NAME
EXNER_NAME = base.EXNER_NAME
MODEL_WINDOW_DURATION_S = base.MODEL_WINDOW_DURATION_S
LIDAR_TIME_RES_S = base.LIDAR_TIME_RES_S
LIDAR_VERT_RES_M = base.LIDAR_VERT_RES_M

CASES = (
    {"cube": 0, "label": "PMAP (Mt Darwin)"},
    {"cube": 1, "label": "PMAP (Rio Grande)"},
)

STRIDE = 3
MATCH_LIDAR_SAMPLING = True

EVENT_WINDOW_MIN = 10.0         # sliding-window length [min]; must hold the full 5-8 min event
MODE_DEFAULT_AMP_K = {"level": 28.0, "box": 32.0, "bwf": 28.0}   # per-mode thresholds (user
#   2026-07-17: box pairs cells across 800 m and needs a higher bar; 28 K for the per-level and
#   BWF modes -- chosen to pull the three climatologies closer together)
AMP_THRESHOLD_K = float(ARGS[1]) if len(ARGS) > 1 else MODE_DEFAULT_AMP_K[DETECT_MODE]
NOISE_AMP_RAISE = 2.0           # obs: the effective per-event threshold is AMP_THRESHOLD_K +
                                # this x mean(retrieval error at the peak, at the trough) -- a
                                # seed below its raised threshold never becomes a candidate
DT_PT_RANGE_MIN = (1.0, 6.0)    # accepted peak->trough time [min]: 2-12 min apparent period band
SEARCH_HALF_Z_KM = float(ARGS[2]) if len(ARGS) > 2 else 3.0
#   half-height of the 2D max/min search box (default +-3 km = 6 km full width): the trough may
#   sit on another level than the warm peak. Non-default values tag the outputs.
AMP_BOX_HALF_Z_KM = 0.4         # (box modes) half-height of the amplitude box (800 m full width)
OUT_SUFFIX = ("" if AMP_THRESHOLD_K == MODE_DEFAULT_AMP_K[DETECT_MODE]
              else f"_amp{AMP_THRESHOLD_K:.0f}") + \
    ("" if SEARCH_HALF_Z_KM == 3.0 else f"_zbox{SEARCH_HALF_Z_KM:g}") + \
    {"level": "_level", "bwf": "_bwf15", "box": "_box"}[DETECT_MODE]
DEDUP_DT_MIN = 1.0              # events whose warm-peak times are within this [min] are the SAME
                                # event seen by two seeds -> merged (strongest amplitude kept,
                                # period = fastest valid swing among the duplicates)
MIN_VALID_FRAC = 0.8            # obs: minimum fraction of RAW-measured samples in the event
                                # window at the peak level; with the raw-valid peak/trough-cell
                                # requirement this replaces the old blunt gap rule
MERGE_DUPLICATES = True         # sweep mode disables this to get raw per-seed events
SWEEP_THRESHOLDS_K = np.arange(45.0, 9.5, -1.0)   # elbow sweep: 45 K down to 10 K in 1-K steps
SWEEP_FIT_HIGH_K = 28.0         # piecewise exponential fits: steep tail >= this ...
SWEEP_FIT_LOW_K = 25.0          # ... and the flatter mixed regime <= this
SWEEP_FIT_EXT_K = 8.0           # extend each fit line this far into the other regime (the fits
                                # coincide with the data inside their own range; the extension
                                # makes the slope break visible)
MAX_VERT_FWHM_KM = 4.0          # confinement: tube-core anomaly FWHM must stay below this [km]
SUPPRESS_Z_KM = 4.0             # non-max suppression half-height around an accepted seed
SEARCH_ZBAND_M = (52000.0, 68000.0)   # detection band; loaded with margin so the 900-m smoothing is clean

# example-curtain column per cube [km in the run frame]; None = auto (most vortex events).
# cube 0 = the user-picked showcase (clean warm-blob row at 58-62 km); cube 1 = picked from the
# claude/tmp/cube1_example_candidates.png scan (5 events spread over TWO bands, 56-58 + 65-66 km).
EXAMPLE_XY_KM = {0: (10.0, -91.0), 1: (126.0, 56.0)}
TEMP_CLIM = base.TEMP_CLIM
TEMP_CMAP = base.TEMP_CMAP
DPI = 150

# CORAL observation mode (`python lidar_event_detect.py coral [AMP]`): the full measured night of
# the compare case, same detection core. The T2Z900 product is already at the 2-min/900-m effective
# resolution (30-s/100-m grid), so no extra smoothing is applied.
OBS_FILE = base.OBS_FILE                # the REFERENCE measurement (the compare-case night)
OBS_DIR = OBS_FILE.parent               # all-nights mode: every *T2Z900.nc in here (2017-2025)
MIN_NIGHT_HOURS = 1.0                   # skip shorter records (window statistics meaningless)
OBS_ZBAND_M = (31000.0, 75000.0)        # detection band = the FULL valid column of the T2Z900
                                        # retrieval (30-75 km >=90% valid; nothing outside 30-77 km)
OBS_FILTER_BAND_M = (30000.0, 76000.0)  # Butterworth background band (all valid data; filtfilt
                                        # odd-padding keeps the 1-km edge margins acceptable)
OBS_MAX_ERR_K = 8.0                     # cap the searched column where the night-median
                                        # `temperature_err` exceeds this: box noise extremes are
                                        # ~4 sigma, so sigma = 8 K produces ~32-K fake swings =
                                        # our thresholds -- above that altitude the noise wins
OBS_BACKGROUND_CUTOFF_M = 15000.0       # vertical low-pass cutoff for the T' anomaly (15 km --
                                        # user choice 2026-07-17, physically motivated; the
                                        # compare/dyn_overview curtains keep their own 20 km)
# fixed top-row x-ranges, IDENTICAL across all figure variants (cubes / single night / archive)
# so the histograms compare directly.
STATS_AMP_XLIM_K = (26.0, 80.0)         # panels a AND e share this amplitude x-range (user)
STATS_YMAX_ARCHIVE = 400.0              # fixed top-row y ceiling for the ARCHIVE figures so the
                                        # three mode variants compare directly (max panel = 393)
STATS_FWHM_XLIM_KM = (0.0, 6.0)         # panel c
STATS_Z_XLIM_KM = (30.0, 75.0)          # altitude-marginal bin range (panel f)
MARGINAL_WIDTH_FRAC = 0.3               # panel-f altitude marginal occupies this fraction of width          # panel d (= the obs column; cube events sit at 52-68)

OBS_ZLIM_KM = (35.0, 75.0)              # fixed display altitude range of the obs panels f/g

XLBL, YPP = base.XLBL, base.YPP
PANEL_LABEL_PAD_Y_PT = 8        # fixed physical inset of the panel letters from the top-right
PANEL_LABEL_PAD_X_PT = 14       # corner (point-offset standard; x pad larger per user request)


def _add_panel_label(ax, letter):
    ax.annotate(letter, xy=(1, 1), xycoords="axes fraction",
                xytext=(-PANEL_LABEL_PAD_X_PT, -PANEL_LABEL_PAD_Y_PT), textcoords="offset points",
                ha="right", va="top", weight="bold", bbox=base.LABEL_BOX_CIRCLE, zorder=7)


def load_cube_band(cube_index):
    """Obs-like T'(col, t, z) for a stride-sampled column grid over the full detection z-band.

    Sampling matches the obs-vs-model comparison's matched+regrid mode exactly
    (``lidar_obs_model_compare.match_lidar_grid_and_sampling``): the model columns are first
    interpolated onto CORAL's 30-s / 100-m grid (``base.LIDAR_GRID_TIME_S/VERT_M``), then the
    2-min / 900-m running mean is applied. Last hour only; T' = T - horizontal cube mean per
    (z, t) computed on the native grid before regridding. The absolute temperature is returned
    alongside T' (the curtain plots T like a real lidar quicklook).
    Returns (temp_abs[ncol, nt, nz], tprime[ncol, nt, nz], times, z_m, col_xy_km[ncol, 2]).
    """
    path = MODEL_ROOT / MODEL_SIM / f"cube_{cube_index}.nc"
    ds = xr.open_dataset(path, decode_times=False)

    z = np.asarray(ds["z"].values, dtype=float)
    times = np.asarray(ds["time"].values, dtype=float)
    start = int(np.searchsorted(times, float(times.max()) - MODEL_WINDOW_DURATION_S))

    band = (z >= SEARCH_ZBAND_M[0] - LIDAR_VERT_RES_M) & (z <= SEARCH_ZBAND_M[1] + LIDAR_VERT_RES_M)
    band_idx = np.where(band)[0]
    z_band = z[band_idx]

    temp = ds[THETA_NAME].isel(time=slice(start, None), z=band_idx).values
    temp *= ds[EXNER_NAME].isel(time=slice(start, None), z=band_idx).values
    x_km = np.asarray(ds["x"].values, dtype=float) / 1000.0
    y_km = np.asarray(ds["y"].values, dtype=float) / 1000.0
    ds.close()

    tmean = temp.mean(axis=(1, 2), keepdims=True)
    xs = np.arange(0, temp.shape[1], STRIDE)
    ys = np.arange(0, temp.shape[2], STRIDE)

    def stride_cols(a):
        sub = a[:, xs][:, :, ys]
        nt, ncx, ncy, nz = sub.shape
        return sub.transpose(1, 2, 0, 3).reshape(ncx * ncy, nt, nz)

    temp_abs = stride_cols(temp)
    tprime = stride_cols(temp - tmean)
    col_xy = np.array([(x_km[i], y_km[j]) for i in xs for j in ys])
    times_w = times[start:]
    z_out = z_band

    if MATCH_LIDAR_SAMPLING:
        target_t = np.arange(times_w[0], times_w[-1] + 1e-6, base.LIDAR_GRID_TIME_S)
        z_lo = np.ceil(z_band.min() / base.LIDAR_GRID_VERT_M) * base.LIDAR_GRID_VERT_M
        z_hi = np.floor(z_band.max() / base.LIDAR_GRID_VERT_M) * base.LIDAR_GRID_VERT_M
        target_z = np.arange(z_lo, z_hi + 1e-6, base.LIDAR_GRID_VERT_M)

        def regrid_and_smooth(a):
            a = interp1d(times_w, a, axis=1, assume_sorted=True)(target_t)
            a = interp1d(z_band, a, axis=2, assume_sorted=True)(target_z).astype(np.float32)
            n_time = max(1, int(round(LIDAR_TIME_RES_S / base.LIDAR_GRID_TIME_S)))
            n_vert = max(1, int(round(LIDAR_VERT_RES_M / base.LIDAR_GRID_VERT_M)))
            a = uniform_filter1d(a, size=n_time, axis=1, mode="nearest")
            return uniform_filter1d(a, size=n_vert, axis=2, mode="nearest")

        temp_abs = regrid_and_smooth(temp_abs)
        tprime = regrid_and_smooth(tprime)
        times_w = target_t
        z_out = target_z

    keep = (z_out >= SEARCH_ZBAND_M[0]) & (z_out <= SEARCH_ZBAND_M[1])
    return temp_abs[..., keep], tprime[..., keep], times_w, z_out[keep], col_xy


def vertical_fwhm(profile, z, iz_core, sign):
    """Half-amplitude vertical extent [m] of the (signed) core anomaly around level iz_core.

    Walks uphill in sign*profile from iz_core to the local extremum, then outward in both
    directions until the anomaly drops below half the core value; half-crossings are linearly
    interpolated. Returns (fwhm_m, z_core_m, core_value_K, edge_truncated) -- edge_truncated is
    True when either half-crossing lies outside the search band, so fwhm_m is only a lower bound
    (a deep saturated layer reaching the band edge must not pass as vertically confined).
    """
    p = sign * np.asarray(profile, dtype=float)
    nz = p.size
    k = int(iz_core)
    while 0 < k < nz - 1 and (p[k + 1] > p[k] or p[k - 1] > p[k]):
        k = k + 1 if p[k + 1] >= p[k - 1] else k - 1
    if p[k] <= 0.0:                # no positive core anomaly -> half level undefined, not a warm tube
        return np.nan, float(z[k]), float(sign * p[k]), True
    half = p[k] / 2.0

    def crossing(step):
        j = k
        while 0 <= j + step < nz and p[j + step] >= half:
            j += step
        jn = j + step
        if jn < 0 or jn >= nz:
            return z[j], True
        frac = (p[j] - half) / (p[j] - p[jn])
        return z[j] + frac * (z[jn] - z[j]), False

    (z_hi, trunc_hi), (z_lo, trunc_lo) = crossing(+1), crossing(-1)
    return abs(z_hi - z_lo), float(z[k]), float(sign * p[k]), trunc_hi or trunc_lo


def detect_events(field, anom, times, z, err=None, valid=None):
    """Vortex-overpass candidate events in one curtain. Returns a list of dicts.

    `field` is the DETECTION field: absolute temperature in the background-free modes
    ("runmean" = per-level windowed swing of T minus the 10-min per-level running mean;
    "box" = the same residual, amplitude = max-min over the window x +/-AMP_BOX_HALF_Z_KM box),
    or the Butterworth anomaly in "bwf" mode (per-level windowed swing). `anom` is always the
    anomaly (obs: T - vertical Butterworth low-pass; model: T - horizontal cube mean), used ONLY
    for the warm-core / FWHM measurement. `err` (obs) is the per-cell 1-sigma retrieval
    uncertainty: every event must exceed AMP_THRESHOLD_K + NOISE_AMP_RAISE * mean(err at the
    peak cell, err at the trough cell) -- a sub-noise seed never becomes a candidate (no grey
    box), replacing the old post-hoc significance gate and the night-median column cap.
    `valid` (obs) is the RAW-measured mask: an event is only accepted if BOTH its peak and trough
    cells are actual measurements AND at least MIN_VALID_FRAC of the window samples at the peak
    level are raw-valid -- fill/interpolation artifacts (bottom-of-retrieval clamping, hole
    edges) can then never define an event, wherever they occur.
    """
    fld = np.asarray(field, dtype=float)
    an = np.asarray(anom, dtype=float)
    dt = float(np.median(np.diff(times)))
    dz = float(np.median(np.diff(z)))
    nwin = max(3, int(round(EVENT_WINDOW_MIN * 60.0 / dt)))
    nsupz = max(1, int(round(SUPPRESS_Z_KM * 1000.0 / dz)))
    nboxz = max(1, int(round(AMP_BOX_HALF_Z_KM * 1000.0 / dz)))
    if DETECT_MODE == "level":
        # ONE altitude level: windowed peak-trough of the raw field per level (background-free
        # by construction -- a time difference at fixed altitude).
        work = maximum_filter1d(fld, nwin, axis=0) - minimum_filter1d(fld, nwin, axis=0)
    else:
        # box modes: amplitude = max-min over the (window x 1-km box) region of the field as is
        # -- RAW absolute T ("box": the ~2-3 K/km stratification across 1 km is accepted) or the
        # Butterworth T' ("bwf").
        size = (nwin, 2 * nboxz + 1)
        work = maximum_filter(fld, size=size) - minimum_filter(fld, size=size)

    events = []
    while True:
        it_c, iz_c = np.unravel_index(np.argmax(work), work.shape)
        if work[it_c, iz_c] < AMP_THRESHOLD_K:
            break
        work[max(0, it_c - nwin):it_c + nwin, max(0, iz_c - nsupz):iz_c + nsupz + 1] = -np.inf

        it0 = max(0, it_c - nwin // 2)
        it1 = min(fld.shape[0], it_c + nwin // 2 + 1)
        seg = fld[it0:it1, iz_c]
        i_max, i_min = int(np.argmax(seg)), int(np.argmin(seg))
        dt_level_min = abs(i_max - i_min) * dt / 60.0
        if DETECT_MODE == "level":
            # strictly single-level: no cross-level timing alternative (raw-T box extrema would
            # be stratification-biased anyway)
            dt_box_min = np.nan
            amp = float(seg[i_max] - seg[i_min])
            it_core = it0 + i_max                  # core = the warm peak at the seed level
            iz_start = iz_c
            pk_cell = (it0 + i_max, iz_c)
            tr_cell = (it0 + i_min, iz_c)
        else:
            # box modes: amplitude, timing and core all from the mode's own 1-km amplitude box
            ab0 = max(0, iz_c - nboxz)
            abox = fld[it0:it1, ab0:iz_c + nboxz + 1]
            am = np.unravel_index(np.argmax(abox), abox.shape)
            an_ = np.unravel_index(np.argmin(abox), abox.shape)
            amp = float(abox[am] - abox[an_])
            dt_box_min = abs(int(am[0]) - int(an_[0])) * dt / 60.0
            it_core = it0 + int(am[0])             # core = the maximum of the amplitude box
            iz_start = ab0 + int(am[1])
            pk_cell = (it0 + int(am[0]), ab0 + int(am[1]))
            tr_cell = (it0 + int(an_[0]), ab0 + int(an_[1]))
        if valid is not None:
            if not (valid[pk_cell] and valid[tr_cell]):
                continue
            if valid[it0:it1, pk_cell[1]].mean() < MIN_VALID_FRAC:
                continue
        noise = float(0.5 * (err[pk_cell] + err[tr_cell])) if err is not None else 0.0
        thr_eff = AMP_THRESHOLD_K + NOISE_AMP_RAISE * noise
        if amp < thr_eff:
            continue

        # the trough may sit on another level (dt_box), but a sub-floor box time (a simultaneous
        # vertical +/- dipole, dt ~ 0) is no overpass evidence and must not override a valid
        # same-level swing -- only times above the lower bound compete.
        dt_valid = [d for d in (dt_level_min, dt_box_min) if d >= DT_PT_RANGE_MIN[0]]
        dt_pt_min = min(dt_valid) if dt_valid else float(np.nanmin([dt_level_min, dt_box_min]))

        fwhm_m, z_core, core_val, truncated = vertical_fwhm(an[it_core, :], z, iz_start, sign=1.0)
        fast = bool(dt_valid) and dt_pt_min <= DT_PT_RANGE_MIN[1]
        warm = core_val > 0.0
        confined = warm and np.isfinite(fwhm_m) and fwhm_m / 1000.0 <= MAX_VERT_FWHM_KM \
            and not truncated
        events.append({
            "t_core_s": float(times[it_core]),
            "z_core_m": z_core,
            "amp_k": amp,
            "dt_peak_trough_min": dt_pt_min,
            "dt_level_min": dt_level_min,
            "dt_box_min": dt_box_min,
            "period_est_min": 2.0 * dt_pt_min,
            "fwhm_km": fwhm_m / 1000.0,
            "core_tprime_k": core_val,
            "noise_k": noise,
            "thr_eff_k": thr_eff,
            "fast": fast,
            "warm": warm,
            "confined": confined,
            "edge_truncated": truncated,
            "vortex": fast and confined,
        })
    return merge_duplicate_events(events) if MERGE_DUPLICATES else events


def merge_duplicate_events(events):
    """Merge events whose warm-peak times are within DEDUP_DT_MIN (the same physical overpass
    caught by two suppression-separated seeds, e.g. 30 s / 100 m apart).

    The kept event is the strongest-amplitude member (events arrive amplitude-ordered from the
    greedy extraction); its period is upgraded to the fastest VALID (>= floor) swing measured by
    any duplicate -- that swing exists in the time series, so discarding it with the weaker seed
    would lose a genuine measurement (observed: amp 52 K seed measured 7.5 min, its 30-s twin
    2.5 min on the same blob).
    """
    merged = []
    for e in events:
        twin = next((k for k in merged
                     if abs(k["t_core_s"] - e["t_core_s"]) <= DEDUP_DT_MIN * 60.0), None)
        if twin is None:
            merged.append(e)
            continue
        e_valid = e["dt_peak_trough_min"] >= DT_PT_RANGE_MIN[0]
        twin_valid = twin["dt_peak_trough_min"] >= DT_PT_RANGE_MIN[0]
        if e_valid and (not twin_valid or e["dt_peak_trough_min"] < twin["dt_peak_trough_min"]):
            for key in ("dt_peak_trough_min", "dt_level_min", "dt_box_min"):
                twin[key] = e[key]
            twin["period_est_min"] = 2.0 * twin["dt_peak_trough_min"]
            twin["fast"] = twin["dt_peak_trough_min"] <= DT_PT_RANGE_MIN[1]
            twin["vortex"] = twin["fast"] and twin["confined"]
    return merged


def pick_example_column(col_xy, events_per_col, cube_index):
    """The curtain column: pinned via EXAMPLE_XY_KM, else the column with the most vortex events
    (ties broken by the strongest event) -- a showcase with several overpasses reads best."""
    pin = EXAMPLE_XY_KM.get(cube_index)
    if pin is not None:
        return int(np.argmin((col_xy[:, 0] - pin[0]) ** 2 + (col_xy[:, 1] - pin[1]) ** 2))
    scores = [(sum(e["vortex"] for e in ev),
               max((e["amp_k"] for e in ev if e["vortex"]), default=0.0)) for ev in events_per_col]
    return max(range(len(scores)), key=lambda c: scores[c])


def _stats_panels(fig, axes, events, ymax=None):
    """Panels a-d: histograms of amplitude / peak->trough time / vertical FWHM / core altitude
    (all candidates grey, vortex BLACK like the curtain boxes). Ticks/labels on top, letters at
    the 8-pt inset."""
    ax_a, ax_b, ax_c, ax_d = axes
    amp = np.array([e["amp_k"] for e in events])
    dtp = np.array([e["dt_peak_trough_min"] for e in events])
    fwhm = np.array([e["fwhm_km"] for e in events])
    vortex = np.array([e["vortex"] for e in events])

    bins_a = np.arange(AMP_THRESHOLD_K, STATS_AMP_XLIM_K[1] + 0.01, 2)
    ax_a.hist(amp, bins=bins_a, color="0.7", label="all candidates")
    ax_a.hist(amp[vortex], bins=bins_a, color="black", label="vortex")
    ax_a.set_xlim(*STATS_AMP_XLIM_K)
    ax_a.set_xlabel(r"$\Delta T_{pt}$ / K")
    ax_a.set_ylabel("events / -")
    ax_a.legend(fontsize=6.5, loc="upper left", framealpha=0.9)

    doys = np.array([e.get("doy", np.nan) for e in events], dtype=float) % 366.0
    has_doy = np.isfinite(doys)
    bins_b = np.sort(np.concatenate([MONTH_START_DOY.astype(float),
                                     MONTH_START_DOY + MONTH_LEN_D / 2.0, [366.0]]))
    if has_doy.any():
        ax_b.hist(doys[has_doy], bins=bins_b, color="0.7")
        ax_b.hist(doys[has_doy & vortex], bins=bins_b, color="black")
    ax_b.set_xlim(1, 366)
    ax_b.set_xticks(MONTH_START_DOY + MONTH_LEN_D / 2.0)
    ax_b.set_xticklabels("JFMAMJJASOND")
    ax_b.set_xlabel("month")

    finite = np.isfinite(fwhm)
    bins_c = np.arange(0, STATS_FWHM_XLIM_KM[1] + 0.01, 0.25)
    ax_c.hist(fwhm[finite & (fwhm <= STATS_FWHM_XLIM_KM[1])], bins=bins_c, color="0.7")
    ax_c.hist(fwhm[vortex], bins=bins_c, color="black")
    ax_c.axvline(MAX_VERT_FWHM_KM, ls="--", color="C3", lw=1.3)
    ax_c.set_xlim(*STATS_FWHM_XLIM_KM)
    ax_c.set_xlabel("vertical FWHM / km")

    bins_d = np.arange(0, EVENT_WINDOW_MIN + 0.75, 0.5)
    ax_d.hist(dtp, bins=bins_d, color="0.7")
    ax_d.hist(dtp[vortex], bins=bins_d, color="black")
    for x in DT_PT_RANGE_MIN:
        ax_d.axvline(x, ls="--", color="C3", lw=1.3)
    ax_d.set_xlim(0, EVENT_WINDOW_MIN + 0.5)
    ax_d.set_xlabel(r"$\Delta t_{pt}$ $(0.5\,T)$ / min")

    if ymax is None:
        ymax = max(ax.get_ylim()[1] for ax in axes)
    for label, ax in zip("abcd", axes):
        ax.set_ylim(0, ymax)
        if ax is not axes[0]:
            plt.setp(ax.get_yticklabels(), visible=False)
        ax.grid(True, alpha=0.25)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        _add_panel_label(ax, label)


MONTH_LEN_D = np.array((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31))
MONTH_START_DOY = np.concatenate(([1], 1 + np.cumsum(MONTH_LEN_D[:-1])))


def _bottom_panels(fig, ax_e, ax_f, events):
    """Panels e/f: (e) amplitude vs core altitude (dots: grey open candidates, black vortex) and
    (f) core altitude vs CONTINUOUS day-of-year with the ALTITUDE MARGINAL overlaid as light step
    lines on a hidden second x-axis anchored to the right edge (replaces the old top-row altitude
    histogram -- same information as f's y-distribution). Events without a "month" key (model
    cubes) leave f's scatter empty but keep the marginal."""
    amp = np.array([e["amp_k"] for e in events])
    z_core = np.array([e["z_core_m"] for e in events]) / 1000.0
    vortex = np.array([e["vortex"] for e in events], dtype=bool)

    ax_e.scatter(amp[~vortex], z_core[~vortex], s=7, facecolors="none", edgecolors="0.6",
                 lw=0.5, zorder=2)
    ax_e.scatter(amp[vortex], z_core[vortex], s=8, color="black", zorder=3)
    if vortex.sum() >= 10:
        # descriptive linear fit of the vortex amplitudes vs altitude (NB the lower envelope is
        # shaped by the altitude-dependent noise-raised threshold, so the slope mixes physical
        # amplitude growth with detection censoring)
        b1, b0 = np.polyfit(z_core[vortex], amp[vortex], 1)
        zz = np.array([z_core[vortex].min(), z_core[vortex].max()])
        ax_e.plot(b0 + b1 * zz, zz, ls="--", color="C3", lw=1.3, zorder=4)
        y_lbl = max(38.0, float(zz[0]))
        ax_e.annotate(rf"{b1:.2f} K$\,$km$^{{-1}}$", (b0 + b1 * y_lbl, y_lbl),
                      xytext=(6, 0), textcoords="offset points", va="center",
                      fontsize=7, color="C3", zorder=6)
    ax_e.set_xlim(*STATS_AMP_XLIM_K)
    ax_e.set_xlabel(r"$\Delta T_{pt}$ / K")
    ax_e.set_ylabel(r"core altitude $z$ / km")

    monthly = [e for e in events if "month" in e]
    if monthly:
        doy = np.array([e["doy"] for e in monthly]) % 366.0
        z_m = np.array([e["z_core_m"] for e in monthly]) / 1000.0
        v_m = np.array([e["vortex"] for e in monthly], dtype=bool)
        ax_f.scatter(doy[~v_m], z_m[~v_m], s=7, facecolors="none", edgecolors="0.6",
                     lw=0.5, zorder=2)
        ax_f.scatter(doy[v_m], z_m[v_m], s=8, color="black", zorder=3)
    # altitude marginal on a hidden second x-axis, anchored to the RIGHT edge of panel f
    if len(events):
        ax_f2 = ax_f.twiny()
        bins_z = np.arange(STATS_Z_XLIM_KM[0], STATS_Z_XLIM_KM[1] + 0.01, 1.0)
        ax_f2.hist(z_core[vortex], bins=bins_z, orientation="horizontal",
                   histtype="bar", color="black", zorder=1.6)
        counts, _, _ = ax_f2.hist(z_core, bins=bins_z, orientation="horizontal",
                                  histtype="step", color="0.65", lw=1.0, ls="--", zorder=1.7)
        ax_f2.set_xlim(max(float(np.max(counts)), 1.0) / MARGINAL_WIDTH_FRAC, 0)
        ax_f2.set_xticks([])
        for sp in ax_f2.spines.values():
            sp.set_visible(False)
    ax_f.set_xlim(1, 366)
    ax_f.set_xticks(MONTH_START_DOY + MONTH_LEN_D / 2.0)
    ax_f.set_xticklabels("JFMAMJJASOND")
    ax_f.set_xlabel("month")
    ax_f.set_ylabel(r"core altitude $z$ / km")
    for ax in (ax_e, ax_f):
        ax.grid(True, alpha=0.25)
    _add_panel_label(ax_e, "e")
    _add_panel_label(ax_f, "f")


def _curtain_panel(fig, ax, temp2d, times, z, events, curtain_label, letter,
                   time_scale, time_xlabel, time_formatter):
    """Absolute-T curtain with event overlays: rectangle = event window x vertical FWHM, circle =
    the warm core; black = vortex, grey = rejected (only defined-FWHM candidates drawn).
    Returns the temperature colorbar."""
    win_plot = EVENT_WINDOW_MIN * 60.0 / time_scale
    pc = ax.pcolormesh(times / time_scale, z / 1000.0, temp2d.T, cmap=TEMP_CMAP,
                       vmin=TEMP_CLIM[0], vmax=TEMP_CLIM[1], rasterized=True)
    for e in events:
        if not np.isfinite(e["fwhm_km"]):
            continue
        color = "black" if e["vortex"] else "0.45"
        zc, dz2 = e["z_core_m"] / 1000.0, e["fwhm_km"] / 2.0
        ax.add_patch(plt.Rectangle((e["t_core_s"] / time_scale - win_plot / 2.0, zc - dz2),
                                   win_plot, 2 * dz2,
                                   fill=False, edgecolor=color, lw=1.3, ls="-", zorder=5))
        ax.plot(e["t_core_s"] / time_scale, zc, "o", ms=4, mfc="none", mec=color, mew=1.2, zorder=6)
    ax.set_xlabel(time_xlabel)
    ax.set_ylabel(r"altitude $z$ / km")
    if time_formatter is not None:
        ax.xaxis.set_major_formatter(time_formatter)
    ax.text(0.012, YPP, curtain_label, transform=ax.transAxes,
            bbox=base.LABEL_BOX_ROUND, zorder=7)
    _add_panel_label(ax, letter)
    cbar_t = fig.colorbar(pc, ax=ax, location="right", shrink=0.9, pad=0.01, extend="both")
    cbar_t.set_label(r"$T$ / K")
    return cbar_t


def draw_event_figure(events, curtain, rate_txt, out, stats_ymax=None):
    """THE standard event figure, shared by all variants (cube / single night / all nights):
    (a-d) amplitude / peak->trough / FWHM / core-altitude histograms over all events, (e) month
    histogram + (f) core altitude vs month (empty for the model, sparse for one night, filled for
    the archive), (g) the example curtain spanning the c+d width.
    `curtain` = dict(temp2d, times, z, events, label, time_scale, time_xlabel, time_formatter)."""
    fig = plt.figure(figsize=(13.0, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=(1.0, 1.45))
    axes = tuple(fig.add_subplot(gs[0, i]) for i in range(4))
    ax_e = fig.add_subplot(gs[1, 0])
    ax_f = fig.add_subplot(gs[1, 1])
    ax_curtain = fig.add_subplot(gs[1, 2:])

    _stats_panels(fig, axes, events, ymax=stats_ymax)
    _bottom_panels(fig, ax_e, ax_f, events)
    _curtain_panel(fig, ax_curtain, curtain["temp2d"], curtain["times"], curtain["z"],
                   curtain["events"], curtain["label"], "g",
                   curtain["time_scale"], curtain["time_xlabel"], curtain["time_formatter"])
    zlim = curtain.get("zlim_km") or (curtain["z"].min() / 1000.0, curtain["z"].max() / 1000.0)
    ax_curtain.set_ylim(*zlim)
    ax_e.set_ylim(*zlim)
    ax_f.set_ylim(*zlim)
    for ax in (ax_f, ax_curtain):
        ax.set_ylabel("")
        plt.setp(ax.get_yticklabels(), visible=False)
    fig.get_layout_engine().set(w_pad=0.01, wspace=0.02)
    ax_e.annotate(rate_txt, xy=(0, 1), xycoords="axes fraction",
                  xytext=(6, -PANEL_LABEL_PAD_Y_PT), textcoords="offset points",
                  ha="left", va="top", fontsize=7, bbox=base.LABEL_BOX_ROUND, zorder=7)
    fig.savefig(out, dpi=DPI, facecolor="w", bbox_inches="tight")
    plt.close(fig)


def write_csv(events, col_xy, out, extra_fields=()):
    fields = ["col", "x_km", "y_km", "t_core_s", "z_core_m", "amp_k", "dt_peak_trough_min",
              "dt_level_min", "dt_box_min", "period_est_min", "fwhm_km", "core_tprime_k",
              "noise_k", "thr_eff_k", "fast", "warm", "confined", "edge_truncated", "vortex"] \
        + list(extra_fields)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for c, ev in enumerate(events):
            for e in ev:
                w.writerow([c, f"{col_xy[c][0]:.1f}", f"{col_xy[c][1]:.1f}"] +
                           [e[k] for k in fields[3:]])


def load_obs_curtain(path=None):
    """One full CORAL night (default: the reference measurement OBS_FILE): absolute T + T'
    anomaly on the detection band.

    T' = T - vertical 5th-order Butterworth low-pass (OBS_BACKGROUND_CUTOFF_M, filtfilt per
    profile) computed on the wide OBS_FILTER_BAND_M, then cropped to OBS_ZBAND_M. The file's
    0-fill is NaN; partial holes are interpolated vertically, laser-off profiles (all-NaN) are
    interpolated in time for the filter and flagged in `gap` so their events can be dropped.
    The returned ABSOLUTE temperature is the PRISTINE raw band (no interpolation) -- gaps stay
    NaN and appear as white gaps in the curtain; only T'/err carry the filled values.
    The retrieval uncertainty `temperature_err` gets the same hole treatment and feeds the
    per-event noise-raised threshold in `detect_events`. Returns (temp_raw (pristine, for the
    curtain), temp_filled (for "t"-mode detection), tprime, err (all [nt, nz]), times_s (since
    00 UTC of the start day), z_m, gap[nt], start_datetime_utc).
    """
    ds = xr.open_dataset(path if path is not None else OBS_FILE, decode_times=False)
    unix = float(ds["time_offset"].values[0]) + np.asarray(ds["time"].values, dtype=float) / 1000.0
    z = np.asarray(ds["altitude"].values, dtype=float) \
        + float(ds["altitude_offset"].values[0]) + float(ds["station_height"].values[0])
    temp = np.asarray(ds["temperature"].values, dtype=float)
    err = np.asarray(ds["temperature_err"].values, dtype=float)
    ds.close()
    err[temp == 0.0] = np.nan
    temp[temp == 0.0] = np.nan

    band = (z >= OBS_FILTER_BAND_M[0]) & (z <= OBS_FILTER_BAND_M[1])
    zb = z[band]
    tb = temp[:, band]
    eb = err[:, band]
    tb_raw = tb.copy()      # pristine (NaN where the raw data has none) -- the curtain shows THIS,
    gap = ~np.isfinite(tb).any(axis=1)   # so laser-off gaps appear as white gaps, not interpolation
    med_err = np.nanmedian(eb, axis=0)
    ok = med_err < OBS_MAX_ERR_K
    z_cap = float(zb[ok].max()) if ok.any() else float(zb.max())
    for arr in (tb, eb):
        for i in np.where(~gap)[0]:
            good = np.isfinite(arr[i])
            if not good.all():
                arr[i] = np.interp(zb, zb[good], arr[i][good])
        if gap.any():
            for k in range(zb.size):
                arr[gap, k] = np.interp(unix[gap], unix[~gap], arr[~gap, k])

    dzb = float(np.median(np.diff(zb)))
    bf, af = signal.butter(5, 2.0 * dzb / OBS_BACKGROUND_CUTOFF_M)
    tprime = tb - signal.filtfilt(bf, af, tb, axis=1)

    keep = (zb >= OBS_ZBAND_M[0]) & (zb <= min(OBS_ZBAND_M[1], z_cap))
    start = datetime.datetime.utcfromtimestamp(unix[0])
    day0 = unix[0] - (start - start.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    return (tb_raw[:, keep], tb[:, keep], tprime[:, keep], eb[:, keep],
            unix - day0, zb[keep], gap, start)


def _fmt_hhmm(hours, _pos=None):
    h = float(hours) % 24.0
    m = int(round((h - int(h)) * 60.0))
    return f"{(int(h) + m // 60) % 24:02d}:{m % 60:02d}"


def analyze_night(path=None, keep_curtain=False):
    """Full detection chain for one CORAL night: load, detect, gap-filter, noise gate.

    Returns dict(night, start, month, hours, gap_count, events[, temp2d/times/z when
    keep_curtain]). Raises on unusable nights (shorter than MIN_NIGHT_HOURS, no low-noise level).
    """
    temp_raw, temp_filled, tprime, err, times, z, gap, start = load_obs_curtain(path)
    hours = (times[-1] - times[0]) / 3600.0
    if hours < MIN_NIGHT_HOURS:
        raise ValueError(f"record too short ({hours:.2f} h)")
    field = tprime if DETECT_MODE == "bwf" else temp_filled
    events = detect_events(field, tprime, times, z, err=err, valid=np.isfinite(temp_raw))
    start_doy = start.timetuple().tm_yday
    for e in events:
        e["month"] = start.month
        e["doy"] = start_doy + e["t_core_s"] / 86400.0
    night = {"night": f"{start:%Y%m%d-%H%M}", "start": start, "month": start.month,
             "hours": float(hours), "gap_count": int(gap.sum()), "events": events}
    if keep_curtain:
        night.update(temp2d=temp_raw, times=times, z=z)
    return night


def _night_worker(path):
    try:
        return analyze_night(path)
    except Exception as exc:
        return {"night": Path(path).stem, "error": str(exc)}


def main_obs():
    """Detection over one CORAL measurement + the single-night combined figure.

    Default: the reference night. argv[4] selects any other night by file stem prefix
    (e.g. `coral 28 3 20230615-2149`); the output tag carries the night's date anyway.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    path = OBS_DIR / f"{ARGS[3]}_T2Z900.nc" if len(ARGS) > 3 else None
    n = analyze_night(path, keep_curtain=True)
    events, start, hours = n["events"], n["start"], n["hours"]
    n_vort = sum(e["vortex"] for e in events)

    print(f"CORAL {start:%Y-%m-%d %H:%M} UTC, {hours:.1f} h, {n['gap_count']} gap profiles -> "
          f"{len(events)} candidates, "
          f"{n_vort} vortex ({n_vort / hours:.2f} / h)  [{time.time() - t0:.1f} s]")
    tag = f"coral_{start:%Y%m%d}_lidar_events{OUT_SUFFIX}"
    curtain = {"temp2d": n["temp2d"], "times": n["times"], "z": n["z"], "events": events,
               "label": f"CORAL | {start:%Y-%m-%d}",
               "time_scale": 3600.0, "time_xlabel": "time / UTC",
               "time_formatter": mticker.FuncFormatter(_fmt_hhmm), "zlim_km": OBS_ZLIM_KM}
    draw_event_figure(events, curtain, f"{n_vort}/{len(events)} vortex",
                      OUTPUT_DIR / f"{tag}.png")
    write_csv([events], np.array([[np.nan, np.nan]]), OUTPUT_DIR / f"{tag}.csv")
    print(f"  wrote {tag}.png / .csv  [{time.time() - t0:.1f} s total]")


def main_obs_all():
    """Detection over ALL CORAL T2Z900 nights (multiprocessing) + the climatology figure.

    The reference night (OBS_FILE) provides the example curtain (panel g). Events get "month"
    attached; the combined csv carries night/month columns. Failed/too-short nights are counted
    and listed at the end.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    files = sorted(OBS_DIR.glob("*T2Z900.nc"))
    print(f"{len(files)} CORAL nights in {OBS_DIR}")
    with Pool(min(16, cpu_count())) as pool:
        results = pool.map(_night_worker, files, chunksize=8)

    failed = [r for r in results if "error" in r]
    nights = [r for r in results if "error" not in r]
    events = []
    for night in nights:
        for e in night["events"]:
            events.append({**e, "night": night["night"]})
    hours_total = sum(night["hours"] for night in nights)
    n_vort = sum(e["vortex"] for e in events)
    print(f"{len(nights)} nights analysed ({hours_total:.0f} h), {len(failed)} skipped -> "
          f"{len(events)} candidates, {n_vort} vortex ({n_vort / hours_total:.3f} / h)  "
          f"[{time.time() - t0:.1f} s]")
    for r in failed[:10]:
        print(f"  skipped {r['night']}: {r['error']}")
    if len(failed) > 10:
        print(f"  ... and {len(failed) - 10} more")

    ref_night = analyze_night(OBS_FILE, keep_curtain=True)
    curtain = {"temp2d": ref_night["temp2d"], "times": ref_night["times"], "z": ref_night["z"],
               "events": ref_night["events"],
               "label": f"CORAL | {ref_night['start']:%Y-%m-%d}",
               "time_scale": 3600.0, "time_xlabel": "time / UTC",
               "time_formatter": mticker.FuncFormatter(_fmt_hhmm), "zlim_km": OBS_ZLIM_KM}

    tag = f"coral_allnights_lidar_events{OUT_SUFFIX}"
    rate_txt = f"{len(nights)} nights, {n_vort}/{len(events)} vortex"
    draw_event_figure(events, curtain, rate_txt, OUTPUT_DIR / f"{tag}.png",
                      stats_ymax=STATS_YMAX_ARCHIVE)

    fields = ["night", "month", "t_core_s", "z_core_m", "amp_k", "dt_peak_trough_min",
              "dt_level_min", "dt_box_min", "period_est_min", "fwhm_km", "core_tprime_k",
              "noise_k", "thr_eff_k", "fast", "warm", "confined", "edge_truncated", "vortex"]
    with open(OUTPUT_DIR / f"{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for e in events:
            w.writerow([e[k] for k in fields])
    print(f"  wrote {tag}.png / .csv  [{time.time() - t0:.1f} s total]")


def main_obs_sweep():
    """Amplitude-threshold sweep ("elbow" figure) over the whole CORAL archive, ALL THREE modes.

    Per mode, detection runs ONCE per night at the sweep minimum without duplicate-merging (the
    greedy seed sequence is threshold-independent -- the threshold only decides where it stops).
    Per threshold the per-night lists are amplitude-filtered INCLUDING the per-event noise raise,
    duplicate-merged and re-flagged exactly like a native run. Figure: (a) linear counts vs
    threshold, (b) log counts + piecewise exponential fits per mode, (c) 2-K amplitude
    distributions as unfilled step lines (linear y). Wide csv with per-mode columns.
    """
    global AMP_THRESHOLD_K, MERGE_DUPLICATES, DETECT_MODE
    AMP_THRESHOLD_K = float(SWEEP_THRESHOLDS_K.min())
    MERGE_DUPLICATES = False
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    files = sorted(OBS_DIR.glob("*T2Z900.nc"))
    modes = ("level", "box", "bwf")
    labels = {"level": "level", "box": "box", "bwf": "bwf15"}
    colors = {"level": "black", "box": "C0", "bwf": "C3"}
    print(f"sweep {SWEEP_THRESHOLDS_K.max():.0f} -> {SWEEP_THRESHOLDS_K.min():.0f} K over "
          f"{len(files)} CORAL nights x {len(modes)} modes")
    nights_by_mode = {}
    for m in modes:
        DETECT_MODE = m
        with Pool(min(16, cpu_count())) as pool:
            results = pool.map(_night_worker, files, chunksize=8)
        nights_by_mode[m] = [r for r in results if "error" not in r]
        print(f"  {labels[m]}: {len(nights_by_mode[m])} nights, "
              f"{sum(len(n['events']) for n in nights_by_mode[m])} raw seeds  "
              f"[{time.time() - t0:.1f} s]")

    counts = {m: ([], []) for m in modes}
    for thr in SWEEP_THRESHOLDS_K:
        for m in modes:
            n_cand = n_vort = 0
            for night in nights_by_mode[m]:
                sel = [dict(e) for e in night["events"]
                       if e["amp_k"] >= thr + NOISE_AMP_RAISE * e["noise_k"]]
                merged = merge_duplicate_events(sel)
                n_cand += len(merged)
                n_vort += sum(e["fast"] and e["confined"] for e in merged)
            counts[m][0].append(n_cand)
            counts[m][1].append(n_vort)
        print(f"  >= {thr:4.1f} K:  " + "  ".join(
            f"{labels[m]} {counts[m][0][-1]}/{counts[m][1][-1]}" for m in modes))

    tag = "coral_sweep_lidar_events"
    with open(OUTPUT_DIR / f"{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["threshold_k"] + [f"{c}_{labels[m]}" for m in modes
                                      for c in ("candidates", "vortex")])
        for i, thr in enumerate(SWEEP_THRESHOLDS_K):
            w.writerow([float(thr)] + [counts[m][j][i] for m in modes for j in (0, 1)])

    dists = {}
    for m in modes:
        amps, vorts = [], []
        for night in nights_by_mode[m]:
            merged = merge_duplicate_events([dict(e) for e in night["events"]])
            amps.extend(e["amp_k"] for e in merged)
            vorts.extend(e["fast"] and e["confined"] for e in merged)
        dists[m] = (np.array(amps), np.array(vorts, dtype=bool))

    thr_arr = SWEEP_THRESHOLDS_K
    sel_hi = thr_arr >= SWEEP_FIT_HIGH_K
    sel_lo = thr_arr <= SWEEP_FIT_LOW_K
    bins_c = np.arange(SWEEP_THRESHOLDS_K.min(), 80.01, 1.0)
    fig, (ax_a, ax_b, ax_c, ax_d) = plt.subplots(1, 4, figsize=(16.2, 3.6),
                                                 constrained_layout=True)
    for m in modes:
        cand = np.array(counts[m][0], dtype=float)
        vort = np.array(counts[m][1], dtype=float)
        col = colors[m]
        ax_a.plot(thr_arr, cand, color=col, lw=1.0, ls="--", alpha=0.55, marker="o", ms=2)
        ax_a.plot(thr_arr, vort, color=col, lw=1.6, marker="o", ms=2.5, label=labels[m])
        ax_b.plot(thr_arr, cand, color=col, lw=1.0, ls="--", alpha=0.55, marker="o", ms=2)
        ax_b.plot(thr_arr, vort, color=col, lw=1.6, marker="o", ms=2.5)
        # piecewise exponential fits (straight in log space), extended into the other regime
        # where the divergence makes the slope break visible
        for series in (cand, vort):
            for sel, xext in ((sel_hi, thr_arr >= SWEEP_FIT_HIGH_K - SWEEP_FIT_EXT_K),
                              (sel_lo, thr_arr <= SWEEP_FIT_LOW_K + SWEEP_FIT_EXT_K)):
                good = series[sel] > 0
                if good.sum() < 2:
                    continue
                slope, icpt = np.polyfit(thr_arr[sel][good], np.log10(series[sel][good]), 1)
                ax_b.plot(thr_arr[xext], 10.0 ** (icpt + slope * thr_arr[xext]),
                          ls=":", color=col, alpha=0.55, lw=1.0, zorder=1.8)
        amps, vorts = dists[m]
        ax_c.hist(amps, bins=bins_c, histtype="step", color=col, lw=1.0, ls="--", alpha=0.55)
        ax_c.hist(amps[vorts], bins=bins_c, histtype="step", color=col, lw=1.6)

    # (d) local e-folding rate of the VORTEX counts: a population change would appear as a step;
    # flat curves = single quasi-exponential (no statistically detectable elbow)
    for m in modes:
        vort = np.array(counts[m][1], dtype=float)
        slope = -np.gradient(np.log(np.maximum(vort, 1.0)), thr_arr)
        ax_d.plot(thr_arr, uniform_filter1d(slope, 3, mode="nearest"), color=colors[m], lw=1.6)
    ax_d.set_ylim(0, 0.4)
    ax_d.set_ylabel(r"$-\,\mathrm{d}\ln N_\mathrm{vortex}/\mathrm{d}\Delta T_{pt}$ / K$^{-1}$")

    ax_a.plot([], [], color="0.5", ls="--", label="candidates")
    ax_a.set_ylabel("events / -")
    ax_a.legend(fontsize=7, loc="center right", framealpha=0.9)
    ax_b.set_yscale("log")
    ax_b.set_ylim(0.8, 2.0 * max(max(c[0]) for c in counts.values()))
    ax_b.set_ylabel("events / -")
    ax_c.set_ylabel("events / -")
    ax_c.set_xlim(bins_c[0], 80.0)
    for letter, ax in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        for m in modes:
            ax.axvline(MODE_DEFAULT_AMP_K[m], ls=":" if m == "bwf" else "--",
                       color=colors[m], lw=1.2, alpha=0.9)
        ax.grid(True, which="both" if ax is ax_b else "major", alpha=0.25)
        ax.set_xlabel(r"$\Delta T_{pt}$" + (" / K" if letter == "c" else " threshold / K"))
        ax.grid(True, alpha=0.25)
        _add_panel_label(ax, letter)

    fig.savefig(OUTPUT_DIR / f"{tag}.png", dpi=DPI, facecolor="w", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {tag}.png / .csv  [{time.time() - t0:.1f} s total]")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"model sim: {MODEL_SIM}  |  window {EVENT_WINDOW_MIN:.0f} min  |  "
          f"amp >= {AMP_THRESHOLD_K:.0f} K  |  dt_pt {DT_PT_RANGE_MIN[0]:.0f}-{DT_PT_RANGE_MIN[1]:.0f} min"
          f"  |  FWHM <= {MAX_VERT_FWHM_KM:.0f} km")
    for case in CASES:
        t0 = time.time()
        temp_abs, tprime, times, z, col_xy = load_cube_band(case["cube"])
        events_per_col = [
            detect_events(tprime[c] if DETECT_MODE == "bwf" else temp_abs[c], tprime[c], times, z)
            for c in range(tprime.shape[0])]
        events = [e for ev in events_per_col for e in ev]
        n_ev = len(events)
        n_vort = sum(e["vortex"] for e in events)
        print(f"cube {case['cube']}: {tprime.shape[0]} columns -> {n_ev} candidates, "
              f"{n_vort} vortex ({n_vort / tprime.shape[0]:.2f} / column h)  "
              f"[{time.time() - t0:.1f} s]")
        tag = f"{MODEL_SIM}_lidar_events_cube{case['cube']}{OUT_SUFFIX}"
        if n_ev:
            col = pick_example_column(col_xy, events_per_col, case["cube"])
            hours = (float(times[-1]) - float(times[0])) / 3600.0
            curtain = {"temp2d": temp_abs[col], "times": times, "z": z,
                       "events": events_per_col[col],
                       "label": f"x {col_xy[col][0]:.0f} km, y {col_xy[col][1]:.0f} km",
                       "time_scale": 60.0, "time_xlabel": "model time / min",
                       "time_formatter": None}
            rate_txt = (f"{n_vort}/{n_ev} vortex, "
                        f"{n_vort / (tprime.shape[0] * hours):.2f} / column h")
            draw_event_figure(events, curtain, rate_txt, OUTPUT_DIR / f"{tag}.png",
                      stats_ymax=STATS_YMAX_ARCHIVE)
        write_csv(events_per_col, col_xy, OUTPUT_DIR / f"{tag}.csv")
        print(f"  wrote {tag}.png / .csv  [{time.time() - t0:.1f} s total]")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "coral-sweep":
        main_obs_sweep()
    elif len(sys.argv) > 1 and sys.argv[1] == "coral-all":
        main_obs_all()
    elif len(sys.argv) > 1 and sys.argv[1] == "coral":
        main_obs()
    else:
        main()
