"""Spectral statistics of model virtual-lidar temperature perturbations over a cube.

Companion to ``lidar_obs_model_compare.py`` (shared config + helpers imported as ``base``). Instead
of a single virtual-lidar column it samples an ensemble of columns across the cube on a coarse grid
(every ``STRIDE``-th point in x and y, i.e. ``STRIDE * dx`` spacing), over the **last hour** of cube
data (``base.MODEL_WINDOW_DURATION_S``, as in the obs-vs-model comparison), takes the T' time series
at each ``PROFILE_ALTITUDES`` level (53-62 km, 1 km steps) and builds median PSD spectra
grouped by perturbation strength. Panels:

  (a) perturbation-strength profile vs altitude (median + inter-column IQR spread)
  (b..) one median-spectra panel per CUMULATIVE strength threshold (``PTP_THRESHOLDS_K``,
        descending: e.g. metric >= 15 K / >= 10 K / >= 5 K; a threshold of 0 = all columns).
        The groups overlap by construction -- each panel adds the weaker columns to the previous
        ones, so the spectral shape can be followed from the most active columns to the bulk.

By default uses **observation-like model data** (``MATCH_LIDAR_SAMPLING``): each column is smoothed to
CORAL's effective resolution (2-min / 900-m running average, on the native cube grid) exactly like the
matched mode of ``lidar_obs_model_compare.py``, so the perturbations are what the lidar would see (the
2-min/900-m smoothing damps the raw ~40-60 K swings to ~15-25 K). Set ``MATCH_LIDAR_SAMPLING = False``
for the raw native cube data instead.

Strength split (``STRENGTH_CRITERION``): ``"ptp"`` (default) categorises each (column, altitude) by
a statistic of its temporally-distinct peak-to-peak T' swings (``PTP_STAT``): ``"median"`` (default)
or ``"mean"`` over ALL distinct swings extracted from the record (greedy largest-first with the
+/-window neighbourhood masked after each), or ``"rank"`` for the old single
``PTP_RANK_FOR_SPLIT``-th-largest swing. ``"variance"`` uses the RMS T' instead (thresholds then
cut on RMS in K). T' = T - horizontal mean over
the whole cube at each (z, t); each column spectrum is variance-normalised (like the compare script),
so the group medians compare spectral *shape* (dominant period) independent of amplitude. A dashed
line in each spectra panel marks that group's dominant spectral-peak period. Two figures (cube 0 =
Mount Darwin, cube 1 = Rio Grande). x-axis follows ``base.SPECTRUM_XAXIS``.
"""

import sys
import time

import numpy as np
import xarray as xr
from scipy import signal
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import lidar_obs_model_compare as base

plt.style.use("/work/bd0620/b309199/mapy/src/latex_default.mplstyle")

# base.MODEL_SIM defaults to the old darwin_240718_400m_r1 run; override to the current production run
# (or pass a jobname as argv[1]) so the ensemble stats match the compare / cube-animation figures.
MODEL_SIM = sys.argv[1] if len(sys.argv) > 1 else "darwin_240718_400m_coralT_ifs_wcoast"
MODEL_ROOT = base.MODEL_ROOT
THETA_NAME = base.THETA_NAME
EXNER_NAME = base.EXNER_NAME
OUTPUT_DIR = base.OUTPUT_DIR
MODEL_WINDOW_DURATION_S = base.MODEL_WINDOW_DURATION_S   # analyse only the last hour of cube data
PTP_WINDOW_MIN = 6.0                                     # swing-detection window [min]; decoupled from base's 4 min
#   (compare uses 4 min for fast-oscillation column selection). 6 min captures the peak->trough of
#   oscillations up to ~12-min period (above the dominant 5-8 min band) without measuring the slow envelope.
LIDAR_TIME_RES_S = base.LIDAR_TIME_RES_S                 # 2 min: observation-like time smoothing
LIDAR_VERT_RES_M = base.LIDAR_VERT_RES_M                 # 900 m: observation-like vertical smoothing
PROFILE_ALTITUDES = tuple(53000.0 + 1000.0 * k for k in range(10))   # 53-62 km, 1 km -- the SPECTRA levels
#   (panels b/c); extends base's 55-62 km three levels down into the lower breaking layer.
PANEL_A_ALTITUDES = tuple(50000.0 + 1000.0 * k for k in range(16))   # 50-65 km, 1 km -- the panel-a strength profile
ALTITUDE_CMAP = base.ALTITUDE_CMAP

CASES = (
    {"cube": 0, "label": "PMAP (Mt Darwin)", "out": "lidar_spectral_cube_darwin.png"},
    {"cube": 1, "label": "PMAP (Rio Grande)", "out": "lidar_spectral_cube_coral.png"},
)

STRIDE = 3

# Match the model to CORAL's effective resolution (2-min / 900-m running average, native grid) before
# the analysis, so the ensemble is observation-like (as in lidar_obs_model_compare's matched mode).
# False = raw native cube data.
MATCH_LIDAR_SAMPLING = True

# Perturbation-strength categorisation of each (column, altitude).
STRENGTH_CRITERION = "ptp"     # "ptp" = peak-to-peak swing statistic | "variance" = RMS T' [K]
# Cumulative spectra groups, one panel per threshold (metric >= thr; 0 = all columns), descending.
# argv[2] overrides as a comma-separated list (e.g. "15,10,5"); tagged into the filename.
PTP_THRESHOLDS_K = (tuple(sorted((float(s) for s in sys.argv[2].split(",")), reverse=True))
                    if len(sys.argv) > 2 else (25.0, 15.0, 5.0))
#   default suits the top2 stat; the ~2-3x smaller all-swing median/mean wants e.g. "15,10,5".
PTP_STAT = sys.argv[3] if len(sys.argv) > 3 else "top2"  # "median" | "mean" over ALL distinct swings |
#   "topN" (e.g. "top2") = median of the N largest distinct swings | "rank" = old Nth-largest only
PTP_RANK_FOR_SPLIT = 2         # ("rank" mode only) the Nth-largest distinct swing (2 = old default)
THRESHOLD_COLORS = ("C3", "C1", "C0", "C2")   # panel-a threshold lines + spectra-panel tag frames

SPREAD_PCT = (25.0, 75.0)      # inter-quartile band (darker gray) in panel a
OUTER_PCT = (10.0, 90.0)       # 10-90th-percentile band (lighter gray) in panel a
PERIOD_CUT_MIN = 20.0
FREQ_MAX_MHZ = 6.0

PROFILE_LW = 1.1
PANEL_A_ZLIM_KM = (50.0, 65.0)   # panel-a altitude range (wider than the 55-62 km spectra band)
PANEL_A_XMAX_K = 20.0 if PTP_STAT in ("median", "mean") else 29.0   # fixed panel-a x-axis (ptp mode) so
#   Darwin/CORAL & thresholds are comparable; the all-swing mean/median statistic is ~2-3x smaller
#   than the rank/topN statistics.
PSD_YMAX = 4.0                  # fixed b/c PSD y-axis (0..PSD_YMAX) so the spectra compare across cubes/thresholds
FIGSIZE = (15.5, 4.32)           # sized for 1 profile + len(PTP_THRESHOLDS_K) spectra panels; height kept
#   small so the fixed-point text reads large vs the figure.
DPI = 150

# panel-corner label insets: symmetric top-left (ha=left) / top-right (ha=right) margins, nudged inward
# from base's 0.04 for this smaller figure. XPP_A is larger because panel a is narrower (width ratio 1.0
# vs 1.8), so it needs a bigger fraction to match b/c's absolute right-margin.
XLBL, YPP = 0.05, base.YPP
XPP = 1.0 - XLBL
XPP_A = 0.91


def load_cube_lidar_ensemble(cube_index, altitudes):
    """T'(column, altitude, time) for a stride-sampled grid of columns, plus the model-time axis.

    Keeps only the last MODEL_WINDOW_DURATION_S of cube data. When MATCH_LIDAR_SAMPLING, the model is
    first smoothed to the observation's effective resolution (2-min / 900-m running average) on the
    native cube grid -- so a z-band around the requested `altitudes` is loaded at native vertical
    resolution, smoothed, then sampled at `altitudes`. T' = T - horizontal mean over the whole cube at
    each (z, t); every STRIDE-th column is kept. Returns (tprime[ncol, nz, nt], times, z_m, grid).
    """
    path = MODEL_ROOT / MODEL_SIM / f"cube_{cube_index}.nc"
    ds = xr.open_dataset(path, decode_times=False)

    z = np.asarray(ds["z"].values, dtype=float)
    times = np.asarray(ds["time"].values, dtype=float)
    start = int(np.searchsorted(times, float(times.max()) - MODEL_WINDOW_DURATION_S))  # last hour only
    dt_seconds = float(np.median(np.diff(times[start:])))

    # load a native-resolution z-band with >= LIDAR_VERT_RES_M margin so the 900-m vertical smoothing is
    # clean at the survey levels; sample `altitudes` out of the (smoothed) band afterwards.
    band = (z >= min(altitudes) - LIDAR_VERT_RES_M) & (z <= max(altitudes) + LIDAR_VERT_RES_M)
    band_idx = np.where(band)[0]
    z_band = z[band_idx]
    lvl_in_band = [int(np.argmin(np.abs(z_band - a))) for a in altitudes]
    z_levels = z_band[lvl_in_band]

    temp = ds[THETA_NAME].isel(time=slice(start, None), z=band_idx).values
    temp *= ds[EXNER_NAME].isel(time=slice(start, None), z=band_idx).values
    ds.close()

    if MATCH_LIDAR_SAMPLING:                    # observation-like 2-min / 900-m running average (native grid)
        dz = float(np.median(np.diff(z_band)))
        n_time = max(1, int(round(LIDAR_TIME_RES_S / dt_seconds)))
        n_vert = max(1, int(round(LIDAR_VERT_RES_M / dz)))
        temp = uniform_filter1d(temp, size=n_time, axis=0, mode="nearest")
        temp = uniform_filter1d(temp, size=n_vert, axis=3, mode="nearest")

    temperature = temp[..., lvl_in_band]        # (time, x, y, nz) at the survey levels
    temperature -= temperature.mean(axis=(1, 2), keepdims=True)

    xs = np.arange(0, temperature.shape[1], STRIDE)
    ys = np.arange(0, temperature.shape[2], STRIDE)
    tprime = temperature[:, xs][:, :, ys].transpose(1, 2, 3, 0)
    tprime = tprime.reshape(len(xs) * len(ys), len(lvl_in_band), tprime.shape[3])
    return tprime, times[start:], z_levels, (len(xs), len(ys))


def compute_psd_batch(series, dt_seconds):
    """Variance-normalised PSD for a batch of time series (axis 1 = time).

    Matches ``lidar_obs_model_compare.compute_psd``: linear detrend, Hann window, then divide by the
    integral so each spectrum integrates to one over frequency (cycles/min). Returns period (min)
    and psd (ncol, nfreq).
    """
    y = signal.detrend(np.asarray(series, dtype=float), type="linear", axis=1)
    window = signal.windows.hann(y.shape[1])
    spectrum = np.fft.rfft(y * window, axis=1)
    freq_cpm = np.fft.rfftfreq(y.shape[1], d=dt_seconds) * 60.0

    df_cpm = 60.0 / (y.shape[1] * dt_seconds)
    nonzero = freq_cpm > 0
    psd = np.abs(spectrum[:, nonzero]) ** 2
    area = psd.sum(axis=1, keepdims=True) * df_cpm
    area[area == 0.0] = 1.0
    return 1.0 / freq_cpm[nonzero], psd / area


def column_swings(series, dt_seconds, n_rank=None):
    """Temporally-distinct peak-to-peak T' swings per column (batch, over the time axis).

    Same short PTP_WINDOW_MIN sliding window as the compare script's annotate_peak_to_peak, applied to
    a whole batch of columns. Greedy largest-first extraction: the +/-nwin neighbourhood of an accepted
    swing is masked before the next, until the record is exhausted (or n_rank swings are collected).
    Returns swings[ncol, n_max] sorted descending along axis 1, NaN-padded where a column runs out of
    distinct swings. (Used only for the strength split -- the dominant period is read from the spectral
    peak, not from the window-capped peak->trough time.)
    """
    series = np.asarray(series, dtype=float)
    ncol, nt = series.shape
    nwin = max(3, int(round(PTP_WINDOW_MIN * 60.0 / dt_seconds)))
    field = maximum_filter1d(series, nwin, axis=1, mode="nearest") - \
        minimum_filter1d(series, nwin, axis=1, mode="nearest")   # windowed peak-to-peak per sample
    cols = np.arange(ncol)
    n_max = n_rank if n_rank is not None else nt // nwin + 1
    swings = np.full((ncol, n_max), np.nan)
    work = field.copy()
    for r in range(n_max):
        idx = np.argmax(work, axis=1)
        val = work[cols, idx]
        alive = np.isfinite(val)
        if not alive.any():
            break
        swings[alive, r] = val[alive]
        for c in cols[alive]:                                    # ncol ~ few hundred; cheap
            work[c, max(0, idx[c] - nwin):idx[c] + nwin] = -np.inf
    return swings


def spectral_peak_period(curves, period_min):
    """Dominant resolved wave period [min] = the strongest *interior* local maximum of the group's
    mean spectrum (averaged over altitude), lightly smoothed. Interior-only so the monotonic
    long-period record edge (which inflates the variance-normalised PSD near PERIOD_CUT_MIN) is
    excluded. NaN if the mean spectrum has no interior peak (purely monotonic). This is the period the
    eye reads off panels b/c -- e.g. ~5 min over CORAL vs ~7 min over Darwin (faster vortices -> shorter
    period) -- unlike the peak->trough time, which is capped at 2*PTP_WINDOW_MIN.
    """
    m = np.nanmean(np.asarray(curves, dtype=float), axis=0)
    band = (period_min >= base._freq_mhz_to_period_min(FREQ_MAX_MHZ)) & (period_min <= PERIOD_CUT_MIN)
    p, mm = period_min[band], m[band]
    if mm.size < 3 or not np.isfinite(mm).all():
        return np.nan
    mm = uniform_filter1d(mm, size=3, mode="nearest")
    interior = np.where((mm[1:-1] > mm[:-2]) & (mm[1:-1] > mm[2:]))[0] + 1
    if interior.size == 0:
        return np.nan
    return float(p[interior[int(np.argmax(mm[interior]))]])


def build_figure(cube_index):
    t0 = time.time()
    tprime, times, z_levels, grid = load_cube_lidar_ensemble(cube_index, PANEL_A_ALTITUDES)
    dt_seconds = float(np.median(np.diff(times)))
    ncol, nz, nt = tprime.shape
    print(f"  {grid[0]}x{grid[1]} = {ncol} columns x {nz} levels x {nt} times  [load {time.time() - t0:.1f} s]")

    def column_metric(series_k):
        if STRENGTH_CRITERION != "ptp":
            return np.sqrt((series_k ** 2).mean(axis=1))
        if PTP_STAT == "rank":
            return column_swings(series_k, dt_seconds, n_rank=PTP_RANK_FOR_SPLIT)[:, PTP_RANK_FOR_SPLIT - 1]
        if PTP_STAT.startswith("top"):
            n_top = int(PTP_STAT[3:])
            return np.nanmedian(column_swings(series_k, dt_seconds, n_rank=n_top), axis=1)
        stat = np.nanmedian if PTP_STAT == "median" else np.nanmean
        return stat(column_swings(series_k, dt_seconds), axis=1)

    # ---- panel a: strength profile (median + IQR + 10-90 band) over the full PANEL_A_ALTITUDES range ----
    metric_all = [column_metric(tprime[:, k, :]) for k in range(nz)]
    p10 = [np.percentile(m, OUTER_PCT[0]) for m in metric_all]
    p25 = [np.percentile(m, SPREAD_PCT[0]) for m in metric_all]
    pmed = [np.median(m) for m in metric_all]
    p75 = [np.percentile(m, SPREAD_PCT[1]) for m in metric_all]
    p90 = [np.percentile(m, OUTER_PCT[1]) for m in metric_all]
    zA_km = z_levels / 1000.0

    # ---- spectra (panels b..): one cumulative group per threshold at the PROFILE_ALTITUDES subset ----
    spec_idx = [int(np.argmin(np.abs(z_levels - a))) for a in PROFILE_ALTITUDES]
    psd_levels = []
    period_min = None
    for k in spec_idx:
        period_min, psd = compute_psd_batch(tprime[:, k, :], dt_seconds)
        psd_levels.append(psd)

    group_curves, group_periods = [], []
    for thr in PTP_THRESHOLDS_K:
        curves = []
        counts = []
        for psd, k in zip(psd_levels, spec_idx):
            sel = metric_all[k] >= thr
            counts.append(int(sel.sum()))
            curves.append(np.median(psd[sel], axis=0) if sel.any() else np.full(psd.shape[1], np.nan))
        curves = np.array(curves)
        group_curves.append(curves)
        group_periods.append(spectral_peak_period(curves, period_min))
        print(f"    >= {thr:.0f} K: counts {counts}  peak {group_periods[-1]:.1f} min")
    for k in spec_idx:
        print(f"    z={z_levels[k] / 1000:.0f} km: median metric {np.median(metric_all[k]):.1f} K")

    # the actual sampled model levels (55.083, 56.083, ... km) -- these genuinely sit +83.3 m above the
    # integer km (the surf_uni_abs grid is regular 200 m but phase-shifted by the surface-up build), so
    # the dashes honestly land just above their ticks rather than being faked to integer km.
    z_spec_km = np.array([z_levels[k] for k in spec_idx]) / 1000.0
    alt_norm = Normalize(vmin=min(PROFILE_ALTITUDES) / 1000.0, vmax=max(PROFILE_ALTITUDES) / 1000.0)
    line_colors = [ALTITUDE_CMAP(alt_norm(zz)) for zz in z_spec_km]
    psd_x = period_min if base.SPECTRUM_XAXIS == "period_log" else base._period_min_to_freq_mhz(period_min)

    n_spec = len(PTP_THRESHOLDS_K)
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(1, 1 + n_spec, width_ratios=(1.0,) + (1.8,) * n_spec)
    ax_strength = fig.add_subplot(gs[0, 0])
    ax_specs = []
    for i in range(n_spec):
        ax_specs.append(fig.add_subplot(gs[0, 1 + i], sharey=ax_specs[0] if i else None))

    for zz, col in zip(z_spec_km, line_colors):   # mark the 55-62 km levels shown in b/c (cmap colours)
        ax_strength.axhline(zz, ls="--", color=col, lw=0.9, alpha=0.75, zorder=1.4)
    band10 = ax_strength.fill_betweenx(zA_km, p10, p90, color="0.86", lw=0, zorder=0, label="10-90th pct")
    bandiqr = ax_strength.fill_betweenx(zA_km, p25, p75, color="0.62", lw=0, zorder=1, label="IQR (25-75)")
    medline, = ax_strength.plot(pmed, zA_km, color="black", lw=2.0, zorder=3, label="median")
    ax_strength.set_ylim(*PANEL_A_ZLIM_KM)
    ax_strength.set_xlim(0, PANEL_A_XMAX_K)
    y_lbl = PANEL_A_ZLIM_KM[0] + 0.97 * (PANEL_A_ZLIM_KM[1] - PANEL_A_ZLIM_KM[0])   # like the T_pk labels
    for thr, tcol in zip(PTP_THRESHOLDS_K, THRESHOLD_COLORS):
        if thr <= 0:
            continue
        ax_strength.axvline(thr, ls="--", color=tcol, lw=1.3, zorder=2)
        ax_strength.text(thr, y_lbl, f"{thr:.0f} K", rotation=90, va="top", ha="right",
                         color=tcol, fontsize=7.5, zorder=7)
    if STRENGTH_CRITERION == "ptp":
        if PTP_STAT == "rank":
            stat_txt = {1: "largest", 2: "2nd-largest"}.get(PTP_RANK_FOR_SPLIT, f"{PTP_RANK_FOR_SPLIT}th-largest")
        elif PTP_STAT.startswith("top"):
            stat_txt = f"median of top-{PTP_STAT[3:]} swings"
        else:
            stat_txt = f"{PTP_STAT} swing"
        ax_strength.set_xlabel(r"$\Delta T'_{pp}$ / K  (" + stat_txt + ")")
    else:
        ax_strength.set_xlabel(r"RMS $T'$ / K")
    ax_strength.set_ylabel("altitude / km")
    ax_strength.grid(True, alpha=0.25)
    ax_strength.legend(handles=[medline, bandiqr, band10], loc="lower right", fontsize=6.5,
                       framealpha=0.9, handlelength=1.3, borderpad=0.4, labelspacing=0.3)

    freq_xlim = (base._period_min_to_freq_mhz(PERIOD_CUT_MIN), FREQ_MAX_MHZ)
    metric_sym = r"$\Delta T'_{pp}$" if STRENGTH_CRITERION == "ptp" else r"RMS $T'$"
    psd_top = PSD_YMAX          # fixed 0..PSD_YMAX so panels compare across cubes/thresholds (clips the long-period edge)
    for ax, curves, per, thr, tcol in zip(ax_specs, group_curves, group_periods,
                                          PTP_THRESHOLDS_K, THRESHOLD_COLORS):
        for k in range(len(line_colors)):        # spectra levels (53-62 km), not the panel-a levels
            ax.plot(psd_x, curves[k], color=line_colors[k], lw=PROFILE_LW, zorder=2)
        ax.grid(True, which="major", alpha=0.25)
        base._configure_spectrum_xaxis(ax, freq_xlim=freq_xlim)
        tag = "all columns" if thr <= 0 else metric_sym + f" $\\geq$ {thr:.0f} K"
        tag_box = dict(base.LABEL_BOX_ROUND, edgecolor=tcol if thr > 0 else "black")
        ax.text(XLBL, YPP, tag, transform=ax.transAxes, bbox=tag_box, zorder=6)
        # mark the dominant resolved wave period = the group spectrum's interior spectral peak
        if np.isfinite(per):
            fx = base._period_min_to_freq_mhz(per)
            ax.axvline(fx, ls="--", color="0.35", lw=1.3, zorder=1)
            ax.text(fx, psd_top * 0.97, rf"$T_\mathrm{{pk}}$ = {per:.1f} min", rotation=90,
                    va="top", ha="right", fontsize=7.5, color="0.35", zorder=5)

    ax_specs[0].set_ylim(0, psd_top)
    ax_specs[0].set_ylabel("median normalised PSD")
    ax_specs[0].yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax_specs[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))   # plain integers, no comma/decimal
    for ax in ax_specs[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    sm = ScalarMappable(norm=alt_norm, cmap=ALTITUDE_CMAP)
    cbar = fig.colorbar(sm, ax=ax_specs, location="right", shrink=0.85, pad=0.015,
                        ticks=[a / 1000.0 for a in PROFILE_ALTITUDES])
    cbar.set_label("profile altitude / km")
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    letters = "abcdefg"[:1 + len(ax_specs)]
    for label, ax, xpp in zip(letters, [ax_strength] + ax_specs, [XPP_A] + [XPP] * len(ax_specs)):
        ax.text(xpp, YPP, label, transform=ax.transAxes, ha="right", weight="bold",
                bbox=base.LABEL_BOX_CIRCLE, zorder=6)

    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    thr_txt = ",".join(f"{t:.0f}" for t in PTP_THRESHOLDS_K)
    print(f"model sim: {MODEL_SIM}  |  data: {'obs-like (2min/900m)' if MATCH_LIDAR_SAMPLING else 'raw'}"
          f"  |  stat: {PTP_STAT}  |  thresholds: {thr_txt} K")
    stat_tag = {"median": "med", "mean": "mean", "rank": f"r{PTP_RANK_FOR_SPLIT}"}.get(PTP_STAT, PTP_STAT)
    thr_tag = "-".join(f"{t:.0f}" for t in PTP_THRESHOLDS_K)
    for case in CASES:
        t0 = time.time()
        out_name = case["out"].replace(".png", f"_ptp{stat_tag}_{thr_tag}K.png")
        out = OUTPUT_DIR / f"{MODEL_SIM}_{out_name}"
        print(f"cube {case['cube']} -> {out.name}")
        fig = build_figure(case["cube"])
        fig.savefig(out, dpi=DPI, facecolor="w", bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}  [{time.time() - t0:.1f} s total]")


if __name__ == "__main__":
    main()
