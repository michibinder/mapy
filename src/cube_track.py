#!/usr/bin/env python3

"""Track vortex/instability features in a horizontal cross-section of a PMAP cube and animate them.

Companion to ``cube_lid.py``: instead of the 3D lambda2 multiview it works on a single horizontal
slice (default z ~ 60 km, the same level as the cube animation's w cross-section) and follows the
coherent structures over time with trackpy (Geach et al. 2020, JGR 125, e2020JD033038, section
3.3.2: locate features -> link across frames -> per-trajectory linear velocity fit).

One detection backend is tracked at a time (the TRACK_FIELD knob), at two kernel sizes:
  * ``l2`` -- lambda2 vortex cores (bright = -lambda2 where lambda2 < 0), the criterion drawn as the
    isosurface in the cube animation;
  * ``w``  -- vertical-wind cores (bright = |w|), larger and more coherent, expected to give a more
    robust median/IQR.

Left panel  : |u_h| horizontal-wind-speed slice (Crameri batlow) + lambda2 contour + the tracked
              features as circles (radius ~ the kernel diameter) filled with their current
              per-frame propagation speed on the same colormap but an own, lower scale (second
              colorbar over the right column) -- wave pattern in the wind vs pattern in the
              feature speeds.
Right panel : time series of the ensemble speed of the tracked features -- median line + IQR band,
              for both methods and for both a ground/domain-relative frame (solid + band) and a
              flow-relative frame (feature velocity minus the box-mean wind at the level; dashed).
              A vertical cursor marks the frame currently shown on the left.

Alongside the mp4, a per-run scatter figure (speedsize_<sim>_cube<N>.png + .csv of the per-track
table) is written: one row per level, scattering each propagating track's median ground speed
against (left column) the equivalent diameter of the connected super-threshold detection-field
region it rides on and (right column) the local horizontal wind speed |u_h| at the feature -- the
quantitative version of the speed-colored circles on the |u_h| maps.

Usage:
    python3 cube_track.py darwin_240718_400m_r1 --cube 0          # single test frame
    python3 cube_track.py darwin_240718_400m_r1 notest --cube 0   # full animation
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cube_track")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import matplotlib
matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np
import pandas as pd
import xarray as xr

if os.path.exists('latex_default.mplstyle'):
    plt.style.use('latex_default.mplstyle')

import trackpy as tp
import plt_helper

from cmcrameri import cm as cmc
from scipy.ndimage import gaussian_filter, label as nd_label

tp.quiet()


DATA_ROOT = Path("/scratch/b/b309199")
ANIMATION_ROOT = Path("/work/bd0620/b309199/mapy/data/pmap-animations")

TIME_NAME = "time"
Z_NAME = "z"
Y_NAME = "y"
X_NAME = "x"
U_NAME = "uvelx"
V_NAME = "uvely"
W_NAME = "uvelz"
UAMB_NAME = "uvelx_ambient"
VAMB_NAME = "uvely_ambient"

# Horizontal levels of the tracked cross-sections (m); one figure row per level, ordered highest at
# the top. The wind shears strongly across this range, so the ambient/mean-wind references and the
# tracked speeds should change with altitude. lambda2 needs a vertical gradient, so a thin z-slab of
# +/- LAMBDA2_SLAB_HALF levels around each level is loaded and lambda2 is taken on the centre plane
# (rest of the cube is never read). Override to a single level with the --level CLI flag.
TRACK_LEVELS_M = (56000.0, 60000.0, 64000.0, 68000.0)
LAMBDA2_SLAB_HALF = 3

# Multi-level figure layout: rows = altitude (highest on top), 2 columns (w map | speed time series).
# The left maps are square; after constrained-layout runs, each right panel is snapped to its left
# map's vertical extent so both columns share the same panel height (AXES_MARGIN_IN estimates the
# horizontal space the axis decorations eat, only to pick a figure height with little whitespace).
FIG_WIDTH = 10.0
RIGHT_WIDTH_RATIO = 1.5
AXES_MARGIN_IN = 1.5
CBAR_HEADROOM_IN = 1.3
SHARE_RIGHT_Y = True
RIGHT_Y_ON_RIGHT = True
PANEL_LETTERS = "abcdefghijklmnop"

# Drop frames in the first N seconds from both the animation and the plotted time axis (the tracking
# itself still runs over every frame). The cube time coordinate is model time (seconds since the run
# reference); MODEL_TIME_ORIGIN is the zero used for the HH:MM:SS labels.
ANIMATION_START_OFFSET_S = 30 * 60
MODEL_TIME_ORIGIN = 0.0

# Light Gaussian pre-smoothing (px) of each detection field before trackpy, and normalisation of the
# field so its PCTL_NORM percentile maps to NORM_SCALE (makes minmass portable between w and lambda2,
# whose raw magnitudes differ by orders of magnitude).
PREFILTER_SIGMA_PX = 1.0
PCTL_NORM = 99.5
NORM_SCALE = 255.0

# trackpy linking. Features advect ~50 m/s; at dt ~ 15.5 s and dx = 400 m that is ~2 px/frame, so a
# few-px search range is ample. MIN_TRACK_LENGTH ~ Geach's "min occurrences" (they scanned 10-50).
SEARCH_RANGE_PX = 5
LINK_MEMORY = 2
MIN_TRACK_LENGTH = 8

# Per-frame feature velocity = slope of a local linear fit of position vs time over this many track
# samples (centred), mirroring Geach's constant-velocity per-trajectory fit but time-resolved.
VELOCITY_WINDOW = 7

# Detection backend. A SINGLE field is tracked per run -- TRACK_FIELD = "w" (|w| features, denser
# ensemble) or "l2" (-lambda2 vortex cores) -- at TWO characteristic feature sizes so we can compare
# small vs large structures (does scale set the
# speed?). ``diameter`` (odd px) is trackpy's characteristic feature size -- the key knob (Geach:
# robust over 11-400 px; the sweep confirms the median speed is insensitive to it). The small/large
# split contrasts vortex-core scale against larger coherent structures. trackpy needs an odd pixel
# diameter, so at dx = 400 m the achievable sizes are odd*0.4 km: SMALL_DIAMETER = 7 px = 2.8 km
# (~3 km, Geach's vortex cores), LARGE_DIAMETER = 23 px = 9.2 km (~9 km, between the vortex scale and
# the ~18 km GW wavelength); the legend rounds these to whole km. Color = field, line style / circle =
# scale. Set ENABLE_SCALE_SPLIT = False to fall back to one diameter per field. ``minmass`` in the
# normalised units above (None -> trackpy default). Tune with track_param_sweep.py.
FIELD_COLORS = {"l2": "#d1495b", "w": "#1b9e77"}
FIELD_LABELS = {"l2": r"$\lambda_2$", "w": r"$w$"}
TRACK_FIELD = "w"
SMALL_DIAMETER = 7
LARGE_DIAMETER = 23
SCALE_LS = {"small": "-", "large": (0, (5, 2))}
ENABLE_SCALE_SPLIT = True


def _build_methods():
    if TRACK_FIELD not in FIELD_COLORS:
        raise ValueError(f"TRACK_FIELD must be one of {tuple(FIELD_COLORS)}, got {TRACK_FIELD!r}")
    scales = ([("small", SMALL_DIAMETER), ("large", LARGE_DIAMETER)] if ENABLE_SCALE_SPLIT
              else [("small", SMALL_DIAMETER)])
    methods = []
    for field in (TRACK_FIELD,):
        for scale, diameter in scales:
            suffix = f" {scale}" if ENABLE_SCALE_SPLIT else ""
            methods.append({
                "key": f"{field}_{scale[0]}",
                "field": field,
                "scale": scale,
                "label": f"{FIELD_LABELS[field]}{suffix}",
                "color": FIELD_COLORS[field],
                "ls": SCALE_LS[scale],
                "band": scale == "small",
                "marker": True,
                "diameter": diameter,
                "minmass": None,
                "separation": None,
            })
    return methods


METHODS = _build_methods()
FIELDS_IN_USE = tuple(dict.fromkeys(m["field"] for m in METHODS))

# Per-track scatter figure (written alongside the mp4 as speedsize_<sim>_cube<N>.png/.csv), one row
# per level, two columns: median ground speed vs feature size (LEFT) and vs the local horizontal
# wind speed |u_h| sampled at the feature positions (RIGHT, with a 1:1 "pure advection" line and a
# vertical ambient-wind line). Size = equivalent diameter 2*sqrt(area/pi) of the 8-connected region
# where the raw detection field exceeds the SEGMENT_PERCENTILE percentile of its positive values at
# that level (for lambda2 the same statistic as the map contour level), looked up at the strongest
# pixel within half a detection diameter of the centroid. One point per track (medians over the
# displayed time window); a track must move a net MIN_NET_DISPLACEMENT_KM to count as a propagating
# feature. Color = field, filled / open circle = small / large detection kernel; per-field binned
# medians over SIZE_BIN_KM / UH_BIN_MS bins with at least MIN_BIN_COUNT tracks.
SEGMENT_PERCENTILE = 92.0
MIN_NET_DISPLACEMENT_KM = 2.0
SIZE_BIN_KM = 2.0
UH_BIN_MS = 10.0
MIN_BIN_COUNT = 5
SCATTER_MARKER_SIZE = 16.0
SCATTER_ALPHA = 0.5
SCATTER_PANEL_WIDTH = 5.4
SCATTER_PANEL_HEIGHT = 2.7
ONE_TO_ONE_COLOR = "0.5"
ONE_TO_ONE_LW = 1.2
ONE_TO_ONE_LS = (0, (3, 2))

# Left panel styling. The map shows the absolute horizontal wind |u_h| at the level (Crameri batlow,
# sequential) so the wave pattern in the wind can be compared with the propagation-speed pattern of
# the tracked features: the circles are FILLED with their current per-frame speed using the SAME
# colormap but an own (lower) scale -- field colorbar over the left column, feature-speed colorbar
# over the right column. *_CLIM = None -> per-run percentile auto-scale (rounded up to 10 / 5 m/s).
MAP_FIELD_CMAP = cmc.batlow
FIELD_SPEED_CLIM = None
FIELD_SPEED_PCTL = 99.5
FIELD_CBAR_TICK_STEP = 20.0
CIRCLE_SPEED_CLIM = None
CIRCLE_SPEED_PCTL = 95.0
CIRCLE_CBAR_TICK_STEP = 10.0
MARKER_EDGE_COLOR = "black"
LAMBDA2_CONTOUR_COLOR = "black"
LAMBDA2_CONTOUR_WIDTH = 0.7
NEGATIVE_LAMBDA2_ONLY = True
LAMBDA2_PERCENTILE = 8.0
MARKER_LW = 1.3
MARKER_ALPHA = 0.9
SHOW_LIDAR_WINDOW = False
LIDAR_HALF_WINDOW_M = 10000.0
WINDOW_CENTER_OVERRIDES = {
    0: (-200.0, -57400.0),
    1: (93800.0, 23800.0),
}

# Right panel styling.
SHOW_IQR = False
IQR_ALPHA = 0.16
MEDIAN_LW = 1.9
SHOW_MEAN = False
LEGEND_FONTSIZE = 12
# Combined ("all features") curve: median + IQR of the pooled distribution of every tracked feature
# (both fields, both scales) per frame, drawn as a heavy black line with a grey IQR band. Each right
# panel also notes the end-of-run gap between this combined median and the ambient wind.
OVERALL_COLOR = "black"
OVERALL_LW = 2.4
OVERALL_BAND_COLOR = "0.55"
OVERALL_BAND_ALPHA = 0.20
DIFF_LABEL_FONTSIZE = 10
# Low-pass every plotted speed curve with a centred running mean of cutoff period SMOOTH_CUTOFF_S
# (5 min) -> sub-5-min spikes are removed so the scale / altitude differences read cleanly.
SMOOTH_CUTOFF_S = 300.0
# Wind reference lines: constant ambient wind speed at the level (1D profile), and the speed of the
# box-mean horizontal wind of the cross-section over time (the evolving "mean wind").
AMBIENT_LINE_COLOR = "0.35"
AMBIENT_LINE_STYLE = (0, (6, 3))
AMBIENT_LINE_LW = 2.3
MEANWIND_LINE_COLOR = "black"
MEANWIND_LINE_STYLE = (0, (1, 1.1))
MEANWIND_LINE_LW = 2.3

XLBL = 0.04
YPP = 0.93
XPP = 1.0 - XLBL

STRIDE_XY = 1  # optional horizontal subsampling for the whole analysis
CLEAR_EXISTING_FRAMES = True


# --------------------------------------------------------------------------------------------------
# reused helpers (pyvista-free copies from cube_lid.py)
# --------------------------------------------------------------------------------------------------

def list_available_cubes(sim_dir: Path):
    def cube_index(path):
        try:
            return int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            return 1_000_000
    cubes = sorted(sim_dir.glob("cube_*.nc"), key=cube_index)
    legacy = sim_dir / "cube.nc"
    if legacy.exists():
        cubes.append(legacy)
    return cubes


def resolve_nc_path(simulation_name: str, cube_index: int = 0) -> Path:
    simulation_name = os.path.basename(os.path.normpath(simulation_name))
    sim_dir = DATA_ROOT / simulation_name
    candidate = sim_dir / f"cube_{int(cube_index)}.nc"
    if candidate.exists():
        return candidate
    if int(cube_index) == 0 and (sim_dir / "cube.nc").exists():
        return sim_dir / "cube.nc"
    available = list_available_cubes(sim_dir)
    available_str = ", ".join(p.name for p in available) if available else "none"
    raise FileNotFoundError(
        f"Could not find cube_{cube_index}.nc for '{simulation_name}' at {sim_dir} "
        f"(available cubes: {available_str})."
    )


def compute_elapsed_seconds(value, time_start_value=MODEL_TIME_ORIGIN):
    return np.asarray(value, dtype=np.float64) - float(time_start_value)


def format_elapsed_hms(total_seconds, with_seconds=True):
    total_seconds = max(0, int(np.rint(float(total_seconds))))
    hrs = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}" if with_seconds else f"{hrs:02d}:{mins:02d}"


def format_elapsed_tick_label(value, pos=None):
    return format_elapsed_hms(value, with_seconds=False)


def add_panel_label(ax, label, x=XPP, y=YPP):
    ax.text(
        x, y, label.strip("()"),
        transform=ax.transAxes, ha="right", va="top",
        fontsize=12, fontweight="bold",
        bbox={"boxstyle": "circle", "facecolor": "white", "edgecolor": "black", "linewidth": 0.67},
        zorder=6,
    )


def add_corner_label(ax, text, x=XLBL, y=YPP):
    ax.text(
        x, y, text,
        transform=ax.transAxes, ha="left", va="top",
        bbox={"boxstyle": "round", "lw": 0.67, "facecolor": "white", "edgecolor": "black"},
        zorder=6,
    )


def window_center_for_cube(x_vals, y_vals, cube_index):
    override = WINDOW_CENTER_OVERRIDES.get(int(cube_index))
    if override is None:
        return float(np.mean(x_vals)), float(np.mean(y_vals))
    xc = min(max(float(override[0]), float(x_vals.min()) + LIDAR_HALF_WINDOW_M),
             float(x_vals.max()) - LIDAR_HALF_WINDOW_M)
    yc = min(max(float(override[1]), float(y_vals.min()) + LIDAR_HALF_WINDOW_M),
             float(y_vals.max()) - LIDAR_HALF_WINDOW_M)
    return xc, yc


# --------------------------------------------------------------------------------------------------
# field preparation
# --------------------------------------------------------------------------------------------------

def compute_lambda2(u, v, w, x, y, z):
    """lambda2 (2nd eigenvalue of S^2 + Omega^2) for a 3D (nz, ny, nx) velocity block."""
    du_dz, du_dy, du_dx = np.gradient(u, z, y, x, edge_order=2)
    dv_dz, dv_dy, dv_dx = np.gradient(v, z, y, x, edge_order=2)
    dw_dz, dw_dy, dw_dx = np.gradient(w, z, y, x, edge_order=2)

    sxy = 0.5 * (du_dy + dv_dx)
    sxz = 0.5 * (du_dz + dw_dx)
    syz = 0.5 * (dv_dz + dw_dy)
    oxy = 0.5 * (du_dy - dv_dx)
    oxz = 0.5 * (du_dz - dw_dx)
    oyz = 0.5 * (dv_dz - dw_dy)

    shape = u.shape
    s = np.zeros((*shape, 3, 3), dtype=np.float64)
    o = np.zeros_like(s)
    s[..., 0, 0] = du_dx
    s[..., 1, 1] = dv_dy
    s[..., 2, 2] = dw_dz
    s[..., 0, 1] = s[..., 1, 0] = sxy
    s[..., 0, 2] = s[..., 2, 0] = sxz
    s[..., 1, 2] = s[..., 2, 1] = syz
    o[..., 0, 1] = oxy
    o[..., 1, 0] = -oxy
    o[..., 0, 2] = oxz
    o[..., 2, 0] = -oxz
    o[..., 1, 2] = oyz
    o[..., 2, 1] = -oyz

    eigvals = np.linalg.eigvalsh(s @ s + o @ o)
    return eigvals[..., 1]


def load_level_fields(ds, level_m):
    """Read the thin z-slab around ``level_m`` and build the per-frame 2D fields used for tracking.

    Returns a dict with the horizontal grid (x, y in m), the time coordinate, the level used, the
    w slice ``W`` and lambda2 slice ``L2`` (both (nt, ny, nx)), the box-mean horizontal wind speed at
    the level over time ``mean_wind_speed`` (nt,), and the constant ambient wind speed there."""
    x = np.asarray(ds[X_NAME].values, dtype=float)[::STRIDE_XY]
    y = np.asarray(ds[Y_NAME].values, dtype=float)[::STRIDE_XY]
    z = np.asarray(ds[Z_NAME].values, dtype=float)
    times = np.asarray(ds[TIME_NAME].values, dtype=float)
    nz = z.size

    center = int(np.argmin(np.abs(z - float(level_m))))
    lo = max(0, center - LAMBDA2_SLAB_HALF)
    hi = min(nz, center + LAMBDA2_SLAB_HALF + 1)
    z_slab = z[lo:hi]
    center_local = center - lo

    def load_slab(name):
        da = ds[name].isel({Z_NAME: slice(lo, hi)}).transpose(TIME_NAME, Z_NAME, Y_NAME, X_NAME)
        arr = np.asarray(da.values, dtype=np.float64)
        return arr[:, :, ::STRIDE_XY, ::STRIDE_XY]

    u_slab = load_slab(U_NAME)
    v_slab = load_slab(V_NAME)
    w_slab = load_slab(W_NAME)

    nt = u_slab.shape[0]
    W = w_slab[:, center_local, :, :].copy()
    S = np.hypot(u_slab[:, center_local, :, :], v_slab[:, center_local, :, :])
    L2 = np.empty((nt, y.size, x.size), dtype=np.float64)
    for t in range(nt):
        L2[t] = compute_lambda2(u_slab[t], v_slab[t], w_slab[t], x, y, z_slab)[center_local]

    umean = np.nanmean(u_slab[:, center_local, :, :], axis=(1, 2))
    vmean = np.nanmean(v_slab[:, center_local, :, :], axis=(1, 2))
    mean_wind_speed = np.hypot(umean, vmean)

    ua = ds[UAMB_NAME].isel({Z_NAME: center}).transpose(TIME_NAME, Y_NAME, X_NAME)
    va = ds[VAMB_NAME].isel({Z_NAME: center}).transpose(TIME_NAME, Y_NAME, X_NAME)
    ambient_speed = float(np.hypot(np.nanmean(np.asarray(ua.values)),
                                   np.nanmean(np.asarray(va.values))))

    return {
        "x": x, "y": y, "times": times,
        "level_m": float(z[center]),
        "W": W, "S": S, "L2": L2,
        "mean_wind_speed": mean_wind_speed,
        "ambient_speed": ambient_speed,
    }


def detection_field(kind, W, L2):
    """Non-negative 2D field stack (nt, ny, nx) fed to trackpy for a detection backend."""
    if kind == "w":
        field = np.abs(W)
    elif kind == "l2":
        field = np.clip(-L2, 0.0, None)
    else:
        raise ValueError(f"unknown detection field {kind!r}")
    return field


def normalize_frame(frame2d):
    frame2d = np.asarray(frame2d, dtype=np.float64)
    if PREFILTER_SIGMA_PX and PREFILTER_SIGMA_PX > 0:
        frame2d = gaussian_filter(frame2d, sigma=PREFILTER_SIGMA_PX)
    ref = np.nanpercentile(frame2d, PCTL_NORM)
    if not np.isfinite(ref) or ref <= 0:
        ref = float(np.nanmax(frame2d)) or 1.0
    return np.clip(frame2d / ref * NORM_SCALE, 0.0, None)


# --------------------------------------------------------------------------------------------------
# detection + linking + velocities  (trackpy)
# --------------------------------------------------------------------------------------------------

def detect_features(field_stack, method):
    """Run trackpy.locate on every frame; return a features DataFrame with a 'frame' column."""
    diameter = int(method["diameter"])
    if diameter % 2 == 0:
        diameter += 1
    separation = method.get("separation") or (diameter + 2)
    frames = []
    for t in range(field_stack.shape[0]):
        img = normalize_frame(field_stack[t])
        feats = tp.locate(img, diameter, minmass=method.get("minmass"),
                          separation=separation, invert=False)
        if feats is not None and len(feats):
            feats = feats.copy()
            feats["frame"] = t
            frames.append(feats)
    if not frames:
        return pd.DataFrame(columns=["x", "y", "mass", "size", "frame"])
    return pd.concat(frames, ignore_index=True)


def _local_slope(t, values, window):
    """Per-sample slope d(values)/dt from a centred local linear fit of half-width window//2."""
    n = len(t)
    out = np.full(n, np.nan)
    half = max(1, int(window) // 2)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        tt = t[lo:hi]
        vv = values[lo:hi]
        if len(tt) >= 2 and np.ptp(tt) > 0:
            out[i] = np.polyfit(tt - tt.mean(), vv - vv.mean(), 1)[0]
    return out


def link_and_measure(features, times_s, x, y):
    """Link features across frames, drop stubs, and attach physical positions and velocities.

    Adds columns: xm, ym (m), u, v (m/s, local linear fit), speed_ground (m/s)."""
    if features is None or not len(features):
        return features
    dx = float(np.mean(np.diff(x))) if x.size > 1 else 1.0
    dy = float(np.mean(np.diff(y))) if y.size > 1 else 1.0
    x0, y0 = float(x[0]), float(y[0])

    traj = tp.link(features, search_range=SEARCH_RANGE_PX, memory=LINK_MEMORY)
    traj = tp.filter_stubs(traj, threshold=MIN_TRACK_LENGTH)
    if not len(traj):
        return traj

    traj = traj.reset_index(drop=True)
    traj = traj.sort_values(["particle", "frame"]).reset_index(drop=True)
    xm = x0 + traj["x"].to_numpy() * dx
    ym = y0 + traj["y"].to_numpy() * dy
    traj["xm"] = xm
    traj["ym"] = ym

    u = np.full(len(traj), np.nan)
    v = np.full(len(traj), np.nan)
    frame_all = traj["frame"].to_numpy().astype(int)
    for _pid, idx in traj.groupby("particle").indices.items():
        idx = np.asarray(idx)
        tt = times_s[frame_all[idx]]
        u[idx] = _local_slope(tt, xm[idx], VELOCITY_WINDOW)
        v[idx] = _local_slope(tt, ym[idx], VELOCITY_WINDOW)
    traj["u"] = u
    traj["v"] = v
    traj["speed_ground"] = np.hypot(u, v)
    return traj


def segmentation_threshold(field_stack):
    """Constant per-level threshold: SEGMENT_PERCENTILE of the positive detection-field values."""
    vals = field_stack[np.isfinite(field_stack) & (field_stack > 0)]
    if vals.size == 0:
        return None
    return float(np.percentile(vals, SEGMENT_PERCENTILE))


def attach_feature_sizes(traj, field_stack, threshold, dx_km, dy_km, half_px):
    """Attach the size of the connected super-threshold region to every tracked feature.

    Per frame, the raw detection field is thresholded and 8-connected regions are labelled; each
    feature takes the region of the strongest pixel within +/- half_px of its centroid (NaN if that
    pixel is below the threshold). Adds the column ``eq_diam_km`` (equivalent diameter of the
    region area)."""
    eq = np.full(len(traj), np.nan)
    if threshold is not None and len(traj):
        cell_km2 = dx_km * dy_km
        structure = np.ones((3, 3), dtype=bool)
        rows = traj["y"].to_numpy(dtype=float)
        cols = traj["x"].to_numpy(dtype=float)
        frames = traj["frame"].to_numpy().astype(int)
        ny, nx = field_stack.shape[1:]
        for f in np.unique(frames):
            fld = field_stack[f]
            labels, _ = nd_label(fld >= threshold, structure=structure)
            counts = np.bincount(labels.ravel())
            for i in np.nonzero(frames == f)[0]:
                r0 = min(max(int(round(rows[i])), 0), ny - 1)
                c0 = min(max(int(round(cols[i])), 0), nx - 1)
                rlo, rhi = max(0, r0 - half_px), min(ny, r0 + half_px + 1)
                clo, chi = max(0, c0 - half_px), min(nx, c0 + half_px + 1)
                win = fld[rlo:rhi, clo:chi]
                rr, cc = np.unravel_index(int(np.argmax(win)), win.shape)
                lab = labels[rlo + rr, clo + cc]
                if lab > 0:
                    eq[i] = 2.0 * np.sqrt(counts[lab] * cell_km2 / np.pi)
    traj = traj.copy()
    traj["eq_diam_km"] = eq
    return traj


def sample_field_at_features(traj, field_stack):
    """Nearest-pixel value of a (nt, ny, nx) field at every tracked feature position."""
    nt, ny, nx = field_stack.shape
    frames = traj["frame"].to_numpy().astype(int)
    rows = np.clip(np.rint(traj["y"].to_numpy(dtype=float)).astype(int), 0, ny - 1)
    cols = np.clip(np.rint(traj["x"].to_numpy(dtype=float)).astype(int), 0, nx - 1)
    return field_stack[frames, rows, cols]


def per_track_speed_size(track_table, start_index, dx_km):
    """One record per propagating track: median size vs median ground speed in the shown window."""
    records = []
    if track_table is None or not len(track_table):
        return records
    sub = track_table[track_table["frame"] >= int(start_index)]
    for pid, g in sub.groupby("particle"):
        if len(g) < MIN_TRACK_LENGTH:
            continue
        g = g.sort_values("frame")
        net_km = float(np.hypot(g["xm"].iloc[-1] - g["xm"].iloc[0],
                                g["ym"].iloc[-1] - g["ym"].iloc[0])) / 1000.0
        if net_km < MIN_NET_DISPLACEMENT_KM:
            continue
        diam_vals = g["eq_diam_km"].to_numpy(dtype=float)
        if not np.isfinite(diam_vals).any():
            continue
        size_km = float(np.nanmedian(diam_vals))
        speed = float(np.nanmedian(g["speed_ground"]))
        if not np.isfinite(speed):
            continue
        records.append({
            "particle": int(pid), "n": int(len(g)), "size_km": size_km,
            "gyration_radius_km": float(np.nanmedian(g["size"])) * dx_km,
            "speed_ms": speed, "uh_ms": float(np.nanmedian(g["uh_ms"])),
            "net_disp_km": net_km,
        })
    return records


def per_frame_stats(traj, nt):
    """Ensemble median / IQR / mean / count of the tracked-feature speed at each frame."""
    keys = ["median_g", "q25_g", "q75_g", "mean_g", "count"]
    stats = {k: np.full(nt, np.nan) for k in keys}
    stats["count"] = np.zeros(nt)
    if traj is None or not len(traj):
        return stats
    full = pd.RangeIndex(nt)
    g = traj.groupby("frame")
    stats["median_g"] = g["speed_ground"].median().reindex(full).to_numpy()
    stats["q25_g"] = g["speed_ground"].quantile(0.25).reindex(full).to_numpy()
    stats["q75_g"] = g["speed_ground"].quantile(0.75).reindex(full).to_numpy()
    stats["mean_g"] = g["speed_ground"].mean().reindex(full).to_numpy()
    stats["count"] = g.size().reindex(full).fillna(0).to_numpy()
    return stats


def smooth_lowpass(y, window):
    """Centred running-mean low-pass of a 1D series (NaN-aware); window in samples."""
    y = np.asarray(y, dtype=float)
    if window is None or window <= 1:
        return y
    return pd.Series(y).rolling(int(window), center=True, min_periods=1).mean().to_numpy()


def features_by_frame(traj):
    """frame -> (xm_km, ym_km, speed_ground) arrays of the tracked features in that frame."""
    out = {}
    if traj is None or not len(traj):
        return out
    for f, sub in traj.groupby("frame"):
        out[int(f)] = (sub["xm"].to_numpy() / 1000.0, sub["ym"].to_numpy() / 1000.0,
                       sub["speed_ground"].to_numpy(dtype=float))
    return out


def run_method(method, fields):
    """Full detect -> link -> size -> stats pipeline for one backend."""
    W, L2 = fields["W"], fields["L2"]
    times_s = compute_elapsed_seconds(fields["times"])
    nt = W.shape[0]
    field_stack = detection_field(method["field"], W, L2)
    feats = detect_features(field_stack, method)
    traj = link_and_measure(feats, times_s, fields["x"], fields["y"])
    has_traj = traj is not None and len(traj)
    if has_traj:
        dx_km = float(np.mean(np.diff(fields["x"]))) / 1000.0
        dy_km = float(np.mean(np.diff(fields["y"]))) / 1000.0
        traj = attach_feature_sizes(traj, field_stack, segmentation_threshold(field_stack),
                                    dx_km, dy_km, half_px=max(1, int(method["diameter"]) // 2))
        traj["uh_ms"] = sample_field_at_features(traj, fields["S"])
    ntracks = 0 if not has_traj else int(traj["particle"].nunique())
    print(f"[i]  method {method['key']}: {0 if feats is None else len(feats)} detections, "
          f"{ntracks} tracks (>= {MIN_TRACK_LENGTH} frames)")
    return {
        "diameter": int(method["diameter"]),
        "stats": per_frame_stats(traj, nt),
        "feats": features_by_frame(traj),
        "traj": traj[["frame", "speed_ground"]] if has_traj else None,
        "track_table": (traj[["particle", "frame", "xm", "ym", "speed_ground", "size",
                              "eq_diam_km", "uh_ms"]].copy() if has_traj else None),
    }


# --------------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------------

def choose_lambda2_level(L2):
    vals = L2[np.isfinite(L2)]
    if NEGATIVE_LAMBDA2_ONLY:
        vals = vals[vals < 0]
    if vals.size == 0:
        return None
    return float(np.percentile(vals, LAMBDA2_PERCENTILE))


def _draw_left_map(axl, state, lev, time_index, dx_km, is_top, is_bottom, current_x, letter):
    x_km = state["x"] / 1000.0
    y_km = state["y"] / 1000.0
    mesh = axl.pcolormesh(x_km, y_km, lev["S"][time_index], cmap=MAP_FIELD_CMAP,
                          vmin=state["field_clim"][0], vmax=state["field_clim"][1],
                          shading="auto", rasterized=True)
    if lev["lambda2_level"] is not None:
        axl.contour(x_km, y_km, lev["L2"][time_index], levels=[lev["lambda2_level"]],
                    colors=LAMBDA2_CONTOUR_COLOR, linewidths=LAMBDA2_CONTOUR_WIDTH)
    circle_norm = plt.Normalize(*state["circle_clim"])
    for method in lev["methods"]:
        if not method.get("marker", True):
            continue
        radius_km = 0.5 * method["diameter"] * dx_km
        pts = method["feats"].get(int(time_index))
        if pts is None:
            continue
        xs, ys, sp = pts
        for xc, yc, sc in zip(xs, ys, sp):
            fc = MAP_FIELD_CMAP(circle_norm(sc)) if np.isfinite(sc) else "none"
            axl.add_patch(Circle((xc, yc), radius_km, facecolor=fc, edgecolor=MARKER_EDGE_COLOR,
                                 lw=MARKER_LW, alpha=MARKER_ALPHA, linestyle=method.get("ls", "-"),
                                 zorder=5))
    axl.set_xlim(x_km.min(), x_km.max())
    axl.set_ylim(y_km.min(), y_km.max())
    xr = float(x_km.max() - x_km.min()) or 1.0
    axl.set_box_aspect(float(y_km.max() - y_km.min()) / xr)
    axl.set_ylabel("y / km")
    if is_bottom:
        axl.set_xlabel("x / km")
    else:
        axl.tick_params(labelbottom=False)
    add_panel_label(axl, letter)
    clabel = f"$z = {lev['level_m'] / 1000.0:.0f}$ km"
    if is_top:
        clabel += f"\n$t = {format_elapsed_hms(current_x)}$"
    add_corner_label(axl, clabel)
    if is_top:
        lh = [Line2D([], [], marker="o", ls="none", markerfacecolor="none",
                     markeredgecolor=MARKER_EDGE_COLOR,
                     markersize=6 if m["diameter"] <= SMALL_DIAMETER else 9,
                     label=f"{m['diameter'] * dx_km:.0f} km")
              for m in lev["methods"] if m.get("marker", True)]
        axl.legend(handles=lh, loc="lower left", fontsize=8, framealpha=0.9, handletextpad=0.3,
                   borderpad=0.3, title=f"tracked {FIELD_LABELS[TRACK_FIELD]}", title_fontsize=8)
    return mesh


def _method_label(m, dx_km):
    """Legend label carrying the tracked feature size rounded to whole km, e.g. '$\\lambda_2$ 3 km'."""
    return f"{FIELD_LABELS[m['field']]} {m['diameter'] * dx_km:.0f} km"


def _draw_right_series(axr, state, lev, time_index, current_x, is_top, is_bottom, letter, dx_km):
    elapsed = state["times_s"]
    for method in lev["methods"]:
        st = method["stats"]
        color = method["color"]
        if SHOW_IQR and method.get("band", True):
            axr.fill_between(elapsed, st["q25_g"], st["q75_g"], color=color, alpha=IQR_ALPHA,
                             linewidth=0, zorder=1)
        axr.plot(elapsed, st["median_plot"], color=color, lw=MEDIAN_LW, ls=method.get("ls", "-"),
                 zorder=3)
    ov = lev["overall"]
    axr.fill_between(elapsed, ov["q25_plot"], ov["q75_plot"], color=OVERALL_BAND_COLOR,
                     alpha=OVERALL_BAND_ALPHA, linewidth=0, zorder=1)
    axr.plot(elapsed, ov["median_plot"], color=OVERALL_COLOR, lw=OVERALL_LW, zorder=4)
    axr.plot(elapsed, lev["mean_wind_plot"], color=MEANWIND_LINE_COLOR, lw=MEANWIND_LINE_LW,
             ls=MEANWIND_LINE_STYLE, zorder=3)
    axr.axhline(lev["ambient_speed"], color=AMBIENT_LINE_COLOR, lw=AMBIENT_LINE_LW,
                ls=AMBIENT_LINE_STYLE, zorder=2)
    axr.axvline(current_x, color="black", lw=1.5, ls="--", zorder=4)
    axr.grid(True, alpha=0.25)
    axr.xaxis.set_major_formatter(mticker.FuncFormatter(format_elapsed_tick_label))
    axr.xaxis.set_major_locator(mticker.MultipleLocator(600))
    if RIGHT_Y_ON_RIGHT:
        axr.yaxis.set_label_position("right")
        axr.yaxis.tick_right()
    axr.set_ylabel(r"feature speed / m$\,$s$^{-1}$")
    if is_bottom:
        axr.set_xlabel("model time / hh:mm")
    else:
        axr.tick_params(labelbottom=False)
    add_panel_label(axr, letter)
    med_now = float(ov["median_plot"][int(time_index)])
    if np.isfinite(med_now):
        diff_now = med_now - float(lev["ambient_speed"])
        rows = (rf"$v_\mathrm{{all}} = {med_now:.0f}$ m$\,$s$^{{-1}}$" + "\n"
                + rf"$v_\mathrm{{all}}\!-\!U_\mathrm{{amb}} = {diff_now:+.0f}$ m$\,$s$^{{-1}}$")
    else:
        rows = (r"$v_\mathrm{all} = $ --" + "\n" + r"$v_\mathrm{all}\!-\!U_\mathrm{amb} = $ --")
    axr.text(0.035, 0.05, rows, transform=axr.transAxes, ha="left", va="bottom",
             fontsize=DIFF_LABEL_FONTSIZE, multialignment="left",
             bbox={"boxstyle": "round", "lw": 0.67, "facecolor": "white", "edgecolor": "black"},
             zorder=6)
    if is_top:
        handles = [Line2D([], [], color=m["color"], lw=MEDIAN_LW, ls=m.get("ls", "-"),
                          label=_method_label(m, dx_km)) for m in lev["methods"]]
        handles.append(Line2D([], [], color=OVERALL_COLOR, lw=OVERALL_LW, label="all (median)"))
        handles.append(Patch(facecolor=OVERALL_BAND_COLOR, alpha=OVERALL_BAND_ALPHA,
                             label="all (IQR)"))
        handles.append(Line2D([], [], color=MEANWIND_LINE_COLOR, lw=MEANWIND_LINE_LW,
                              ls=MEANWIND_LINE_STYLE, label="mean wind"))
        handles.append(Line2D([], [], color=AMBIENT_LINE_COLOR, lw=AMBIENT_LINE_LW,
                              ls=AMBIENT_LINE_STYLE, label="ambient wind"))
        axr.legend(handles=handles, loc="upper left", fontsize=LEGEND_FONTSIZE, framealpha=0.92,
                   ncol=2, columnspacing=1.2, handlelength=2.4, borderpad=0.5, labelspacing=0.4)


def render_frame(state, time_index):
    x_km = state["x"] / 1000.0
    current_x = float(state["times_s"][time_index])
    levels = state["levels"]
    n = len(levels)
    dx_km = float(np.mean(np.diff(x_km))) if x_km.size > 1 else 0.4

    left_axes_w = (FIG_WIDTH - AXES_MARGIN_IN) / (1.0 + RIGHT_WIDTH_RATIO)
    fig = plt.figure(figsize=(FIG_WIDTH, left_axes_w * n + CBAR_HEADROOM_IN), layout="constrained")
    gs = fig.add_gridspec(n, 2, width_ratios=[1.0, RIGHT_WIDTH_RATIO])
    left_axes, right_axes = [], []
    for r in range(n):
        axl = fig.add_subplot(gs[r, 0], sharex=left_axes[0] if left_axes else None,
                              sharey=left_axes[0] if left_axes else None)
        axr = fig.add_subplot(gs[r, 1], sharex=right_axes[0] if right_axes else None,
                              sharey=(right_axes[0] if (right_axes and SHARE_RIGHT_Y) else None))
        left_axes.append(axl)
        right_axes.append(axr)

    mesh = None
    for r, lev in enumerate(levels):
        is_top, is_bottom = r == 0, r == n - 1
        mesh = _draw_left_map(left_axes[r], state, lev, time_index, dx_km,
                              is_top, is_bottom, current_x, PANEL_LETTERS[2 * r])
        _draw_right_series(right_axes[r], state, lev, time_index, current_x, is_top, is_bottom,
                           PANEL_LETTERS[2 * r + 1], dx_km)

    right_axes[0].set_xlim(state["time_axis_limits"])
    right_axes[0].set_ylim(0.0, 1.08 * float(state["right_ymax"]))

    cbar = fig.colorbar(mesh, ax=left_axes, location="top", shrink=0.8, aspect=40, pad=0.015,
                        extend="max")
    cbar.set_ticks(np.arange(0.0, state["field_clim"][1] + 1e-6, FIELD_CBAR_TICK_STEP))
    cbar.set_label(r"$|\mathbf{u}_\mathrm{h}|$ / m$\,$s$^{-1}$")
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(*state["circle_clim"]), cmap=MAP_FIELD_CMAP)
    cbar2 = fig.colorbar(sm, ax=right_axes, location="top", shrink=0.8, aspect=40, pad=0.015,
                         extend="max")
    cbar2.set_ticks(np.arange(0.0, state["circle_clim"][1] + 1e-6, CIRCLE_CBAR_TICK_STEP))
    cbar2.set_label(r"feature speed / m$\,$s$^{-1}$")

    # Let constrained layout place everything, then snap each right panel's vertical extent to its
    # (square) left map so the two columns share exactly the same panel height.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for axl, axr in zip(left_axes, right_axes):
        pl, pr = axl.get_position(), axr.get_position()
        axr.set_position([pr.x0, pl.y0, pr.width, pl.height])

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb


def _binned_median(values, speeds, bin_width):
    """Median speed in ``bin_width``-wide bins of ``values`` with at least MIN_BIN_COUNT tracks."""
    values = np.asarray(values, dtype=float)
    speeds = np.asarray(speeds, dtype=float)
    ok = np.isfinite(values) & np.isfinite(speeds)
    values, speeds = values[ok], speeds[ok]
    if values.size == 0:
        return np.array([]), np.array([])
    edges = np.arange(0.0, values.max() + bin_width, bin_width)
    idx = np.digitize(values, edges) - 1
    centers, medians = [], []
    for b in range(edges.size - 1):
        sel = idx == b
        if int(sel.sum()) >= MIN_BIN_COUNT:
            centers.append(0.5 * (edges[b] + edges[b + 1]))
            medians.append(float(np.median(speeds[sel])))
    return np.asarray(centers), np.asarray(medians)


def _scatter_by_scale(ax, df, xcol, bin_width):
    """Scatter speed vs ``xcol`` per field (filled/open = small/large kernel) + binned medians."""
    for field in FIELDS_IN_USE:
        color = FIELD_COLORS[field]
        fsub = df[df["field"] == field]
        for scale, filled in (("small", True), ("large", False)):
            sub = fsub[fsub["scale"] == scale]
            if len(sub):
                ax.scatter(sub[xcol], sub["speed_ms"], s=SCATTER_MARKER_SIZE, marker="o",
                           facecolors=color if filled else "none", edgecolors=color,
                           linewidths=0.8, alpha=SCATTER_ALPHA, zorder=3)
        centers, medians = _binned_median(fsub[xcol], fsub["speed_ms"], bin_width)
        if centers.size:
            ax.plot(centers, medians, color=color, lw=2.2, marker="D", ms=4.5, zorder=4)


def render_speed_size_figure(state, out_path):
    """Per-track scatters, one row per level (highest on top): median ground speed vs feature size
    (left column) and vs the local horizontal wind speed |u_h| at the feature (right column)."""
    levels = state["levels"]
    n = len(levels)
    start = int(state.get("start_index", 0))
    x_km = state["x"] / 1000.0
    dx_km = float(np.mean(np.diff(x_km))) if x_km.size > 1 else 0.4
    pooled = pd.concat([lev["speed_size"] for lev in levels], ignore_index=True)
    sizes = pooled["size_km"].to_numpy(dtype=float)
    sizes = sizes[np.isfinite(sizes)]
    xmax_size = max(4.0, 1.08 * float(np.percentile(sizes, 99.0))) if sizes.size else 10.0
    xmax_uh = float(state["field_clim"][1])

    fig, axes = plt.subplots(n, 2, figsize=(1.85 * SCATTER_PANEL_WIDTH, SCATTER_PANEL_HEIGHT * n),
                             sharex="col", sharey=True, layout="constrained")
    axes = np.asarray(axes).reshape(n, 2)
    for r, lev in enumerate(levels):
        axl, axr = axes[r]
        df = lev["speed_size"]
        _scatter_by_scale(axl, df, "size_km", SIZE_BIN_KM)
        axl.axhline(lev["ambient_speed"], color=AMBIENT_LINE_COLOR, lw=AMBIENT_LINE_LW,
                    ls=AMBIENT_LINE_STYLE, zorder=2)
        mean_ref = float(np.nanmean(np.asarray(lev["mean_wind_speed"], dtype=float)[start:]))
        if np.isfinite(mean_ref):
            axl.axhline(mean_ref, color=MEANWIND_LINE_COLOR, lw=MEANWIND_LINE_LW,
                        ls=MEANWIND_LINE_STYLE, zorder=2)
        _scatter_by_scale(axr, df, "uh_ms", UH_BIN_MS)
        axr.plot([0.0, xmax_uh], [0.0, xmax_uh], color=ONE_TO_ONE_COLOR, lw=ONE_TO_ONE_LW,
                 ls=ONE_TO_ONE_LS, zorder=2)
        axr.axvline(lev["ambient_speed"], color=AMBIENT_LINE_COLOR, lw=AMBIENT_LINE_LW,
                    ls=AMBIENT_LINE_STYLE, zorder=2)
        for ax, letter in ((axl, PANEL_LETTERS[2 * r]), (axr, PANEL_LETTERS[2 * r + 1])):
            ax.grid(True, alpha=0.25)
            add_panel_label(ax, letter)
        axl.set_ylabel(r"feature speed / m$\,$s$^{-1}$")
        add_corner_label(axl, f"$z = {lev['level_m'] / 1000.0:.0f}$ km\n$N = {len(df)}$")
        if r == n - 1:
            axl.set_xlabel("feature diameter / km")
            axr.set_xlabel(r"local $|\mathbf{u}_\mathrm{h}|$ / m$\,$s$^{-1}$")
    handles = [Line2D([], [], marker="o", ls="none", mfc=FIELD_COLORS[f],
                      mec=FIELD_COLORS[f], label=FIELD_LABELS[f]) for f in FIELDS_IN_USE]
    if ENABLE_SCALE_SPLIT:
        handles += [Line2D([], [], marker="o", ls="none", mfc="black", mec="black",
                           label=f"{SMALL_DIAMETER * dx_km:.0f} km kernel"),
                    Line2D([], [], marker="o", ls="none", mfc="none", mec="black",
                           label=f"{LARGE_DIAMETER * dx_km:.0f} km kernel")]
    handles += [Line2D([], [], color="0.3", lw=2.2, marker="D", ms=4.5,
                       label="binned median"),
                Line2D([], [], color=ONE_TO_ONE_COLOR, lw=ONE_TO_ONE_LW, ls=ONE_TO_ONE_LS,
                       label=r"$v = |\mathbf{u}_\mathrm{h}|$"),
                Line2D([], [], color=MEANWIND_LINE_COLOR, lw=MEANWIND_LINE_LW,
                       ls=MEANWIND_LINE_STYLE, label="mean wind (time mean)"),
                Line2D([], [], color=AMBIENT_LINE_COLOR, lw=AMBIENT_LINE_LW,
                       ls=AMBIENT_LINE_STYLE, label="ambient wind")]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, fontsize=9, frameon=False,
               columnspacing=1.2, handletextpad=0.5, labelspacing=0.35)
    axes[0, 0].set_xlim(0.0, xmax_size)
    axes[0, 1].set_xlim(0.0, xmax_uh)
    axes[0, 0].set_ylim(0.0, 1.08 * float(state["right_ymax"]))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[i]  Wrote {out_path}")


# --------------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------------

def frame_png_path(frames_dir, time_index):
    return Path(frames_dir) / f"frame_{int(time_index):04d}.png"


def clear_frame_directory(frames_dir):
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    if CLEAR_EXISTING_FRAMES:
        for png_path in frames_dir.glob("*.png"):
            png_path.unlink()
    return frames_dir


def select_time_indices(times):
    nt = int(times.size)
    start = 0
    if ANIMATION_START_OFFSET_S is not None:
        start = int(np.searchsorted(times - times[0], float(ANIMATION_START_OFFSET_S)))
    start = max(0, min(start, nt - 1))
    return list(range(start, nt))


def build_state(simulation_name, cube_index):
    nc_path = resolve_nc_path(simulation_name, cube_index)
    print(f"[i]  Using cube file: {nc_path}")
    ds = xr.open_dataset(nc_path, decode_times=False)

    display_levels = sorted({float(v) for v in TRACK_LEVELS_M}, reverse=True)
    x = y = times = None
    levels = []
    for level_m in display_levels:
        fields = load_level_fields(ds, level_m)
        x, y, times = fields["x"], fields["y"], fields["times"]
        print(f"[i]  level {fields['level_m'] / 1000.0:.2f} km   grid {x.size} x {y.size}   "
              f"frames {fields['W'].shape[0]}   ambient {fields['ambient_speed']:.1f} m/s")
        methods = []
        pooled_trajs = []
        track_tables = []
        for method in METHODS:
            result = run_method(method, fields)
            methods.append({
                "key": method["key"], "field": method["field"], "label": method["label"],
                "color": method["color"], "ls": method["ls"], "band": method["band"],
                "marker": method["marker"], "diameter": result["diameter"],
                "stats": result["stats"], "feats": result["feats"],
            })
            if result["traj"] is not None:
                pooled_trajs.append(result["traj"])
            if result["track_table"] is not None:
                track_tables.append((method["field"], method["scale"], result["track_table"]))
        nt = fields["W"].shape[0]
        pooled = pd.concat(pooled_trajs, ignore_index=True) if pooled_trajs else None
        levels.append({
            "level_m": fields["level_m"], "S": fields["S"], "L2": fields["L2"],
            "lambda2_level": choose_lambda2_level(fields["L2"]),
            "methods": methods,
            "overall": per_frame_stats(pooled, nt),
            "mean_wind_speed": fields["mean_wind_speed"],
            "ambient_speed": fields["ambient_speed"],
            "track_tables": track_tables,
        })
    ds.close()

    times_s = compute_elapsed_seconds(times)
    time_indices = select_time_indices(times)
    tlo = float(times_s[time_indices[0]])
    thi = float(times_s[time_indices[-1]])
    if tlo == thi:
        thi = tlo + 1.0

    dt = float(np.median(np.diff(times_s))) if times_s.size > 1 else 1.0
    window = max(1, int(round(SMOOTH_CUTOFF_S / dt)))
    if window % 2 == 0:
        window += 1
    print(f"[i]  low-pass window: {window} frames (~{window * dt / 60.0:.1f} min, dt {dt:.1f} s)")

    right_ymax = 0.0
    for lev in levels:
        lev["mean_wind_plot"] = smooth_lowpass(lev["mean_wind_speed"], window)
        cand = [lev["ambient_speed"], float(np.nanmax(lev["mean_wind_plot"]))]
        for m in lev["methods"]:
            m["stats"]["median_plot"] = smooth_lowpass(m["stats"]["median_g"], window)
            mm = np.nanmax(m["stats"]["median_plot"])
            if np.isfinite(mm):
                cand.append(float(mm))
        ov = lev["overall"]
        ov["median_plot"] = smooth_lowpass(ov["median_g"], window)
        ov["q25_plot"] = smooth_lowpass(ov["q25_g"], window)
        ov["q75_plot"] = smooth_lowpass(ov["q75_g"], window)
        qq = np.nanmax(ov["q75_plot"])
        if np.isfinite(qq):
            cand.append(float(qq))
        right_ymax = max(right_ymax, max(cand))

    if FIELD_SPEED_CLIM is not None:
        field_clim = (float(FIELD_SPEED_CLIM[0]), float(FIELD_SPEED_CLIM[1]))
    else:
        smax = max(float(np.nanpercentile(lev["S"], FIELD_SPEED_PCTL)) for lev in levels)
        field_clim = (0.0, max(10.0, float(np.ceil(smax / 10.0) * 10.0)))
    if CIRCLE_SPEED_CLIM is not None:
        circle_clim = (float(CIRCLE_SPEED_CLIM[0]), float(CIRCLE_SPEED_CLIM[1]))
    else:
        pooled_speeds = [sp for lev in levels for m in lev["methods"]
                         for (_xs, _ys, sp) in m["feats"].values()]
        pooled_speeds = (np.concatenate(pooled_speeds) if pooled_speeds else np.array([0.0]))
        pooled_speeds = pooled_speeds[np.isfinite(pooled_speeds)]
        cmax = (float(np.percentile(pooled_speeds, CIRCLE_SPEED_PCTL))
                if pooled_speeds.size else 10.0)
        circle_clim = (0.0, max(5.0, float(np.ceil(cmax / 5.0) * 5.0)))
    print(f"[i]  color scales: |u_h| {field_clim[0]:.0f}-{field_clim[1]:.0f} m/s, "
          f"feature speed {circle_clim[0]:.0f}-{circle_clim[1]:.0f} m/s")

    start_index = int(time_indices[0])
    dx_km = float(np.mean(np.diff(x))) / 1000.0 if x.size > 1 else 0.4
    speed_size_columns = ["level_km", "field", "scale", "particle", "n", "size_km",
                          "gyration_radius_km", "speed_ms", "uh_ms", "net_disp_km"]
    for lev in levels:
        records = []
        for field, scale, table in lev.pop("track_tables"):
            for rec in per_track_speed_size(table, start_index, dx_km):
                rec.update({"field": field, "scale": scale, "level_km": lev["level_m"] / 1000.0})
                records.append(rec)
        lev["speed_size"] = pd.DataFrame(records, columns=speed_size_columns)
        print(f"[i]  level {lev['level_m'] / 1000.0:.0f} km: {len(lev['speed_size'])} propagating "
              f"tracks in the speed-size scatter")

    return {
        "x": x, "y": y, "times_s": times_s,
        "levels": levels,
        "right_ymax": float(right_ymax),
        "time_axis_limits": (tlo, thi),
        "start_index": start_index,
        "field_clim": field_clim,
        "circle_clim": circle_clim,
    }, time_indices


_worker_state = None


def _init_worker(state, pbar=None):
    global _worker_state
    _worker_state = state
    _worker_state["pbar"] = pbar


def _render_worker(time_index):
    rgb = render_frame(_worker_state, int(time_index))
    imageio.imwrite(frame_png_path(_worker_state["frames_dir"], time_index), rgb)
    pbar = _worker_state.get("pbar")
    if pbar is not None:
        plt_helper.show_progress(pbar["progress_counter"], pbar["lock"], pbar["stime"], pbar["ntasks"])
    return int(time_index)


def run_animation(simulation_name, cube_index=0, render_all=True):
    simulation_name = os.path.basename(os.path.normpath(simulation_name))
    state, time_indices = build_state(simulation_name, cube_index)

    out_dir = ANIMATION_ROOT / f"{simulation_name}_track{int(cube_index)}"
    frames_dir = clear_frame_directory(out_dir)
    state["frames_dir"] = str(frames_dir)
    outfile = out_dir / f"anime_track_{simulation_name}_cube{int(cube_index)}.mp4"

    if not render_all:
        time_indices = [time_indices[-1]]
        print(f"[i]  Test mode: rendering a single frame at time index {time_indices[0]}.")

    print(f"[i]  Rendering {len(time_indices)} frames -> {frames_dir}")
    if len(time_indices) == 1:
        _init_worker(state)
        _render_worker(time_indices[0])
    else:
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
        ncpus = max(1, min(mp.cpu_count(), len(time_indices)))
        if ctx is None:
            _init_worker(state)
            for ti in time_indices:
                _render_worker(ti)
        else:
            manager = ctx.Manager()
            pbar = {
                "progress_counter": manager.Value("i", 0),
                "lock": manager.Lock(),
                "stime": time.time(),
                "ntasks": len(time_indices),
            }
            print(f"[i]  CPUs for rendering: {ncpus}")
            with ctx.Pool(processes=ncpus, initializer=_init_worker, initargs=(state, pbar)) as pool:
                for _ in pool.imap_unordered(_render_worker, time_indices):
                    pass
        plt_helper.create_animation(str(frames_dir), outfile.name, fps=10)
        generated = frames_dir / outfile.name
        if generated.resolve() != outfile.resolve():
            shutil.move(str(generated), str(outfile))
        print(f"[i]  Wrote {outfile}")

    scatter_path = out_dir / f"speedsize_{simulation_name}_cube{int(cube_index)}.png"
    render_speed_size_figure(state, scatter_path)
    pooled = pd.concat([lev["speed_size"] for lev in state["levels"]], ignore_index=True)
    if len(pooled):
        csv_path = scatter_path.with_suffix(".csv")
        pooled.to_csv(csv_path, index=False)
        print(f"[i]  Wrote {csv_path}")

    print("[i]  Done.")


def parse_args():
    parser = argparse.ArgumentParser(description="Track vortex features in a PMAP cube slice.")
    parser.add_argument("simulation", help="simulation jobname (scratch dir under /scratch/b/b309199)")
    parser.add_argument("notest", nargs="?", default=None,
                        help="'notest' renders the full animation; omit for a single test frame")
    parser.add_argument("--cube", type=int, default=0, help="cube index -> cube_<CUBE>.nc")
    parser.add_argument("--level", type=float, default=None,
                        help="track a single level (m) instead of the default multi-level set")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.level is not None:
        global TRACK_LEVELS_M
        TRACK_LEVELS_M = (float(args.level),)
    render_all = str(args.notest).lower() == "notest"
    run_animation(args.simulation, cube_index=args.cube, render_all=render_all)


if __name__ == "__main__":
    main()
