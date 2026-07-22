#!/usr/bin/env python3

"""Vertical-profile vortex-feature tracking of a PMAP cube: track w features on EVERY ~600-m level.

Companion of ``cube_track.py`` (imported as a library for the detection/linking/stats machinery),
shifting the focus from a few fixed levels to the vertical dimension: TWO w detection kernels
(~3 and ~9 km, one color each) are tracked on every cube level at PROFILE_DZ_M (600 m) vertical
resolution, and the median feature speeds become time-evolving vertical profiles to compare
against the ambient and box-mean wind profiles.

Figure layout (2 rows; the top-row x ticks/labels sit on TOP so the rows pack closely):
  top (a-d):   (a) speed-vs-altitude profile panel -- constant ambient wind profile, evolving
               box-mean wind profile, and the evolving median feature-speed profile of each kernel
               (colored, with per-kernel IQR bands); (b) direction profile (met. convention,
               degrees FROM) -- constant ambient + evolving box-mean wind direction + the median
               feature-propagation direction per kernel (from the tracked u/v velocity fits,
               masked below MIN_DIR_COUNT features); (c) DENSITY-WEIGHTED box-mean vertical
               momentum-flux profiles rho*<u'w'> / rho*<v'w'> in mPa; (d) feature-count profile
               per kernel. A dashed horizontal line marks the display level of the bottom row.
  bottom (e,f): one row of the cube_track animation at DISPLAY_LEVEL_M (default 58 km): |u_h| map +
               lambda2 contour + speed-colored feature circles (radius/edge color = kernel) (e),
               and the per-kernel median feature-speed time series with IQR bands, mean/ambient
               wind references, time cursor and the live value box (pooled median) (f).

Usage:
    python3 cube_track_profile.py darwin_240718_400m_coralT_ifs_wcoast --cube 0        # test frame
    python3 cube_track_profile.py darwin_240718_400m_coralT_ifs_wcoast notest --cube 0 # full mp4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import time

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

from scipy import ndimage

import cmaps
import cube_track as ct
import plt_helper

# Tracking: TWO kernels of ONE field on every profile level, one color each (used consistently in
# panels a/b/d/e/f). PROFILE_TRACK_FIELD selects the field (CLI --field overrides): "w" = |w|
# (vertical-wind features) or "t" = |T'| (temperature perturbation vs the horizontal box mean per
# level, T = theta*exner -- the quantity the virtual lidar measures, so tracked T' peaks/valleys
# connect to the lidar-diagnosed periods). trackpy needs an odd pixel diameter: 7 px = 2.8 km
# ("3 km", dense ensemble -- smaller would chase grid-scale noise at the ~4-5 dx effective
# resolution) and 23 px = 9.2 km ("9 km", the large scale of the 4-level analysis). The level
# stride is anchored on DISPLAY_LEVEL_M so the display level is always an exact analysis level.
PROFILE_TRACK_FIELD = "w"
FIELD_SYMBOL = {"w": r"$w$", "t": r"$T'$", "tpv": r"$T'$", "l2seg": r"$\lambda_2$"}
KERNEL_DIAMETERS_PX = (7, 23)
KERNEL_COLORS = ("#1b9e77", "#7570b3")
KERNEL_IQR_ALPHA = 0.16
PROFILE_DZ_M = 600.0
DISPLAY_LEVEL_M = 57000.0

# Panel-e background: "t" = T' (divergent wave cmap, the virtual-lidar quantity) or "uh" = |u_h|
# (sequential batlow). The feature circles keep the feature-speed colour scale either way.
MAP_FIELD = "t"
TPRIME_CLIM = None
TPRIME_PCTL = 99.5
TPRIME_COLORMAP = cmaps.get_wave_cmap()

# "l2seg" mode (segmentation tubes vs small-scale): TWO populations from the lambda2 field --
# SMALL = trackpy L2SEG_SMALL_PX kernel on -lambda2 (the small-scale debris) and TUBE = lambda2
# SEGMENTATION (8-connected -lambda2 regions above SEG_PERCENTILE, centroid-linked -- shape-
# agnostic so elongated tubes are caught whole). For each tube a physical "size" is measured: the
# distance from its WARM CORE (argmax T' inside the lambda2 region) to the nearest T' TROUGH
# (centroid of the nearest cold region, T' below the 100-VALLEY_PERCENTILE pct) -- the spatial
# peak-to-trough separation d that, with the tracked speed v, predicts the lidar-apparent period
# 2d/v. Panel c becomes the size profile; an end scatter relates size to speed and to the deficit.
SEG_PERCENTILE = 92.0
MIN_TUBE_AREA_PX = 20
# More robust tube definition: intersect the lambda2 core mask with a WARM T' mask (the warm
# core sits inside the tube), so strain-only / cold lambda2 features are dropped and the segmented
# region is a compact warm-vortex core that links more cleanly. WARM_PERCENTILE = warmest pct kept.
COMBINE_L2_TPRIME = True
WARM_PERCENTILE = 80.0
L2SEG_SMALL_PX = 7
L2SEG_TUBE_MARKER_PX = 10
# The trough is found by sampling T' ALONG THE TUBE'S PROPAGATION RAY (from its tracked u/v),
# outward from the warm core in both directions, and taking the FIRST local minimum -- exactly the
# 1D transect the lidar sees as the pattern advects past it (tau = d / v). Rays are lightly
# smoothed (PV_RAY_SMOOTH_PX) so grid-scale noise cannot fake a trough, and the search is capped at
# PV_MAX_SEARCH_KM. (An earlier version paired with the centroid of the nearest thresholded cold
# region; for large cold regions that centroid sits far away and the distances were ~2x too big.)
# A trough only counts if it is a GENUINE swing: T' must drop at least PV_MIN_AMPLITUDE_K below
# the warm core (mirrors the lidar analysis, which diagnoses peak-trough events by their
# amplitude). Without it the "first local minimum" is a small-scale wiggle 1-3 km inside the
# turbulent warm region rather than the wave's trough.
PV_MAX_SEARCH_KM = 30.0
PV_RAY_SMOOTH_PX = 7
PV_MIN_AMPLITUDE_K = 10.0
# All rays at a frame use the per-frame MEDIAN tube propagation direction (the wave packet has one
# propagation direction; per-tube velocity fits scatter and make the pairing lines fan out).
PV_USE_MEDIAN_DIRECTION = True
PV_LINE = {"color": "black", "lw": 1.0, "zorder": 6}
L2SEG_COLORS = {"small": "#1b9e77", "tube": "#7570b3"}
SCATTER_MARKER_SIZE = 16.0
SCATTER_ALPHA = 0.6

# "tpv" mode (peak-to-valley): SIGNED T' is tracked as two populations -- warm peaks (clip(T',0))
# and cold valleys (clip(-T',0)) -- with ONE kernel PV_KERNEL_PX for both (default 23 px = 9.2 km
# ~ the T' half-wavelength cell scale; trackpy's bandpass suppresses structures much LARGER than
# the kernel, so smaller kernels lose the smooth cells). The two "methods" become peak/valley
# (warm red / cold blue in all panels), and panel c shows the per-level median nearest
# PEAK-VALLEY DISTANCE instead of the momentum flux -- with the tracked speed v this predicts the
# lidar-apparent period 2d/v.
PV_KERNEL_PX = 23
PV_COLORS = {"peak": "#d1495b", "valley": "#2c7fb8"}

# Fixed x-range of the wind-profile panel = clim of the |u_h| map (None -> per-run auto).
PROFILE_FIELD_CLIM = (0.0, 130.0)


def _kernel_km(px):
    return f"{px * 0.4:.0f}"


def _build_methods():
    if PROFILE_TRACK_FIELD == "tpv":
        return [
            {"key": f"tpv_{sign}", "field": "tpv", "scale": sign, "detector": "trackpy",
             "label": rf"$T'$ {sign}", "color": PV_COLORS[sign], "ls": "-", "band": True,
             "marker": True, "diameter": int(PV_KERNEL_PX), "minmass": None, "separation": None}
            for sign in ("peak", "valley")
        ]
    if PROFILE_TRACK_FIELD == "l2seg":
        return [
            {"key": "l2_small", "field": "l2seg", "scale": "small", "detector": "trackpy",
             "label": "small", "color": L2SEG_COLORS["small"], "ls": "-", "band": True,
             "marker": True, "diameter": int(L2SEG_SMALL_PX), "minmass": None, "separation": None},
            {"key": "l2_tube", "field": "l2seg", "scale": "tube", "detector": "segment",
             "label": "tube", "color": L2SEG_COLORS["tube"], "ls": "-", "band": True,
             "marker": True, "diameter": int(L2SEG_TUBE_MARKER_PX), "minmass": None,
             "separation": None},
        ]
    return [
        {"key": f"{PROFILE_TRACK_FIELD}_{_kernel_km(px)}km", "field": PROFILE_TRACK_FIELD,
         "scale": "single", "detector": "trackpy",
         "label": rf"{FIELD_SYMBOL[PROFILE_TRACK_FIELD]} {_kernel_km(px)} km",
         "color": color, "ls": "-", "band": True, "marker": True, "diameter": int(px),
         "minmass": None, "separation": None}
        for px, color in zip(KERNEL_DIAMETERS_PX, KERNEL_COLORS)
    ]


METHODS = _build_methods()


def _method_legend_label(m):
    if PROFILE_TRACK_FIELD == "tpv":
        return rf"$T'$ {m['scale']}"
    if PROFILE_TRACK_FIELD == "l2seg":
        return m["scale"]
    return f"{_kernel_km(m['diameter'])} km"

# Layout: top row = wind profile + momentum-flux profile + feature-count profile (shared z axis,
# x ticks/labels on TOP; a and b equal width), bottom row = map + time series with the two
# colorbars BELOW the panels. Both rows are snapped to equal panel height after layout.
FIG_WIDTH = 10.0
RIGHT_WIDTH_RATIO = 1.35
TOP_WIDTH_RATIOS = (1.0, 1.0, 1.0, 1.0)
AXES_MARGIN_IN = 1.5
CBAR_HEADROOM_IN = 1.3
PROFILE_HEIGHT_RATIO = 1.0
BOTTOM_GAP_FRAC = 0.015
PANEL_LABEL_PAD_PT = 12.0
MIN_DIR_COUNT = 3


def _add_panel_label(ax, label):
    """Circled panel letter at a fixed physical inset from the panel's top-right corner."""
    ax.annotate(label, xy=(1.0, 1.0), xycoords="axes fraction",
                xytext=(-PANEL_LABEL_PAD_PT, -PANEL_LABEL_PAD_PT), textcoords="offset points",
                ha="right", va="top", fontsize=12, fontweight="bold",
                bbox={"boxstyle": "circle", "facecolor": "white", "edgecolor": "black",
                      "linewidth": 0.67}, zorder=6)
DISPLAY_LEVEL_LINE = {"color": "black", "lw": 1.2, "ls": (0, (4, 3))}
PROFILE_LEGEND_LOC = "lower left"
FLUX_UW_LINE = {"color": "black", "lw": 1.6, "ls": "-"}
FLUX_VW_LINE = {"color": "0.45", "lw": 1.6, "ls": (0, (5, 2))}


def per_frame_median_uv(traj, nt):
    """Per-frame median feature velocity (u, v) + direction-IQR deviations (deg).

    The direction quartiles are computed circularly: per frame, each feature's propagation
    direction is expressed as a deviation from the median-vector direction (wrapped to +/-180),
    and dev_q25/dev_q75 are the 25/75 percentiles of those deviations."""
    med_u = np.full(nt, np.nan)
    med_v = np.full(nt, np.nan)
    dev_q25 = np.full(nt, np.nan)
    dev_q75 = np.full(nt, np.nan)
    if traj is not None and len(traj):
        for f, g in traj.groupby("frame"):
            f = int(f)
            u = g["u"].to_numpy(dtype=float)
            v = g["v"].to_numpy(dtype=float)
            ok = np.isfinite(u) & np.isfinite(v)
            if not ok.any():
                continue
            mu = float(np.median(u[ok]))
            mv = float(np.median(v[ok]))
            med_u[f] = mu
            med_v[f] = mv
            m_ang = np.degrees(np.arctan2(-mu, -mv))
            ang = np.degrees(np.arctan2(-u[ok], -v[ok]))
            dev = (ang - m_ang + 180.0) % 360.0 - 180.0
            dev_q25[f] = float(np.percentile(dev, 25.0))
            dev_q75[f] = float(np.percentile(dev, 75.0))
    return med_u, med_v, dev_q25, dev_q75


def peak_valley_distance(feats_peak, feats_valley, nt):
    """Per-frame median nearest peak-valley distance (km) between the two tracked populations.

    For each tracked peak the distance to the nearest tracked valley is taken (and vice versa);
    the per-frame statistic is the median of the union of both nearest-neighbour sets."""
    d = np.full(nt, np.nan)
    for f in range(nt):
        pa = feats_peak.get(f)
        pb = feats_valley.get(f)
        if pa is None or pb is None or not len(pa[0]) or not len(pb[0]):
            continue
        dist = np.hypot(pa[0][:, None] - pb[0][None, :], pa[1][:, None] - pb[1][None, :])
        d[f] = float(np.median(np.concatenate([dist.min(axis=1), dist.min(axis=0)])))
    return d


def per_frame_col_stats(traj, nt, col):
    """Per-frame median / q25 / q75 of a track-table column (NaN where no features)."""
    med = np.full(nt, np.nan)
    q25 = np.full(nt, np.nan)
    q75 = np.full(nt, np.nan)
    if traj is not None and len(traj) and col in traj:
        full = pd.RangeIndex(nt)
        g = traj.groupby("frame")[col]
        med = g.median().reindex(full).to_numpy()
        q25 = g.quantile(0.25).reindex(full).to_numpy()
        q75 = g.quantile(0.75).reindex(full).to_numpy()
    return med, q25, q75


def _lambda2_center_plane(u, v, w, x, y, z_slab, c):
    """lambda2 (2nd eigenvalue of S^2+Omega^2) on the centre plane ``c`` of a (nz,ny,nx) slab.

    Gradients use the whole slab (for d/dz) but the eigen-decomposition is done only on the centre
    plane -- ~nz times cheaper than eigvalsh over the full slab, which matters at 34 levels."""
    du_dz, du_dy, du_dx = np.gradient(u, z_slab, y, x, edge_order=2)
    dv_dz, dv_dy, dv_dx = np.gradient(v, z_slab, y, x, edge_order=2)
    dw_dz, dw_dy, dw_dx = np.gradient(w, z_slab, y, x, edge_order=2)
    dudx, dudy, dudz = du_dx[c], du_dy[c], du_dz[c]
    dvdx, dvdy, dvdz = dv_dx[c], dv_dy[c], dv_dz[c]
    dwdx, dwdy, dwdz = dw_dx[c], dw_dy[c], dw_dz[c]
    ny, nx = dudx.shape
    s = np.zeros((ny, nx, 3, 3), dtype=np.float64)
    o = np.zeros_like(s)
    s[..., 0, 0] = dudx
    s[..., 1, 1] = dvdy
    s[..., 2, 2] = dwdz
    s[..., 0, 1] = s[..., 1, 0] = 0.5 * (dudy + dvdx)
    s[..., 0, 2] = s[..., 2, 0] = 0.5 * (dudz + dwdx)
    s[..., 1, 2] = s[..., 2, 1] = 0.5 * (dvdz + dwdy)
    oxy, oxz, oyz = 0.5 * (dudy - dvdx), 0.5 * (dudz - dwdx), 0.5 * (dvdz - dwdy)
    o[..., 0, 1], o[..., 1, 0] = oxy, -oxy
    o[..., 0, 2], o[..., 2, 0] = oxz, -oxz
    o[..., 1, 2], o[..., 2, 1] = oyz, -oyz
    return np.linalg.eigvalsh(s @ s + o @ o)[..., 1]


def level_neg_lambda2(ds, z_full, center_zidx, x, y):
    """Per-time non-negative -lambda2 stack (nt,ny,nx) at one level (vortex cores = large)."""
    h = ct.LAMBDA2_SLAB_HALF
    lo = max(0, center_zidx - h)
    hi = min(z_full.size, center_zidx + h + 1)
    z_slab = z_full[lo:hi]
    c = center_zidx - lo

    def load(name):
        da = ds[name].isel({ct.Z_NAME: slice(lo, hi)}).transpose(
            ct.TIME_NAME, ct.Z_NAME, ct.Y_NAME, ct.X_NAME)
        return np.asarray(da.values, dtype=np.float32)

    us, vs, ws = load(ct.U_NAME), load(ct.V_NAME), load(ct.W_NAME)
    nt = us.shape[0]
    out = np.empty((nt, y.size, x.size), dtype=np.float64)
    for t in range(nt):
        out[t] = _lambda2_center_plane(us[t], vs[t], ws[t], x, y, z_slab, c)
    return np.clip(-out, 0.0, None)


def detect_segments(l2_stack, tprime_stack, l2_thresh, warm_thresh, dx_km):
    """Segment lambda2 tubes per frame and locate each tube's warm core (argmax T' inside it).

    The tube mask is (-lambda2 >= l2_thresh), optionally intersected with (T' >= warm_thresh) when
    COMBINE_L2_TPRIME. Returns a features DataFrame with x, y (centroid px), frame, warm_r, warm_c
    (px) and eq_diam_km. The warm-core -> trough pairing needs the propagation direction and is
    therefore done after linking, in ``attach_pv_size_directional``."""
    structure = np.ones((3, 3), dtype=bool)
    rows = {"x": [], "y": [], "frame": [], "warm_r": [], "warm_c": [], "eq_diam_km": []}
    if l2_thresh is None:
        return pd.DataFrame(rows)
    for t in range(l2_stack.shape[0]):
        tp = tprime_stack[t]
        mask = l2_stack[t] >= l2_thresh
        if COMBINE_L2_TPRIME and warm_thresh is not None:
            mask &= tp >= warm_thresh
        labels, n = ndimage.label(mask, structure=structure)
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        area = np.bincount(labels.ravel(), minlength=n + 1)[1:]
        cent = np.asarray(ndimage.center_of_mass(np.ones_like(tp), labels, idx))
        warm = np.asarray(ndimage.maximum_position(tp, labels, idx), dtype=float)
        for j in range(n):
            if area[j] < MIN_TUBE_AREA_PX:
                continue
            rows["x"].append(cent[j, 1])
            rows["y"].append(cent[j, 0])
            rows["frame"].append(t)
            rows["warm_r"].append(warm[j, 0])
            rows["warm_c"].append(warm[j, 1])
            rows["eq_diam_km"].append(2.0 * np.sqrt(area[j] / np.pi) * dx_km)
    return pd.DataFrame(rows)


def _first_local_min_offset(rays):
    """Offset (in samples, >=1) of the first QUALIFYING local minimum of each outward ray.

    Qualifying = a local minimum that also lies at least PV_MIN_AMPLITUDE_K below the warm-core
    value in column 0, so shallow small-scale wiggles are not mistaken for the wave's trough.
    Returns NaN where no such minimum exists within the ray."""
    if rays.shape[1] < 3:
        return np.full(rays.shape[0], np.nan)
    interior = rays[:, 1:-1]
    is_min = (interior < rays[:, :-2]) & (interior <= rays[:, 2:])
    is_min &= interior <= (rays[:, [0]] - PV_MIN_AMPLITUDE_K)
    found = is_min.any(axis=1)
    off = np.argmax(is_min, axis=1).astype(float) + 1.0
    off[~found] = np.nan
    return off


def attach_pv_size_directional(traj, tprime_stack, x, y):
    """Pair each tracked tube's warm core with the first T' trough along its propagation ray.

    T' is sampled outward from the warm core in both directions along the tube's tracked
    propagation axis (the 1D transect the lidar effectively sees); the nearer of the two first
    local minima sets ``pv_size_km``. Also attaches the warm-core / trough positions in km so the
    pairing can be drawn on the map."""
    traj = traj.copy()
    n = len(traj)
    pv = np.full(n, np.nan)
    wx, wy, tx, ty = (np.full(n, np.nan) for _ in range(4))
    dxm = float(np.mean(np.diff(x))) if x.size > 1 else 400.0
    dym = float(np.mean(np.diff(y))) if y.size > 1 else 400.0
    x0, y0 = float(x[0]), float(y[0])
    dx_km = dxm / 1000.0
    kmax = max(3, int(round(PV_MAX_SEARCH_KM / dx_km)))
    ks = np.arange(0, kmax + 1, dtype=float)
    warm_r = traj["warm_r"].to_numpy(dtype=float)
    warm_c = traj["warm_c"].to_numpy(dtype=float)
    u = traj["u"].to_numpy(dtype=float)
    v = traj["v"].to_numpy(dtype=float)
    frames = traj["frame"].to_numpy().astype(int)
    for f in np.unique(frames):
        sel = np.nonzero(frames == f)[0]
        spd = np.hypot(u[sel], v[sel])
        good = np.isfinite(spd) & (spd > 0.0)
        sel = sel[good]
        if not sel.size:
            continue
        if PV_USE_MEDIAN_DIRECTION:
            mu, mv = float(np.median(u[sel])), float(np.median(v[sel]))
            msp = np.hypot(mu, mv)
            if msp <= 0.0:
                continue
            er = np.full(sel.size, mv / msp)
            ec = np.full(sel.size, mu / msp)
        else:
            spd = np.hypot(u[sel], v[sel])
            er, ec = v[sel] / spd, u[sel] / spd
        tp = tprime_stack[int(f)]
        best_k = np.full(sel.size, np.nan)
        best_sign = np.zeros(sel.size)
        for sign in (1.0, -1.0):
            rr = warm_r[sel][:, None] + sign * er[:, None] * ks[None, :]
            cc = warm_c[sel][:, None] + sign * ec[:, None] * ks[None, :]
            rays = ndimage.map_coordinates(tp, [rr.ravel(), cc.ravel()], order=1,
                                           mode="constant", cval=np.nan).reshape(rr.shape)
            rays = ndimage.uniform_filter1d(rays, size=PV_RAY_SMOOTH_PX, axis=1, mode="nearest")
            off = _first_local_min_offset(rays)
            take = np.isfinite(off) & (~np.isfinite(best_k) | (off < best_k))
            best_k[take] = off[take]
            best_sign[take] = sign
        ok = np.isfinite(best_k)
        if not ok.any():
            continue
        idx_ok = sel[ok]
        pv[idx_ok] = best_k[ok] * dx_km
        tr_r = warm_r[idx_ok] + best_sign[ok] * er[ok] * best_k[ok]
        tr_c = warm_c[idx_ok] + best_sign[ok] * ec[ok] * best_k[ok]
        wx[idx_ok] = (x0 + warm_c[idx_ok] * dxm) / 1000.0
        wy[idx_ok] = (y0 + warm_r[idx_ok] * dym) / 1000.0
        tx[idx_ok] = (x0 + tr_c * dxm) / 1000.0
        ty[idx_ok] = (y0 + tr_r * dym) / 1000.0
    traj["pv_size_km"] = pv
    traj["warm_x_km"], traj["warm_y_km"] = wx, wy
    traj["trough_x_km"], traj["trough_y_km"] = tx, ty
    return traj


def pv_segments_by_frame(traj):
    """frame -> (warm_x, warm_y, trough_x, trough_y) km arrays of the paired peak-trough segments."""
    out = {}
    if traj is None or not len(traj):
        return out
    sub = traj[np.isfinite(traj["pv_size_km"])]
    for f, g in sub.groupby("frame"):
        out[int(f)] = (g["warm_x_km"].to_numpy(), g["warm_y_km"].to_numpy(),
                       g["trough_x_km"].to_numpy(), g["trough_y_km"].to_numpy())
    return out


def track_profile_levels(ds, x, y, times_s):
    """Track both w kernels on every ~PROFILE_DZ_M level; return per-level stats + display feats.

    The level stride through the cube's z grid is anchored on DISPLAY_LEVEL_M. Returns
    (z_levels_m, stats_all [per kernel: list of per-level stats], mean_wind (nlev, nt),
    ambient (nlev,), uw (nlev, nt), vw (nlev, nt), display_feats [per kernel],
    pooled_display [per-frame stats of both kernels pooled at the display level],
    display_index) where uw/vw are the DENSITY-WEIGHTED box-mean vertical momentum-flux
    components rho*<u'w'> / rho*<v'w'> in mPa (perturbations from the horizontal box mean per
    level; for conservative waves rho*<u'w'> is constant with height, so a decrease with
    altitude flags deposition/breaking)."""
    z = np.asarray(ds[ct.Z_NAME].values, dtype=float)
    dz = float(np.median(np.diff(z))) if z.size > 1 else PROFILE_DZ_M
    step = max(1, int(round(PROFILE_DZ_M / dz)))
    disp_zidx = int(np.argmin(np.abs(z - DISPLAY_LEVEL_M)))
    idx = list(range(disp_zidx % step, z.size, step))
    display_index = (disp_zidx - idx[0]) // step

    def load_levels(name):
        da = ds[name].isel({ct.Z_NAME: idx}).transpose(ct.TIME_NAME, ct.Z_NAME, ct.Y_NAME,
                                                       ct.X_NAME)
        return np.asarray(da.values, dtype=np.float32)

    u = load_levels(ct.U_NAME)
    v = load_levels(ct.V_NAME)
    w = load_levels(ct.W_NAME)
    ua = load_levels(ct.UAMB_NAME)
    va = load_levels(ct.VAMB_NAME)
    tprime = None
    if PROFILE_TRACK_FIELD in ("t", "tpv", "l2seg") or MAP_FIELD == "t":
        temp = load_levels("theta_total") * load_levels("exner_total")
        tprime = temp - np.nanmean(temp, axis=(2, 3), keepdims=True)
        del temp
    if PROFILE_TRACK_FIELD == "tpv":
        stacks = [np.clip(tprime, 0.0, None), np.clip(-tprime, 0.0, None)]
    elif PROFILE_TRACK_FIELD == "t":
        stacks = [np.abs(tprime)] * len(METHODS)
    elif PROFILE_TRACK_FIELD == "l2seg":
        stacks = None
    else:
        stacks = [np.abs(w)] * len(METHODS)

    nlev = len(idx)
    nt = w.shape[0]
    umean = np.nanmean(u, axis=(2, 3)).T
    vmean = np.nanmean(v, axis=(2, 3)).T
    mean_wind = np.hypot(umean, vmean)
    ua_mean = np.nanmean(ua, axis=(0, 2, 3))
    va_mean = np.nanmean(va, axis=(0, 2, 3))
    ambient = np.hypot(ua_mean, va_mean)
    ambient_dir = np.degrees(np.arctan2(-ua_mean, -va_mean)) % 360.0
    rho_mean = np.nanmean(load_levels("density"), axis=(2, 3)).T
    wp = w - np.nanmean(w, axis=(2, 3), keepdims=True)
    uw = np.nanmean((u - np.nanmean(u, axis=(2, 3), keepdims=True)) * wp, axis=(2, 3)).T
    vw = np.nanmean((v - np.nanmean(v, axis=(2, 3), keepdims=True)) * wp, axis=(2, 3)).T
    del wp
    uw = 1.0e3 * rho_mean * uw
    vw = 1.0e3 * rho_mean * vw

    dx_km = float(np.mean(np.diff(x))) / 1000.0 if x.size > 1 else 0.4
    stats_all = [[] for _ in METHODS]
    feat_u = np.full((len(METHODS), nlev, nt), np.nan)
    feat_v = np.full((len(METHODS), nlev, nt), np.nan)
    feat_dq25 = np.full((len(METHODS), nlev, nt), np.nan)
    feat_dq75 = np.full((len(METHODS), nlev, nt), np.nan)
    pvd = np.full((nlev, nt), np.nan)
    size_med = np.full((nlev, nt), np.nan)
    size_q25 = np.full((nlev, nt), np.nan)
    size_q75 = np.full((nlev, nt), np.nan)
    tube_tables = [None] * nlev
    display_feats = [dict() for _ in METHODS]
    display_pv_segments = {}
    pooled_display = None
    for i in range(nlev):
        if PROFILE_TRACK_FIELD == "l2seg":
            l2f = level_neg_lambda2(ds, z, idx[i], x, y)
            pos = l2f[l2f > 0]
            l2_thresh = float(np.percentile(pos, SEG_PERCENTILE)) if pos.size else None
            tp_i = tprime[:, i]
            warm_thresh = float(np.percentile(tp_i, WARM_PERCENTILE)) if COMBINE_L2_TPRIME else None
        pooled = []
        msg = []
        level_feats = []
        for k, method in enumerate(METHODS):
            if method.get("detector") == "segment":
                feats = detect_segments(l2f, tp_i, l2_thresh, warm_thresh, dx_km)
            elif PROFILE_TRACK_FIELD == "l2seg":
                feats = ct.detect_features(l2f, method)
            else:
                feats = ct.detect_features(stacks[k][:, i].astype(np.float64), method)
            traj = ct.link_and_measure(feats, times_s, x, y)
            if method.get("detector") == "segment" and traj is not None and len(traj):
                traj = attach_pv_size_directional(traj, tp_i, x, y)
            stats_all[k].append(ct.per_frame_stats(traj, nt))
            (feat_u[k, i], feat_v[k, i],
             feat_dq25[k, i], feat_dq75[k, i]) = per_frame_median_uv(traj, nt)
            fbf = ct.features_by_frame(traj)
            level_feats.append(fbf)
            ntracks = 0 if (traj is None or not len(traj)) else int(traj["particle"].nunique())
            msg.append(f"{method['key']}: {0 if feats is None else len(feats):6d} det "
                       f"/ {ntracks:4d} trk")
            if method.get("detector") == "segment" and traj is not None and len(traj):
                size_med[i], size_q25[i], size_q75[i] = per_frame_col_stats(traj, nt, "pv_size_km")
                tube_tables[i] = traj[["particle", "frame", "xm", "ym", "speed_ground",
                                       "pv_size_km"]].copy()
                if i == display_index:
                    display_pv_segments = pv_segments_by_frame(traj)
            if i == display_index:
                display_feats[k] = fbf
                if traj is not None and len(traj):
                    pooled.append(traj[["frame", "speed_ground"]])
        if PROFILE_TRACK_FIELD == "tpv":
            pvd[i] = peak_valley_distance(level_feats[0], level_feats[1], nt)
        if i == display_index:
            pooled_display = ct.per_frame_stats(
                pd.concat(pooled, ignore_index=True) if pooled else None, nt)
        print(f"[i]  level {z[idx[i]] / 1000.0:6.2f} km:   " + "   ".join(msg))
    disp_T = tprime[:, display_index].copy() if tprime is not None else None
    return (z[idx], stats_all, feat_u, feat_v, feat_dq25, feat_dq75, mean_wind, umean, vmean,
            ambient, ambient_dir, uw, vw, pvd, size_med, size_q25, size_q75, tube_tables,
            display_feats, display_pv_segments, disp_T, pooled_display, display_index)


def build_tube_scatter(tube_tables, mean_wind, z_levels, start_index):
    """One record per propagating tube: median size, median speed, deficit vs box-mean wind."""
    cols = ["level_km", "particle", "n", "size_km", "speed_ms", "deficit_ms", "net_disp_km"]
    if all(t is None for t in tube_tables):
        return pd.DataFrame(columns=cols)
    recs = []
    for i, tt in enumerate(tube_tables):
        if tt is None or not len(tt):
            continue
        sub = tt[tt["frame"] >= start_index]
        mw = mean_wind[i]
        for pid, g in sub.groupby("particle"):
            if len(g) < ct.MIN_TRACK_LENGTH:
                continue
            g = g.sort_values("frame")
            net_km = float(np.hypot(g["xm"].iloc[-1] - g["xm"].iloc[0],
                                    g["ym"].iloc[-1] - g["ym"].iloc[0])) / 1000.0
            if net_km < ct.MIN_NET_DISPLACEMENT_KM:
                continue
            size_km = float(np.nanmedian(g["pv_size_km"]))
            speed = float(np.nanmedian(g["speed_ground"]))
            if not (np.isfinite(size_km) and np.isfinite(speed)):
                continue
            frames = g["frame"].to_numpy().astype(int)
            deficit = float(np.nanmedian(g["speed_ground"].to_numpy() - mw[frames]))
            recs.append({"level_km": float(z_levels[i]) / 1000.0, "particle": int(pid),
                         "n": int(len(g)), "size_km": size_km, "speed_ms": speed,
                         "deficit_ms": deficit, "net_disp_km": net_km})
    return pd.DataFrame(recs, columns=cols)


def build_state(simulation_name, cube_index):
    nc_path = ct.resolve_nc_path(simulation_name, cube_index)
    print(f"[i]  Using cube file: {nc_path}")
    ds = xr.open_dataset(nc_path, decode_times=False)

    disp = ct.load_level_fields(ds, DISPLAY_LEVEL_M)
    x, y, times = disp["x"], disp["y"], disp["times"]
    times_s = ct.compute_elapsed_seconds(times)
    print(f"[i]  display level {disp['level_m'] / 1000.0:.2f} km   grid {x.size} x {y.size}   "
          f"frames {disp['W'].shape[0]}")

    (z_levels, stats_all, feat_u, feat_v, feat_dq25, feat_dq75, mean_wind, umean, vmean, ambient,
     ambient_dir, uw, vw, pvd, size_med, size_q25, size_q75, tube_tables, display_feats,
     display_pv_segments, disp_T, pooled_display, display_index) = \
        track_profile_levels(ds, x, y, times_s)
    ds.close()

    nt = int(times_s.size)
    time_indices = ct.select_time_indices(times)
    tlo = float(times_s[time_indices[0]])
    thi = float(times_s[time_indices[-1]])
    if tlo == thi:
        thi = tlo + 1.0

    dt = float(np.median(np.diff(times_s))) if times_s.size > 1 else 1.0
    window = max(1, int(round(ct.SMOOTH_CUTOFF_S / dt)))
    if window % 2 == 0:
        window += 1
    print(f"[i]  low-pass window: {window} frames (~{window * dt / 60.0:.1f} min, dt {dt:.1f} s)")

    nlev = z_levels.size
    nkern = len(METHODS)
    med_plot = np.full((nkern, nlev, nt), np.nan)
    q25_plot = np.full((nkern, nlev, nt), np.nan)
    q75_plot = np.full((nkern, nlev, nt), np.nan)
    count_plot = np.full((nkern, nlev, nt), np.nan)
    mean_wind_plot = np.full((nlev, nt), np.nan)
    uw_plot = np.full((nlev, nt), np.nan)
    vw_plot = np.full((nlev, nt), np.nan)
    feat_u_plot = np.full((nkern, nlev, nt), np.nan)
    feat_v_plot = np.full((nkern, nlev, nt), np.nan)
    feat_dq25_plot = np.full((nkern, nlev, nt), np.nan)
    feat_dq75_plot = np.full((nkern, nlev, nt), np.nan)
    for k in range(nkern):
        for i, st in enumerate(stats_all[k]):
            med_plot[k, i] = ct.smooth_lowpass(st["median_g"], window)
            q25_plot[k, i] = ct.smooth_lowpass(st["q25_g"], window)
            q75_plot[k, i] = ct.smooth_lowpass(st["q75_g"], window)
            count_plot[k, i] = ct.smooth_lowpass(st["count"], window)
            feat_u_plot[k, i] = ct.smooth_lowpass(feat_u[k, i], window)
            feat_v_plot[k, i] = ct.smooth_lowpass(feat_v[k, i], window)
            feat_dq25_plot[k, i] = ct.smooth_lowpass(feat_dq25[k, i], window)
            feat_dq75_plot[k, i] = ct.smooth_lowpass(feat_dq75[k, i], window)
    umean_plot = np.full((nlev, nt), np.nan)
    vmean_plot = np.full((nlev, nt), np.nan)
    pvd_plot = np.full((nlev, nt), np.nan)
    for i in range(nlev):
        mean_wind_plot[i] = ct.smooth_lowpass(mean_wind[i], window)
        umean_plot[i] = ct.smooth_lowpass(umean[i], window)
        vmean_plot[i] = ct.smooth_lowpass(vmean[i], window)
        uw_plot[i] = ct.smooth_lowpass(uw[i], window)
        vw_plot[i] = ct.smooth_lowpass(vw[i], window)
        pvd_plot[i] = ct.smooth_lowpass(pvd[i], window)
    pvd_finite = pvd_plot[np.isfinite(pvd_plot)]
    pvd_xmax = (max(5.0, float(np.ceil(1.08 * np.percentile(pvd_finite, 99.0) / 5.0) * 5.0))
                if pvd_finite.size else 10.0)
    size_med_plot = np.full((nlev, nt), np.nan)
    size_q25_plot = np.full((nlev, nt), np.nan)
    size_q75_plot = np.full((nlev, nt), np.nan)
    for i in range(nlev):
        size_med_plot[i] = ct.smooth_lowpass(size_med[i], window)
        size_q25_plot[i] = ct.smooth_lowpass(size_q25[i], window)
        size_q75_plot[i] = ct.smooth_lowpass(size_q75[i], window)
    size_finite = size_q75_plot[np.isfinite(size_q75_plot)]
    size_xmax = (max(5.0, float(np.ceil(1.08 * np.percentile(size_finite, 99.0) / 5.0) * 5.0))
                 if size_finite.size else 10.0)
    tube_scatter = build_tube_scatter(tube_tables, mean_wind, z_levels, int(time_indices[0]))
    if TPRIME_CLIM is not None:
        tprime_clim = (float(TPRIME_CLIM[0]), float(TPRIME_CLIM[1]))
    elif disp_T is not None:
        tmax = float(np.nanpercentile(np.abs(disp_T), TPRIME_PCTL))
        tmax = max(1.0, float(np.ceil(tmax / 2.0) * 2.0))
        tprime_clim = (-tmax, tmax)
    else:
        tprime_clim = (-1.0, 1.0)
    mean_dir_plot = np.degrees(np.arctan2(-umean_plot, -vmean_plot)) % 360.0
    feat_dir_plot = np.degrees(np.arctan2(-feat_u_plot, -feat_v_plot)) % 360.0
    feat_dir_plot[count_plot < MIN_DIR_COUNT] = np.nan
    dir_q25_plot = feat_dir_plot + feat_dq25_plot
    dir_q75_plot = feat_dir_plot + feat_dq75_plot
    pooled_med_plot = (ct.smooth_lowpass(pooled_display["median_g"], window)
                       if pooled_display is not None else np.full(nt, np.nan))

    dirs = np.concatenate([np.asarray(ambient_dir, dtype=float), mean_dir_plot.ravel(),
                           feat_dir_plot.ravel(), dir_q25_plot.ravel(), dir_q75_plot.ravel()])
    dirs = dirs[np.isfinite(dirs)]
    if dirs.size:
        dlo, dhi = np.percentile(dirs, [1.0, 99.0])
    else:
        dlo, dhi = 0.0, 360.0
    if dhi - dlo <= 300.0:
        dir_xlim = (float(np.floor((dlo - 10.0) / 30.0) * 30.0),
                    float(np.ceil((dhi + 10.0) / 30.0) * 30.0))
    else:
        dir_xlim = (0.0, 360.0)

    count_xmax = max(5.0, float(np.ceil(1.08 * np.nanmax(count_plot) / 5.0) * 5.0))
    flux_xmax = float(np.nanmax(np.abs(np.concatenate([uw_plot, vw_plot]))))
    flux_xmax = max(1.0, float(np.ceil(1.08 * flux_xmax / 5.0) * 5.0))

    if PROFILE_FIELD_CLIM is not None:
        field_clim = (float(PROFILE_FIELD_CLIM[0]), float(PROFILE_FIELD_CLIM[1]))
    else:
        smax = max(float(np.nanmax(mean_wind_plot)), float(np.nanmax(ambient)),
                   float(np.nanpercentile(disp["S"], ct.FIELD_SPEED_PCTL)))
        field_clim = (0.0, max(10.0, float(np.ceil(smax / 10.0) * 10.0)))
    speeds = np.concatenate([sp for feats in display_feats for (_xs, _ys, sp) in feats.values()]
                            or [np.array([0.0])])
    speeds = speeds[np.isfinite(speeds)]
    cmax = float(np.percentile(speeds, ct.CIRCLE_SPEED_PCTL)) if speeds.size else 10.0
    circle_clim = (ct.CIRCLE_SPEED_CLIM if ct.CIRCLE_SPEED_CLIM is not None
                   else (0.0, max(5.0, float(np.ceil(cmax / 5.0) * 5.0))))
    print(f"[i]  color scales: |u_h| {field_clim[0]:.0f}-{field_clim[1]:.0f} m/s, "
          f"feature speed {circle_clim[0]:.0f}-{circle_clim[1]:.0f} m/s")

    d = display_index
    cand = [float(ambient[d]), float(np.nanmax(mean_wind_plot[d]))]
    qq = np.nanmax(q75_plot[:, d])
    if np.isfinite(qq):
        cand.append(float(qq))
    right_ymax = max(cand)

    return {
        "x": x, "y": y, "times_s": times_s,
        "z_levels_km": z_levels / 1000.0,
        "med_plot": med_plot, "q25_plot": q25_plot, "q75_plot": q75_plot,
        "mean_wind_plot": mean_wind_plot, "ambient_prof": ambient,
        "count_plot": count_plot, "count_xmax": count_xmax,
        "uw_plot": uw_plot, "vw_plot": vw_plot, "flux_xmax": flux_xmax,
        "pvd_plot": pvd_plot, "pvd_xmax": pvd_xmax,
        "size_med_plot": size_med_plot, "size_q25_plot": size_q25_plot,
        "size_q75_plot": size_q75_plot, "size_xmax": size_xmax,
        "tube_scatter": tube_scatter,
        "pooled_med_plot": pooled_med_plot,
        "ambient_dir": ambient_dir, "mean_dir_plot": mean_dir_plot,
        "feat_dir_plot": feat_dir_plot, "dir_xlim": dir_xlim,
        "dir_q25_plot": dir_q25_plot, "dir_q75_plot": dir_q75_plot,
        "display_index": display_index,
        "display_z_km": float(z_levels[display_index]) / 1000.0,
        "disp_S": disp["S"], "disp_L2": disp["L2"], "disp_T": disp_T,
        "tprime_clim": tprime_clim,
        "lambda2_level": ct.choose_lambda2_level(disp["L2"]),
        "display_feats": display_feats, "pv_segments": display_pv_segments,
        "field_clim": field_clim, "circle_clim": circle_clim,
        "right_ymax": float(right_ymax),
        "time_axis_limits": (tlo, thi),
    }, time_indices


def _draw_direction_panel(axd, state, time_index, letter="b"):
    """Direction profiles (met. convention: degrees FROM): wind + feature propagation."""
    z_km = state["z_levels_km"]
    for k, method in enumerate(METHODS):
        q25 = state["dir_q25_plot"][k, :, time_index]
        q75 = state["dir_q75_plot"][k, :, time_index]
        ok = np.isfinite(q25) & np.isfinite(q75)
        axd.fill_betweenx(z_km, q25, q75, where=ok, color=method["color"],
                          alpha=KERNEL_IQR_ALPHA, linewidth=0, zorder=1)
        axd.plot(state["feat_dir_plot"][k, :, time_index], z_km, color=method["color"], lw=1.9,
                 zorder=4)
    axd.plot(state["ambient_dir"], z_km, color=ct.AMBIENT_LINE_COLOR, lw=ct.AMBIENT_LINE_LW,
             ls=ct.AMBIENT_LINE_STYLE, zorder=2)
    axd.plot(state["mean_dir_plot"][:, time_index], z_km, color=ct.MEANWIND_LINE_COLOR,
             lw=ct.MEANWIND_LINE_LW, ls=ct.MEANWIND_LINE_STYLE, zorder=3)
    axd.axhline(state["display_z_km"], zorder=2, **DISPLAY_LEVEL_LINE)
    axd.grid(True, alpha=0.25)
    lo, hi = state["dir_xlim"]
    axd.set_xlim(lo, hi)
    ticks = np.arange(np.ceil(lo / 60.0) * 60.0, hi + 1e-6, 60.0)
    axd.set_xticks(ticks[(ticks > lo + 1.0) & (ticks < hi - 1.0)])
    axd.xaxis.set_minor_locator(mticker.MultipleLocator(30))
    axd.set_xlabel(r"wind direction / $^\circ$")
    axd.tick_params(labelleft=False)
    _add_panel_label(axd, letter)


def _draw_profile_panel(axp, state, time_index, letter="a"):
    z_km = state["z_levels_km"]
    for k, method in enumerate(METHODS):
        q25 = state["q25_plot"][k, :, time_index]
        q75 = state["q75_plot"][k, :, time_index]
        ok = np.isfinite(q25) & np.isfinite(q75)
        axp.fill_betweenx(z_km, q25, q75, where=ok, color=method["color"],
                          alpha=KERNEL_IQR_ALPHA, linewidth=0, zorder=1)
        axp.plot(state["med_plot"][k, :, time_index], z_km, color=method["color"], lw=2.0,
                 zorder=4)
    axp.plot(state["ambient_prof"], z_km, color=ct.AMBIENT_LINE_COLOR, lw=ct.AMBIENT_LINE_LW,
             ls=ct.AMBIENT_LINE_STYLE, zorder=2)
    axp.plot(state["mean_wind_plot"][:, time_index], z_km, color=ct.MEANWIND_LINE_COLOR,
             lw=ct.MEANWIND_LINE_LW, ls=ct.MEANWIND_LINE_STYLE, zorder=3)
    axp.axhline(state["display_z_km"], zorder=2, **DISPLAY_LEVEL_LINE)
    axp.grid(True, alpha=0.25)
    axp.set_xlim(state["field_clim"])
    axp.set_ylim(z_km.min() - 0.4, z_km.max() + 0.4)
    axp.set_xlabel(r"wind speed / m$\,$s$^{-1}$")
    axp.set_ylabel(r"$z$ / km")
    _add_panel_label(axp, letter)


def _draw_pvdist_panel(axf, state, time_index, letter="c"):
    """Per-level median nearest peak-valley distance of the tracked signed-T' populations."""
    z_km = state["z_levels_km"]
    axf.plot(state["pvd_plot"][:, time_index], z_km, color="black", lw=1.9, zorder=3)
    axf.axhline(state["display_z_km"], zorder=2, **DISPLAY_LEVEL_LINE)
    axf.grid(True, alpha=0.25)
    px = float(state["pvd_xmax"])
    axf.set_xlim(0.0, px)
    axf.set_xticks([t for t in axf.get_xticks() if 1e-6 < t < px - 1e-6])
    axf.set_xlim(0.0, px)
    axf.set_xlabel("peak-valley distance / km")
    axf.tick_params(labelleft=False)
    _add_panel_label(axf, letter)


def _draw_size_panel(axf, state, time_index, letter="d"):
    """Per-level median peak-to-trough distance (tube warm-core -> nearest T' trough) + IQR."""
    z_km = state["z_levels_km"]
    color = L2SEG_COLORS["tube"]
    q25 = state["size_q25_plot"][:, time_index]
    q75 = state["size_q75_plot"][:, time_index]
    ok = np.isfinite(q25) & np.isfinite(q75)
    axf.fill_betweenx(z_km, q25, q75, where=ok, color=color, alpha=KERNEL_IQR_ALPHA,
                      linewidth=0, zorder=1)
    axf.plot(state["size_med_plot"][:, time_index], z_km, color=color, lw=1.9, zorder=3)
    axf.axhline(state["display_z_km"], zorder=2, **DISPLAY_LEVEL_LINE)
    axf.grid(True, alpha=0.25)
    sx = float(state["size_xmax"])
    axf.set_xlim(0.0, sx)
    axf.set_xticks([t for t in axf.get_xticks() if 1e-6 < t < sx - 1e-6])
    axf.set_xlim(0.0, sx)
    axf.set_xlabel("peak-trough dist. / km")
    axf.tick_params(labelleft=False)
    _add_panel_label(axf, letter)


def _draw_count_panel(axc, state, time_index, letter="d"):
    z_km = state["z_levels_km"]
    for k, method in enumerate(METHODS):
        axc.plot(state["count_plot"][k, :, time_index], z_km, color=method["color"], lw=1.9,
                 zorder=3)
    axc.axhline(state["display_z_km"], zorder=2, **DISPLAY_LEVEL_LINE)
    axc.grid(True, alpha=0.25)
    cx = float(state["count_xmax"])
    axc.set_xlim(0.0, cx)
    axc.set_xticks([t for t in axc.get_xticks() if 1e-6 < t < cx - 1e-6])
    axc.set_xlim(0.0, cx)
    axc.set_xlabel("feature count / -")
    axc.yaxis.tick_right()
    axc.yaxis.set_label_position("right")
    axc.set_ylabel(r"$z$ / km")
    axc.tick_params(labelleft=False, labelright=True)
    _add_panel_label(axc, letter)


def _draw_flux_panel(axf, state, time_index, letter="c"):
    z_km = state["z_levels_km"]
    axf.axvline(0.0, color="0.6", lw=0.8, zorder=1)
    axf.plot(state["uw_plot"][:, time_index], z_km, zorder=3, **FLUX_UW_LINE)
    axf.plot(state["vw_plot"][:, time_index], z_km, zorder=3, **FLUX_VW_LINE)
    axf.axhline(state["display_z_km"], zorder=2, **DISPLAY_LEVEL_LINE)
    axf.grid(True, alpha=0.25)
    fx = float(state["flux_xmax"])
    axf.set_xlim(-fx, fx)
    axf.set_xticks([t for t in axf.get_xticks() if -fx + 1e-6 < t < fx - 1e-6])
    axf.set_xlim(-fx, fx)
    axf.set_xlabel(r"momentum flux / mPa")
    axf.tick_params(labelleft=False)
    _add_panel_label(axf, letter)
    handles = [Line2D([], [], label=r"$\bar{\rho}\langle u'w'\rangle$", **FLUX_UW_LINE),
               Line2D([], [], label=r"$\bar{\rho}\langle v'w'\rangle$", **FLUX_VW_LINE)]
    axf.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.92,
               borderpad=0.4, labelspacing=0.35, handlelength=1.8)


def _draw_display_map(axl, state, time_index, dx_km, current_x, letter="e"):
    x_km = state["x"] / 1000.0
    y_km = state["y"] / 1000.0
    if MAP_FIELD == "t" and state["disp_T"] is not None:
        mesh = axl.pcolormesh(x_km, y_km, state["disp_T"][time_index], cmap=TPRIME_COLORMAP,
                              vmin=state["tprime_clim"][0], vmax=state["tprime_clim"][1],
                              shading="auto", rasterized=True)
    else:
        mesh = axl.pcolormesh(x_km, y_km, state["disp_S"][time_index], cmap=ct.MAP_FIELD_CMAP,
                              vmin=state["field_clim"][0], vmax=state["field_clim"][1],
                              shading="auto", rasterized=True)
    seg = state.get("pv_segments", {}).get(int(time_index))
    if seg is not None:
        wx, wy, tx, ty = seg
        for a, b, c, d in zip(wx, wy, tx, ty):
            axl.plot([a, c], [b, d], **PV_LINE)
            axl.plot([c], [d], marker="o", ms=2.5, color=PV_LINE["color"], zorder=6)
    if state["lambda2_level"] is not None:
        axl.contour(x_km, y_km, state["disp_L2"][time_index], levels=[state["lambda2_level"]],
                    colors=ct.LAMBDA2_CONTOUR_COLOR, linewidths=ct.LAMBDA2_CONTOUR_WIDTH)
    circle_norm = plt.Normalize(*state["circle_clim"])
    for k, method in enumerate(METHODS):
        radius_km = 0.5 * method["diameter"] * dx_km
        pts = state["display_feats"][k].get(int(time_index))
        if pts is None:
            continue
        xs, ys, sp = pts
        for xc, yc, sc in zip(xs, ys, sp):
            fc = ct.MAP_FIELD_CMAP(circle_norm(sc)) if np.isfinite(sc) else "none"
            axl.add_patch(Circle((xc, yc), radius_km, facecolor=fc,
                                 edgecolor=method["color"], lw=ct.MARKER_LW,
                                 alpha=ct.MARKER_ALPHA, zorder=5))
    lh = [Line2D([], [], marker="o", ls="none", markerfacecolor="none",
                 markeredgecolor=m["color"],
                 markersize=6 if m["diameter"] <= KERNEL_DIAMETERS_PX[0] else 9,
                 label=_method_legend_label(m))
          for m in METHODS]
    axl.legend(handles=lh, loc="lower left", fontsize=8, framealpha=0.9, handletextpad=0.3,
               borderpad=0.3, title=f"tracked {FIELD_SYMBOL[PROFILE_TRACK_FIELD]}",
               title_fontsize=8)
    axl.set_xlim(x_km.min(), x_km.max())
    axl.set_ylim(y_km.min(), y_km.max())
    axl.set_xlabel("x / km")
    axl.set_ylabel("y / km")
    _add_panel_label(axl, letter)
    ct.add_corner_label(axl, f"$z = {state['display_z_km']:.0f}$ km\n"
                             f"$t = {ct.format_elapsed_hms(current_x)}$")
    return mesh


def _draw_display_series(axr, state, time_index, current_x, letter="f"):
    elapsed = state["times_s"]
    d = state["display_index"]
    for k, method in enumerate(METHODS):
        axr.fill_between(elapsed, state["q25_plot"][k, d], state["q75_plot"][k, d],
                         color=method["color"], alpha=KERNEL_IQR_ALPHA, linewidth=0, zorder=1)
        axr.plot(elapsed, state["med_plot"][k, d], color=method["color"], lw=1.9, zorder=4)
    axr.plot(elapsed, state["mean_wind_plot"][d], color=ct.MEANWIND_LINE_COLOR,
             lw=ct.MEANWIND_LINE_LW, ls=ct.MEANWIND_LINE_STYLE, zorder=3)
    ambient = float(state["ambient_prof"][d])
    axr.axhline(ambient, color=ct.AMBIENT_LINE_COLOR, lw=ct.AMBIENT_LINE_LW,
                ls=ct.AMBIENT_LINE_STYLE, zorder=2)
    axr.axvline(current_x, color="black", lw=1.5, ls="--", zorder=4)
    axr.grid(True, alpha=0.25)
    axr.xaxis.set_major_formatter(mticker.FuncFormatter(ct.format_elapsed_tick_label))
    axr.xaxis.set_major_locator(mticker.MultipleLocator(600))
    axr.yaxis.set_label_position("right")
    axr.yaxis.tick_right()
    axr.set_ylabel(r"feature speed / m$\,$s$^{-1}$")
    axr.set_xlabel("model time / hh:mm")
    axr.set_xlim(state["time_axis_limits"])
    axr.set_ylim(0.0, 1.08 * float(state["right_ymax"]))
    _add_panel_label(axr, letter)
    handles = [Line2D([], [], color=m["color"], lw=1.9, label=_method_legend_label(m))
               for m in METHODS]
    handles += [
        Patch(facecolor="0.6", alpha=0.25, label="IQR"),
        Line2D([], [], color=ct.MEANWIND_LINE_COLOR, lw=ct.MEANWIND_LINE_LW,
               ls=ct.MEANWIND_LINE_STYLE, label="mean wind"),
        Line2D([], [], color=ct.AMBIENT_LINE_COLOR, lw=ct.AMBIENT_LINE_LW,
               ls=ct.AMBIENT_LINE_STYLE, label="ambient wind"),
    ]
    axr.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.92,
               borderpad=0.4, labelspacing=0.35, ncol=2, columnspacing=1.0)
    med_now = float(state["pooled_med_plot"][int(time_index)])
    if np.isfinite(med_now):
        rows = (rf"$v_\mathrm{{all}} = {med_now:.0f}$ m$\,$s$^{{-1}}$" + "\n"
                + rf"$v_\mathrm{{all}}\!-\!U_\mathrm{{amb}} = {med_now - ambient:+.0f}$"
                + r" m$\,$s$^{-1}$")
    else:
        rows = (r"$v_\mathrm{all} = $ --" + "\n" + r"$v_\mathrm{all}\!-\!U_\mathrm{amb} = $ --")
    axr.text(0.035, 0.05, rows, transform=axr.transAxes, ha="left", va="bottom",
             fontsize=ct.DIFF_LABEL_FONTSIZE, multialignment="left",
             bbox={"boxstyle": "round", "lw": 0.67, "facecolor": "white", "edgecolor": "black"},
             zorder=6)


def render_frame(state, time_index):
    x_km = state["x"] / 1000.0
    dx_km = float(np.mean(np.diff(x_km))) if x_km.size > 1 else 0.4
    current_x = float(state["times_s"][time_index])

    left_axes_w = (FIG_WIDTH - AXES_MARGIN_IN) / (1.0 + RIGHT_WIDTH_RATIO)
    fig_height = left_axes_w * (1.0 + PROFILE_HEIGHT_RATIO) + CBAR_HEADROOM_IN
    fig = plt.figure(figsize=(FIG_WIDTH, fig_height), layout="constrained")
    outer = fig.add_gridspec(2, 1, height_ratios=[PROFILE_HEIGHT_RATIO, 1.0])
    fig.get_layout_engine().set(w_pad=1.5 / 72.0, wspace=0.0)
    n_top = 5 if PROFILE_TRACK_FIELD == "l2seg" else 4
    gs_top = outer[0].subgridspec(1, n_top, width_ratios=[1.0] * n_top)
    gs_bot = outer[1].subgridspec(1, 2, width_ratios=[1.0, RIGHT_WIDTH_RATIO])
    top_axes = [fig.add_subplot(gs_top[0])]
    for j in range(1, n_top):
        top_axes.append(fig.add_subplot(gs_top[j], sharey=top_axes[0]))
    axl = fig.add_subplot(gs_bot[0])
    axr = fig.add_subplot(gs_bot[1])
    for ax in top_axes:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

    letters = "abcdefg"
    _draw_profile_panel(top_axes[0], state, time_index, letters[0])
    _draw_direction_panel(top_axes[1], state, time_index, letters[1])
    if PROFILE_TRACK_FIELD == "l2seg":
        _draw_flux_panel(top_axes[2], state, time_index, letters[2])
        _draw_size_panel(top_axes[3], state, time_index, letters[3])
        _draw_count_panel(top_axes[4], state, time_index, letters[4])
    elif PROFILE_TRACK_FIELD == "tpv":
        _draw_pvdist_panel(top_axes[2], state, time_index, letters[2])
        _draw_count_panel(top_axes[3], state, time_index, letters[3])
    else:
        _draw_flux_panel(top_axes[2], state, time_index, letters[2])
        _draw_count_panel(top_axes[3], state, time_index, letters[3])
    mesh = _draw_display_map(axl, state, time_index, dx_km, current_x, letters[n_top])
    _draw_display_series(axr, state, time_index, current_x, letters[n_top + 1])

    if MAP_FIELD == "t" and state["disp_T"] is not None:
        cbar = fig.colorbar(mesh, ax=axl, location="bottom", shrink=0.9, aspect=30, pad=0.015,
                            extend="both")
        cbar.set_label(r"$T'$ / K")
    else:
        cbar = fig.colorbar(mesh, ax=axl, location="bottom", shrink=0.9, aspect=30, pad=0.015,
                            extend="max")
        cbar.set_ticks(np.arange(0.0, state["field_clim"][1] + 1e-6, ct.FIELD_CBAR_TICK_STEP))
        cbar.set_label(r"$|\mathbf{u}_\mathrm{h}|$ / m$\,$s$^{-1}$")
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(*state["circle_clim"]), cmap=ct.MAP_FIELD_CMAP)
    cbar2 = fig.colorbar(sm, ax=axr, location="bottom",
                         shrink=0.9 / RIGHT_WIDTH_RATIO, aspect=30, pad=0.015, extend="max")
    cbar2.set_ticks(np.arange(0.0, state["circle_clim"][1] + 1e-6, ct.CIRCLE_CBAR_TICK_STEP))
    cbar2.set_label(r"feature speed / m$\,$s$^{-1}$")

    fig.canvas.draw()
    fig.set_layout_engine("none")
    pl = axl.get_position()
    span_x0 = top_axes[0].get_position().x0
    span_x1 = top_axes[-1].get_position().x1
    e_width = (span_x1 - span_x0 - BOTTOM_GAP_FRAC) / (1.0 + RIGHT_WIDTH_RATIO)
    r_x0 = span_x0 + e_width + BOTTOM_GAP_FRAC
    r_width = span_x1 - r_x0
    axl.set_position([span_x0, pl.y0, e_width, pl.height])
    axr.set_position([r_x0, pl.y0, r_width, pl.height])
    for ax in top_axes:
        p = ax.get_position()
        ax.set_position([p.x0, p.y0, p.width, pl.height])
    p1, p2 = cbar.ax.get_position(), cbar2.ax.get_position()
    cbar_y0 = min(p1.y0, p2.y0)
    cbar_h = min(p1.height, p2.height)
    cbar_w = min(p1.width, p2.width)
    c1 = span_x0 + 0.5 * e_width
    c2 = r_x0 + 0.5 * r_width
    delta = 0.5 * (1.0 - (c2 + 0.5 * cbar_w) - (c1 - 0.5 * cbar_w))
    cbar.ax.set_position([c1 - 0.5 * cbar_w + delta, cbar_y0, cbar_w, cbar_h])
    cbar2.ax.set_position([c2 - 0.5 * cbar_w + delta, cbar_y0, cbar_w, cbar_h])

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb


def render_size_speed_figure(state, out_path):
    """Two-panel per-tube scatter: peak-trough size vs feature speed and vs deficit (speed-wind)."""
    df = state["tube_scatter"]
    if df is None or not len(df):
        print("[i]  no propagating tubes -> skipping size-speed scatter")
        return
    zc = df["level_km"].to_numpy()
    norm = plt.Normalize(float(zc.min()), float(zc.max()))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.0), sharex=True,
                                   layout="constrained")
    for ax, ycol, ylabel, zeroline in (
            (ax1, "speed_ms", r"feature speed / m$\,$s$^{-1}$", False),
            (ax2, "deficit_ms", r"speed $-$ mean wind / m$\,$s$^{-1}$", True)):
        ax.scatter(df["size_km"], df[ycol], c=zc, cmap=ct.MAP_FIELD_CMAP, norm=norm,
                   s=SCATTER_MARKER_SIZE, alpha=SCATTER_ALPHA, linewidths=0, zorder=3)
        centers, medians = ct._binned_median(df["size_km"], df[ycol], 2.0)
        if centers.size:
            ax.plot(centers, medians, color="black", lw=2.0, marker="D", ms=4.5, zorder=4)
        if zeroline:
            ax.axhline(0.0, color="0.5", lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("peak-trough distance / km")
        ax.set_ylabel(ylabel)
    ax1.set_xlim(left=0.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=ct.MAP_FIELD_CMAP)
    cbar = fig.colorbar(sm, ax=(ax1, ax2), location="right", shrink=0.85, aspect=30)
    cbar.set_label(r"$z$ / km")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[i]  Wrote {out_path}")
    csv_path = out_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"[i]  Wrote {csv_path}")


_worker_state = None


def _init_worker(state, pbar=None):
    global _worker_state
    _worker_state = state
    _worker_state["pbar"] = pbar


def _render_worker(time_index):
    rgb = render_frame(_worker_state, int(time_index))
    imageio.imwrite(ct.frame_png_path(_worker_state["frames_dir"], time_index), rgb)
    pbar = _worker_state.get("pbar")
    if pbar is not None:
        plt_helper.show_progress(pbar["progress_counter"], pbar["lock"], pbar["stime"],
                                 pbar["ntasks"])
    return int(time_index)


def run_animation(simulation_name, cube_index=0, render_all=True):
    simulation_name = os.path.basename(os.path.normpath(simulation_name))
    state, time_indices = build_state(simulation_name, cube_index)

    tag = "" if PROFILE_TRACK_FIELD == "w" else f"_{PROFILE_TRACK_FIELD}"
    out_dir = ct.ANIMATION_ROOT / f"{simulation_name}_trackprof{int(cube_index)}{tag}"
    frames_dir = ct.clear_frame_directory(out_dir)
    state["frames_dir"] = str(frames_dir)
    outfile = out_dir / f"anime_trackprof_{simulation_name}_cube{int(cube_index)}{tag}.mp4"

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
            with ctx.Pool(processes=ncpus, initializer=_init_worker,
                          initargs=(state, pbar)) as pool:
                for _ in pool.imap_unordered(_render_worker, time_indices):
                    pass
        plt_helper.create_animation(str(frames_dir), outfile.name, fps=10)
        generated = frames_dir / outfile.name
        if generated.resolve() != outfile.resolve():
            shutil.move(str(generated), str(outfile))
        print(f"[i]  Wrote {outfile}")

    if PROFILE_TRACK_FIELD == "l2seg":
        scatter_path = out_dir / f"sizespeed_{simulation_name}_cube{int(cube_index)}{tag}.png"
        render_size_speed_figure(state, scatter_path)

    print("[i]  Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vertical-profile vortex-feature tracking of a PMAP cube.")
    parser.add_argument("simulation", help="simulation jobname (scratch dir)")
    parser.add_argument("notest", nargs="?", default=None,
                        help="'notest' renders the full animation; omit for a single test frame")
    parser.add_argument("--cube", type=int, default=0, help="cube index -> cube_<CUBE>.nc")
    parser.add_argument("--display-level", type=float, default=None,
                        help="display level (m) for the bottom map/series row")
    parser.add_argument("--dz", type=float, default=None,
                        help="vertical analysis spacing (m); overrides PROFILE_DZ_M (coarser = "
                             "fewer levels = faster, useful for test frames)")
    parser.add_argument("--field", choices=tuple(FIELD_SYMBOL), default=None,
                        help="tracked field: 'w' (|w|, default), 't' (|T'|), 'tpv' (signed T' "
                             "peaks & valleys) or 'l2seg' (lambda2 segmentation tubes vs "
                             "small-scale + peak-trough size panel & scatter)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.display_level is not None:
        global DISPLAY_LEVEL_M
        DISPLAY_LEVEL_M = float(args.display_level)
    if args.dz is not None:
        global PROFILE_DZ_M
        PROFILE_DZ_M = float(args.dz)
    if args.field is not None:
        global PROFILE_TRACK_FIELD, METHODS
        PROFILE_TRACK_FIELD = str(args.field)
        METHODS = _build_methods()
    render_all = str(args.notest).lower() == "notest"
    run_animation(args.simulation, cube_index=args.cube, render_all=render_all)


if __name__ == "__main__":
    main()
