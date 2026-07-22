"""Dynamics-overview animation for the ERA5-nested (sam) runs -- v4 two-row design.

TOP ROW (both modes): (a) xz section through CORAL (GW field + isentropes),
(b) yz section at x=0 (GW field + smoothed u contours, u=0 bold), (c) CORAL
virtual-lidar T' curtain (Butterworth 20 km, time cursor, UTC top axis).
BOTTOM ROW (mode-dependent horizontal maps):
  mode "wind":   |V| at 60 km, 40 km (batlow, shared cbar) and Z_LOW (tokyo)
  mode "tprime": T' (horizontal separable Butterworth high-pass, GW_HP_CUTOFF_M)
                 at 60 km and 40 km + |V| at Z_LOW as tropospheric context.
All maps keep isobars/barbs (wind maps), terrain coastline, section lines, cube boxes.

    dyn_overview.py SIM [notest] [w|t] [wind|tprime]

(old 6-panel v3 docstring below)


Row-major panels (a-f), maps left, sections+lidar right, colorbars far right:

  (a) |V| at 60 km map          (b) CORAL virtual-lidar T' curtain (time-height,
      batlow, shared scale w/ c     Butterworth vertical low-pass LIDAR_CUTOFF_M,
                                    moving time cursor + UTC clock)
  (c) |V| at 40 km map          (d) xz section through CORAL: GW color field
      batlow                        (variant t: T' vs level mean | w) + isentropes
  (e) |V| surface map           (f) yz section at x=0: GW color field + u contours
      Crameri tokyo                 (10 m/s steps, negative dashed, u=0 bold)

Maps carry sparse barbs, many thin smoothed isobars (p = p00*exner^(cp/Rd), the
geopotential analog on z surfaces), terrain coastline, and dashed section-location
lines. Panels b, d, f share the T' colormap/scale (variant w recolors d+f only).

    /home/b/b309199/venvs/post-venv/bin/python dyn_overview.py SIM [notest] [w|t]

Output: mapy/data/pmap-animations/<SIM>_dyn_<var>/ + dyn_<var>_<SIM>.mp4
(mp4 via plt_helper.create_animation / imageio -- no bare ffmpeg on the nodes).
Submit with rdyn_pmap.sh SIM [notest] [w|t]. T0_UTC must match the run (20:00 UTC
for the darwin_240718_era5 family).
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from cmcrameri import cm as cmc
import scipy.ndimage
import scipy.signal

import yaml

import cmaps
import plt_helper

plt.style.use("/work/bd0620/b309199/mapy/src/latex_default.mplstyle")

T0_UTC = "2024-07-18T20:00"
Z_TOP_MAP = 60000.0
Z_MID_MAP = 40000.0
Z_LOW_MAP = 5000.0
Y_XZ = 0.0
X_YZ = 0.0
SPEED_LEV_UP = np.arange(0, 171, 10)
SPEED_LEV_LOW = np.arange(0, 41, 2.5)
W_CLIM = 16.0
T_CLIM = 20.0
U_CONTOUR_STEP = 15.0
BARB_STRIDE = 100
N_PRESSURE_LEVELS = 26
PRESSURE_LW = 0.35
PRESSURE_SMOOTH_CELLS = 6.0
U_SMOOTH_CELLS = (6.0, 10.0)
ISENTROPES = np.array([275, 285, 295, 305, 315, 330, 350, 375, 400, 430, 470, 520,
                       580, 650, 730, 830, 950, 1100, 1300, 1550, 1900, 2400, 3100,
                       4100, 5500, 7300, 9500])
SPONGE_COLOR = "black"
SPONGE_LW = 1.0
SPONGE_LS = (0, (1, 1.4))
SPONGE_CURRENT = {"wx": 0.0, "wy": 0.0, "wtop": 0.0}
LIDAR_CUTOFF_M = 20000.0
LIDAR_DZ_M = 200.0
GW_HP_CUTOFF_M = 500e3
TPRIME_LOW_FACTOR = 4.0
NPROC = int(os.environ.get("DYN_NPROC", "32"))
FPS = 12

RUNDIR = "/scratch/b/b309199"
OUTBASE = "/work/bd0620/b309199/mapy/data/pmap-animations"

G, RD, CP, P0 = 9.80616, 287.05, 1004.0, 1.0e5

XLBL, YPP = 0.04, 0.93
XPP = 1 - XLBL
BOX_RND = {"boxstyle": "round", "lw": 0.67, "facecolor": "white", "edgecolor": "black"}
BOX_CIR = {"boxstyle": "circle", "lw": 0.67, "facecolor": "white", "edgecolor": "black"}


def panel_letter(ax, letter):
    ax.text(XPP, YPP, letter, transform=ax.transAxes, weight="bold", ha="right", bbox=BOX_CIR)


def panel_label(ax, text):
    ax.text(XLBL, YPP, text, transform=ax.transAxes, ha="left", bbox=BOX_RND)


def open_slices(sim):
    d = f"{RUNDIR}/{sim}"
    dsz = xr.open_dataset(f"{d}/slices_z.nc", decode_times=False)
    dsy = xr.open_dataset(f"{d}/slices_y.nc", decode_times=False)
    dsx = xr.open_dataset(f"{d}/slices_x.nc", decode_times=False)
    return dsz, dsy, dsx


def pressure_hpa(exner):
    return P0 / 100.0 * exner ** (CP / RD)


def fixed_pressure_levels(dsz, kz, n=N_PRESSURE_LEVELS):
    p = pressure_hpa(dsz.exner_total.isel(time=1, z=kz).values)
    lo, hi = np.nanpercentile(p, 0.5), np.nanpercentile(p, 99.5)
    return np.linspace(lo, hi, n)


def surface_k(dsz):
    return 0 if float(dsz.z.values[0]) < 100.0 else None


def surface_height(dsz):
    k = surface_k(dsz)
    if k is None:
        return None
    zt = dsz.zcr.isel(z=k)
    if "time" in zt.dims:
        zt = zt.isel(time=1)
    return zt.values


def butter_lowpass_2d(field, dx_m, cutoff_m):
    """Separable Butterworth low-pass along both map axes (edge-safe filtfilt)."""
    b, a = scipy.signal.butter(5, 2.0 * dx_m / cutoff_m)
    bg = scipy.signal.filtfilt(b, a, field, axis=0)
    return scipy.signal.filtfilt(b, a, bg, axis=1)


def plot_tprime_map(ax, dsz, it, kz, cmap, clim, terr, rects, plev, factor=1.0):
    """Same cross-section as the wind map (isobars, terrain, lines, cube boxes) but color =
    T' from the separable zonal+meridional Butterworth high-pass; no barbs."""
    x = dsz.x.values / 1e3
    y = dsz.y.values / 1e3
    th = dsz.theta_total.isel(time=it, z=kz).values
    ex = dsz.exner_total.isel(time=it, z=kz).values
    t = th * ex
    tp = t - butter_lowpass_2d(t, float(dsz.x.values[1] - dsz.x.values[0]), GW_HP_CUTOFF_M)
    pc = ax.pcolormesh(x, y, factor * tp.T, cmap=cmap, vmin=-clim, vmax=clim, rasterized=True)
    p = pressure_hpa(dsz.exner_total.isel(time=it, z=kz).values)
    p = scipy.ndimage.gaussian_filter(p, PRESSURE_SMOOTH_CELLS)
    cs = ax.contour(x, y, p.T, levels=plev, colors="grey", linewidths=0.28)
    ax.clabel(cs, cs.levels[::5], fontsize=6, fmt="%.3g", colors="grey")
    if terr is not None:
        ax.contour(x, y, terr.T, levels=[20.0], colors="dimgray", linewidths=0.5)
    ax.axvline(X_YZ / 1e3, color="black", ls="--", lw=0.8)
    ax.axhline(Y_XZ / 1e3, color="black", ls="--", lw=0.8)
    for x0, x1, y0, y1 in rects:
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="black", ls="--", lw=1.0)
    shade_sponges_map(ax, x, y, SPONGE_CURRENT)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    return pc


def gw_field(ds, it, sel_dim, sel_idx, variant, dh_m):
    """GW color field on a vertical slice: w directly, or T' = T minus a HORIZONTAL
    Butterworth low-pass (cutoff GW_HP_CUTOFF_M) applied per level along the slice --
    vertical filtering is avoided on purpose (jet-region mountain waves reach
    lambda_z ~ 2*pi*U/N ~ 30-45 km and would be damaged)."""
    if variant == "w":
        return ds.uvelz.isel(time=it, **{sel_dim: sel_idx}).values
    th = ds.theta_total.isel(time=it, **{sel_dim: sel_idx}).values
    ex = ds.exner_total.isel(time=it, **{sel_dim: sel_idx}).values
    t = th * ex
    b, a = scipy.signal.butter(5, 2.0 * dh_m / GW_HP_CUTOFF_M)
    bg = scipy.signal.filtfilt(b, a, t, axis=0)
    return t - bg


def lidar_curtain(sim):
    """CORAL-column T'(t, z): profiles on a uniform z grid, Butterworth vertical
    low-pass (cutoff LIDAR_CUTOFF_M) as background, perturbation = full - background."""
    _, _, dsx = open_slices(sim)
    ix = int(np.argmin(np.abs(dsx.x.values - X_YZ)))
    jy = int(np.argmin(np.abs(dsx.y.values - 0.0)))
    th = dsx.theta_total.isel(x=ix, y=jy).values
    ex = dsx.exner_total.isel(x=ix, y=jy).values
    t_full = th * ex
    zcr = dsx.zcr.isel(x=ix, y=jy)
    if "time" in zcr.dims:
        zcr = zcr.isel(time=1)
    zcr = zcr.values
    times = dsx.time.values
    dsx.close()

    zu = np.arange(0.0, 90000.0 + LIDAR_DZ_M / 2, LIDAR_DZ_M)
    b, a = scipy.signal.butter(5, 2.0 * LIDAR_DZ_M / LIDAR_CUTOFF_M)
    tp = np.full((len(times), len(zu)), np.nan)
    for i in range(len(times)):
        if not np.isfinite(times[i]):
            continue
        prof = np.interp(zu, zcr, t_full[i])
        bg = scipy.signal.filtfilt(b, a, prof)
        tp[i] = prof - bg
    return times, zu, tp


def sponge_params(sim):
    """Absorber widths [m] from the run's config snapshot (any yml in the run dir)."""
    for f in sorted(glob.glob(f"{RUNDIR}/{sim}/*.yml")):
        try:
            c = yaml.safe_load(open(f))
        except Exception:
            continue
        if isinstance(c, dict) and "absorber" in c:
            a = c["absorber"]
            return {"wx": float(a.get("widthx", 0.0)), "wy": float(a.get("widthy", 0.0)),
                    "wtop": float(a.get("depth", 0.0))}
    return {"wx": 0.0, "wy": 0.0, "wtop": 0.0}


def shade_sponges_map(ax, x, y, sp):
    """Thin dotted line along the absorber inner edge (a rectangle when both wx,wy>0)."""
    kw = dict(color=SPONGE_COLOR, lw=SPONGE_LW, ls=SPONGE_LS, zorder=3.0,
              solid_capstyle="butt")
    wx, wy = sp["wx"] / 1e3, sp["wy"] / 1e3
    x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
    xin0, xin1, yin0, yin1 = x0 + wx, x1 - wx, y0 + wy, y1 - wy
    ya, yb = (yin0, yin1) if sp["wy"] > 0 else (y0, y1)
    xa, xb = (xin0, xin1) if sp["wx"] > 0 else (x0, x1)
    if sp["wx"] > 0:
        ax.plot([xin0, xin0], [ya, yb], **kw)
        ax.plot([xin1, xin1], [ya, yb], **kw)
    if sp["wy"] > 0:
        ax.plot([xa, xb], [yin0, yin0], **kw)
        ax.plot([xa, xb], [yin1, yin1], **kw)


def shade_sponges_section(ax, h, ztop_km, sp, lateral_key):
    """Dotted line along the absorber inner edge: lateral edges + top edge (inverted-U)."""
    kw = dict(color=SPONGE_COLOR, lw=SPONGE_LW, ls=SPONGE_LS, zorder=3.0,
              solid_capstyle="butt")
    w = sp[lateral_key] / 1e3
    hin0, hin1 = h[0] + w, h[-1] - w
    ztop_in = ztop_km - sp["wtop"] / 1e3
    za, zb = (0.0, ztop_in) if sp["wtop"] > 0 else (0.0, ztop_km)
    xa, xb = (hin0, hin1) if sp[lateral_key] > 0 else (h[0], h[-1])
    if sp[lateral_key] > 0:
        ax.plot([hin0, hin0], [za, zb], **kw)
        ax.plot([hin1, hin1], [za, zb], **kw)
    if sp["wtop"] > 0:
        ax.plot([xa, xb], [ztop_in, ztop_in], **kw)


def cube_footprints(sim):
    """Footprints of the run's cube outputs; tolerates partial/corrupt files (e.g. a
    cube_N.tmp.nc left behind by an aborted run)."""
    rects = []
    for f in sorted(glob.glob(f"{RUNDIR}/{sim}/cube_[0-9]*.nc")):
        if ".tmp" in os.path.basename(f):
            continue
        try:
            with xr.open_dataset(f, decode_times=False) as c:
                rects.append((float(c.x.min()) / 1e3, float(c.x.max()) / 1e3,
                              float(c.y.min()) / 1e3, float(c.y.max()) / 1e3))
        except Exception as e:
            print(f"[warn] skipping unreadable cube file {f}: {e}", flush=True)
    return rects


def plot_speed_map(ax, dsz, it, kz, lev, cmap, plev, terr, barbcolor, rects, barb_kz=None):
    x = dsz.x.values / 1e3
    y = dsz.y.values / 1e3
    u = dsz.uvelx.isel(time=it, z=kz).values
    v = dsz.uvely.isel(time=it, z=kz).values
    spd = np.sqrt(u**2 + v**2)
    cf = ax.contourf(x, y, spd.T, levels=lev, cmap=cmap, extend="max")
    p = pressure_hpa(dsz.exner_total.isel(time=it, z=kz).values)
    p = scipy.ndimage.gaussian_filter(p, PRESSURE_SMOOTH_CELLS)
    cs = ax.contour(x, y, p.T, levels=plev, colors="black", linewidths=PRESSURE_LW)
    ax.clabel(cs, cs.levels[::5], fontsize=6, fmt="%.3g")
    s = BARB_STRIDE
    kb = kz if barb_kz is None else barb_kz
    ub = dsz.uvelx.isel(time=it, z=kb).values
    vb = dsz.uvely.isel(time=it, z=kb).values
    ax.barbs(x[::s], y[::s], ub[::s, ::s].T, vb[::s, ::s].T, length=4.5,
             linewidth=0.6, color=barbcolor)
    shade_sponges_map(ax, x, y, SPONGE_CURRENT)
    if terr is not None:
        ax.contour(x, y, terr.T, levels=[20.0], colors="dimgray", linewidths=0.5)
    ax.axvline(X_YZ / 1e3, color="black", ls="--", lw=0.8)
    ax.axhline(Y_XZ / 1e3, color="black", ls="--", lw=0.8)
    for x0, x1, y0, y1 in rects:
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="black", ls="--", lw=1.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    return cf


def render_frame(args):
    (it, sim, outdir, plevs, tval, variant, mode, cur_t, cur_z, cur_tp, rects, sp) = args
    global SPONGE_CURRENT
    SPONGE_CURRENT = sp
    dsz, dsy, dsx = open_slices(sim)
    kz_low = int(np.argmin(np.abs(dsz.z.values - Z_LOW_MAP)))
    jy = int(np.argmin(np.abs(dsy.y.values - Y_XZ)))
    ix = int(np.argmin(np.abs(dsx.x.values - X_YZ)))
    terr = surface_height(dsz)

    tp_cmap = cmaps.get_wave_cmap()
    gw_cmap = cmaps.get_vik_white_cmap() if variant == "w" else tp_cmap
    gw_clim = W_CLIM if variant == "w" else T_CLIM
    gw_lab = r"$w$ / m$\,$s$^{-1}$" if variant == "w" else r"$T'$ / K"

    fig = plt.figure(figsize=(12.6, 11.0), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.01)
    gs = fig.add_gridspec(5, 3, height_ratios=[1.15, 1.0, 0.012, 1.0, 0.06])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    ax_g = fig.add_subplot(gs[3, 0])
    ax_h = fig.add_subplot(gs[3, 1])
    ax_i = fig.add_subplot(gs[3, 2])
    cax_low = fig.add_subplot(gs[4, 0]).inset_axes([0.06, 0.3, 0.88, 0.55])
    cax_up = fig.add_subplot(gs[4, 1]).inset_axes([0.06, 0.3, 0.88, 0.55])
    cax_tp = fig.add_subplot(gs[4, 2]).inset_axes([0.06, 0.3, 0.88, 0.55])
    map_axes = (ax_d, ax_e, ax_f, ax_g, ax_h, ax_i)
    for h in fig.axes:
        if h not in (ax_a, ax_b, ax_c) + map_axes + (cax_low, cax_up, cax_tp):
            h.axis("off")

    x = dsy.x.values / 1e3
    zk = dsy.z.values / 1e3
    dx_m = float(dsy.x.values[1] - dsy.x.values[0])
    gw_xz = gw_field(dsy, it, "y", jy, variant, dx_m)
    ax_a.pcolormesh(x, zk, gw_xz.T, cmap=gw_cmap, vmin=-gw_clim, vmax=gw_clim, rasterized=True)
    th_xz = dsy.theta_total.isel(time=it, y=jy).values
    cs_th = ax_a.contour(x, zk, th_xz.T, levels=ISENTROPES, colors="dimgray", linewidths=0.35)
    ax_a.clabel(cs_th, ISENTROPES[::4], fontsize=6, fmt="%g K")
    zda = dsy.zcr.isel(y=jy)
    if "time" in zda.dims:
        zda = zda.isel(time=1)
    ax_a.plot(x, zda.values[:, 0] / 1e3, lw=1.5, color="black")
    ax_a.axvline(X_YZ / 1e3, color="black", ls="--", lw=0.8)
    shade_sponges_section(ax_a, x, float(zk[-1]), sp, "wx")
    panel_letter(ax_a, "a")
    d_extra = "" if variant == "w" else rf", $T'$: $\lambda_h<{GW_HP_CUTOFF_M/1e3:.0f}$ km"
    panel_label(ax_a, f"y: {dsy.y.values[jy]/1e3:.0f} km" + d_extra)
    ax_a.set_ylabel("altitude z / km")
    ax_a.xaxis.tick_top()
    ax_a.xaxis.set_label_position("top")
    ax_a.set_xlabel("streamwise x / km")

    yy = dsx.y.values / 1e3
    dy_m = float(dsx.y.values[1] - dsx.y.values[0])
    gw_yz = gw_field(dsx, it, "x", ix, variant, dy_m)
    ax_b.pcolormesh(yy, zk, gw_yz.T, cmap=gw_cmap, vmin=-gw_clim, vmax=gw_clim, rasterized=True)
    uu = dsx.uvelx.isel(time=it, x=ix).values
    uu = scipy.ndimage.gaussian_filter(uu, U_SMOOTH_CELLS)
    ulev = np.arange(-165.0, 165.1, U_CONTOUR_STEP)
    ulev = ulev[np.abs(ulev) > 0.1]
    ax_b.contour(yy, zk, uu.T, levels=ulev, colors="black", linewidths=0.35,
                 negative_linestyles="dashed")
    ax_b.contour(yy, zk, uu.T, levels=[0.0], colors="black", linewidths=1.7)
    zdx = dsx.zcr.isel(x=ix)
    if "time" in zdx.dims:
        zdx = zdx.isel(time=1)
    ax_b.plot(yy, zdx.values[:, 0] / 1e3, lw=1.5, color="black")
    ax_b.axvline(Y_XZ / 1e3, color="black", ls="--", lw=0.8)
    shade_sponges_section(ax_b, yy, float(zk[-1]), sp, "wy")
    panel_letter(ax_b, "b")
    panel_label(ax_b, f"x: {dsx.x.values[ix]/1e3:.0f} km")
    ax_b.xaxis.tick_top()
    ax_b.xaxis.set_label_position("top")
    ax_b.set_xlabel("spanwise y / km")
    ax_b.tick_params(labelleft=False)

    th_h = np.array(cur_t) / 3600.0
    pc_c = ax_c.pcolormesh(th_h, np.array(cur_z) / 1e3, np.array(cur_tp).T, cmap=tp_cmap,
                           vmin=-T_CLIM, vmax=T_CLIM, rasterized=True)
    ax_c.axvline(tval / 3600.0, color="black", lw=1.2)
    utc = np.datetime64(T0_UTC) + np.timedelta64(int(tval), "s")
    ax_c.text(XLBL, YPP, rf"CORAL $T'_{{\mathrm{{BWF\,{LIDAR_CUTOFF_M/1e3:.0f}\,km}}}}$  |  "
              f"{str(utc)[11:16]} UTC", transform=ax_c.transAxes, ha="left", bbox=BOX_RND)
    if sp["wtop"] > 0:
        ax_c.axhline(float(np.array(cur_z)[-1]) / 1e3 - sp["wtop"] / 1e3,
                     color=SPONGE_COLOR, lw=SPONGE_LW, ls=SPONGE_LS, zorder=3.0)
    panel_letter(ax_c, "c")
    ax_c.set_ylabel("altitude z / km")
    ax_c.yaxis.tick_right()
    ax_c.yaxis.set_label_position("right")
    hmax = np.nanmax(cur_t) / 3600.0
    hh = np.arange(1.0, hmax, 1.0)
    ax_c.set_xticks(hh)
    ax_c.set_xticklabels([f"{h:.0f} h" for h in hh], fontsize=7)
    ax_c.tick_params(axis="x", direction="in", pad=-14)
    ax_c.yaxis.set_minor_locator(AutoMinorLocator())
    secx = ax_c.secondary_xaxis("top")
    secx.set_xticks(hh)
    secx.set_xticklabels([str(np.datetime64(T0_UTC) + np.timedelta64(int(h * 3600), "s"))[11:16]
                          for h in hh], fontsize=7)
    secx.set_xlabel("UTC")

    kz20 = int(np.argmin(np.abs(dsz.z.values - 20000.0)))
    kz_mid = int(np.argmin(np.abs(dsz.z.values - Z_MID_MAP)))
    kz_top = int(np.argmin(np.abs(dsz.z.values - Z_TOP_MAP)))
    kz_sfc = surface_k(dsz)

    for ax, kz, let in ((ax_d, kz20, "d"), (ax_e, kz_mid, "e"), (ax_f, kz_top, "f")):
        plot_tprime_map(ax, dsz, it, kz, tp_cmap, T_CLIM, terr, rects, plevs[kz][::2])
        panel_letter(ax, let)
        panel_label(ax, f"z: {dsz.z.values[kz]/1e3:.0f} km")

    cf_low = plot_speed_map(ax_g, dsz, it, kz_low, SPEED_LEV_LOW, cmc.tokyo,
                            plevs["low"], terr, "white", rects, barb_kz=kz_sfc)
    panel_letter(ax_g, "g")
    panel_label(ax_g, f"z: {dsz.z.values[kz_low]/1e3:.0f} km")
    cf_up = plot_speed_map(ax_h, dsz, it, kz_mid, SPEED_LEV_UP, cmc.batlow,
                           plevs[kz_mid], terr, "white", rects)
    panel_letter(ax_h, "h")
    panel_label(ax_h, f"z: {dsz.z.values[kz_mid]/1e3:.0f} km")
    plot_speed_map(ax_i, dsz, it, kz_top, SPEED_LEV_UP, cmc.batlow,
                   plevs[kz_top], terr, "white", rects)
    panel_letter(ax_i, "i")
    panel_label(ax_i, f"z: {dsz.z.values[kz_top]/1e3:.0f} km")

    for ax in (ax_d, ax_e, ax_f):
        ax.tick_params(labelbottom=False)
    for ax in (ax_e, ax_f, ax_h, ax_i):
        ax.tick_params(labelleft=False)
    ax_d.set_ylabel("spanwise y / km")
    ax_g.set_ylabel("spanwise y / km")
    for ax in (ax_g, ax_h, ax_i):
        ax.set_xlabel("streamwise x / km")

    z_low_km = dsz.z.values[kz_low] / 1e3
    fig.colorbar(cf_low, cax=cax_low, orientation="horizontal",
                 label=rf"$|\mathbf{{v}}_h|$ ({z_low_km:.0f} km) / m$\,$s$^{{-1}}$")
    fig.colorbar(cf_up, cax=cax_up, orientation="horizontal",
                 label=r"$|\mathbf{v}_h|$ (40, 60 km) / m$\,$s$^{-1}$")
    fig.colorbar(pc_c, cax=cax_tp, orientation="horizontal", label=r"$T'$ / K", extend="both")

    fig.savefig(f"{outdir}/frame_{it:04d}.png", dpi=110)
    plt.close(fig)
    dsz.close(); dsy.close(); dsx.close()
    return it


def main():
    sim = sys.argv[1]
    full = "notest" in sys.argv[2:]
    variant = "w" if "w" in [a for a in sys.argv[2:] if a in ("w", "t")] else "t"
    mode = "tprime" if "tprime" in sys.argv[2:] else "wind"
    outdir = f"{OUTBASE}/{sim}_dyn_{variant}_{mode}"
    os.makedirs(outdir, exist_ok=True)

    dsz, dsy, dsx = open_slices(sim)
    kz_top = int(np.argmin(np.abs(dsz.z.values - Z_TOP_MAP)))
    kz_mid = int(np.argmin(np.abs(dsz.z.values - Z_MID_MAP)))
    kz_low = int(np.argmin(np.abs(dsz.z.values - Z_LOW_MAP)))
    kz20 = int(np.argmin(np.abs(dsz.z.values - 20000.0)))
    plevs = {kz_top: fixed_pressure_levels(dsz, kz_top),
             kz_mid: fixed_pressure_levels(dsz, kz_mid),
             kz20: fixed_pressure_levels(dsz, kz20),
             "low": fixed_pressure_levels(dsz, kz_low)}
    times = dsz.time.values
    valid = [i for i in range(len(times)) if np.isfinite(times[i])]
    dsz.close(); dsy.close(); dsx.close()

    cur_t, cur_z, cur_tp = lidar_curtain(sim)
    m = np.isfinite(cur_t)
    cur_t, cur_tp = cur_t[m], cur_tp[m]

    frames = valid if full else [valid[-1]]
    if full:
        for f in glob.glob(f"{outdir}/*.png"):
            os.remove(f)
    print(f"[i] {sim} variant={variant} mode={mode}: rendering {len(frames)} frame(s)", flush=True)
    rects = cube_footprints(sim)
    sp = sponge_params(sim)
    print(f"[i] sponges: wx={sp['wx']/1e3:.0f} km, wy={sp['wy']/1e3:.0f} km, wtop={sp['wtop']/1e3:.0f} km")
    args = [(it, sim, outdir, plevs, float(times[it]), variant, mode, cur_t, cur_z, cur_tp, rects, sp)
            for it in frames]
    if len(args) == 1:
        render_frame(args[0])
    else:
        with Pool(NPROC) as pool:
            for it in pool.imap_unordered(render_frame, args):
                print(f"[ok] frame {it}", flush=True)

    if full:
        plt_helper.create_animation(outdir, f"dyn_{variant}_{mode}_{sim}.mp4", fps=FPS)


if __name__ == "__main__":
    main()
