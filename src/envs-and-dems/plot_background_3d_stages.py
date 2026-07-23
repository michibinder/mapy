"""Visualize the ERA5 -> JAWARA-blend -> balance modification of the 3D nest input data.

Meridional (yz) cross-section at a fixed x (upstream of the Andes, over water) showing the
three stages the background field passes through in build_background_3d.py, on the shared
working grid (uniform WORK_DZ) so the stages are directly comparable:

  col 1  ERA5              raw ERA5 (horizontally interpolated onto the model grid)
  col 2  + JAWARA blend    ERA5->JAWARA vertical blend (BLEND_M ramp, ~40-60 km)
  col 3  balanced          adjust_theta balance -> what is written to input_<N>.nc

  row 1  horizontal wind speed |v_h|
         col 1  colour = |v_h|                        (state, sequential)
         col 2  colour = |v_h|(blend) - |v_h|(ERA5)   (diverging)   -- the JAWARA modification
         col 3  colour = |v_h|(bal)   - |v_h|(blend)  (diverging)   -- ~0 (adjust_theta keeps winds)
         contours = |v_h| of that stage (WIND_CONTOUR_STEP m/s)

  row 2  potential temperature / isentropes
         col 1  colour = T                            (state, sequential)
         col 2,3 colour = difference to the stage on the left
                 DIFF_MODE_THETA = "percent"  -> 100*(theta_this-theta_left)/theta_left  (~ frac. T change)
                                 = "absolute" -> T_this - T_left  [K]
         contours = theta isentropes of that stage

Reuses the builder's own physics (imported, CONFIG overridden to the darwin_240718 case):
source_to_work / blend_weight / balance_none (hydrostatic theta of a T field) /
balance_adjust_theta. Reproduces one build_one() timestamp keeping the intermediate stages,
so it needs the builder's memory footprint (~35 GB on the z90 grid) -> run on a compute node.

    /home/b/b309199/venvs/post-venv/bin/python plot_background_3d_stages.py [TIDX] [X_KM]

Paper-bound -> mapy/data/figures/.
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm

sys.path.insert(0, "/work/bd0620/b309199/mapy/src")   # cmaps lives one level up
import cmaps
import build_background_3d as B

plt.style.use("/work/bd0620/b309199/mapy/src/latex_default.mplstyle")

# --- override the builder CONFIG to the darwin_240718 case (its sources exist) ------------
B.ERA5_FILE = "/work/bd0620/b309199/data/era5-data/sam3d_240718-ml-int.nc"
B.JAWARA_FILE = "/work/bd0620/b309199/mapy/data/jawara/jawara_sam_240718.nc"
B.GRID_FILE = "/work/bd0620/b309199/mapy/data/pmap-topos/sam_4000m_1100x900_fullorog.nc"
B.T0_UTC = "2024-07-18T18:00"
B.DT_BC_S = 3600.0
B.WORK_ZTOP = 90000.0          # darwin grid zcr top; fewer levels than the 128 km default
# CORIOLIS_MODE from argv[3] (default betaplane = production run; "fplane" for the contrast
# figure -- f-plane has a larger f_model-f_real -> a clearly visible balance correction).
B.CORIOLIS_MODE = sys.argv[3] if len(sys.argv) > 3 else "betaplane"
B.BALANCE_MODE = "adjust_theta"

# --- figure CONFIG -----------------------------------------------------------------------
TIDX = int(sys.argv[1]) if len(sys.argv) > 1 else 2        # -> T0_UTC + TIDX h (20:00 UTC)
X_SECTION_KM = float(sys.argv[2]) if len(sys.argv) > 2 else -394.0   # main Andes ridge
Z_TOP_KM = 90.0

WIND_CONTOUR_STEP = 20.0
WIND_STATE_CLIM = (0.0, 130.0)
T_STATE_CLIM = (160.0, 290.0)
PCT_CLIM = 2.0                 # +/- % (shared by all difference panels b, e, f)
WIND_PCT_DENOM_FLOOR = 1.0     # m/s floor on the |v_h| denominator (avoids div-by-~0 aloft)

OUTFILE = (f"/work/bd0620/b309199/mapy/data/figures/"
           f"background_3d_stages_{B.CORIOLIS_MODE}_t{TIDX:02d}_x{int(X_SECTION_KM)}.png")

STATE_WIND_CMAP = cm.batlow
STATE_T_CMAP = cm.lipari
DIFF_CMAP = cmaps.get_coolwarm_soft_cmap()   # calm teal->white->rose, white centre
YPP, XLBL = 0.93, 0.04
XPP = 1 - XLBL


def build_stages(n):
    """Reproduce build_one(n) keeping the three stages on the working grid.

    Each stage is sliced to the meridional section (nearest x index to X_SECTION_KM)
    immediately, so only sections (ny, nz) survive -- the full (nx, ny, nz) arrays are freed
    before the memory-heavy balance step. Returns a dict of section arrays + axes + terrain.
    """
    grid = B.GridTemplate(B.GRID_FILE)
    era, jaw, hi_e, hi_j, z_work = B.load_sources(grid)
    shape = (grid.nx, grid.ny)
    dz = B.WORK_DZ
    grid_dx = float(grid.x_m[1] - grid.x_m[0])
    tstamp = np.datetime64(B.T0_UTC) + np.timedelta64(int(n * B.DT_BC_S), "s")
    print(f"[stages] {tstamp}  x={X_SECTION_KM} km", flush=True)

    i0 = int(np.argmin(np.abs(grid.x_m - X_SECTION_KM * 1e3)))

    def sec(a):
        return np.asarray(a[i0, :, :], dtype=np.float64)

    psurf = hi_e(era["p"].sel(time=tstamp).isel(level=0).values)

    # -- temperature / theta stages -------------------------------------------------------
    w_t = B.blend_weight(z_work, B.BLEND_M)
    t_e = B.source_to_work(era, hi_e, "t", tstamp, z_work, shape)
    t_j = B.source_to_work(jaw, hi_j, "t", tstamp, z_work, shape)
    t_blend = (1 - w_t) * t_e + w_t * t_j
    del t_j
    T_e_sec = sec(t_e)                                      # ERA5 temperature (state panel)
    theta_e_sec = sec(B.balance_none(t_e, psurf, dz)[0])
    del t_e
    theta_bl_sec = sec(B.balance_none(t_blend, psurf, dz)[0])

    # -- wind stages (speed is rotation-invariant; adjust_theta keeps winds) --------------
    w_u = B.blend_weight(z_work, B.BLEND_M_WIND or B.BLEND_M)
    ue = B.source_to_work(era, hi_e, "u", tstamp, z_work, shape)
    ve = B.source_to_work(era, hi_e, "v", tstamp, z_work, shape)
    spd_e_sec = sec(np.hypot(ue, ve))
    uj = B.source_to_work(jaw, hi_j, "u", tstamp, z_work, shape)
    vj = B.source_to_work(jaw, hi_j, "v", tstamp, z_work, shape)
    u_geo = (1 - w_u) * ue + w_u * uj
    v_geo = (1 - w_u) * ve + w_u * vj
    del ue, ve, uj, vj
    spd_bl_sec = sec(np.hypot(u_geo, v_geo))

    # -- balanced stage (needs grid-rotated winds) ----------------------------------------
    u, v = grid.rotate_wind(u_geo, v_geo)
    del u_geo, v_geo
    f_real = (B.OMEGA2 * np.sin(np.deg2rad(grid.lat))).astype(np.float32)
    if B.CORIOLIS_MODE == "fplane":
        f_model = np.full_like(f_real, np.float32(B.F0))
    else:
        f_model = np.float32(B.F0) + np.float32(B.BETA0) * np.broadcast_to(
            grid.y_m[np.newaxis, :], f_real.shape).astype(np.float32)
    theta_bal_sec = sec(B.balance_adjust_theta(u, v, t_blend, psurf, f_model - f_real, grid_dx, dz)[0])
    del u, v, t_blend

    return {
        "y": grid.y_m / 1e3,
        "z": z_work / 1e3,
        "orog": grid.zcr[i0, :, 0] / 1e3,
        "xpos": grid.x_m[i0] / 1e3,
        "tstamp": str(tstamp),
        "spd": [spd_e_sec, spd_bl_sec],      # ERA5, blend (== kept input winds)
        "theta": [theta_e_sec, theta_bl_sec, theta_bal_sec],
        "T_state": T_e_sec,
    }


def add_labels(ax, letter, text):
    ax.text(XLBL, YPP, text, transform=ax.transAxes, ha="left", va="center",
            bbox={"boxstyle": "round", "lw": 0.67, "facecolor": "white", "edgecolor": "black"})
    ax.text(XPP, YPP, letter, transform=ax.transAxes, ha="right", va="center", weight="bold",
            bbox={"boxstyle": "circle", "lw": 0.67, "facecolor": "white", "edgecolor": "black"})


def theta_levels(theta_all):
    lo = max(250.0, np.nanmin([t.min() for t in theta_all]))
    hi = np.nanmax([np.percentile(t, 99.5) for t in theta_all])
    return np.unique(np.round(np.geomspace(lo, hi, 22)))


def main():
    S = build_stages(TIDX)
    y, z, orog = S["y"], S["z"], S["orog"]
    Y, Z = np.meshgrid(y, z, indexing="ij")
    # stage name folded into the corner label (no separate column headers)
    wind_labels = [r"ERA5,  $|\mathbf{v}_h|$", r"+ JAWARA,  $\Delta|\mathbf{v}_h|$",
                   r"balanced,  $|\mathbf{v}_h|$ kept"]
    theta_labels = [r"ERA5,  $T$", r"+ JAWARA,  $\Delta\theta$", r"balanced,  $\Delta\theta$"]
    letters = ["a", "b", "c", "d", "e", "f"]

    fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.4), sharex=True, sharey=True,
                            gridspec_kw={"wspace": 0.03, "hspace": 0.05})
    fig.subplots_adjust(left=0.055, right=0.995, top=0.885, bottom=0.20)
    blend_km = (B.BLEND_M[0] / 1e3, B.BLEND_M[1] / 1e3)

    # ---- row 1: wind speed --------------------------------------------------------------
    wlev = np.arange(0, 200 + WIND_CONTOUR_STEP, WIND_CONTOUR_STEP)
    for c in range(3):
        ax = axs[0, c]
        if c == 0:
            p_state_w = ax.pcolormesh(Y, Z, S["spd"][0], cmap=STATE_WIND_CMAP,
                                      vmin=WIND_STATE_CLIM[0], vmax=WIND_STATE_CLIM[1], rasterized=True)
            contour_fld = S["spd"][0]
        elif c == 1:
            denom = np.maximum(S["spd"][0], WIND_PCT_DENOM_FLOOR)
            pct = 100.0 * (S["spd"][1] - S["spd"][0]) / denom
            p_diff = ax.pcolormesh(Y, Z, pct, cmap=DIFF_CMAP,
                                   vmin=-PCT_CLIM, vmax=PCT_CLIM, rasterized=True)
            contour_fld = S["spd"][1]
        else:
            contour_fld = S["spd"][1]           # winds kept by adjust_theta -> contours only
        ax.contour(Y, Z, contour_fld, levels=wlev[wlev > 0], colors="k", linewidths=0.4)
        ax.plot(y, orog, color="k", lw=1.2)
        if c == 1:                              # mark the ERA5->JAWARA blend band
            for zb in blend_km:
                ax.axhline(zb, ls="--", color="k", lw=1.0)
        add_labels(ax, letters[c], wind_labels[c])

    # ---- row 2: theta / isentropes ------------------------------------------------------
    tlev = theta_levels(S["theta"])
    for c in range(3):
        ax = axs[1, c]
        th = S["theta"][c]
        if c == 0:
            p_state_t = ax.pcolormesh(Y, Z, S["T_state"], cmap=STATE_T_CMAP, rasterized=True,
                                      vmin=T_STATE_CLIM[0], vmax=T_STATE_CLIM[1])
        else:
            left = S["theta"][c - 1]
            pct = 100.0 * (th - left) / left
            p_diff = ax.pcolormesh(Y, Z, pct, cmap=DIFF_CMAP,
                                   vmin=-PCT_CLIM, vmax=PCT_CLIM, rasterized=True)
        ax.contour(Y, Z, th, levels=tlev, colors="k", linewidths=0.4)
        ax.plot(y, orog, color="k", lw=1.2)
        if c == 1:
            for zb in blend_km:
                ax.axhline(zb, ls="--", color="k", lw=1.0)
        add_labels(ax, letters[3 + c], theta_labels[c])

    for c in (0, 1, 2):
        axs[0, c].tick_params(labelbottom=False)
    for r in (0, 1):
        axs[r, 1].tick_params(labelleft=False)
        axs[r, 2].tick_params(labelleft=False)
    for ax in axs[1, :]:
        ax.set_xlabel("spanwise y / km")
    for ax in axs[:, 0]:
        ax.set_ylabel("altitude z / km")
    axs[0, 0].set_ylim(0, Z_TOP_KM)

    # ---- horizontal colorbars: states over/under col-1, shared % centred over b|c --------
    CB_H, GAP = 0.020, 0.014
    pa = axs[0, 0].get_position(); pb = axs[0, 1].get_position()
    pc = axs[0, 2].get_position(); pd = axs[1, 0].get_position()

    LF = 0.8                                                           # left-column cbar fraction
    cax_w = fig.add_axes([pa.x0 + (1 - LF) / 2 * pa.width, pa.y1 + GAP, LF * pa.width, CB_H])
    cb_w = fig.colorbar(p_state_w, cax=cax_w, orientation="horizontal")   # |v_h| : top, over a
    cb_w.set_label(r"$|\mathbf{v}_h|$ / m$\,$s$^{-1}$")
    cax_w.xaxis.set_ticks_position("top"); cax_w.xaxis.set_label_position("top")

    x_lo, x_hi = pb.x0, pc.x1                                           # % : top, centred b|c
    cw = 0.70 * (x_hi - x_lo); cx = 0.5 * (x_lo + x_hi)
    cax_p = fig.add_axes([cx - cw / 2, pa.y1 + GAP, cw, CB_H])
    cb_p = fig.colorbar(p_diff, cax=cax_p, orientation="horizontal", extend="both")
    cb_p.set_label(r"relative change / %")
    cax_p.xaxis.set_ticks_position("top"); cax_p.xaxis.set_label_position("top")

    cax_t = fig.add_axes([pd.x0 + (1 - LF) / 2 * pd.width, 0.10, LF * pd.width, CB_H])
    cb_t = fig.colorbar(p_state_t, cax=cax_t, orientation="horizontal")   # T : bottom, under d
    cb_t.set_label(r"$T$ / K")

    fig.savefig(OUTFILE, dpi=150, bbox_inches="tight")
    print(f"[out] {OUTFILE}", flush=True)


if __name__ == "__main__":
    main()
