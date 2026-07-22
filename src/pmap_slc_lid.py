import sys
import os
import glob
import time
from datetime import datetime
from types import SimpleNamespace
import shutil

import numpy as np
import pandas as pd
import xarray as xr
import scipy.ndimage
import scipy.signal

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, FuncFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle

# - for reloading libraries/modules - #
import importlib
import multiprocessing as mp

import warnings
warnings.filterwarnings('ignore')

from cmcrameri import cm
import cmaps, plt_helper, filter

# USAGE
# python3 slc_and_lid.py vortex darwin_240718_400m <notest>

"""Config"""
data_folder = "/scratch/b/b309199"
# data_folder = "/work/bd0620/b309199/patagonia"
# data_folder = "/work/bd0620/b309199/scratch"
pbar_interval = 5 # %
animation_folder = "../data/pmap-animations"

VERTICAL_CUTOFF = 20 # km (LAMBDA_CUT)
TEMPORAL_CUTOFF = 8*60 # min (TAU_CUT)
# panel-d (lidar) perturbation method: "data0" = T minus the simulation's t=0 ambient (data_0 of
# the base run); "bwf" = emulate a measurement, background from a vertical Butterworth low-pass
LIDAR_BG = "bwf"

if os.path.exists('latex_default.mplstyle'):
    plt.style.use('latex_default.mplstyle')

"""Colormaps"""
mycmap = {}
mynorm = {}
myclev = {}
myclevl = {}
mylabel = {}

xvar = "vorticity"
# mycmap[xvar] = plt.get_cmap('RdBu')
mycmap[xvar] = plt.get_cmap('PiYG')
# clev_vort, clev_l_vort = plt_helper.get_colormap_bins_and_labels(max_level=0.2)
myclev[xvar] = np.linspace(-80,80,100) * 10**(-3)
myclevl[xvar] = [np.min(myclev[xvar]), np.max(myclev[xvar])]
mynorm[xvar] = BoundaryNorm(boundaries=myclev[xvar], ncolors=mycmap[xvar].N, clip=True)
mylabel[xvar] = "Vorticity / s$^{-1}$"

xvar = "t"
# mycmap[xvar] = plt.get_cmap('seismic')
mycmap[xvar] = cmaps.get_wave_cmap()
# clev_t, clev_l_t = plt_helper.get_colormap_bins_and_labels(max_level=32)
# myclev[xvar] = np.linspace(-16,16,100)
# myclev[xvar] = np.linspace(-30,30,100)
myclev[xvar] = np.linspace(-12,12,100)
myclev[xvar] = np.linspace(-1,1,100)
myclevl[xvar] = [np.min(myclev[xvar]), np.max(myclev[xvar])]
mynorm[xvar] = BoundaryNorm(boundaries=myclev[xvar], ncolors=mycmap[xvar].N, clip=True)
mylabel[xvar] = r"T' / K"

xvar = "u"
mycmap[xvar] = cmaps.get_vik_white_cmap()
# mycmap[xvar] = cmaps.get_wave_cmap()
# mycmap[xvar] = plt.get_cmap('RdBu_r')
# myclev[xvar] = np.linspace(-40,40,100)
myclev[xvar] = np.linspace(-5,5,100)
myclevl[xvar] = [np.min(myclev[xvar]), np.max(myclev[xvar])]
mynorm[xvar] = BoundaryNorm(boundaries=myclev[xvar], ncolors=mycmap[xvar].N, clip=True)
mylabel[xvar] = r"u' / m$\,$s$^{-1}$"

xvar = "w"
mycmap[xvar] = cmaps.get_vik_white_cmap()
# mycmap[xvar] = plt.get_cmap('RdBu_r')
# mycmap[xvar] = cmaps.get_wave_cmap()
# myclev[xvar] = np.linspace(-16,16,100)
myclev[xvar] = np.linspace(-0.25,0.25,100)
# myclev[xvar] = np.linspace(-45,45,100)
myclevl[xvar] = [np.min(myclev[xvar]), np.max(myclev[xvar])]
mynorm[xvar] = BoundaryNorm(boundaries=myclev[xvar], ncolors=mycmap[xvar].N, clip=True)
mylabel[xvar] = r"w / m$\,$s$^{-1}$"

# cmap = cmaps.get_wave_cmap()
# # clev   = [-32,-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16,32]
# # clev_l = [-16,-4,-1,1,4,16]
# clev, clev_l = plt_helper.get_colormap_bins_and_labels(max_level=2)
# norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
# cbar_label = r"P' / Pa"

# cmap = plt.get_cmap('YlOrRd')
# # clev = [8,16,32,64,128,256,512,1024]
# clev = [32,64,128,256,512,1024,2048,4096]
# clev_l = clev
# norm = BoundaryNorm(boundaries=clev, ncolors=cmap.N, clip=True)
# cbar_label = r"E$_{pm}$ / J$\,$kg$^{-1}$"

# cmap = plt.get_cmap('PiYG_r')
# cmap = plt.get_cmap('RdBu_r')
# cmap = cmaps.get_wave_cmap()
# clev, clev_l = plt_helper.get_colormap_bins_and_labels(max_level=64)
# cbar_label = r"MF$_x$ / mPa"


"""Scenario configuration: everything that changes per region/preset lives here."""
LAMBDAX = 50 * 1000
LAMBDAZ = 15 * 1000
ZCUT_MF = 75000
WIND_LEVELS = np.arange(-120, 120, 10)
YZ_TPRIME_CUTOFF_M = 500e3
YZ_U_CONTOUR_STEP = 15.0
YZ_U_SMOOTH_M = (2000.0, 24000.0)
XPP, YPP = 0.88, 0.93
XLBL = 0.04
LW1, LW2, LW_SPONGE = 1.2, 0.8, 1.5
CSPONGE = "lightgrey"

CUBE_RECT_COLOR = "black"
CUBE_RECT_LW = 1.3
CUBE_RECT_DASHES = (0, (6, 4))
cube_rects = []
DIM_ORDER = "txyz"

CLEV_LARGE = {"w": (-16, 16), "t": (-16, 16), "u": (-40, 40)}

# region (= simulation-name prefix) -> zoom window (x0, x1, y0, y1) [km], optional marker / color + wind ranges
REGIONS = {
    "pata": {"zoom": (-400, -100, 100, 400), "clev": CLEV_LARGE, "wind_lims": (-103, 153)},
    "darwin": {"zoom": (-50, 150, -100, 100), "marker": (-8.8, -61.6), "clev": CLEV_LARGE, "wind_lims": (-103, 153),
               "yslice": 1, "zslice": 4, "xlid": 114.4},
}

# preset (= VAR argument).  limits: "auto" (full domain) | "zoom" (region window) | (x0,x1,y0,y1)
#   zlim: "auto" (0..domain top) | (z0, z1).  vars: physical field per (xz, xz2, lidar/xy2) panel.
#   layout: figure layout name.  optional: surf_factor, thlev.
PRESETS = {
    "over": {"limits": "auto", "zlim": "auto", "vars": ("w", "t", "t"), "layout": "full"},
    "vortex": {"limits": "zoom", "zlim": (45, 70), "vars": ("w", "vorticity", "t"), "layout": "full"},
    "amtm": {"limits": "auto", "zlim": "auto", "vars": ("u", "t", "t"), "layout": "full"},
    "alima_x": {"limits": "auto", "zlim": "auto", "vars": ("t", "t", "t"), "layout": "full"},
    "alima_y": {"limits": "auto", "zlim": "auto", "vars": ("t", "t", "t"), "layout": "full"},
    "surf": {"limits": (-100, 100, -100, 100), "zlim": (0, 9), "vars": ("u", "t", "t"),
             "surf_factor": 1, "thlev": "surf", "layout": "full"},
    "small": {"limits": "auto", "zlim": "auto", "vars": ("w", "vorticity", "t"), "layout": "row3"},
}


def resolve(region, preset, domain):
    """Combine region defaults + preset + the simulation domain into one settings object."""
    p = PRESETS.get(preset, PRESETS["over"])
    reg = REGIONS.get(region, {})
    lim = p["limits"]
    if lim == "auto":
        xlim, ylim = list(domain["xlim"]), list(domain["ylim"])
    elif lim == "zoom":
        z = reg.get("zoom", (*domain["xlim"], *domain["ylim"]))
        xlim, ylim = [z[0], z[1]], [z[2], z[3]]
    else:
        xlim, ylim = [lim[0], lim[1]], [lim[2], lim[3]]
    zlim = [0.0, domain["zlim"][1]] if p["zlim"] == "auto" else list(p["zlim"])
    if p.get("thlev") == "surf":
        thlev = np.exp(4 + 0.03 * np.arange(40, 100, 0.2))
    else:
        thlev = np.exp(4 + 0.03 * np.arange(1, 350, 5))
    clev, norm, clevl = dict(myclev), dict(mynorm), dict(myclevl)
    for v, (lo, hi) in reg.get("clev", {}).items():
        lv = np.linspace(lo, hi, 100)
        clev[v], clevl[v] = lv, [lo, hi]
        norm[v] = BoundaryNorm(boundaries=lv, ncolors=mycmap[v].N, clip=True)
    return SimpleNamespace(
        xlim=xlim, ylim=ylim, zlim=zlim,
        var1=p["vars"][0], var2=p["vars"][1], var3=p["vars"][2],
        surf_factor=p.get("surf_factor", 5), thlev=thlev,
        layout=p["layout"], marker=reg.get("marker"),
        wind_lims=list(reg.get("wind_lims", (-10, 20))),
        clev=clev, norm=norm, clevl=clevl,
    )


def get_pmap_bv(cfg):
    stb00 = cfg.get("ambient_fields", {}).get("stb00", np.nan)
    return np.sqrt(stb00) if stb00 is not None and stb00 >= 0 else np.nan


def get_cube_footprints(fpath):
    """Horizontal (x, y) extent [km] of every output cube_<N>.nc in the run dir.

    Returns a list of (x0, x1, y0, y1, label) tuples (label = the cube index), read
    once from each cube's x/y coordinates (stored in metres) so the horizontal slice
    panels can mark where the 3D sub-volumes were dumped.
    """
    rects = []
    for f in sorted(glob.glob(os.path.join(fpath, "cube_*.nc"))):
        try:
            with xr.open_dataset(f, decode_times=False) as ds:
                x = np.asarray(ds["x"].values, dtype=float) / 1000.0
                y = np.asarray(ds["y"].values, dtype=float) / 1000.0
        except (OSError, KeyError, ValueError):
            continue
        label = os.path.splitext(os.path.basename(f))[0].split("_")[-1]
        rects.append((float(np.nanmin(x)), float(np.nanmax(x)),
                      float(np.nanmin(y)), float(np.nanmax(y)), label))
    return rects


def nearest_zslice_index(fpath, target_km):
    """(index, altitude_km) of the output z-slice nearest to target_km.

    Matches against the slice's nominal output altitude — the `z` coordinate of
    slices_z.nc, i.e. the config `output_slices.z` values, which is what the panel
    label reports (via `zpos`). This is picked over the `zcr` field because some runs
    (e.g. darwin_240718_400m base) wrote a bogus `zcr`; the `z` coordinate stays the
    authoritative requested altitude. Runs with different `output_slices.z` grids map
    the same target km to different indices.
    """
    with xr.open_dataset(os.path.join(fpath, "slices_z.nc"), decode_times=False) as ds:
        z_km = np.asarray(ds["z"].values, dtype=float) / 1000.0
    k = int(np.argmin(np.abs(z_km - float(target_km))))
    return k, float(z_km[k])


def overlay_cube_footprints(ax, S):
    """Draw a dashed rectangle for each output-cube footprint on a horizontal panel."""
    for (x0, x1, y0, y1, _label) in getattr(S, "cube_rects", []):
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0, fill=False,
            edgecolor=CUBE_RECT_COLOR, linestyle=CUBE_RECT_DASHES,
            linewidth=CUBE_RECT_LW, zorder=5,
        ))


def build_full():
    """Figure stuff"""
    gskw  = {'hspace':0.05, 'wspace':0.03, 'height_ratios': [7,7,0.4,1], 'width_ratios': [3,3,3]} #  , 'width_ratios': [5,5]}
    # gskw2 = {'hspace':0.14, 'wspace':0.06, 'height_ratios': [7,7,0.1,2], 'width_ratios': [1,1,1,3,3]}
    fig, axes = plt.subplots(4,3,figsize=(15,10), gridspec_kw=gskw)
    for ax in axes[-1,0:3]:
        ax.set_axis_off()
    for ax in axes[-2,0:3]:
        ax.set_axis_off()

    # --- Replace top left axis --- #
    gs_topleft = axes[0,0].get_gridspec()
    axes[0,0].remove()
    
    # gs_top2 = fig.add_gridspec(4,5, hspace=0.05, wspace=0.03, height_ratios=[7,7,0.4,1], width_ratios=[0.94,0.94,0.94,3,3])
    # ax_wind = fig.add_subplot(gs_top2[0])
    # ax_stab = fig.add_subplot(gs_top2[1])
    # ax_t    = fig.add_subplot(gs_top2[2])
    gs_top2 = fig.add_gridspec(4,4, hspace=0.05, wspace=0.03, height_ratios=[7,7,0.4,1], width_ratios=[1.46,1.46,3,3])
    ax_wind = fig.add_subplot(gs_top2[0])
    # ax_stab = fig.add_subplot(gs_top2[1])
    ax_t    = fig.add_subplot(gs_top2[1])

    ax0 = axes[0,1] # xz
    ax1 = axes[1,0] # xy
    ax2 = axes[1,1] # xy2
    ax3 = axes[1,2] # xy2
    axlid = axes[0,2] # lid

    ax0.grid(False)
    axlid.grid(False)
    return fig, axes, ax_wind, ax_t, ax0, ax1, ax2, ax3, axlid


def build_row3():
    gskw = {"hspace": 0.05, "wspace": 0.07, "height_ratios": [7, 0.4], "width_ratios": [1.5, 3, 3]}
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), gridspec_kw=gskw)
    for ax in axes[-1, :]:
        ax.set_axis_off()
    ax_wind = axes[0, 0]
    ax0 = axes[0, 1]
    axlid = axes[0, 2]
    ax0.grid(False)
    axlid.grid(False)
    return fig, axes, ax_wind, ax0, axlid


def slc_and_lid(t, var, fpath, slices, dslid, image_folder, pbar, sema):
    """Visualize PMAP slices and virtual lidar time-height diagrams."""
    
    cfg, dsxz, dsyz, ds_xyslices = plt_helper.preprocess_pmap_tstep(fpath, t, slices=slices, dimorder=DIM_ORDER)
    xy_ref = ds_xyslices[0]

    """Limits and timestamp and thlev"""
    zsponge = [cfg["zmax"] / 1000, dsxz.zcr.max().values]

    domain = {
        "xlim": (float(xy_ref.xcr.min().values), float(xy_ref.xcr.max().values)),
        "ylim": (float(xy_ref.ycr.min().values), float(xy_ref.ycr.max().values)),
        "zlim": (0.0, float(dsxz.zcr.max().values)),
    }
    S = resolve(region, var, domain)
    xlim, ylim, zlim = S.xlim, S.ylim, S.zlim
    var1, var2, var3 = S.var1, S.var2, S.var3
    surf_factor, thlev = S.surf_factor, S.thlev
    zcut_mf, wind_levels = ZCUT_MF, WIND_LEVELS

    if itopo == 1:
        amp = int(np.max(ds_xyslices[0].zcrtopo))
        topo_levels = np.linspace(20, surf_factor * 0.5 * amp, 2)
    elif amp0 < 0:
        topo_levels = np.linspace(surf_factor * amp0, -surf_factor * amp0, 12)
    else:
        topo_levels = np.linspace(-2 * surf_factor * amp0, 2 * surf_factor * amp0, 24)
    S.topo_levels = topo_levels
    S.region = region
    S.xlim_lid = xlim_lid
    S.cube_rects = cube_rects

    lambdax, lambdaz = LAMBDAX, LAMBDAZ
    lw1, lw2, lw_sponge, csponge = LW1, LW2, LW_SPONGE, CSPONGE
    xpp, ypp = XPP, YPP
    alpha_box = 0.9
    alpha_sponge = 0.7
    ipp = 0
    numb_str = ['a','b','c','d','e','f','g','h','i','j']

    if S.layout == "row3":
        fig, axes, ax_wind, ax0, axlid = build_row3()
        ax_t = ax1 = ax2 = ax3 = None
    else:
        fig, axes, ax_wind, ax_t, ax0, ax1, ax2, ax3, axlid = build_full()

    if 'i' in dslid.attrs and 'j' in dslid.attrs:
        iprof = int(dslid.i)
        # jprof = int(dslid.j)
    else:
        iprof = int(np.shape(dsxz['w'].values)[-1]/2)
        if var == 'alima_y':
            dslid.attrs['xpos'] = float(np.asarray(dsyz.xpos).item())
            dslid.attrs['ypos'] = alima_locs[t] / 1000
        else:
            dslid.attrs['xpos'] = alima_locs[t] / 1000
            dslid.attrs['ypos'] = float(np.asarray(dsxz.ypos).item())
    zcr_prof = dslid.zcr.isel(time=0) if 'time' in dslid.zcr.dims else dslid.zcr
    dslid.attrs['kprof'] = int(np.argmin(np.abs(np.asarray(zcr_prof.values) - ds_xyslices[0].zpos)))
        
    """Wind axis"""
    cu = 'darkorchid'
    cv = 'lightseagreen'
    lw_wind = 2
    lws_wind = 0.4
    wind_lims = S.wind_lims
    x0=0
    # print(np.max(dsxz['u'][:,iprof+15]-dsxz['ue'][:,iprof+15]))
    ax_wind.plot(dsxz['ue'][:,iprof], dsxz.zcr[:,iprof], lw=lw_wind, ls='--', color='black', label=r'u$_{env}$')
    ax_wind.plot(np.mean(dsxz['u'][:,:],axis=1), dsxz.zcr[:,x0], lw=lw_wind, ls='dotted', color='red', label=r'u$_{mean}$')
    # ax_wind.plot(dslidars[0]['u'][3*t,:], dslidars[0]['zcr'][3*t,:], lw=1, ls='dotted', color=cu, label='u')
    ax_wind.plot(dsxz['u'][:,iprof], dsxz.zcr[:,iprof], lw=lws_wind, ls='-', color=cu, label=r'u$_{mtn}$')
    ax_wind.plot(dsxz['ve'][:,iprof], dsxz.zcr[:,iprof], lw=lw_wind, ls='--', color=cv, label=r'v$_{env}$')
    ax_wind.plot(dsxz['v'][:,iprof], dsxz.zcr[:,iprof], lw=lws_wind, ls='-', color=cv, label=r'v$_{mtn}$')
    ax_wind.set_xlabel('(u,v) / m$\,$s$^{-1}$')
    ax_wind.set_ylabel('altitude z / km')
    ax_wind.set_ylim(zlim)
    ax_wind.set_xlim(wind_lims)
    # ax_wind.yaxis.set_major_locator(MultipleLocator(10))
    
    ax_wind.xaxis.set_label_position('top')
    ax_wind.vlines(x=[0], ymin=0,ymax=zlim[1], colors="grey", lw=0.75, ls='-.')
    ##ax_wind.xaxis.set_major_locator(MultipleLocator(50))
    ax_wind.tick_params(axis='x', which='both', top=True, bottom=True, labelbottom=False, labeltop=True, labelleft=True)
    ax_wind.xaxis.set_minor_locator(AutoMinorLocator())
    ax_wind.yaxis.set_minor_locator(AutoMinorLocator())
    ax_wind.legend(loc="lower left", fontsize=8, markerscale=0.5, handlelength=1, handletextpad=0.1)
    ax_wind.grid()
    ax_wind.text(xpp, ypp, numb_str[ipp], transform=ax_wind.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ipp += 1
    
    
    if S.layout == "full":
        """Temperature and temperature gradient (mark altitudes with gradient below threshold)""" 
        zcr = dsxz['zcr'][:,iprof].values
        if var == "mf":
            zcr = dsxz['zcr'][:,0].values
            uw  = dsxz['w'].values * (dsxz['u'].values-dsxz['ue'].values)
            mfx = dsxz['rh0'].values * uw
            mfx = np.mean(mfx, axis=1)
            mfx = filter.gaussian_filter_fft_1D(mfx, lambdaz, cfg["dz"])
            mfx = mfx * 10**6
            ax_t.plot(mfx, zcr, lw=1.5, ls='-', color="royalblue")
            # ax_t.plot(tenv, zcr, lw=1.5, ls='--', color="coral")
            ax_t.set_xlabel('MF$_x$ / Pa $\cdot 10^{-6}$')
            ax_t.set_xlim([-10**6,10**3])
            ax_t.set_xscale("symlog")
            ax_t.set_xticks([-10**4,-10**2, 10**2])
        elif var == "ep":
            zcr = dsxz['zcr'][:,0].values
            tloc, tenv = plt_helper.get_eulag_t_and_tenv(dsxz['thprime'].values, dsxz['the'].values, dsxz['pprime'].values, dsxz['ppe'].values, cfg["constants"]["cap"], cfg["constants"]["p0"])
            epm = 1/2*(cfg["constants"]["gravity0"]/get_pmap_bv(cfg))**2 * ((tloc-tenv)/tenv)**2
            # epm = plt_helper.gaussian_filter_fft(epm, lambdaz, lambdax, ds.dz00, ds.dx00)
            epm = np.mean(epm, axis=1)
            epm = filter.gaussian_filter_fft_1D(epm, lambdaz, cfg["dz"])
            ax_t.plot(epm, zcr, lw=1.5, ls='-', color="red")
            # ax_t.plot(tenv, zcr, lw=1.5, ls='--', color="coral")
            ax_t.set_xlabel(r"E$_{pm}$ / J$\,$kg$^{-1}$")
            ax_t.set_xlim([0,2400])
        else:
            zcr = dsxz['zcr'][:,iprof].values
            tloc, tenv = plt_helper.get_eulag_t_and_tenv(dsxz['thprime'][:,iprof].values, dsxz['the'][:,iprof].values, dsxz['pprime'][:,iprof].values, dsxz['ppe'][:,iprof].values, cfg["constants"]["cap"], cfg["constants"]["p0"])
            ax_t.plot(tloc, zcr, lw=lw2, ls='-', color="coral")
            ax_t.plot(tenv, zcr, lw=1.5, ls='--', color="coral")
            ax_t.set_xlabel('T / K')
            ax_t.set_xlim([155,295])

            if var != "surf":
                xlo, xhi = 155, 295
                ylo, yhi = zlim
                x = np.linspace(xlo, xhi, 70)
                y_top = ylo + 0.85 * (yhi - ylo)
                y0 = -1 / 9.8 * (x - xlo) + y_top
                ax_t.plot(x, y0, lw=1.5, ls='--', color='k')
                ax_t.text(xlo + 0.03 * (xhi - xlo), y_top, "-9.8 K/km", ha='left', va='bottom')

        ## tgrad = np.gradient(tloc,zcr)
        ## tprime = tloc-tte

        # ---- TEMPERATURE ------------------ # 
        ax_t.xaxis.set_label_position('top')
        # ax_grad.vlines(x=[0], ymin=0,ymax=zlim[1], colors="grey", lw=0.75, ls='-.')
        ##ax_wind.xaxis.set_major_locator(MultipleLocator(50))
        ax_t.tick_params(which='both', top=True, bottom=True, labelbottom=False, labeltop=True, labelleft=False, labelright=False)
        ax_t.xaxis.set_minor_locator(AutoMinorLocator())
        ax_t.yaxis.set_minor_locator(AutoMinorLocator())
        ax_t.set_ylim(zlim)
        # ax_t.yaxis.set_major_locator(MultipleLocator(10))
        ax_t.grid()
        ax_t.text(xpp, ypp, numb_str[ipp], transform=ax_t.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
        ipp += 1
    xpp = 0.96
    
    ############## SLICES ##############
    """Plot xz-slice"""
    ax0, contf_1 = plt_xzslc(ax0, dsxz, cfg, S, var=var1)
    ax0.axvline(x=dslid.xpos, color='black', lw=lw1, ls='--')
    if var == 'alima_x' or var == 'alima_y':
        ax0.text(dslid.xpos, zlim[0] + 7, '✈', fontsize=30, fontname='DejaVu Sans', ha='center', va='center')
    ax0.axhline(y=ds_xyslices[0].zpos, color='black', lw=lw1, ls='--')
    # ax0.axhline(y=ds_xyslices[1].zpos, color='black', lw=lw1, ls='--')
    ax0.text(xpp, ypp, numb_str[ipp], transform=ax0.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ipp += 1

    """Plot virtual lidars"""
    axlid, contf_3 = plot_vlidar(axlid, dslid, cfg, S, var=var3, t=t, fpath=fpath)
    axlid.axhline(y=ds_xyslices[0].zpos, color='black', lw=lw1, ls='--')

    axlid.text(xpp, ypp, numb_str[ipp], transform=axlid.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ipp += 1
        
    if S.layout == "full":
        """Plot SLICES"""
        ax1, contf_1 = plt_xyslc(ax1, ds_xyslices[0], cfg, S, var=var1)
        ax1.axhline(y=dslid.ypos, color='black', lw=lw1, ls='--')
        ax1.axvline(x=dslid.xpos, color='black', lw=lw1, ls='--')
        # ax1.text(dslid.xpos, dslid.ypos, "d", weight='bold', fontsize=8, bbox={"boxstyle" : "circle", "lw":0.4, "facecolor":"white", "edgecolor":"black"})
        ax1.set_ylabel('spanwise y / km')

        ## - AXES 2 - ##
        if var == 'amtm':        
            ax2, contf_2 = plt_xyslc(ax2, ds_xyslices[3], cfg, S, var=var2)
            ax2.axvline(x=dslid.xpos, color='black', lw=lw1, ls='--')
            ax2.axhline(y=dslid.ypos, color='black', lw=lw1, ls='--')
            # amtm_domain = 200 
            # ax2.set_xlim(dslid.xpos-amtm_domain/2, dslid.xpos+amtm_domain/2)
            # ax2.set_ylim(dslid.ypos-amtm_domain/2, dslid.ypos+amtm_domain/2)
        elif var == 'alima_x' or var == 'alima_y':
            ax2, contf_2 = plt_yzslc(ax2, dsyz, cfg, S, var=var2)
            ax2.axvline(x=dslid.ypos, color='black', lw=lw1, ls='--')
            ax2.axhline(y=ds_xyslices[0].zpos, color='black', lw=lw1, ls='--')
            ax2.text(dslid.ypos, zlim[0] + 7, '✈', fontsize=30, fontname='DejaVu Sans', ha='center', va='center')
        else:
            ax2, contf_2 = plt_yzslc(ax2, dsyz, cfg, S, var=var2)
            ax2.axvline(x=dslid.ypos, color='black', lw=lw1, ls='--')
            ax2.axhline(y=ds_xyslices[0].zpos, color='black', lw=lw1, ls='--')
        ax2.xaxis.set_label_position('bottom')
        ax2.tick_params(labelleft=False, labeltop=False, labelbottom=True)

        ## - AXES 3 (bottom left) - ##
        # ax3, _ = plt_yzslc(ax3, dsyz, ds, var='w')
        # ax3.axvline(x=dslid.ypos, color='black', lw=lw1, ls='--')
        # ax3.axhline(y=ds_xyslices[0].zpos, color='black', lw=lw1, ls='--')
        # ax3.set_ylabel('altitude z / km')

        if var == 'amtm':
            ax3, _ = plt_amtm(ax3, ds_xyslices[1:6], cfg, dslid, S)
        elif var == 'alima_x'  or var == 'alima_y':
            ax3, _ = plt_xyslc(ax3, ds_xyslices[1], cfg, S, var=var3)
        else:
            ax3, _ = plt_xyslc(ax3, ds_xyslices[0], cfg, S, var=var3)
        ax3.axvline(x=dslid.xpos, color='black', lw=lw1, ls='--')
        ax3.axhline(y=dslid.ypos, color='black', lw=lw1, ls='--')
        ax3.set_ylabel('spanwise y / km')

        ax3.tick_params(labelleft=False,labelright=True, labeltop=False, labelbottom=True)
        ax3.yaxis.set_label_position('right')
        ax3.xaxis.set_label_position('bottom')
        ############## SLICES ##############

        ax1.text(xpp, ypp, numb_str[ipp], transform=ax1.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
        ipp += 1
        ax2.text(xpp, ypp, numb_str[ipp], transform=ax2.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
        ipp += 1
        ax3.text(xpp, ypp, numb_str[ipp], transform=ax3.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
        ipp += 1

        # - Sponge layer - #
        # if zsponge[0] > 0:
        #     ax0.fill_between(xlim, [zsponge[1],zsponge[1]], [zsponge[0],zsponge[0]], facecolor=csponge, alpha=alpha_sponge)
        #     axlid.fill_between(xlim_lid, [zsponge[1],zsponge[1]], [zsponge[0],zsponge[0]], facecolor=csponge, alpha=alpha_sponge)
        # ax2.axvspan(ds.xcr[0,0], ds.xcr[0,0]+ds.dxabL/1000, alpha=alpha_sponge, color=csponge)
        # ax2.axvspan(ds.xcr[0,-1]-ds.dxabR/1000, ds.xcr[0,-1], alpha=alpha_sponge, color=csponge)
        # xsponge = [ds.xcr[0,0] + ds.dxabL/1000, ds.xcr[0,-1] - ds.dxabR/1000]
        # ax2.fill_between(xsponge, [ds.ycr[0,0]+ds.dyab/1000,ds.ycr[0,0]+ds.dyab/1000], [ds.ycr[0,0],ds.ycr[0,0]], facecolor=csponge, alpha=alpha_sponge)
        # ax2.fill_between(xsponge, [ds.ycr[-1,0],ds.ycr[-1,0]], [ds.ycr[-1,0]-ds.dyab/1000,ds.ycr[-1,0]-ds.dyab/1000], facecolor=csponge, alpha=alpha_sponge)

    """Colorbar"""
    if S.layout == "full":
        fig.colorbar(contf_1, ax=axes[-1,0], location='bottom', shrink=0.8, fraction=1, ticks=S.clevl[var1], label=mylabel[var1], pad=0, extend='both', aspect=30)
        if var2 == "vorticity":
            fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((-3, -3))
        else:
            fmt = matplotlib.ticker.ScalarFormatter()
        fig.colorbar(contf_2, ax=axes[-1,1], location='bottom', shrink=0.8, fraction=1, ticks=S.clevl[var2], label=mylabel[var2], format=fmt, pad=0, extend='both', aspect=30)
        fig.colorbar(contf_3, ax=axes[-1,2], location='bottom', shrink=0.8, fraction=1, ticks=S.clevl[var3], label=mylabel[var3], pad=0, extend='both', aspect=30)
    else:
        fig.colorbar(contf_1, ax=axes[-1,1], location='bottom', shrink=0.8, fraction=1, ticks=S.clevl[var1], label=mylabel[var1], pad=0, extend='both', aspect=30)
        fig.colorbar(contf_3, ax=axes[-1,2], location='bottom', shrink=0.8, fraction=1, ticks=S.clevl[var3], label=mylabel[var3], pad=0, extend='both', aspect=30)

    """Save figure"""
    os.makedirs(image_folder,exist_ok=True)
    if t<10:
        buffer = "00"
    elif t<100:
        buffer = "0"
    else:
        buffer = ""
    fig_title = "slice_" + buffer + str(t) + ".png"
    dpi = 200
    if test:
        dpi = 300
    fig.savefig(os.path.join(image_folder,fig_title), facecolor='w', edgecolor='w',
                    format='png', dpi=dpi, bbox_inches='tight')
    """Finish"""
    plt_helper.show_progress(pbar['progress_counter'], pbar['lock'], pbar["stime"], pbar['ntasks'])
    sema.release()


def plt_xzslc(ax, dsxz, cfg, S, var='w'):
    jslc = int(dsxz.j)
    xlim, zlim, thlev, surf_factor = S.xlim, S.zlim, S.thlev, S.surf_factor
    xpp, ypp, lw2, zcut_mf = XPP, YPP, LW2, ZCUT_MF
    lambdax, lambdaz = LAMBDAX, LAMBDAZ
    clev = S.clev[var]

    if var == "vorticity":
        cvar = plt_helper.vorticity_xz(dsxz['u'].values, dsxz['w'].values, cfg["dx"], cfg["dz"])

    elif var == "mf":
        uw   = dsxz['w'].values * (dsxz['u'].values-dsxz['ue'].values)
        mfx  = dsxz['rh0'].values * uw
        mfx = filter.gaussian_fft_smoothing(mfx, lambdaz, lambdax, cfg["dz"], cfg["dx"])
        izcut = int(zcut_mf / cfg["dz"])
        mfx[izcut:,:] = mfx[izcut:,:] * 1000
        cvar = mfx*1000
            
    elif var == "ep":
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dsxz['thprime'].values, dsxz['the'].values, dsxz['pprime'].values, dsxz['ppe'].values, cfg["constants"]["cap"], cfg["constants"]["p0"])
        epm = 1/2*(cfg["constants"]["gravity0"]/get_pmap_bv(cfg))**2 * ((tloc-tenv)/tenv)**2
        epm = filter.gaussian_fft_smoothing(epm, lambdaz, lambdax, cfg["dz"], cfg["dx"])
        epm = np.where(epm>clev[0],epm,np.nan)
        cvar = epm
    elif var == "u":
        cvar = dsxz["u"].values - dsxz["ue"].values
    elif var == "t":
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dsxz['thprime'].values, dsxz['the'].values, dsxz['pprime'].values, dsxz['ppe'].values, cfg["constants"]["cap"], cfg["constants"]["p0"])
        cvar = tloc - tenv
    else:
        cvar = dsxz[var].values

    cmap = mycmap[var]
    norm = S.norm[var]
    clev = S.clev[var]
    contf = ax.pcolormesh(dsxz.xcr, dsxz.zcr, cvar,
                            cmap=cmap, norm=norm, shading='nearest', rasterized=True)
    ax.contour(dsxz.xcr, dsxz.zcr, dsxz['the']+dsxz['thprime'], 
                            colors='k', alpha=0.7, levels=thlev, lw=lw2)
        
    ax.plot(dsxz.xcr[0, :], surf_factor*dsxz.zcr[0,:], lw=1.5, color='black')
    #ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(labelbottom=False,labeltop=False)
    ax.set_xlabel('streamwise x / km') # change to longitudes, latitude 10$^3$
    ax.xaxis.set_label_position('top') 
    ax.grid()

    ##ax.set_ylabel('altitude z-z$_{trp}$ / km')
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.tick_params(which='both', top=True, right=True, bottom=False, labelbottom=False, labeltop=True, labelleft=False, labelright=False)
    ax.text(XLBL, ypp, f"y: {float(np.asarray(dsxz.ypos).item()):.1f}km", transform=ax.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    return ax, contf


def plt_yzslc(ax, dsyz, cfg, S, var="w"):
    """Plot xy-slice"""
    ylim, zlim, surf_factor = S.ylim, S.zlim, S.surf_factor
    xpp, ypp, lambdax = XPP, YPP, LAMBDAX
    clev = S.clev[var]

    if var == "mf":
        uw   = dsyz['w'].values * (dsyz['u'].values-dsyz['ue'].values)
        vw   = dsyz['w'].values * (dsyz['v'].values-dsyz['ve'].values)
        mfx  = dsyz['rh0'].values * uw
        mfy  = dsyz['rh0'].values * vw
        mfx = filter.gaussian_fft_smoothing(mfx,lambdax, lambdax, cfg["dx"], cfg["dx"])
        mfy = filter.gaussian_fft_smoothing(mfy,lambdax, lambdax, cfg["dx"], cfg["dx"])
        cvar = mfx*1000
    elif var == "ep":
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dsyz['thprime'].values, dsyz['the'].values, dsyz['pprime'].values, dsyz['ppe'].values, cfg["constants"]["cap"], cfg["constants"]["p0"])
        epm = 1/2*(cfg["constants"]["gravity0"]/get_pmap_bv(cfg))**2 * ((tloc-tenv)/tenv)**2
        epm = filter.gaussian_fft_smoothing(epm,lambdax, lambdax, cfg["dx"], cfg["dx"])
        epm = np.where(epm>clev[0],epm,np.nan)
        cvar = epm
    elif var == "u":
        cvar = dsyz["u"] - dsyz["ue"]
    elif var == "t":
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dsyz['thprime'].values, dsyz['the'].values, dsyz['pprime'].values, dsyz['ppe'].values, cfg["constants"]["cap"], cfg["constants"]["p0"])
        bb, ba = scipy.signal.butter(5, 2.0 * cfg["dx"] / YZ_TPRIME_CUTOFF_M)
        cvar = tloc - scipy.signal.filtfilt(bb, ba, tloc, axis=1)
    else:
        cvar = dsyz[var].values
    islc = int(dsyz.i)
    cmap = mycmap[var]
    norm = S.norm[var]
    clev = S.clev[var]
    contf = ax.pcolormesh(dsyz.ycr, dsyz.zcr, cvar, cmap=cmap, norm=norm, shading='nearest', rasterized=True)
    if var == "t":
        uu = scipy.ndimage.gaussian_filter(
            dsyz['u'].values, (YZ_U_SMOOTH_M[0] / cfg["dz"], YZ_U_SMOOTH_M[1] / cfg["dx"]))
        ulev = np.arange(-165.0, 165.1, YZ_U_CONTOUR_STEP)
        ulev = ulev[np.abs(ulev) > 0.1]
        ax.contour(dsyz.ycr, dsyz.zcr, uu, levels=ulev, colors='black', linewidths=0.35,
                   negative_linestyles='dashed')
        ax.contour(dsyz.ycr, dsyz.zcr, uu, levels=[0.0], colors='black', linewidths=1.7)

    ### - Topography - ###
    ax.plot(dsyz.ycr[0, :], surf_factor*dsyz.zcr[0,:], lw=1.5, color='black')

    # ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('spanwise y / km')

    ax.set_xlim(ylim)
    ax.set_ylim(zlim)
    ax.grid()
    lbl = f"x: {float(np.asarray(dsyz.xpos).item()):.1f}km"
    if var == "t":
        lbl += rf", $T'$: $\lambda_h<{YZ_TPRIME_CUTOFF_M/1e3:.0f}$ km"
    ax.text(XLBL, ypp, lbl, transform=ax.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})

    return ax, contf


def plt_amtm(ax, dsxy_list, cfg, dslid, S):
    """Plot AMTM measurement (variable is t)"""
    surf_factor, topo_levels = S.surf_factor, S.topo_levels
    xpp, ypp = XPP, YPP

    weights = [0.0625,0.25,0.375,0.25,0.0625] # Pascal's triangle
    # weights = [1, 8, 28, 56, 70, 56, 28, 8, 1] / 256
    # dsxy = sum(w * ds for w, ds in zip(weights, dsxy_list))
    # print(dsxy)

    var = "t"

    tlocs = []
    tenvs = []
    for dsxy in dsxy_list:
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dsxy['thprime'].values, dsxy['the'].values, dsxy['pprime'].values, dsxy['ppe'].values, cfg["constants"]["cap"], cfg["constants"]["p0"])
        tlocs.append(tloc)
        tenvs.append(tenv)
    tloc = sum(w * layer for w, layer in zip(weights, tlocs))
    tenv = sum(w * layer for w, layer in zip(weights, tenvs))
    cvar = tloc - tenv
    dsxy = dsxy_list[0]

    cmap = mycmap[var]
    norm = S.norm[var]
    clev = S.clev[var]
    contf = ax.pcolormesh(dsxy.xcr, dsxy.ycr, cvar, cmap=cmap, norm=norm, shading='nearest', rasterized=True)

    ### - Topography - ###
    ax.contour(dsxy.xcr, dsxy.ycr, surf_factor*dsxy.zcrtopo, colors='k', levels=topo_levels, linewidths=0.3)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('streamwise x / km')

    amtm_domain = 200
    ax.set_xlim(dslid.xpos-amtm_domain/2, dslid.xpos+amtm_domain/2)
    ax.set_ylim(dslid.ypos-amtm_domain/2, dslid.ypos+amtm_domain/2)
    ax.grid()

    overlay_cube_footprints(ax, S)

    # - Lidar and Slice positions - #
    ax.text(dslid.xpos, dslid.ypos, "d", weight='bold', fontsize=8, bbox={"boxstyle" : "circle", "lw":0.4, "facecolor":"white", "edgecolor":"black"})
    ax.text(XLBL, ypp, f"AMTM (80-88km)", transform=ax.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})

    return ax, contf


def plt_xyslc(ax, dsxy, cfg, S, var="w"):
    """Plot xy-slice"""
    xlim, ylim, surf_factor, topo_levels, region = S.xlim, S.ylim, S.surf_factor, S.topo_levels, S.region
    xpp, ypp, zcut_mf, lambdax = XPP, YPP, ZCUT_MF, LAMBDAX
    clev = S.clev[var]

    if var == "vorticity":
        cvar = 5*plt_helper.vorticity_xz(dsxy['v'].values, dsxy['u'].values, cfg["dy"], cfg["dx"]) # dv/dx - du/dy

    elif var == "mf":
        uw   = dsxy['w'].values * (dsxy['u'].values-dsxy['ue'].values)
        vw   = dsxy['w'].values * (dsxy['v'].values-dsxy['ve'].values)
        mfx  = dsxy['rh0'].values * uw
        mfy  = dsxy['rh0'].values * vw
        mfx = filter.gaussian_fft_smoothing(mfx,lambdax, lambdax, cfg["dx"], cfg["dx"])
        mfy = filter.gaussian_fft_smoothing(mfy,lambdax, lambdax, cfg["dx"], cfg["dx"])
        if dsxy.zpos > zcut_mf / 1000:
            mfx = mfx*1000
        cvar = mfx*1000
        # efx = (dsxy['u']-dsxy['ue']) * ds['pprime']
        # efy = (dsxy['v']-dsxy['ve']) * ds['pprime']
        # efx = subroutines.fft_gaussian_xy(efx,nx_avg)
        # efy = subroutines.fft_gaussian_xy(efy,nx_avg)
        # efmax = np.max(np.sqrt(efx**2+efy**2))
        # print(efmax)
        # efx = efx/efmax
        # efy = efy/efmax
        
        # efx = np.where(np.abs(efy)>0.165,efx,np.nan) # 0.07
        # efy = np.where(np.abs(efy)>0.165,efy,np.nan)
    elif var == "ep":
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dsxy['thprime'].values, dsxy['the'].values, dsxy['pprime'].values, dsxy['ppe'].values, cfg["constants"]["cap"], cfg["constants"]["p0"])
        epm = 1/2*(cfg["constants"]["gravity0"]/get_pmap_bv(cfg))**2 * ((tloc-tenv)/tenv)**2
        epm = filter.gaussian_fft_smoothing(epm,lambdax, lambdax, cfg["dx"], cfg["dx"])
        epm = np.where(epm>clev[0],epm,np.nan)
        cvar = epm
    elif var == "u":
        cvar = dsxy["u"] - dsxy["ue"]
    elif var == "t":
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dsxy['thprime'].values, dsxy['the'].values, dsxy['pprime'].values, dsxy['ppe'].values, cfg["constants"]["cap"], cfg["constants"]["p0"])
        cvar = tloc - tenv
    else:
        cvar = dsxy[var].values

    cmap = mycmap[var]
    norm = S.norm[var]
    clev = S.clev[var]
    contf = ax.pcolormesh(dsxy.xcr, dsxy.ycr, cvar, cmap=cmap, norm=norm, shading='nearest', rasterized=True)
    
    ### - Topography - ###
    # print("Topo-Levels", topo_levels)
    ax.contour(dsxy.xcr, dsxy.ycr, surf_factor*dsxy.zcrtopo, colors='k', levels=topo_levels, linewidths=0.3)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('streamwise x / km')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid()

    if region=="darwin":
        ax.plot(-8.8, -61.6, marker='^', mfc='none', mec='black', markersize=10)

    overlay_cube_footprints(ax, S)

    ax.text(XLBL, ypp, f"z: {float(np.asarray(dsxy.zpos).item()):.1f}km", transform=ax.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})

    return ax, contf


def lidar_bwf_perturbation(tloc, zcr_km):
    """Emulate a measured perturbation: estimate the background by a vertical Butterworth low-pass
    of the full temperature on a uniform z grid (cutoff VERTICAL_CUTOFF), then perturbation = full
    - background. Inputs are (time, z); the vertical grid is time-invariant so one profile of
    heights drives the interpolation to/from the uniform grid the filter needs."""
    z_ref = np.asarray(zcr_km)[0] if np.ndim(zcr_km) > 1 else np.asarray(zcr_km)
    zuni = np.arange(float(z_ref.min()), float(z_ref.max()), 0.2)
    t_uni = np.array([np.interp(zuni, z_ref, tloc[tt]) for tt in range(tloc.shape[0])])
    _, bg_uni = filter.butterworth_filter(t_uni, cutoff=1/VERTICAL_CUTOFF, fs=1/0.2, order=5, mode='low')
    bg = np.array([np.interp(z_ref, zuni, bg_uni[tt]) for tt in range(tloc.shape[0])])
    return tloc - bg


def plot_vlidar(axlid, dslid, cfg, S, var="t", t=0, fpath=None):
    zlim, surf_factor, xlim_lid = S.zlim, S.surf_factor, S.xlim_lid
    xpp, ypp, lw1, lambdaz, zcut_mf = XPP, YPP, LW1, LAMBDAZ, ZCUT_MF
    clev = S.clev[var]

    time_res = (dslid.time[-1] - dslid.time[-2]).values * 60 # min
    if var == "mf":
        uw   = dslid['w'].values * (dslid['u'].values-dslid['ue'].values)
        mfx  = dslid['rh0'].values * uw
        mfx = filter.gaussian_fft_smoothing(mfx, 2*60, lambdaz, time_res, cfg["dz"])
        izcut = int(zcut_mf / cfg["dz"])
        mfx[:,izcut:] = mfx[:,izcut:] * 1000
        cvar = mfx*1000
    elif var == "ep":
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dslid['thprime'].values, dslid['the'].values, dslid['pprime'].values, dslid['ppe'].values, cfg["constants"]["cap"], cfg["constants"]["p0"])
        epm = 1/2*(cfg["constants"]["gravity0"]/get_pmap_bv(cfg))**2 * ((tloc-tenv)/tenv)**2
        epm = filter.gaussian_fft_smoothing(epm, 1, lambdaz, 1, cfg["dz"])
        epm = np.where(epm>clev[0],epm,np.nan)
        cvar = epm
    elif var == "u":
        cvar = dslid["u"] - dslid["ue"]
    elif var == "t":
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dslid['thprime'].values, dslid['the'].values, dslid['pprime'].values, dslid['ppe'].values, cfg["constants"]["cap"], cfg["constants"]["p0"])
        if LIDAR_BG == "bwf":
            # emulate a measurement: background from a vertical Butterworth low-pass of the full T
            cvar = lidar_bwf_perturbation(tloc, np.asarray(dslid.zcr.values))
        else:
            # perturbation relative to the simulation's t=0 ambient (data_0 of the base run):
            # T_ambient = theta_ambient * exner_ambient interpolated to the lidar heights
            prof = plt_helper._load_pmap_ambient(plt_helper._get_pmap_reference_path(fpath)) if fpath else None
            if prof is not None:
                z = np.asarray(dslid.zcr.values)
                z_m = (z[0] if z.ndim > 1 else z) * 1000.0
                tenv = (np.interp(z_m, prof["zcr"], prof["theta_total"])
                        * np.interp(z_m, prof["zcr"], prof["exner_total"]))[None, :]
            cvar = tloc - tenv

        ### Second temperature axis ####
        axt = axlid.twinx()
        axt.plot(dslid.time, cvar[:,dslid.kprof], lw=1.5, color='red')
        axt.tick_params(labelbottom=False,labeltop=False, labelleft=True, labelright=False, left=True, right=False)

        axt.tick_params(axis='y', color='red', labelcolor='red', direction="in", pad=-7) # pad=-27
        axt.spines['left'].set_color('red')
        axt.spines['left'].set_linewidth(1.5)
        axt.set_ylim([-15,110])
        axt.yaxis.set_major_formatter(FuncFormatter(kelvin_formatter))
        axt.set_yticks([-10,0,10,20]) # labels=["-10 K", "0 K", "10 K", "20 K"]
        axt.minorticks_off()  # the twinx K-axis must not draw its own (right-side) minor ticks
        [label.set_horizontalalignment('left') for label in axt.get_yticklabels()]
        # axt.set_ylabel(r"T' / K", color='red')
    else:
        cvar = dslid[var].values
    
    cmap = mycmap[var]
    norm = S.norm[var]
    clev = S.clev[var]
    contf = axlid.pcolormesh(dslid.time.expand_dims({'z':dslid.z},axis=1), dslid.zcr, cvar,
                        cmap=cmap, norm=norm, shading='nearest', rasterized=True)

    # isentropes = axlid.contour(dslid.time, dslid.zcr, dslid.the + dslid.th, levels=thlev, colors='k', lw=lw2)
    # ax.clabel(isentropes, thlev[1::], fontsize=8, fmt='%1.0f K', inline_spacing=1, inline=True, 
    #             manual=[(8,ds.zcr[10,0,x]), (8,ds.zcr[-15,0,x])]) # ha='left', thlev[1::3]

    axlid.plot(dslid.time, surf_factor*dslid.zcr[:,0], lw=2, color='black')
    
    # axlid.yaxis.set_major_locator(MultipleLocator(10))
    axlid.xaxis.set_minor_locator(AutoMinorLocator())
    axlid.yaxis.set_minor_locator(AutoMinorLocator())
    axlid.tick_params(which='both', labelbottom=False,labeltop=True, labelleft=False, labelright=True, left=False, right=True)
    axlid.xaxis.set_label_position('top')
    axlid.yaxis.set_label_position('right')
    axlid.set_xlabel('time / h')
    axlid.set_ylabel('altitude z / km')
    axlid.set_xlim(xlim_lid)
    axlid.set_ylim(zlim)
    axlid.grid()
    
    hrs, rem = divmod(dslid.time[t].values*3600, 3600)
    mins, secs = divmod(rem, 60)
    axlid.axvline(x=[dslid.time[t].values], color='black', lw=lw1, ls='--') # ymin=zlim[0],ymax=zlim[1]
    axt.text(XLBL, ypp-0.1, f"Time: {int(hrs):02d}:{int(mins):02d}:{int(secs):02d}s ({t:03d})", transform=axt.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    if 'xpos' in dslid.attrs:
        axt.text(XLBL, ypp, f"x: {float(dslid.attrs['xpos']):.1f}km, y: {float(dslid.attrs['ypos']):.1f}km", transform=axt.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})

    return axlid, contf


def kelvin_formatter(x, pos):
    return f"{x:.0f}K"
    # return f"{x:.0f}\u2009K"


def emulate_airplane_measurement(ds, speed=220, xrange=None, dim='xcr'):
    """
    Emulate an airplane flying along one slice (x-axis) with dimensions (time, z, x).
    
    Parameters:
    - ds: xarray.Dataset with dimensions (time, z, x)
    - speed: speed of the airplane in m/s
    - range: Boundaries for airplane to turn around
    
    Returns:
    - xarray.DataArray with dimensions (time, z), representing the vertical
      profile at the airplane's position at each timestep.
    """

    ## Format dataset
    # ds = ds.set_coords('time')
    if dim == 'ycr':
        ds['y'] = ds['ycr'][:,0]
        ds = ds.set_coords('y')
        ds = ds.set_index({'y':'y'})
    else:
        ds['x'] = ds['xcr'][0,:]
        ds = ds.set_coords('x')
        ds = ds.set_index({'x':'x'})
    ds = ds.reset_coords('zcr', drop=False)

    ## Get x-locations of airplane for each timestep
    if xrange is None:
        xrange  = [ds[dim].min().values, ds[dim].max().values]
    xrange = [x * 1000 for x in xrange] # meter
    
    T = (xrange[1] - xrange[0]) / speed  # time for one leg
    tau = np.mod(ds.time.values * 3600, 2 * T)
    xlocs = np.where(tau < T, # tau < T -> forward flying
                    xrange[0] + speed * tau,
                    xrange[1] - speed * (tau - T))
    
    # Extract vertical profiles for each x-location
    if dim == 'ycr':
        vertical_profiles = [
            ds.isel(t=tstep).sel(y=xloc/1000, method='nearest')
            for tstep, xloc in enumerate(xlocs)
        ]
    else:
        vertical_profiles = [
            ds.isel(t=tstep).sel(x=xloc/1000, method='nearest')
            for tstep, xloc in enumerate(xlocs)
        ]
    result = xr.concat(vertical_profiles, dim='time')
    return result, xlocs


if __name__ == '__main__':
    """Generate animation of PMAP simulation based on NETCDF slice output."""

    """Example: 
        >> python3 slc_and_lid.py <var> <simulation> <notest>
        >> python3 slc_and_lid.py vortex <simulation> <notest>
        >> python3 slc_and_lid.py amtm <simulation> <notest>
        >> python3 slc_and_lid.py surf <simulation> <notest>
        >> python3 slc_and_lid.py alima_x ideal_topo_L20 <notest>
        >> python3 slc_and_lid.py alima_y ideal_topo_L20 <notest>
    """    
    global region, xlim_lid, test, itopo, amp0

    var = sys.argv[1]
    simulation_arg = sys.argv[2]
    fpath = simulation_arg if os.path.isdir(simulation_arg) else os.path.join(data_folder, simulation_arg)
    simulation = os.path.basename(os.path.normpath(fpath))
    cube_rects = get_cube_footprints(fpath)
    print(f"[i]  Output-cube footprints: {[(r[4], round(r[0]), round(r[1]), round(r[2]), round(r[3])) for r in cube_rects]}")
    inventory = plt_helper.get_slice_inventory(fpath)
    if inventory["model"] != "pmap":
        raise ValueError(f"{os.path.basename(__file__)} now only supports PMAP output, got '{inventory['model']}' for {fpath}")

    test = True
    zslice_km = None
    for a in sys.argv[3:]:
        if a == "notest":
            test = False
        elif a.startswith("--zkm="):
            try:
                zslice_km = float(a.split("=", 1)[1])
            except ValueError:
                zslice_km = None
    region = simulation.split("_")[0]
    ncpus = mp.cpu_count()-2 # use maximum here but check number of tasks --> ntasks
    ncpus = 40  # ~5GB/worker (per-frame data+figure) -> 40 fit in --mem=240G; 50 -> OOM/swap
    image_folder = os.path.join(animation_folder, simulation) + "_" + var
    # shutil.rmtree(image_folder, ignore_errors=True)
    if os.path.isdir(image_folder):
        for png_file in glob.glob(os.path.join(image_folder, "*.png")):
            try:
                os.remove(png_file)
            except OSError:
                pass

    """Slices"""
    reg = REGIONS.get(region, {})
    xlim_lid = None
    xlid = reg.get("xlid", 0)
    kslice = reg.get("zslice", 2)
    if zslice_km is not None:
        kslice, _kz = nearest_zslice_index(fpath, zslice_km)
        print(f"[i]  z-slice override: target {zslice_km:g} km -> index {kslice} = {_kz:.2f} km")
    kdefault = min(inventory['z'] - 1, kslice)
    yslice = reg.get("yslice", 0)
    slices = {"x": 0, "y": yslice, "z": [kdefault]}
    if var == 'alima_x' or var == 'alima_y':
        xlid = None
        slices = {"x": 0, "y": yslice, "z": [0, kdefault]}

    stime = time.time()
    print(f"[i]  Test run (one timestamp): Opening NETCDF files for simulation: {fpath}")
    print("Slices: ", slices)

    _cfg, dsxz, dsyz = plt_helper.preprocess_pmap_xzyz(fpath, slices=slices)
    elapsed = time.time() - stime
    print(f"[i]  Files loaded in {elapsed / 60:.2f}min.")

    if xlim_lid is None:
        xlim_lid = [float(dsxz.time.min().values), float(dsxz.time.max().values)]

    amp0 = _cfg.get("define_orography", {}).get("args", {}).get("amplitude", 0.0)
    itopo = int(0 if amp0 else 1)
    if itopo == 0:
            print("[i]  Using idealized topography.")
    else:
            print("[i]  Using realistic topography.")

    """Lidar location in xz slice"""
    if xlid is not None:
        xmin = float(dsxz.xcr.min().values)
        xmax = float(dsxz.xcr.max().values)
        if xlid < xmin or xlid > xmax:
            xlid = float(dsxz.xcr[0, np.shape(dsxz.xcr)[-1] // 2].values)
        ilid = int((xlid - xmin) / (xmax - xmin) * np.shape(dsxz.xcr)[-1])
        xlid = dsxz.xcr[0,ilid].values
        dslid = dsxz.isel(x=ilid) # x location in km
        dslid.attrs['i'] = ilid 
        dslid.attrs['xpos'] = xlid
        # print(f"[i]  Virtual ground-based lidar profile: i={dslid.i}, j={dslid.j}, x={float(dslid.attrs['xpos']):.1f}km, y={float(dslid.attrs['ypos']):.1f}km")
        
    else:
        global alima_locs
        if var == 'alima_y':
            dslid, alima_locs = emulate_airplane_measurement(dsyz, speed=220, xrange=[-200,200], dim='ycr')
        else:
            dslid, alima_locs = emulate_airplane_measurement(dsxz, speed=220, xrange=[-200,200], dim='xcr')
        print(f"[i]  Virtual airplane-based profile (ALIMA)!")

    # xlim_lid = [0,ds_xz.time.max().values]
    trange = [np.abs(dsxz.time.values - xlim_lid[0]).argmin(), np.abs(dsxz.time.values - xlim_lid[1]).argmin()]
    # trange = [300,trange[1]]
    # trange = [0,90]

    trange = np.arange(trange[0], trange[1])
    if test:
        trange = [len(dsxz.time.values)-1]
        # trange = [0]
    print(f"Plotting time steps {trange[0]} to {trange[-1]}")

    """Parallel processing"""
    progress_counter = mp.Manager().Value('i', 0)
    lock = mp.Manager().Lock()
    stime = time.time()
    pbar = {"progress_counter": progress_counter, "lock": lock, "stime": stime, "ntasks": len(trange)}
    ncpus = np.min([ncpus, pbar['ntasks']])
    sema = mp.Semaphore(ncpus)

    print(f"[i]  CPUs available: {mp.cpu_count()}")
    print(f"[i]  CPUs for visualization: {ncpus}")

    # compute the ambient reference for every slice ONCE here, before forking the render workers,
    # so data_0.nc is read a single time and each worker only subtracts the inherited reference
    _amb = plt_helper.precompute_ambient(fpath, slices)
    print(f"[i]  Ambient reference {'precomputed' if _amb else 'unavailable (legacy reference)'}", flush=True)

    running_procs = []
    for t in trange:
        for p in running_procs[:]:
            if not p.is_alive():
                p.join()
                running_procs.remove(p)
        sema.acquire()
        args = (t, var, fpath, slices, dslid, image_folder, pbar, sema)
        proc = mp.Process(target=slc_and_lid, args=args)
        running_procs.append(proc)
        proc.start()
    
    for proc in running_procs:
        proc.join()

    plt_helper.create_animation(image_folder, "anime_slc_" + simulation + "_" + var + ".mp4")
