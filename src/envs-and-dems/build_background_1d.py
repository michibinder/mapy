"""Build a 1D PMAP background column for the darwin_240718 case.

Rewrite of env_jawara_1D_240718_2.ipynb with a single CONFIG block and a clean pipeline:

    JAWARA reanalysis  --(lower atmosphere + upper winds)-->
    CORAL lidar T      --(de-waved, mesosphere temperature)-->  blended T(z), u(z), v(z)
       --> hydrostatic reconstruction --> (th, rho, exner) --> PMAP init .nc

Key differences from the notebook, and why (see the session analysis):

  * Temperature in ~55-82 km is taken from a DE-WAVED CORAL mean, not JAWARA. JAWARA is a
    coarse reanalysis: its mesospheric T is smooth and nearly time-invariant, with its
    least-stable layer stuck at ~67 km, so no averaging/smoothing of JAWARA can lower the
    60 km stability. CORAL resolves the real, lower/less-stable layer (~62 km); blending it
    in is the only lever that lowers stability where the wave breaks.
  * De-waving: the CORAL mean still contains the transient wave packet near 0 UTC. We remove
    it by (optionally) excluding that time window and by vertical smoothing (CORAL_VSMOOTH_KM),
    so we import the background structure, not the wave itself.
  * ALL smoothing/blending is done on T, u, v BEFORE the hydrostatic integration, so the saved
    (th, rho, exner) are mutually consistent (the notebook smoothed th and rho independently,
    which is slightly inconsistent).
  * Winds use a single snapshot at T_INDEX (01-02 UTC) -- already the lowest data-valid
    critical level (~72 km). Time-averaging the winds washes out the reversal and RAISES it,
    so we do not average winds.
  * JET_* are explicit, clearly ad-hoc sensitivity knobs (NOT data-constrained) to lower/weaken
    the mesospheric jet and its critical level, for testing how far breaking would descend.

Run with the post-venv:
    /home/b/b309199/venvs/post-venv/bin/python build_background_1d.py
"""
from pathlib import Path
import datetime as dt

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("/work/bd0620/b309199/mapy/src/latex_default.mplstyle")

# ======================================================================================
# CONFIG -- every lever lives here
# ======================================================================================
JAWARA_FILE = Path("/work/bd0620/b309199/mapy/data/jawara/jawara_patagonia_240718.nc")
ERA5_FILE = Path("/work/bd0620/b309199/data/era5-data/20240718-2136-ml-int.nc")  # native 0.25 deg, 0-70 km
CORAL_FILE = Path("/work/bd0620/b309199/mapy/data/coral/20240718-2136_T15Z900.nc")
OUT_NC = Path("/work/bd0620/b309199/mapy/data/envs/jawara_240719T01_coralT_200m_l641.nc")
OUT_FIG = Path("/work/bd0620/b309199/claude/tmp/background_1d_coralT.png")
OUT_FIG_FULL = Path("/work/bd0620/b309199/claude/tmp/background_1d_coralT_fullcolumn.png")
REF_NC = Path("/work/bd0620/b309199/mapy/data/envs/jawara_240719T01_blend_400m_l321.nc")  # current, for comparison

# --- model vertical grid ---
NZ = 641
DZ = 200.0                                   # m  -> 0 .. 128 km  (was 321 @ 400 m)

# --- horizontal sampling of JAWARA ---
# Geography: Mt Darwin (the mountain) ~ (-54.7, 290.7 E); CORAL lidar (in the NE lee, at the coast)
# ~ (-53.79, 292.25 E). Forcing is southwesterly, so the low-level source is UPSTREAM = SW of the
# range, just off the coast; the upper wind (breaking region above CORAL) uses the CORAL column.
CORAL_LAT, CORAL_LON = -53.5, 292.5          # CORAL grid column -> upper-atmosphere wind + temperature/psurf

# Low-level (surface) wind forcing source. ERA5 is the native 0.25-deg product (terrain-resolving,
# most exact) but its sponge damps winds above ~45-50 km, so it is used ONLY for the low levels
# (blended to the JAWARA/CORAL upper wind by WIND_BLEND_KM, well below the sponge). JAWARA is the
# coarse (~2.8 deg) fallback.
WIND_LL_DATASET = "era5"                       # "era5" (0.25 deg, use only < ~45 km) | "jawara" (coarse)
WIND_LL_SOURCE = "area"                        # "point" (single column) | "area" (band mean)
WIND_LL_POINT = (-55.5, 284.0)                 # SW-upstream over open water (~76 W), |U|~28 m/s, WSW (252 deg)
# WIND_LL_AREA = (slice(-56.0, -58.0), slice(283.0, 288.0))  # original open-ocean box 56-58S/77-72W: |U|@3km~26.3 WSW (255)
WIND_LL_AREA = (slice(-52.5, -54.0), slice(285.0, 286.5))  # coastal box 52.5-54S/75-73.5W (SW-Chile fjords, low-wind pocket); ~4 m/s weaker than the open-ocean box
WIND_LL_TAG = "wcoast"                          # filename tag for a non-default WIND_LL_AREA (avoids clobbering vsm20); "" = none

# --- time selection ---
T_INDEX = 25                                 # 2024-07-19 01:00 UTC (lowest data-valid critical level)
T_AVG_HALFWIN = 4                            # JAWARA temperature (below the CORAL band) averaged over +/- this many hours

# --- temperature: CORAL blend (km) ---
BLEND_JAW_TO_CORAL = (30.0, 36.0)            # quintic JAWARA -> CORAL; down to ~30 km so the 15 km
                                             #   filter also removes the wave bumps in the ERA5/JAWARA
                                             #   T at 36-44 km (broadens the stratopause by ~9 K, well
                                             #   below the breaking region). ERA5 below 30 km untouched.
BLEND_CORAL_TO_JAW = (82.0, 86.0)            # quintic CORAL -> JAWARA (CORAL invalid above ~86 km)
CORAL_VSMOOTH_KM = 20.0                       # vertical smoothing to de-wave the CORAL mean (matches the 20 km GW Butterworth
                                              #   cutoff used in the GW analysis; only acts on CORAL, valid >15 km & used only
                                              #   >50 km, so the ERA5 tropopause below is untouched). 10-15 km => least-stable
                                              #   layer 57-59 km; 20 km smooths that out further -> higher/weaker breaking.
CORAL_EXCLUDE_WAVE = False                    # also drop the 0-UTC wave packet from the CORAL mean
CORAL_WAVE_WINDOW = ("2024-07-18T23:30", "2024-07-19T03:30")
T_SMOOTH_KM = 0.8                            # final light smoothing of the blended T (continuity)

# --- winds ---
WIND_BLEND_KM = (20.0, 40.0)                 # quintic area-avg -> CORAL-point
WIND_SMOOTH_KM = 1.6

# --- jet-shaping knobs (sensitivity; only the upper wind above the pivot) ---
# JET_MESO_DECAY="era5": keep JAWARA's magnitude up to JET_DECAY_PIVOT_KM, then make the upper wind
#   decay with ERA5's (faster) normalised shape instead of following JAWARA's coarse jet bump.
#   Anchored to JAWARA at the pivot, so it stays above the sponged ERA5 magnitude but decays like it.
#   Motivated by the measured ~12% JAWARA-vs-ERA5 overestimate + the spurious 58-60 km JAWARA bump.
JET_MESO_DECAY = "none"                       # "none" | "era5"
JET_DECAY_PIVOT_KM = 52.0
# uniform shift/scale (kept for reference; "era5" decay is preferred over these):
JET_PIVOT_KM = 45.0                          # shift/scale act only above this altitude
JET_SHIFT_KM = 0.0                           # >0 lowers the jet + critical level (relocates the bump - avoid)
JET_SCALE = 1.0                              # <1 weakens the jet uniformly above the pivot

# --- thermodynamic constants (PMAP-consistent) ---
G, RD, CP, P0 = 9.80616, 287.05, 1004.0, 1.0e5
KAPPA = RD / CP


# ======================================================================================
# helpers
# ======================================================================================
def smoothstep(z, z1, z2):
    """Quintic 0->1 ramp from z1 to z2 (zero slope/curvature at both ends)."""
    t = np.clip((z - z1) / (z2 - z1), 0.0, 1.0)
    return t ** 3 * (10.0 - 15.0 * t + 6.0 * t ** 2)


def quintic_blend(lo_field, hi_field, z, z1, z2):
    """Smoothstep blend lo->hi between z1 and z2."""
    w = smoothstep(z, z1, z2)
    return (1.0 - w) * lo_field + w * hi_field


def n2_theta(z, th):
    return G * np.gradient(np.log(th), z)


def n2_from_t(z, t):
    return (G / t) * (np.gradient(t, z) + G / CP)


# ======================================================================================
# data loaders
# ======================================================================================
def _select_column(ds, src):
    """Pick a single upstream column ('point') or a band mean ('area') from a dataset."""
    if WIND_LL_SOURCE == "area":
        col = ds.sel(latitude=WIND_LL_AREA[0], longitude=WIND_LL_AREA[1]).mean(["latitude", "longitude"])
        where = f"{src} band {WIND_LL_AREA[0].start}..{WIND_LL_AREA[0].stop} lat"
    else:
        col = ds.sel(latitude=WIND_LL_POINT[0], longitude=WIND_LL_POINT[1], method="nearest")
        where = f"{src} ({float(col.latitude):.2f}, {float(col.longitude):.2f}E)"
    return col, where


def _reshape_upper_to_era5_decay(u_pt, v_pt, zmodel):
    """Above JET_DECAY_PIVOT_KM, replace JAWARA's jet shape with ERA5's normalised decay, anchored to
    JAWARA's magnitude at the pivot (keeps direction; only reduces). |U|(z>piv) = |U|_jaw(piv) *
    |U|_era5(z)/|U|_era5(piv). ERA5 (CORAL column) is used where valid (<=70 km); its 70 km ratio is
    held above."""
    e = xr.open_dataset(ERA5_FILE)
    ze = e["level"].values.astype(float)
    ec = e.sel(latitude=CORAL_LAT, longitude=CORAL_LON, method="nearest")
    Ue = np.interp(zmodel, ze, np.hypot(ec["u"].isel(time=T_INDEX).values, ec["v"].isel(time=T_INDEX).values))
    e.close()

    zk = zmodel / 1e3
    Uj = np.hypot(u_pt, v_pt)
    ip = int(np.argmin(np.abs(zk - JET_DECAY_PIVOT_KM)))
    ratio = Ue / Ue[ip]
    ratio[zk > 70.0] = ratio[int(np.argmin(np.abs(zk - 70.0)))]     # ERA5 tops out ~70 km
    target = np.where(zk > JET_DECAY_PIVOT_KM, np.minimum(Uj[ip] * ratio, Uj), Uj)
    scale = np.where(Uj > 1e-6, target / Uj, 1.0)
    return u_pt * scale, v_pt * scale


def load_jawara_column(zmodel):
    """Profiles interpolated onto the model grid.

    Temperature (+psurf) and the upper-atmosphere wind come from the CORAL column of the JAWARA
    whole-atmosphere file; the low-level forcing wind comes from the SW-upstream column of either
    ERA5 (native 0.25 deg, default) or JAWARA (coarse). Returns a dict of profiles.
    """
    ds = xr.open_dataset(JAWARA_FILE)
    zj = ds["level"].values.astype(float)
    pt = ds.sel(latitude=CORAL_LAT, longitude=CORAL_LON, method="nearest")

    sl = slice(T_INDEX - T_AVG_HALFWIN, T_INDEX + T_AVG_HALFWIN)
    t_jaw = pt["t"].isel(time=sl).mean("time").values           # temperature background
    psurf = float(pt["p"].isel(time=sl).mean("time").values[0])  # surface p anchor
    u_pt = pt["u"].isel(time=T_INDEX).values                    # upper-atmosphere wind (CORAL col, JAWARA)
    v_pt = pt["v"].isel(time=T_INDEX).values
    jtime = str(ds["time"].values[T_INDEX])[:16]
    ds.close()

    # low-level forcing wind: ERA5 (fine, terrain-resolving) or JAWARA (coarse), interpolated from
    # its own vertical grid. Only matters below WIND_BLEND_KM, so ERA5's sponged upper levels are
    # blended out before they are reached.
    lowds = xr.open_dataset(ERA5_FILE if WIND_LL_DATASET == "era5" else JAWARA_FILE)
    zlow = lowds["level"].values.astype(float)
    low, where = _select_column(lowds, WIND_LL_DATASET)
    u_low = np.interp(zmodel, zlow, low["u"].isel(time=T_INDEX).values)
    v_low = np.interp(zmodel, zlow, low["v"].isel(time=T_INDEX).values)
    lowds.close()

    interp = lambda f: np.interp(zmodel, zj, f)
    u_up, v_up = interp(u_pt), interp(v_pt)
    upper_note = ""
    if JET_MESO_DECAY == "era5":
        u_up, v_up = _reshape_upper_to_era5_decay(u_up, v_up, zmodel)
        upper_note = f" | upper wind: ERA5 decay above {JET_DECAY_PIVOT_KM:.0f} km"
    return dict(t=interp(t_jaw), u_pt=u_up, v_pt=v_up,
                u_area=u_low, v_area=v_low, psurf=psurf,
                time=f"{jtime} | low-level wind: {where}{upper_note}")


def load_coral_background(zmodel):
    """De-waved CORAL temperature on the model grid (NaN where CORAL is invalid)."""
    ds = xr.open_dataset(CORAL_FILE, decode_times=False)
    toff = float(ds["time_offset"].values[0])
    unix = toff + ds["time"].values.astype(float) / 1000.0
    z = (ds["altitude"].values.astype(float) + float(ds["altitude_offset"].values[0])
         + float(ds["station_height"].values[0]))
    T = np.where(ds["temperature"].values.astype(float) == 0.0, np.nan,
                 ds["temperature"].values.astype(float))
    ds.close()

    if CORAL_EXCLUDE_WAVE:
        epoch = dt.datetime(1970, 1, 1)
        w0 = (dt.datetime.fromisoformat(CORAL_WAVE_WINDOW[0]) - epoch).total_seconds()
        w1 = (dt.datetime.fromisoformat(CORAL_WAVE_WINDOW[1]) - epoch).total_seconds()
        keep = ~((unix >= w0) & (unix <= w1))
        T = T[keep]

    valid = np.isfinite(T).any(axis=0)
    z, T = z[valid], T[:, valid]
    tmean = np.nanmean(T, axis=0)

    n = max(1, int(round(CORAL_VSMOOTH_KM * 1000.0 / np.median(np.diff(z)))))
    tmean = uniform_filter1d(tmean, size=n, mode="nearest")

    t_on_model = np.full_like(zmodel, np.nan)
    inside = (zmodel >= z.min()) & (zmodel <= z.max())
    t_on_model[inside] = np.interp(zmodel[inside], z, tmean)
    return t_on_model


def coral_raw_mean(zmodel, window="nightly"):
    """Raw (no vertical filter) CORAL temperature mean on the model grid, for a time window.

    window="nightly" -> mean over all profiles; "first3h" -> mean over the first 3 h only
    (pre-breaking). Diagnostic reference only; the stacking uses the smoothed nightly mean.
    """
    ds = xr.open_dataset(CORAL_FILE, decode_times=False)
    toff = float(ds["time_offset"].values[0])
    unix = toff + ds["time"].values.astype(float) / 1000.0
    z = (ds["altitude"].values.astype(float) + float(ds["altitude_offset"].values[0])
         + float(ds["station_height"].values[0]))
    T = np.where(ds["temperature"].values.astype(float) == 0.0, np.nan,
                 ds["temperature"].values.astype(float))
    ds.close()

    if window == "first3h":
        T = T[unix <= unix.min() + 3.0 * 3600.0]
    valid = np.isfinite(T).any(axis=0)
    z, T = z[valid], T[:, valid]
    tmean = np.nanmean(T, axis=0)

    out = np.full_like(zmodel, np.nan)
    inside = (zmodel >= z.min()) & (zmodel <= z.max())
    out[inside] = np.interp(zmodel[inside], z, tmean)
    return out


# ======================================================================================
# profile builders
# ======================================================================================
def build_temperature(zmodel, jaw, t_coral):
    """JAWARA everywhere, replaced by de-waved CORAL over the mesospheric window.

    The CORAL weight rises 0->1 across BLEND_JAW_TO_CORAL, holds at 1, then falls 1->0 across
    BLEND_CORAL_TO_JAW, so JAWARA is untouched below/above and CORAL owns the middle band.
    """
    zk = zmodel / 1000.0
    t_jaw = jaw["t"]
    t_coral_filled = np.where(np.isfinite(t_coral), t_coral, t_jaw)
    w = smoothstep(zk, *BLEND_JAW_TO_CORAL) * (1.0 - smoothstep(zk, *BLEND_CORAL_TO_JAW))
    t = (1.0 - w) * t_jaw + w * t_coral_filled
    if T_SMOOTH_KM > 0:
        t = gaussian_filter1d(t, T_SMOOTH_KM * 1000.0 / DZ)
    return t


def apply_jet_knob(u, v, zmodel):
    """Explicit, ad-hoc modification of the mesospheric jet (sensitivity only)."""
    if JET_SHIFT_KM == 0.0 and JET_SCALE == 1.0:
        return u, v
    zk = zmodel / 1000.0
    us, vs = u.copy(), v.copy()
    if JET_SHIFT_KM != 0.0:                                     # pull the profile down by JET_SHIFT_KM
        us = np.interp(zmodel + JET_SHIFT_KM * 1000.0, zmodel, u)
        vs = np.interp(zmodel + JET_SHIFT_KM * 1000.0, zmodel, v)
    us = JET_SCALE * us
    vs = JET_SCALE * vs
    w = np.clip((zk - JET_PIVOT_KM) / 5.0, 0.0, 1.0)            # blend in above the pivot only
    return (1 - w) * u + w * us, (1 - w) * v + w * vs


def build_winds(zmodel, jaw):
    zk = zmodel / 1000.0
    u = quintic_blend(jaw["u_area"], jaw["u_pt"], zk, *WIND_BLEND_KM)
    v = quintic_blend(jaw["v_area"], jaw["v_pt"], zk, *WIND_BLEND_KM)
    u, v = apply_jet_knob(u, v, zmodel)
    if WIND_SMOOTH_KM > 0:
        u = gaussian_filter1d(u, WIND_SMOOTH_KM * 1000.0 / DZ)
        v = gaussian_filter1d(v, WIND_SMOOTH_KM * 1000.0 / DZ)
    return u, v


def hydrostatic(zmodel, t, psurf):
    """Integrate hydrostatic balance upward from the surface pressure to get (p, th, rho, exner)
    consistent with the temperature profile."""
    p = np.empty_like(t)
    p[0] = psurf
    for k in range(1, len(t)):
        dz = zmodel[k] - zmodel[k - 1]
        integrand = 0.5 * (1.0 / t[k - 1] + 1.0 / t[k])         # trapezoid of g/(Rd T)
        p[k] = p[k - 1] * np.exp(-G / RD * dz * integrand)
    exner = (p / P0) ** KAPPA
    th = t / exner
    rho = p / (RD * t)
    return p, th, rho, exner


# ======================================================================================
# main
# ======================================================================================
def _knob_tag():
    """Filename suffix so data-pure and jet-shaped runs never clobber each other."""
    tag = ""
    if CORAL_VSMOOTH_KM != 15.0:
        tag += f"_vsm{CORAL_VSMOOTH_KM:g}"
    if WIND_LL_TAG:
        tag += f"_{WIND_LL_TAG}"
    if JET_MESO_DECAY == "era5":
        tag += f"_era5decay{JET_DECAY_PIVOT_KM:g}"
    if JET_SHIFT_KM != 0.0 or JET_SCALE != 1.0:
        tag += f"_jet{JET_SHIFT_KM:g}km_s{JET_SCALE:g}"
    return tag


def main():
    zmodel = np.arange(NZ) * DZ

    jaw = load_jawara_column(zmodel)
    t_coral = load_coral_background(zmodel)
    t = build_temperature(zmodel, jaw, t_coral)
    u, v = build_winds(zmodel, jaw)
    p, th, rho, exner = hydrostatic(zmodel, t, jaw["psurf"])

    # --- sanity: hydrostatic column reproduces JAWARA where we did not touch T ---
    _, th_j, rho_j, _ = hydrostatic(zmodel, jaw["t"], jaw["psurf"])
    zlo = min(BLEND_JAW_TO_CORAL[0] - 2.0, 28.0) * 1000.0
    lo = zmodel < zlo
    print(f"time {jaw['time']}")
    print(f"  hydrostatic self-check: max |th_rebuilt - th(JAWARA-T)| below {zlo/1e3:.0f} km = "
          f"{np.max(np.abs(th_j - th)[lo]):.2f} K   (CORAL blend starts at {BLEND_JAW_TO_CORAL[0]:.0f} km)")
    k3 = int(np.argmin(np.abs(zmodel - 3000.0)))
    print(f"  low-level forcing |U| @ 3 km = {np.hypot(u[k3], v[k3]):.1f} m/s  "
          f"(u,v = {u[k3]:.1f}, {v[k3]:.1f})")

    out = xr.Dataset(
        {"u": ("level", u), "v": ("level", v), "w": ("level", np.zeros_like(u)),
         "th": ("level", th), "rho": ("level", rho), "exner": ("level", exner)},
        coords={"level": zmodel.astype("int64")},
    )
    out_nc = OUT_NC.with_name(OUT_NC.stem + _knob_tag() + OUT_NC.suffix)
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_nc)
    print(f"wrote {out_nc}")

    t_coral_night_raw = coral_raw_mean(zmodel, "nightly")
    t_coral_3h_raw = coral_raw_mean(zmodel, "first3h")
    make_diagnostics(zmodel, t, u, v, th, t_coral_night_raw, t_coral_3h_raw)
    make_ambient_plot(zmodel, u, v, t, rho, exner)


def _source_profiles(path, lat, lon):
    """(z, |U|, T, N, rho) for a source column: T time-averaged (build window), winds at T_INDEX."""
    ds = xr.open_dataset(path)
    z = ds["level"].values.astype(float)
    col = ds.sel(latitude=lat, longitude=lon, method="nearest")
    sl = slice(T_INDEX - T_AVG_HALFWIN, T_INDEX + T_AVG_HALFWIN)
    t = col["t"].isel(time=sl).mean("time").values
    p = col["p"].isel(time=sl).mean("time").values
    u = col["u"].isel(time=T_INDEX).values
    v = col["v"].isel(time=T_INDEX).values
    ds.close()
    n = uniform_filter1d(n2_from_t(z, t), 5)
    return dict(z=z / 1e3, U=np.hypot(u, v), T=t, N=np.sqrt(np.clip(n, 0, None)) * 1e3,
                rho=p / (RD * t))


def _coral_profile_N(zmodel, tprof):
    """(z_km, N[1e-3 s^-1]) on a CORAL temperature profile's valid range (1 km-smoothed N)."""
    m = np.isfinite(tprof)
    z, tt = zmodel[m], tprof[m]
    n = np.sqrt(np.clip(uniform_filter1d((G / tt) * (np.gradient(tt, z) + G / CP),
                                         int(1000.0 / DZ) | 1), 0, None)) * 1e3
    return z / 1e3, n


def make_ambient_plot(zmodel, u, v, t, rho, exner):
    """Full-column view of the stacked background vs its pure sources (ERA5 / JAWARA / CORAL)."""
    era = _source_profiles(ERA5_FILE, CORAL_LAT, CORAL_LON)          # native ERA5 at CORAL column
    jaw = _source_profiles(JAWARA_FILE, CORAL_LAT, CORAL_LON)        # JAWARA at CORAL column

    # forcing wind = exactly what the build injects at low levels (ERA5/JAWARA, point or area)
    lowds = xr.open_dataset(ERA5_FILE if WIND_LL_DATASET == "era5" else JAWARA_FILE)
    zforce = lowds["level"].values.astype(float) / 1e3
    fcol, flabel = _select_column(lowds, WIND_LL_DATASET)
    uforce = np.hypot(fcol["u"].isel(time=T_INDEX).values, fcol["v"].isel(time=T_INDEX).values)
    lowds.close()
    tcor = load_coral_background(zmodel)                             # de-waved CORAL T on model grid
    cor_m = np.isfinite(tcor)
    zc = zmodel[cor_m] / 1e3                                         # valid CORAL altitudes (km)
    tc = tcor[cor_m]
    # N on the valid CORAL profile only (avoids fill-value gradient spikes at its 15/86 km edges);
    # trim 2 points each end to drop the one-sided-gradient artefacts.
    ncor = np.sqrt(np.clip(uniform_filter1d((G / tc) * (np.gradient(tc, zmodel[cor_m]) + G / CP),
                                            int(1000.0 / DZ) | 1), 0, None)) * 1e3
    cs = slice(2, -2)

    zk = zmodel / 1e3
    n_st = np.sqrt(np.clip(uniform_filter1d(n2_from_t(zmodel, t), int(1000.0 / DZ) | 1), 0, None)) * 1e3
    CE, CJ, CC, CS = "tab:blue", "0.55", "tab:green", "tab:red"     # ERA5 / JAWARA / CORAL / stacked
    C3H = "tab:purple"                                             # CORAL 3 h raw (nightly raw = dotted CORAL green)

    # raw (unfiltered) CORAL means, diagnostic reference: nightly + pre-breaking first-3 h
    tc_night_raw = coral_raw_mean(zmodel, "nightly")
    tc_3h_raw = coral_raw_mean(zmodel, "first3h")
    zc_nr, N_nr = _coral_profile_N(zmodel, tc_night_raw)
    zc_3r, N_3r = _coral_profile_N(zmodel, tc_3h_raw)

    fig, ax = plt.subplots(1, 5, figsize=(16.0, 6.0), sharey=True, constrained_layout=True)
    for a in ax:
        a.set_ylim(0, 90)
        a.axhspan(*WIND_BLEND_KM, color="tab:blue", alpha=0.05, lw=0)
        a.axhspan(BLEND_JAW_TO_CORAL[0], BLEND_CORAL_TO_JAW[1], color="tab:red", alpha=0.05, lw=0)

    # (0) wind speed
    ax[0].axvline(0, color="0.8", lw=0.6)
    ax[0].plot(era["U"], era["z"], color=CE, lw=1.0, label="ERA5 (CORAL col)")
    ax[0].plot(uforce, zforce, color=CE, lw=1.3, ls=":", label="ERA5 (SW forcing)")
    ax[0].plot(jaw["U"], jaw["z"], color=CJ, lw=1.0, label="JAWARA (CORAL col)")
    ax[0].plot(np.hypot(u, v), zk, color=CS, lw=2.0, label="stacked")
    ax[0].set_xlabel(r"$|U|$ / m$\,$s$^{-1}$"); ax[0].set_ylabel("altitude / km")
    ax[0].set_xlim(0, 130); ax[0].legend(loc="upper left", fontsize=9)

    # (1) temperature
    ax[1].plot(era["T"], era["z"], color=CE, lw=1.0, label="ERA5")
    ax[1].plot(jaw["T"], jaw["z"], color=CJ, lw=1.0, label="JAWARA")
    ax[1].plot(tc, zc, color=CC, lw=1.2, label="CORAL (de-waved)")
    ax[1].plot(tc_night_raw, zk, color=CC, ls=":", lw=0.9, label="CORAL nightly (raw)")
    ax[1].plot(tc_3h_raw, zk, color=C3H, ls="--", lw=0.9, label="CORAL 3 h (raw)")
    ax[1].plot(t, zk, color=CS, lw=2.0, label="stacked")
    ax[1].set_xlabel("temperature / K"); ax[1].set_xlim(180, 300); ax[1].legend(loc="upper left", fontsize=8)

    # (2) stability N
    ax[2].plot(era["N"], era["z"], color=CE, lw=1.0)
    ax[2].plot(jaw["N"], jaw["z"], color=CJ, lw=1.0)
    ax[2].plot(ncor[cs], zc[cs], color=CC, lw=1.2)
    ax[2].plot(N_nr[cs], zc_nr[cs], color=CC, ls=":", lw=0.9)
    ax[2].plot(N_3r[cs], zc_3r[cs], color=C3H, ls="--", lw=0.9)
    ax[2].plot(n_st, zk, color=CS, lw=2.0)
    ax[2].set_xlabel(r"$N$ / $10^{-3}\,$s$^{-1}$"); ax[2].set_xlim(0, 32)

    # (3) density
    ax[3].plot(era["rho"], era["z"], color=CE, lw=1.0)
    ax[3].plot(jaw["rho"], jaw["z"], color=CJ, lw=1.0)
    ax[3].plot(rho, zk, color=CS, lw=2.0)
    ax[3].set_xscale("log"); ax[3].set_xlabel(r"density / kg$\,$m$^{-3}$")

    # (4) exner (stacked; sources shown as (p/p0)^kappa)
    ax[4].plot(exner, zk, color=CS, lw=2.0)
    ax[4].set_xlabel("exner"); ax[4].set_xlim(0, 1)

    circ = {"boxstyle": "circle", "lw": 0.8, "facecolor": "white", "edgecolor": "black"}
    for lab, a in zip("abcde", ax):
        a.text(0.93, 0.98, lab, transform=a.transAxes, ha="right", va="top",
               weight="bold", fontsize=13, bbox=circ)

    out = OUT_FIG_FULL.with_name(OUT_FIG_FULL.stem + _knob_tag() + OUT_FIG_FULL.suffix)
    fig.savefig(out, dpi=150, facecolor="w", bbox_inches="tight")
    print(f"wrote {out}")


def _least_stable(zmodel, n2, lo=55000, hi=72000):
    m = (zmodel >= lo) & (zmodel <= hi)
    i = np.nanargmin(n2[m])
    return zmodel[m][i] / 1e3, n2[m][i] * 1e4


def make_diagnostics(zmodel, t, u, v, th, t_coral_night_raw=None, t_coral_3h_raw=None):
    ref = xr.open_dataset(REF_NC)
    zr = ref["level"].values.astype(float)
    tr = (ref["th"] * ref["exner"]).values.astype(float)
    ur, vr = ref["u"].values.astype(float), ref["v"].values.astype(float)
    ref.close()

    n2_new = uniform_filter1d(n2_from_t(zmodel, t), 5)
    n2_old = uniform_filter1d(n2_from_t(zr, tr), 5)
    spd, spr = np.hypot(u, v), np.hypot(ur, vr)
    tsat_new = t * np.sqrt(np.clip(n2_new, 1e-8, None)) * spd / G
    tsat_old = tr * np.sqrt(np.clip(n2_old, 1e-8, None)) * spr / G

    zk = zmodel / 1e3
    box = {"boxstyle": "round", "lw": 0.67, "facecolor": "white", "edgecolor": "black"}
    fig, ax = plt.subplots(1, 4, figsize=(13.5, 6.2), constrained_layout=True)
    for a in ax:
        a.axhspan(56, 60, color="gold", alpha=0.22, lw=0)
        a.set_ylim(48, 74)

    if t_coral_night_raw is not None:
        ax[0].plot(t_coral_night_raw, zk, color="tab:blue", lw=0.9, alpha=0.85,
                   zorder=1, label="CORAL nightly (raw)")
    if t_coral_3h_raw is not None:
        ax[0].plot(t_coral_3h_raw, zk, color="tab:green", lw=0.9, alpha=0.85,
                   zorder=1, label="CORAL 3 h (raw)")
    ax[0].plot(tr, zr / 1e3, "k", lw=1.6, zorder=3, label="current")
    ax[0].plot(t, zk, "tab:red", lw=1.6, zorder=3, label="rebuilt (20 km, nightly)")
    ax[0].set_xlabel("temperature / K"); ax[0].set_ylabel("altitude / km")
    ax[0].set_xlim(198, 262); ax[0].legend(loc="lower left", fontsize=7)
    ax[0].text(0.04, 0.93, "T", transform=ax[0].transAxes, weight="bold", bbox=box)

    ax[1].plot(n2_old * 1e4, zr / 1e3, "k", lw=1.6)
    ax[1].plot(n2_new * 1e4, zk, "tab:red", lw=1.6)
    for prof, zc, c in [(n2_old, zr, "k"), (n2_new, zmodel, "tab:red")]:
        zl, n2l = _least_stable(zc, uniform_filter1d(prof, 5))
        ax[1].scatter([n2l], [zl], color=c, s=30, edgecolor="k", lw=0.5, zorder=5)
        ax[1].annotate(f"{zl:.0f} km", (n2l, zl), textcoords="offset points",
                       xytext=(4, 4), fontsize=7, color=c)
    ax[1].set_xlabel(r"$N^2$ / $10^{-4}\,$s$^{-2}$"); ax[1].set_xlim(1.5, 4.5)
    ax[1].text(0.04, 0.93, "stability", transform=ax[1].transAxes, weight="bold", bbox=box)

    ax[2].axvline(0, color="0.7", lw=0.6)
    ax[2].plot(spr, zr / 1e3, "k", lw=1.6, label="|U| current")
    ax[2].plot(spd, zk, "tab:red", lw=1.6, label="|U| rebuilt")
    ax[2].plot(u, zk, "tab:blue", lw=1.0, label="u"); ax[2].plot(v, zk, "tab:orange", lw=1.0, label="v")
    ax[2].set_xlabel(r"wind / m$\,$s$^{-1}$"); ax[2].set_xlim(-20, 125)
    ax[2].legend(loc="lower left", fontsize=7)
    ax[2].text(0.04, 0.93, "wind", transform=ax[2].transAxes, weight="bold", bbox=box)

    ax[3].plot(tsat_old, zr / 1e3, "k", lw=1.6, label="current")
    ax[3].plot(tsat_new, zk, "tab:red", lw=1.6, label="rebuilt")
    ax[3].set_xlabel(r"$T'_\mathrm{sat}=\bar T N|U|/g$ / K"); ax[3].set_xlim(0, 55)
    ax[3].legend(loc="upper right", fontsize=8)
    ax[3].text(0.04, 0.93, "overturning\nthreshold", transform=ax[3].transAxes, weight="bold",
               bbox=box, va="top")

    out_fig = OUT_FIG.with_name(OUT_FIG.stem + _knob_tag() + OUT_FIG.suffix)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, facecolor="w", bbox_inches="tight")
    print(f"wrote {out_fig}")
    zl_old, _ = _least_stable(zr, n2_old)
    zl_new, _ = _least_stable(zmodel, n2_new)
    print(f"least-stable layer: current {zl_old:.1f} km  ->  rebuilt {zl_new:.1f} km")


if __name__ == "__main__":
    main()
