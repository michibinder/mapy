"""Build the 3D transient background for the ERA5-nested PMAP (sam) runs.

3D counterpart of build_background_1d.py: for every boundary-condition timestamp it blends
ERA5 (lower atmosphere) with JAWARA (mesosphere), balances the state, interpolates onto the
run's terrain-following grid, and writes PMAP-readable input files:

    <outdir>/input_<N>.nc   one per timestamp, dims (time=1, x, y, z), float32,
                            vars xcr/ycr/zcr + uvelx/uvely/uvelz/density/theta_total/
                            exner_total (+ uvel[xy]_ambient copies), exactly the schema of
                            PMAP's own data_<N>.nc output (PMAP interpolates NOTHING on read).

Pipeline per timestamp (all on a uniform working z grid, PMAP-local Cartesian x/y):
  1. horizontal bilinear interpolation of ERA5-int and JAWARA fields onto the model grid
     (winds rotated from geographic east/north to grid components via numerically computed
     local basis vectors -- exact for the conformal Lambert projection);
  2. vertical blend ERA5 -> JAWARA with a 5th-order smooth ramp over BLEND_KM (knob);
  3. balancing (BALANCE_MODE knob):
       "adjust_theta" (default) -- keep the blended winds, apply the MINIMAL mass-field
           correction restoring f0-thermal-wind consistency: the data state is already
           ~balanced with the local f(lat), so per level a correction theta'_c is solved
           (Neumann-Poisson least squares) from the (F0 - f(x,y))-weighted shear of the
           vertically LOW-PASSED wind (TW_SHEAR_SMOOTH_M), windowed by the BL/top tapers
           (TW_BL_TAPER_M / TW_TOP_TAPER_M -- frictional, GW and tidal shear carry no
           thermal-wind signal), and added to the hydrostatic data theta. Stability floor
           N2 >= N2_MIN swept upward; exner = data exner at the BL-taper top + the
           analogous (F0-f) geostrophic correction, integrated hydrostatically up/down.
           Winds survive exactly; the real T'/N2 structure survives up to the O(f0/f-1)
           correction. (A FULL re-derivation of theta' from F0 thermal wind doubles the
           pole-subtropics contrast and drives exner negative aloft -- see the function
           docstring.)
       "adjust_wind" -- keep the blended thermodynamics: hydrostatic integration of exner
           from the ERA5 surface pressure + blended T per column, then replace the winds by
           the geostrophic wind of that mass field with constant F0. Ageostrophic flow
           (jet streaks, fronts) is lost; wind errors ~ (f_local-F0)/F0 where the real
           atmosphere was balanced.
       "none" -- per-column hydrostatics only, winds kept: thermodynamics and wind are each
           self-consistent but not in thermal-wind balance with F0 (model adjusts by itself).
  4. vertical interpolation of the balanced columns onto zcr (read from the topo/grid file);
  5. write input_<N>.nc.

Run with the post-venv (CWD = this dir). A single timestamp takes a few minutes; the full
series runs sequentially:
    /home/b/b309199/venvs/post-venv/bin/python build_background_3d.py [N_ONLY]
optional N_ONLY = build only input_<N_ONLY>.nc (quick test: N_ONLY=0).
"""
import sys
from pathlib import Path

import numpy as np
import pyproj
import scipy.fft
import scipy.ndimage
import xarray as xr

# ======================================================================================
# CONFIG -- every lever lives here
# ======================================================================================
ERA5_FILE = "/work/bd0620/b309199/data/era5-data/sam3d_180616-ml-int.nc"
JAWARA_FILE = "/work/bd0620/b309199/mapy/data/jawara/jawara_sam_180616.nc"
GRID_FILE = "/work/bd0620/b309199/mapy/data/pmap-topos/sam_4000m_1100x900_fullorog_z128.nc"
OUTDIR = "/work/bd0620/b309199/data/nests/sam_180616_era5_input_z128_beta_moist"

# 2018-06-16 case, first day only: input_0..input_24 = 00:00..24:00 UTC, i.e. 25 files for a
# 24 h run (model_time 0 = 06-16T00:00, time_dependent_bc.nstep = 24). ERA5 + JAWARA both
# reach 06-18T00:00, so day 2 can be appended later as input_25..input_48 without a refetch.
T0_UTC = "2018-06-16T00:00"
N_FILES = 25
DT_BC_S = 3600.0

# ERA5 -> JAWARA transition (m): 5th-order smooth ramp, ERA5-pure below, JAWARA-pure above.
# BLEND_M_WIND = None -> winds use BLEND_M too; set e.g. (20e3, 40e3) to hand the jet to
# JAWARA lower down (ERA5 winds are sponge-damped above ~45 km).
BLEND_M = (40e3, 60e3)
BLEND_M_WIND = None

# "adjust_theta" (default) | "adjust_wind" | "none"  -- see module docstring.
BALANCE_MODE = "adjust_theta"

# Coriolis of the TARGET RUN -- MUST match the run config's model.beta_plane:
#   "fplane"    -> f_model = F0 = 2*Omega*sin(angle0)                  (beta_plane: False)
#   "betaplane" -> f_model = F0 + BETA0*(y - y_center)                 (beta_plane: True)
# (PMAP forces.py: coriolis_z = fcr0 + beta_plane*beta0*(ycr - ycr0), beta0 =
# 2*Omega*cos(angle0)/radius_sphere, radius_sphere = 6368 km, ycr0 = domain centre.)
# The adjust_theta correction targets (f_model - f_real(lat)); on the beta plane the
# residual shrinks to the meridian-convergence/curvature part.
CORIOLIS_MODE = "betaplane"
OMEGA2 = 1.4584e-4
ANGLE0 = -55.0
RADIUS_SPHERE = 6368.0e3
F0 = OMEGA2 * np.sin(np.deg2rad(ANGLE0))
BETA0 = OMEGA2 * np.cos(np.deg2rad(ANGLE0)) / RADIUS_SPHERE

UVELZ_CONST = 0.0
WRITE_AMBIENT_WINDS = True

# Water vapour: interpolate ERA5 specific humidity q -> vapour mixing ratio r = q/(1-q),
# tapered to zero across the ERA5->JAWARA blend (the mesosphere is dry; JAWARA carries no q).
# Written as `rvapour` (PMAP's TimeDependentBC.rvapour_varname_in_file default) ALONGSIDE the
# unchanged, dry-balanced theta/density/exner -- so a moist run's dynamical state equals the
# dry beta run's exactly, plus vapour. The mass field is NOT re-derived for virtual-temperature
# effects (q <~ 2% -> <2% density bias, which the model relaxes away in the first steps).
WRITE_RVAPOUR = True

# Static-stability floor applied to the balanced theta (adjust_theta mode): N2 >= N2_MIN
# everywhere, enforced by an upward sweep. 0 disables. Tropospheric N2 ~ 1e-4, mesospheric
# ~4e-4; 1e-5 only removes genuine instabilities introduced by the f0 thermal-wind fit.
N2_MIN = 1.0e-5

# The thermal-wind fit must see only the BALANCED flow: gravity-wave shear (small vertical
# scales, kept in the winds on purpose) and frictional boundary-layer shear carry no
# thermal-wind temperature signal and would destroy the stratification if inverted.
# TW_SHEAR_SMOOTH_M = Gaussian sigma [m] of the vertical low-pass applied to the winds
# before taking the shear target; TW_BL_TAPER_M = quintic ramp (zero target below, full
# above) removing the boundary-layer shear; exner is anchored at the ramp top.
TW_SHEAR_SMOOTH_M = 3000.0
TW_BL_TAPER_M = (2000.0, 5000.0)

# Same argument at the top: above ~70 km the (JAWARA) shear is tide/GW-dominated -- not
# balanced flow -- and inverting it at theta_ref ~ 8000 K produced theta' extremes that
# drove the hydrostatic exner integration through zero (negative pi -> NaN density).
# The thermal-wind target is ramped to zero over TW_TOP_TAPER_M; above it theta is the
# plain hydrostatic data state (plus the N2_MIN floor).
# The 70-85 km range is set by that PHYSICS, not by the sponge: on the 128 km grid the top
# absorber starts at 100 km, so 70-85 km is now free atmosphere and the taper deliberately
# stays put. Raising it into the new sponge would be actively dangerous -- theta_ref keeps
# growing with height (~60000 K at 128 km), which is exactly the regime that drove exner
# negative before.
TW_TOP_TAPER_M = (70000.0, 85000.0)

# Working grid must span the run's zcr: 128 km for the sam_180616 grid (was 90 km). ERA5
# stops at its own z_top (80 km) and source_to_work clamps above it, but the ERA5 blend
# weight is exactly 0 above BLEND_M[1]=60 km, so nothing of that clamp survives; JAWARA
# carries 60-128 km on its own (cutout valid to 139.6 km).
WORK_DZ = 100.0
WORK_ZTOP = 128000.0

G, RD, CP, P0 = 9.80616, 287.05, 1004.0, 1.0e5
KAPPA = RD / CP


# ======================================================================================
# grid template + horizontal interpolation machinery
# ======================================================================================
def tf(zeta):
    """5th-order polynomial transition 0 -> 1 (same as topo taper / 1D builder ramps)."""
    zeta = np.clip(zeta, 0.0, 1.0)
    return zeta**3 * (10.0 - 15.0 * zeta + 6.0 * zeta**2)


class GridTemplate:
    """Model grid (from the topo file): lat/lon of every point, wind-rotation basis, zcr."""

    def __init__(self, path):
        ds = xr.open_dataset(path)
        self.ds = ds
        self.nx, self.ny = ds.sizes["x"], ds.sizes["y"]
        self.nz = ds.sizes["z"]
        self.x_m = ds["x_m"].values
        self.y_m = ds["y_m"].values
        self.zcr = ds["zcr"].values
        self.xcr = ds["xcr"].values
        self.ycr = ds["ycr"].values
        self.z_comp = ds["z"].values

        proj = pyproj.Proj(ds.attrs["proj4"])
        cx, cy = ds.attrs["center_x_m"], ds.attrs["center_y_m"]
        px = cx + self.x_m[:, None] + 0.0 * self.y_m[None, :]
        py = cy + 0.0 * self.x_m[:, None] + self.y_m[None, :]
        self.lon, self.lat = proj(px, py, inverse=True)

        dl = 1e-4
        ex, ey = proj(self.lon + dl, self.lat)
        wx, wy = proj(self.lon - dl, self.lat)
        nxc, nyc = proj(self.lon, self.lat + dl)
        sxc, syc = proj(self.lon, self.lat - dl)
        e = np.stack([ex - wx, ey - wy])
        n = np.stack([nxc - sxc, nyc - syc])
        self.e_hat = (e / np.linalg.norm(e, axis=0)).astype(np.float32)
        self.n_hat = (n / np.linalg.norm(n, axis=0)).astype(np.float32)

    def rotate_wind(self, u_east, v_north):
        """Geographic (east, north) components -> grid (x, y) components."""
        ex, ey, nx, ny = self.e_hat[0], self.e_hat[1], self.n_hat[0], self.n_hat[1]
        if u_east.ndim == 3:
            ex, ey, nx, ny = (a[..., np.newaxis] for a in (ex, ey, nx, ny))
        ux = u_east * ex + v_north * nx
        uy = u_east * ey + v_north * ny
        return ux, uy


class HorizontalInterp:
    """Precomputed bilinear weights from a regular (lat, lon) grid onto the model points.

    Target coordinates are clipped into the source box (edge-hold outside) -- covers the
    current too-small JAWARA cutout; becomes a no-op once the full-envelope cutout exists.
    """

    def __init__(self, lat_src, lon_src, lat_tgt, lon_tgt):
        lat_src = np.asarray(lat_src, dtype=np.float64)
        lon_src = np.asarray(lon_src, dtype=np.float64)
        self.flip = lat_src[0] > lat_src[-1]
        if self.flip:
            lat_src = lat_src[::-1]
        lon_t = np.mod(lon_tgt, 360.0)
        lon_s = np.mod(lon_src, 360.0)
        if not np.all(np.diff(lon_s) > 0):
            raise ValueError("source longitudes not monotonic after mod 360")

        la = np.clip(lat_tgt, lat_src[0], lat_src[-1])
        lo = np.clip(lon_t, lon_s[0], lon_s[-1])
        i = np.clip(np.searchsorted(lat_src, la) - 1, 0, len(lat_src) - 2)
        j = np.clip(np.searchsorted(lon_s, lo) - 1, 0, len(lon_s) - 2)
        wi = (la - lat_src[i]) / (lat_src[i + 1] - lat_src[i])
        wj = (lo - lon_s[j]) / (lon_s[j + 1] - lon_s[j])
        self.i, self.j = i, j
        self.w00 = (1 - wi) * (1 - wj)
        self.w01 = (1 - wi) * wj
        self.w10 = wi * (1 - wj)
        self.w11 = wi * wj

    def __call__(self, field_latlon):
        """field (lat, lon) -> field on target points (target shape)."""
        f = field_latlon[::-1, :] if self.flip else field_latlon
        return (self.w00 * f[self.i, self.j] + self.w01 * f[self.i, self.j + 1]
                + self.w10 * f[self.i + 1, self.j] + self.w11 * f[self.i + 1, self.j + 1])


# ======================================================================================
# balance building blocks
# ======================================================================================
def poisson_neumann(rhs, dx, dy):
    """Solve laplace(phi) = rhs with homogeneous Neumann BCs via DCT-II; mean(phi) = 0.

    Least-squares integration of a target gradient field; the O(edge) error lives in the
    lateral absorber zones.
    """
    nx, ny = rhs.shape
    r = scipy.fft.dctn(rhs, type=2, norm="ortho")
    kx = (2.0 * np.cos(np.pi * np.arange(nx) / nx) - 2.0) / dx**2
    ky = (2.0 * np.cos(np.pi * np.arange(ny) / ny) - 2.0) / dy**2
    lam = kx[:, None] + ky[None, :]
    lam[0, 0] = 1.0
    r /= lam
    r[0, 0] = 0.0
    return scipy.fft.idctn(r, type=2, norm="ortho")


def hydrostatic_ref(t_ref, psurf_ref, dz):
    """Mean-profile hydrostatics (trapezoid of 1/T, as in build_background_1d.hydrostatic)."""
    p = np.empty_like(t_ref)
    p[0] = psurf_ref
    for k in range(1, len(t_ref)):
        integrand = 0.5 * (1.0 / t_ref[k - 1] + 1.0 / t_ref[k])
        p[k] = p[k - 1] * np.exp(-G / RD * dz * integrand)
    exner = (p / P0) ** KAPPA
    return p, t_ref / exner, exner


def hydrostatic_columns(t3d, psurf2d, dz):
    """Column-wise hydrostatic integration (vectorized over x, y) -> p(x,y,z)."""
    p = np.empty_like(t3d)
    p[:, :, 0] = psurf2d
    for k in range(1, t3d.shape[2]):
        integrand = 0.5 * (1.0 / t3d[:, :, k - 1] + 1.0 / t3d[:, :, k])
        p[:, :, k] = p[:, :, k - 1] * np.exp(-G / RD * dz * integrand)
    return p


def ddx(f, dx):
    return np.gradient(f, dx, axis=0)


def ddy(f, dy):
    return np.gradient(f, dy, axis=1)


def div_faces(gx, gy, dx, dy):
    """Mimetic divergence of a target gradient field: face-averaged fluxes, zero flux
    through the domain boundary (matches the DCT-II Neumann Laplacian exactly, so
    poisson_neumann returns the exact discrete least-squares potential)."""
    fx = np.zeros((gx.shape[0] + 1, gx.shape[1]), dtype=gx.dtype)
    fx[1:-1] = 0.5 * (gx[1:] + gx[:-1])
    fy = np.zeros((gy.shape[0], gy.shape[1] + 1), dtype=gy.dtype)
    fy[:, 1:-1] = 0.5 * (gy[:, 1:] + gy[:, :-1])
    return (fx[1:] - fx[:-1]) / dx + (fy[:, 1:] - fy[:, :-1]) / dy


def balance_adjust_theta(u, v, t_blend, psurf, df2d, grid_dx, dz):
    """Keep winds; adjust (theta, exner, density) to f0-thermal-wind via a MINIMAL
    correction of the data state.

    The blended data mass field is already ~balanced with the LOCAL f(lat); only the
    f-plane mismatch needs fixing. Per level, a correction theta'_c is solved from the
    Neumann-Poisson least-squares integration of
        grad(theta'_c) = W(z) * (theta_ref*(F0 - f(x,y))/g) * (dv~/dz, -du~/dz),
    with (u~,v~) the vertically low-passed winds (TW_SHEAR_SMOOTH_M) and W(z) the
    boundary-layer/top taper (TW_BL_TAPER_M / TW_TOP_TAPER_M: frictional and tidal/GW
    shear carry no thermal-wind signal). The correction vanishes at the domain-centre
    latitude and stays a fraction of the data gradients at the edges, so exner remains
    close to the (positive) data exner. A full re-derivation of theta' from F0-thermal
    wind was tried first and FAILED: it roughly doubles the pole-subtropics contrast
    (f0 inflation + level-mean-zero) and drives the hydrostatic exner integration
    negative above ~84 km in a quarter of the columns.

    exner: anchored at the BL-taper top by the analogous (F0 - f) geostrophic correction
    Poisson added to the data exner, then integrated hydrostatically up/down with the
    corrected theta (no vertical differentiation of solved fields). Stability floor
    N2 >= N2_MIN swept upward before the integration.
    """
    nz = u.shape[2]
    p_data = hydrostatic_columns(t_blend.astype(np.float64), psurf, dz)
    pi_data = ((p_data / P0) ** KAPPA).astype(np.float32)
    del p_data
    theta = (t_blend / pi_data)
    t_ref = t_blend.mean(axis=(0, 1)).astype(np.float64)
    _, th_ref, _ = hydrostatic_ref(t_ref, float(np.mean(psurf)), dz)

    z_work = np.arange(nz) * dz
    taper = tf((z_work - TW_BL_TAPER_M[0]) / (TW_BL_TAPER_M[1] - TW_BL_TAPER_M[0])) * (
        1.0 - tf((z_work - TW_TOP_TAPER_M[0]) / (TW_TOP_TAPER_M[1] - TW_TOP_TAPER_M[0]))
    )
    df = np.squeeze(np.asarray(df2d)).astype(np.float32)

    us = scipy.ndimage.gaussian_filter1d(u, sigma=TW_SHEAR_SMOOTH_M / dz, axis=2, mode="nearest")
    dudz = np.gradient(us, dz, axis=2)
    del us
    vs = scipy.ndimage.gaussian_filter1d(v, sigma=TW_SHEAR_SMOOTH_M / dz, axis=2, mode="nearest")
    dvdz = np.gradient(vs, dz, axis=2)
    del vs

    for k in range(nz):
        coef = np.float32(taper[k] * th_ref[k] / G) * df
        rhs = div_faces(coef * dvdz[:, :, k], -coef * dudz[:, :, k], grid_dx, grid_dx)
        theta[:, :, k] += poisson_neumann(rhs, grid_dx, grid_dx)
    del dudz, dvdz

    if N2_MIN > 0.0:
        n_touched = 0
        for k in range(1, nz):
            floor = theta[:, :, k - 1] * (1.0 + np.float32(N2_MIN * dz / G))
            low = theta[:, :, k] < floor
            n_touched += int(low.sum())
            np.maximum(theta[:, :, k], floor, out=theta[:, :, k])
        print(f"      stability floor touched {n_touched / theta.size * 100:.2f}% of cells",
              flush=True)

    ka = int(round(TW_BL_TAPER_M[1] / dz))
    coefa = df / np.float32(CP * th_ref[ka])
    rhsa = div_faces(coefa * v[:, :, ka], -coefa * u[:, :, ka], grid_dx, grid_dx)
    pia = pi_data[:, :, ka] + poisson_neumann(rhsa, grid_dx, grid_dx)
    del pi_data

    pi = np.empty_like(u)
    pi[:, :, ka] = pia
    for k in range(ka + 1, nz):
        integ = 0.5 * (1.0 / theta[:, :, k - 1] + 1.0 / theta[:, :, k])
        pi[:, :, k] = pi[:, :, k - 1] - np.float32(G / CP * dz) * integ
    for k in range(ka - 1, -1, -1):
        integ = 0.5 * (1.0 / theta[:, :, k + 1] + 1.0 / theta[:, :, k])
        pi[:, :, k] = pi[:, :, k + 1] + np.float32(G / CP * dz) * integ

    if float(pi.min()) <= 0.0:
        raise RuntimeError(f"exner went non-positive (min {pi.min():.4g}) -- balance invalid")
    print(f"      exner(top) min {pi[:, :, -1].min():.4f} (ref-like ~0.02)", flush=True)
    p = P0 * pi ** (1.0 / KAPPA)
    rho = p / (RD * (theta * pi))
    return theta, pi, rho


def balance_adjust_wind(t_blend, psurf, grid_dx, dz):
    """Keep thermodynamics; replace winds by the F0-geostrophic wind of the mass field."""
    p = hydrostatic_columns(t_blend, psurf, dz)
    pi = (p / P0) ** KAPPA
    theta = t_blend / pi
    rho = p / (RD * t_blend)
    u = np.empty_like(t_blend)
    v = np.empty_like(t_blend)
    for k in range(t_blend.shape[2]):
        cpth = CP * theta[:, :, k]
        u[:, :, k] = -cpth / F0 * ddy(pi[:, :, k], grid_dx)
        v[:, :, k] = cpth / F0 * ddx(pi[:, :, k], grid_dx)
    return theta, pi, rho, u, v


def balance_none(t_blend, psurf, dz):
    """Column hydrostatics only."""
    p = hydrostatic_columns(t_blend, psurf, dz)
    pi = (p / P0) ** KAPPA
    return t_blend / pi, pi, p / (RD * t_blend)


# ======================================================================================
# vertical interpolation working grid -> zcr
# ======================================================================================
def to_zcr(field_work, zcr, dz):
    """Linear interpolation from the uniform working grid onto zcr (vectorized)."""
    k = zcr / dz
    k0 = np.clip(np.floor(k).astype(np.int32), 0, field_work.shape[2] - 2)
    w = np.clip(k - k0, 0.0, 1.0).astype(field_work.dtype)
    lo = np.take_along_axis(field_work, k0, axis=2)
    hi = np.take_along_axis(field_work, k0 + 1, axis=2)
    return (lo * (1.0 - w) + hi * w).astype(np.float32)


# ======================================================================================
# main
# ======================================================================================
def load_sources(grid):
    """Open ERA5/JAWARA, build interpolators + the shared working-grid level mapping."""
    era = xr.open_dataset(ERA5_FILE)
    jaw = xr.open_dataset(JAWARA_FILE)
    hi_e = HorizontalInterp(era.latitude.values, era.longitude.values, grid.lat, grid.lon)
    hi_j = HorizontalInterp(jaw.latitude.values, jaw.longitude.values, grid.lat, grid.lon)
    z_work = np.arange(0.0, WORK_ZTOP + WORK_DZ / 2, WORK_DZ)
    return era, jaw, hi_e, hi_j, z_work


def source_to_work(ds, hi, varname, tstamp, z_work, grid_shape):
    """One variable at one timestamp -> (nx, ny, nz_work), horizontal interp + z clamp."""
    arr = ds[varname].sel(time=tstamp).transpose("level", "latitude", "longitude").values
    z_src = ds["level"].values.astype(np.float64)
    out = np.empty(grid_shape + (len(z_work),), dtype=np.float32)
    kk = np.clip(np.searchsorted(z_src, z_work) - 1, 0, len(z_src) - 2)
    for m, zt in enumerate(z_work):
        k = kk[m]
        wz = np.clip((zt - z_src[k]) / (z_src[k + 1] - z_src[k]), 0.0, 1.0)
        out[:, :, m] = hi(arr[k] * (1 - wz) + arr[k + 1] * wz)
    return out


def blend_weight(z_work, blend_m):
    return tf((z_work - blend_m[0]) / (blend_m[1] - blend_m[0])).astype(np.float32)


def build_one(n, grid, era, jaw, hi_e, hi_j, z_work):
    tstamp = np.datetime64(T0_UTC) + np.timedelta64(int(n * DT_BC_S), "s")
    print(f"[t{n:02d}] {tstamp}", flush=True)
    shape = (grid.nx, grid.ny)
    dz = WORK_DZ
    grid_dx = float(grid.x_m[1] - grid.x_m[0])

    w_t = blend_weight(z_work, BLEND_M)
    w_u = blend_weight(z_work, BLEND_M_WIND or BLEND_M)

    t_e = source_to_work(era, hi_e, "t", tstamp, z_work, shape)
    t_j = source_to_work(jaw, hi_j, "t", tstamp, z_work, shape)
    t_blend = (1 - w_t) * t_e + w_t * t_j
    del t_e, t_j

    ue = source_to_work(era, hi_e, "u", tstamp, z_work, shape)
    uj = source_to_work(jaw, hi_j, "u", tstamp, z_work, shape)
    u_geo = (1 - w_u) * ue + w_u * uj
    del ue, uj
    ve = source_to_work(era, hi_e, "v", tstamp, z_work, shape)
    vj = source_to_work(jaw, hi_j, "v", tstamp, z_work, shape)
    v_geo = (1 - w_u) * ve + w_u * vj
    del ve, vj
    u, v = grid.rotate_wind(u_geo, v_geo)
    del u_geo, v_geo

    psurf = hi_e(era["p"].sel(time=tstamp).isel(level=0).values)

    if BALANCE_MODE == "adjust_theta":
        f_real = (OMEGA2 * np.sin(np.deg2rad(grid.lat))).astype(np.float32)
        if CORIOLIS_MODE == "betaplane":
            f_model = np.float32(F0) + np.float32(BETA0) * np.broadcast_to(
                grid.y_m[np.newaxis, :], f_real.shape).astype(np.float32)
        elif CORIOLIS_MODE == "fplane":
            f_model = np.full_like(f_real, np.float32(F0))
        else:
            raise ValueError(f"unknown CORIOLIS_MODE {CORIOLIS_MODE}")
        theta, pi, rho = balance_adjust_theta(u, v, t_blend, psurf, f_model - f_real, grid_dx, dz)
        del t_blend
    elif BALANCE_MODE == "adjust_wind":
        theta, pi, rho, u, v = balance_adjust_wind(t_blend, psurf, grid_dx, dz)
        del t_blend
    elif BALANCE_MODE == "none":
        theta, pi, rho = balance_none(t_blend, psurf, dz)
        del t_blend
    else:
        raise ValueError(f"unknown BALANCE_MODE {BALANCE_MODE}")

    n_unstable = int((np.diff(theta[:, ::37, :], axis=2) < 0).sum())
    print(f"      theta sfc..top {theta[..., 0].mean():.1f}..{theta[..., -1].mean():.0f} K, "
          f"unstable cells (subsampled) {n_unstable}, "
          f"|u|max {np.abs(u).max():.0f} m/s", flush=True)

    data = {}
    data["uvelx"] = to_zcr(u, grid.zcr, dz)
    del u
    data["uvely"] = to_zcr(v, grid.zcr, dz)
    del v
    data["uvelz"] = np.full((grid.nx, grid.ny, grid.nz), UVELZ_CONST, dtype=np.float32)
    data["density"] = to_zcr(rho, grid.zcr, dz)
    del rho
    data["theta_total"] = to_zcr(theta, grid.zcr, dz)
    del theta
    data["exner_total"] = to_zcr(pi, grid.zcr, dz)
    del pi
    if WRITE_RVAPOUR:
        q_e = source_to_work(era, hi_e, "q", tstamp, z_work, shape)   # ERA5 specific humidity [kg/kg]
        r_e = q_e / np.maximum(1.0 - q_e, np.float32(1e-6))           # -> vapour mixing ratio
        r_work = np.clip((1.0 - w_t) * r_e, 0.0, None).astype(np.float32)  # taper to 0 in JAWARA band
        del q_e, r_e
        data["rvapour"] = to_zcr(r_work, grid.zcr, dz)
        del r_work
        print(f"      rvapour max {data['rvapour'].max() * 1e3:.2f} g/kg", flush=True)
    for name, arr in data.items():
        if not np.isfinite(arr).all():
            raise RuntimeError(f"non-finite values in {name} -- refusing to write input_{n}")
    if float(data["exner_total"].min()) <= 0.0 or float(data["density"].min()) <= 0.0:
        raise RuntimeError(f"non-positive exner/density -- refusing to write input_{n}")

    if WRITE_AMBIENT_WINDS:
        data["uvelx_ambient"] = data["uvelx"]
        data["uvely_ambient"] = data["uvely"]

    out = xr.Dataset(
        {k: (("time", "x", "y", "z"), val[np.newaxis, ...]) for k, val in data.items()},
        coords={
            "time": np.array([n * DT_BC_S]),
            "x": grid.x_m.astype(np.float32),
            "y": grid.y_m.astype(np.float32),
            "z": grid.z_comp.astype(np.float32),
        },
    )
    out["xcr"] = (("x", "y"), grid.xcr)
    out["ycr"] = (("x", "y"), grid.ycr)
    out["zcr"] = (("x", "y", "z"), grid.zcr)
    out.attrs.update(
        valid_time_utc=str(tstamp), balance_mode=BALANCE_MODE, coriolis_mode=CORIOLIS_MODE,
        blend_m=str(BLEND_M), blend_m_wind=str(BLEND_M_WIND or BLEND_M),
        f0=float(F0), era5=ERA5_FILE, jawara=JAWARA_FILE, grid=GRID_FILE,
    )
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    fname = f"{OUTDIR}/input_{n}.nc"
    out.to_netcdf(fname)
    print(f"      -> {fname}", flush=True)


def main():
    sel = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if "-" in arg:
            lo, hi = arg.split("-")
            sel = range(int(lo), int(hi) + 1)
        else:
            sel = [int(arg)]
    grid = GridTemplate(GRID_FILE)
    era, jaw, hi_e, hi_j, z_work = load_sources(grid)
    for n in range(N_FILES):
        if sel is not None and n not in sel:
            continue
        build_one(n, grid, era, jaw, hi_e, hi_j, z_work)


if __name__ == "__main__":
    main()
