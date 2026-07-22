#!/usr/bin/env python
"""Spectral / scale-decomposition diagnostic of a horizontal PMAP z-slice.

Maps (w, full domain, vik-white cmap):
  (a) full field   (b) high-pass lambda<CUT   (c) low-pass lambda>CUT (~IFS)
  2D isotropic Butterworth, high-pass = exact complement of low-pass (b+c=a).
  Terrain (coastline) contours + the analysis-window rectangle overlaid.

Panel (d) PSD, computed from data INSIDE the analysis window only:
  - w 1D PSD on the y=0 transect (along x) and the x=0 transect (along y); thin lines.
  - 2D radial spectrum for w (thick solid) and theta' (thick dashed): LINE = the
    variance-preserving omnidirectional E(k) (keeps the spectral valley). The 25-75
    IQR band (directional spread across azimuthal sectors) is shown for w only.
  Per-field normalization (1D by its peak; radial by E(k) peak).
  x-axis linear 20-200 km; linear-y; y-axis labels on the right.
  4 equal-size panels: maps use aspect='auto'; colorbar attached to all 4 axes (top).

Usage: python spectral_zslice.py [SIM] [TIDX] [ZIDX] [WINDOW]
  WINDOW = 'sponge' (default; domain minus absorber widthx/widthy) | 'full' | <half-width km>
"""
import sys

sys.path.append('/work/bd0620/b309199/mapy/src')

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle
from scipy import signal

plt.style.use('/work/bd0620/b309199/mapy/src/latex_default.mplstyle')
import cmaps
import plt_helper

# ----------------------------------------------------------------------------- config
SIM  = sys.argv[1] if len(sys.argv) > 1 else 'darwin_240718_400m_r1'
TIDX = int(sys.argv[2]) if len(sys.argv) > 2 else -1
ZIDX = int(sys.argv[3]) if len(sys.argv) > 3 else 1

LAMBDA_CUT = 70e3                 # m   filter cutoff wavelength
BW_ORDER   = 4
BOX_KM     = 200                  # half-width of the +-BOX_KM analysis rectangle (all PSDs use ONLY data inside it)
XLIM_KM    = (20, 400)            # panel-d wavelength axis (linear), extended toward the data limit
YMAX_LINY  = 120                  # fixed panel-d y-limit (shared across time steps for comparison; sharp 1D transect spikes may clip)
PCT        = (25, 75)             # radial-band percentiles across azimuthal sectors
OUTDIR     = '/work/bd0620/b309199/mapy/data/figures'

CMAP_W = cmaps.get_vik_white_cmap()
CLEV_W, CLEV_W_LABELS = plt_helper.get_colormap_bins_and_labels(max_level=32)   # power-of-2 (log) bins; w std~4 m/s

# ----------------------------------------------------------------------------- load
ds = xr.open_dataset(f"/scratch/b/b309199/{SIM}/slices_z.nc")
x  = ds.x.values.astype(float)
y  = ds.y.values.astype(float)
dx = float(x[1] - x[0])
dy = float(y[1] - y[0])
nx, ny = x.size, y.size
zkm  = float(ds.z.values[ZIDX]) / 1e3
tstr = np.datetime_as_string(ds.time.values[TIDX], unit='s')

w = ds.uvelz.isel(time=TIDX, z=ZIDX).values.astype(float)             # (x,y)
w = w - w.mean()
th = ds.theta_total.isel(time=TIDX, z=ZIDX).values.astype(float)
tprime = th - th.mean()
topo = ds.zcr.isel(time=0, z=0).values.astype(float)                  # surface elev (m)
TOPO_LEV = np.linspace(20, 0.5 * np.nanmax(topo), 2)                  # coastline + half-peak, as pmap_slc_lid (itopo==1)

# ----------------------------------------------------------------------------- 2D Butterworth on w (maps, full domain)
kx = np.fft.fftfreq(nx, d=dx)
ky = np.fft.fftfreq(ny, d=dy)
KR = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
Hlow = 1.0 / (1.0 + (KR / (1.0 / LAMBDA_CUT)) ** (2 * BW_ORDER))
W = np.fft.fft2(w)
w_low = np.real(np.fft.ifft2(W * Hlow))
w_high = w - w_low

# ----------------------------------------------------------------------------- spectra analysis window
def _sponge_box():
    """Interior box = domain minus the absorber (sponge) widths, read from the run config."""
    import re
    cfg = open(f"/scratch/b/b309199/{SIM}/config.yml").read()

    def num(k, d=0.0):
        m = re.search(rf'{k}:\s*([\d.eE+-]+)', cfg)
        return float(m.group(1)) if m else d

    def on(k):
        return bool(re.search(rf'{k}:\s*[Tt]rue', cfg))

    wx = num('widthx') if on('xboundaries') else 0.0
    wy = num('widthy') if on('yboundaries') else 0.0
    return x[-1] - wx, y[-1] - wy


BOX_ARG = (sys.argv[4] if len(sys.argv) > 4 else 'sponge').lower()
FULL = BOX_ARG in ('full', 'all')
if FULL:
    ixb, iyb = np.arange(nx), np.arange(ny)
    BOXX_KM = BOXY_KM = None
    BOX_TAG = 'full'
else:
    bx, by = _sponge_box() if BOX_ARG == 'sponge' else (float(BOX_ARG) * 1e3, float(BOX_ARG) * 1e3)
    BOX_TAG = 'sponge' if BOX_ARG == 'sponge' else 'box%g' % float(BOX_ARG)
    BOXX_KM, BOXY_KM = bx / 1e3, by / 1e3
    ixb = np.where(np.abs(x) <= bx)[0]
    iyb = np.where(np.abs(y) <= by)[0]
boxsel = np.ix_(ixb, iyb)                         # PSDs use ONLY this data


def detrend2d(a):
    """Remove the best-fit plane (2D linear detrend) so a large-scale gradient/ramp
    doesn't leak into the longest-wavelength FFT bins."""
    sx, sy = a.shape
    xi, yi = np.meshgrid(np.arange(sx), np.arange(sy), indexing='ij')
    G = np.column_stack([xi.ravel(), yi.ravel(), np.ones(a.size)])
    c, *_ = np.linalg.lstsq(G, a.ravel(), rcond=None)
    return a - (G @ c).reshape(a.shape)


def psd_line(line, d):
    """One-sided variance-preserving PSD of a single 1D transect (linearly detrended)."""
    n = line.size
    win = np.hanning(n)
    a = signal.detrend(line, type='linear') * win
    P = np.abs(np.fft.rfft(a)) ** 2 * 2.0 * d / (n * (win ** 2).mean())
    freq = np.fft.rfftfreq(n, d=d)
    return freq[1:], P[1:]


def radial_psd(arr, dx, dy, nsec=8):
    """2D radial PSD line = variance-preserving omnidirectional E(k) (azimuthal sum; keeps the
    spectral valley). 10-90 band = E(k) scaled by the directional spread across NSEC azimuthal
    sectors (sector = mean power over its modes), so the band brackets E(k) where resolvable."""
    sx, sy = arr.shape
    win2 = np.outer(np.hanning(sx), np.hanning(sy))
    a2 = detrend2d(arr) * win2
    P2 = (np.abs(np.fft.fft2(a2)) ** 2 * (dx * dy) / (sx * sy * (win2 ** 2).mean())).ravel()
    kxs, kys = np.fft.fftfreq(sx, dx), np.fft.fftfreq(sy, dy)
    KX = (kxs[:, None] * np.ones((1, sy)))
    KY = (np.ones((sx, 1)) * kys[None, :])
    krr = np.sqrt(KX ** 2 + KY ** 2).ravel()
    ang = np.mod(np.arctan2(KY.ravel(), KX.ravel()), np.pi)        # fold (real field -> symmetric)
    sel = krr > 0
    kbins = np.logspace(np.log10(krr[sel].min()), np.log10(krr[sel].max()), 24)
    secbins = np.linspace(0, np.pi, nsec + 1)
    kidx = np.digitize(krr, kbins)
    sidx = np.digitize(ang, secbins)
    kc, El, lo, hi = [], [], [], []
    for b in range(1, len(kbins)):
        ring = kidx == b
        n = ring.sum()
        if n < 4:
            continue
        Ek = P2[ring].sum() / (sx * dx * sy * dy * (kbins[b] - kbins[b - 1]))   # variance-preserving line
        kc.append(np.sqrt(kbins[b - 1] * kbins[b]))
        El.append(Ek)
        if n >= 2 * nsec:
            sv = np.array([P2[ring & (sidx == s)].mean() for s in range(1, nsec + 1)
                           if (ring & (sidx == s)).sum()])
            m = sv.mean()
            lo.append(Ek * np.percentile(sv, PCT[0]) / m)
            hi.append(Ek * np.percentile(sv, PCT[1]) / m)
        else:                                                                  # too few modes for sectors
            lo.append(Ek)
            hi.append(Ek)
    return (np.array(a) for a in (kc, El, lo, hi))


def spectra(field):
    sub = field[boxsel]
    sub = sub - sub.mean()
    ix0 = int(np.argmin(np.abs(x[ixb])))         # x=0 column within the box
    iy0 = int(np.argmin(np.abs(y[iyb])))         # y=0 row within the box
    fx, Px = psd_line(sub[:, iy0], dx)           # 1D along x, on the y=0 transect
    fy, Py = psd_line(sub[ix0, :], dy)           # 1D along y, on the x=0 transect
    kc, rmed, rlo, rhi = radial_psd(sub, dx, dy)
    S = dict(lamx=1.0 / fx / 1e3, Px=Px, lamy=1.0 / fy / 1e3, Py=Py,
             lamrad=1.0 / kc / 1e3, rmed=rmed, rlo=rlo, rhi=rhi)
    # variance normalization: divide each curve by its own integral over wavenumber (cyc/km),
    # so int(PSD dk) = 1 (Parseval: that integral = the signal variance). Matches the lidar-compare
    # convention; compares spectral SHAPE (removes amplitude) across curves and times.
    S['vx'] = np.trapz(Px, fx * 1e3)
    S['vy'] = np.trapz(Py, fy * 1e3)
    S['vr'] = np.trapz(rmed, kc * 1e3)
    S['x0'], S['y0'] = float(x[ixb][ix0]) / 1e3, float(y[iyb][iy0]) / 1e3
    return S


Sw = spectra(w)
St = spectra(tprime)
print(f"[i] {SIM} z={zkm:.1f}km t={tstr}  window={BOX_TAG} ({ixb.size}x{iyb.size})  "
      f"var w-1dx/1dy/2d={Sw['vx']:.3g}/{Sw['vy']:.3g}/{Sw['vr']:.3g}  theta'-2d={St['vr']:.3g}")

# ----------------------------------------------------------------------------- map color scale (log bins)
norm = BoundaryNorm(CLEV_W, ncolors=CMAP_W.N, clip=True)
ext = [x[0] / 1e3, x[-1] / 1e3, y[0] / 1e3, y[-1] / 1e3]
Xkm, Ykm = x / 1e3, y / 1e3


RBOX = {"boxstyle": "round", "lw": 0.67, "facecolor": "white", "edgecolor": "black"}
CBOX = {"boxstyle": "circle", "lw": 0.67, "facecolor": "white", "edgecolor": "black"}
XPP, YPP, XLBL = 0.96, 0.93, 0.04   # letters ha='right' at XPP -> same edge gap as XLBL on the left


def corner_labels(ax, header, letter):
    ax.text(XLBL, YPP, header, transform=ax.transAxes, bbox=RBOX)                 # info label: not bold
    ax.text(XPP, YPP, letter, transform=ax.transAxes, ha='right', weight='bold', bbox=CBOX)


def draw_map(ax, field, header, letter, show_x, show_y):
    im = ax.imshow(field.T, extent=ext, origin='lower', aspect='auto',
                   cmap=CMAP_W, norm=norm, interpolation='nearest', rasterized=True)
    ax.contour(Xkm, Ykm, topo.T, levels=TOPO_LEV, colors='k', linewidths=0.3)
    if not FULL:
        ax.add_patch(Rectangle((-BOXX_KM, -BOXY_KM), 2 * BOXX_KM, 2 * BOXY_KM,
                               fill=False, ec='k', ls=(0, (4, 2)), lw=1.2))
    # transect lines used for the 1D FFT (y=0 row -> 1D x; x=0 col -> 1D y); dotted, white-haloed
    hx = (Xkm[0], Xkm[-1]) if FULL else (-BOXX_KM, BOXX_KM)
    vy = (Ykm[0], Ykm[-1]) if FULL else (-BOXY_KM, BOXY_KM)
    halo = [pe.Stroke(linewidth=2.4, foreground='w'), pe.Normal()]
    ax.plot(hx, (0, 0), color='k', ls=':', lw=1.3, path_effects=halo, zorder=4)
    ax.plot((0, 0), vy, color='k', ls=':', lw=1.3, path_effects=halo, zorder=4)
    corner_labels(ax, header, letter)
    if show_x:
        ax.set_xlabel('x / km')
    else:
        ax.tick_params(labelbottom=False)
    if show_y:
        ax.set_ylabel('y / km')
    else:
        ax.tick_params(labelleft=False)
    return im


def radial_band(ax, S, ls, lw, alpha):
    """2D radial median (black) + 10-90 percentile band (azimuthal spread)."""
    f = S['normrad']
    ax.fill_between(S['lamrad'], S['rlo'] / f, S['rhi'] / f, color='0.45', alpha=alpha, lw=0)
    ax.plot(S['lamrad'], S['rmed'] / f, color='k', ls=ls, lw=lw)


def build(yscale):
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.015, h_pad=0.015, wspace=0.01, hspace=0.01)
    axa, axb = axs[0]
    axc, axd = axs[1]
    im = draw_map(axa, w,       f'full field ($z={zkm:.1f}$ km)',                    'a', show_x=False, show_y=True)
    draw_map(axb, w_high,       r'high-pass ($\lambda<%g$ km)' % (LAMBDA_CUT / 1e3), 'b', show_x=False, show_y=False)
    draw_map(axc, w_low,        r'low-pass ($\lambda>%g$ km)' % (LAMBDA_CUT / 1e3),  'c', show_x=True, show_y=True)
    cb = fig.colorbar(im, ax=[axa, axb, axc, axd], location='top', shrink=0.55, aspect=40, pad=0.02,
                      extend='both', spacing='uniform', ticks=CLEV_W_LABELS)
    cb.set_label(r"w / m$\,$s$^{-1}$")

    # x-axis LINEAR IN WAVENUMBER (k = 1/lambda, cyc/km); wavelength on the top secondary axis.
    # each curve variance-normalised (divided by its own integral over k) -> on-screen area = variance
    axd.plot(1.0 / Sw['lamrad'], Sw['rmed'] / Sw['vr'], color='k', lw=2.0, zorder=4)
    axd.plot(1.0 / St['lamrad'], St['rmed'] / St['vr'], color='k', ls='--', lw=2.0, zorder=4)
    axd.plot(1.0 / Sw['lamx'], Sw['Px'] / Sw['vx'], color='C0', ls=':', lw=1.3, zorder=5)
    axd.plot(1.0 / Sw['lamy'], Sw['Py'] / Sw['vy'], color='C3', ls=':', lw=1.3, zorder=5)
    axd.axvline(1e3 / LAMBDA_CUT, color='grey', lw=1.0, ls=':')
    axd.set_xlim(1.0 / XLIM_KM[0], 1.0 / XLIM_KM[1])          # short lambda (high k) on the left, as before
    axd.set_yscale(yscale)
    axd.set_ylim((2e-3, 50) if yscale == 'log' else (0, YMAX_LINY))
    axd.grid(True, which='both', alpha=0.2)
    axd.set_xlabel(r'wavenumber / km$^{-1}$')
    axd.set_ylabel(r'PSD$_\sigma$  (variance-normalised)')
    axd.yaxis.set_label_position('right')
    axd.tick_params(labelleft=False, labelright=True)
    secax = axd.secondary_xaxis('top', functions=(lambda k: 1.0 / k, lambda l: 1.0 / l))
    secax.set_xlabel('horizontal wavelength / km')
    lamticks = [400, 200, 100, 50, 30, 20]
    secax.set_xticks(lamticks)
    secax.set_xticklabels([f'{l:g}' for l in lamticks])
    secax.minorticks_off()
    corner_labels(axd, 'PSD', 'd')
    h = [plt.Line2D([], [], color='C0', ls=':', lw=1.3, label=r"w 1D FFT$_x$ ($y=%g$ km)" % round(Sw['y0'])),
         plt.Line2D([], [], color='C3', ls=':', lw=1.3, label=r"w 1D FFT$_y$ ($x=%g$ km)" % round(Sw['x0'])),
         plt.Line2D([], [], color='k', lw=2.0, label=r"w 2D FFT"),
         plt.Line2D([], [], color='k', ls='--', lw=2.0, label=r"$\theta'$ 2D FFT")]
    axd.legend(handles=h, loc='upper center', fontsize=8.5, framealpha=0.95, handlelength=1.8)

    tag = 'logy' if yscale == 'log' else 'liny'
    out = f"{OUTDIR}/spectral_{SIM}_z{zkm:.0f}km_t{TIDX}_{BOX_TAG}_{tag}.png"
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"[i] wrote {out}")


build('linear')
