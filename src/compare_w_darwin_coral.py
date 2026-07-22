#!/usr/bin/env python
"""Compare vertical velocity w above Mount Darwin and the CORAL site.

For each darwin run, the w column is taken from the appropriate y-slice (Darwin at the
y = -62.4 km slice, CORAL at the y = +35.2 km slice) at the location's x position, and the two
profiles are overlaid. The 800m and 400m runs are shown in separate panels for a resolution
comparison.
"""
import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("/work/bd0620/b309199/mapy/src/latex_default.mplstyle")

SIMS = {
    "800m": "/scratch/b/b309199/darwin_240718_800m",
    "400m": "/scratch/b/b309199/darwin_240718_400m",
}
# (x, y) of the two columns in km
LOCS = {
    "Mt Darwin": dict(x=-8.8, y=-62.4, color="tab:blue"),
    "CORAL": dict(x=114.0, y=35.2, color="tab:red"),
}
TARGET_TIME_S = 7200.0   # model time [s] to compare at (within both runs)
ZMAX_KM = 90.0
OUT = "/work/bd0620/b309199/mapy/data/figures/w_darwin_coral_comparison.png"


def _tsec(ds):
    tv = ds["time"].values
    if np.issubdtype(tv.dtype, np.datetime64):
        return (tv - tv[0]) / np.timedelta64(1, "s")
    return np.asarray(tv, float)


def w_column(path, x_km, y_km, target_s):
    """Return (z_km, w, y_real_km, x_real_km, t_real_s) for the column nearest (x_km, y_km)."""
    ds = xr.open_dataset(f"{path}/slices_y.nc")
    y = np.asarray(ds["y"]) / 1000.0
    x = np.asarray(ds["x"]) / 1000.0
    jy = int(np.argmin(np.abs(y - y_km)))
    ix = int(np.argmin(np.abs(x - x_km)))
    tsec = _tsec(ds)
    it = int(np.argmin(np.abs(tsec - target_s)))
    w = np.asarray(ds["uvelz"].isel(time=it, y=jy, x=ix))
    zc = ds["zcr"].isel(y=jy, x=ix)
    if "time" in zc.dims:
        zc = zc.isel(time=0)
    return np.asarray(zc) / 1000.0, w, float(y[jy]), float(x[ix]), float(tsec[it])


RES_STYLE = {"800m": "-", "400m": "--"}   # linestyle by resolution; colour by location

fig, ax = plt.subplots(figsize=(6.5, 7))
for tag, path in SIMS.items():
    for name, p in LOCS.items():
        z, w, yreal, xreal, treal = w_column(path, p["x"], p["y"], TARGET_TIME_S)
        ax.plot(w, z, color=p["color"], lw=1.0, ls=RES_STYLE[tag], label=f"{name}, {tag}")
ax.axvline(0, color="grey", lw=0.5, ls="--")
ax.set(xlabel=r"w / m$\,$s$^{-1}$", ylabel="altitude z / km", ylim=(0, ZMAX_KM),
       title=f"w above Mt Darwin vs CORAL   (t = {TARGET_TIME_S / 3600:.1f} h)")
ax.legend(fontsize=9, loc="upper right")
fig.text(0.5, 0.012, "Mt Darwin: x=-9, y=-63 km     |     CORAL: x=114, y=35 km",
         ha="center", fontsize=8, color="grey")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT, dpi=120)
print("saved", OUT)
