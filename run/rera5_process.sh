#!/bin/bash
# Submit the ERA5 model-level -> geometric-height interpolation (download_era5_ml.py's
# 'process' stage) as a CPU batch job. The 'download' stage is network-bound and belongs in
# the background on the login node; THIS stage is the memory/CPU-heavy one and belongs here.
# Case/window/grid (CONFIG: name, area, start/end, z_top_m, dz_m, proc_chunk_h) is edited IN
# download_era5_ml.py, not here. Re-running is safe: an existing -ml-int.nc is NOT skipped,
# so delete it first if you want a rebuild.
# usage: ./rera5_process.sh
USER=b309199
SRC=/work/bd0620/$USER/mapy/src/envs-and-dems
VENV=/home/b/$USER/venvs/post-venv

LOGDIR=/work/bd0620/$USER/mapy/plot_logs
mkdir -p "$LOGDIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=era5_int
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --account=bd0620
#SBATCH --output=$LOGDIR/era5_int_%j.out
#SBATCH --error=$LOGDIR/era5_int_%j.out

set -e
cd $SRC
$VENV/bin/python -u download_era5_ml.py process
EOF
