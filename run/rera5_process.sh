#!/bin/bash
# Submit the ERA5 model-level -> geometric-height interpolation (download_era5_ml.py's
# 'process' stage) as a CPU batch job. The 'download' stage is network-bound and belongs in
# the background on the login node; THIS stage is the memory/CPU-heavy one and belongs here.
# Case/window/grid (CONFIG: name, area, start/end, z_top_m, dz_m, proc_chunk_h) is edited IN
# download_era5_ml.py, not here. Re-running is safe: an existing -ml-int.nc is NOT skipped,
# so delete it first if you want a rebuild.
# usage: ./rera5_process.sh

# --- site detection: Levante (DKRZ) vs JUPITER (JSC) -------------------------
if [ -d /e/project1/gwturb ]; then                      # JUPITER
    USER=binder5
    MAPY_ROOT=/e/project1/gwturb/binder5/mapy
    VENV=/e/project1/gwturb/binder5/venvs/post-venv
    SCRATCH_BASE=/e/scratch/gwturb/binder5
    SB_PARTITION=booster
    SB_ACCOUNT=gwturb
    SB_MODULES="module --force purge; module load Stages/2026 GCC/14.3.0 Python/3.13.5"
else                                                    # Levante
    USER=b309199
    MAPY_ROOT=/work/bd0620/$USER/mapy
    VENV=/home/b/$USER/venvs/post-venv
    SCRATCH_BASE=/scratch/b/$USER
    SB_PARTITION=compute
    SB_ACCOUNT=bd0620
    SB_MODULES=":"   # Levante post-venv (conda) is self-contained
fi
SRC=$MAPY_ROOT/src/envs-and-dems
LOGDIR=$MAPY_ROOT/plot_logs
# -----------------------------------------------------------------------------

mkdir -p "$LOGDIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=era5_int
#SBATCH --partition=$SB_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --account=$SB_ACCOUNT
#SBATCH --output=$LOGDIR/era5_int_%j.out
#SBATCH --error=$LOGDIR/era5_int_%j.out

set -e
$SB_MODULES
cd $SRC
$VENV/bin/python -u download_era5_ml.py process
EOF
