#!/bin/bash
# Submit build_background_3d.py (ERA5+JAWARA -> PMAP nest input files) as a CPU batch job.
# Config (OUTDIR, CORIOLIS_MODE, WRITE_RVAPOUR, blend, ...) is edited IN THE SCRIPT, not here.
# usage: ./rbuild_background3d.sh [RANGE]     RANGE = e.g. 0  or  1-8  (default: all N_FILES)

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

RANGE=${1:-}

mkdir -p "$LOGDIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=bg3d_${RANGE:-all}
#SBATCH --partition=$SB_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --account=$SB_ACCOUNT
#SBATCH --output=$LOGDIR/bg3d_${RANGE:-all}_%j.out
#SBATCH --error=$LOGDIR/bg3d_${RANGE:-all}_%j.out

set -e
$SB_MODULES
cd $SRC
$VENV/bin/python build_background_3d.py $RANGE
EOF
