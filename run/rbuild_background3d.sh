#!/bin/bash
# Submit build_background_3d.py (ERA5+JAWARA -> PMAP nest input files) as a CPU batch job.
# Config (OUTDIR, CORIOLIS_MODE, WRITE_RVAPOUR, blend, ...) is edited IN THE SCRIPT, not here.
# usage: ./rbuild_background3d.sh [RANGE]     RANGE = e.g. 0  or  1-8  (default: all N_FILES)
USER=b309199
SRC=/work/bd0620/$USER/mapy/src/envs-and-dems
VENV=/home/b/$USER/venvs/post-venv
RANGE=${1:-}

LOGDIR=/work/bd0620/$USER/mapy/plot_logs
mkdir -p "$LOGDIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=bg3d_${RANGE:-all}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --account=bd0620
#SBATCH --output=$LOGDIR/bg3d_${RANGE:-all}_%j.out
#SBATCH --error=$LOGDIR/bg3d_${RANGE:-all}_%j.out

set -e
cd $SRC
$VENV/bin/python build_background_3d.py $RANGE
EOF
