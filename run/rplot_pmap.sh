#!/bin/bash
# Submit the pmap_slc_lid.py plotting/animation script as a CPU batch job.
# usage: ./rplot_pmap.sh VAR SIMULATION [notest] [ZKM]
#        e.g. ./rplot_pmap.sh vortex linhydro_lowN notest
#        e.g. ./rplot_pmap.sh over darwin_240718_400m notest 58   (pin horizontal slice to ~58 km)
#   ZKM = target altitude [km] for the horizontal (z) slice; picks the nearest output level,
#         overriding the region default zslice index (runs with different output_slices.z grids
#         map the same km to different indices).

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
MAPY_SRC=$MAPY_ROOT/src
PLOT_VENV=$VENV
LOGDIR=$MAPY_ROOT/plot_logs
# -----------------------------------------------------------------------------

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 VAR SIMULATION [notest] [ZKM]" >&2
    exit 1
fi
VAR=$1
SIM=$2
NOTEST=${3:-}
ZKM=${4:-}
ZKM_ARG=""
if [ -n "$ZKM" ]; then ZKM_ARG="--zkm=$ZKM"; fi

mkdir -p "$LOGDIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=plot_${SIM}_${VAR}
#SBATCH --partition=$SB_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --account=$SB_ACCOUNT
#SBATCH --output=$LOGDIR/plot_${SIM}_${VAR}_%j.out
#SBATCH --error=$LOGDIR/plot_${SIM}_${VAR}_%j.out

set -e
$SB_MODULES
cd $MAPY_SRC
$PLOT_VENV/bin/python pmap_slc_lid.py $VAR $SIM $NOTEST $ZKM_ARG
EOF
