#!/bin/bash
# Submit the dyn_overview.py dynamics-overview animation as a CPU batch job.
# usage: ./rdyn_pmap.sh SIMULATION [notest] [w|t] [wind|tprime]

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

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 SIMULATION [notest]" >&2
    exit 1
fi
SIM=$1
NOTEST=${2:-}
VAR=${3:-t}
MODE=${4:-wind}

mkdir -p "$LOGDIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=dyn_${SIM}_${VAR}_${MODE}
#SBATCH --partition=$SB_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --account=$SB_ACCOUNT
#SBATCH --output=$LOGDIR/dyn_${SIM}_${VAR}_${MODE}_%j.out
#SBATCH --error=$LOGDIR/dyn_${SIM}_${VAR}_${MODE}_%j.out

set -e
$SB_MODULES
cd $MAPY_SRC
$PLOT_VENV/bin/python dyn_overview.py $SIM $NOTEST $VAR $MODE
EOF
