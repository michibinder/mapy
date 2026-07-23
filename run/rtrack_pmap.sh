#!/bin/bash
# Submit cube_track.py (vortex-feature tracking animation of a horizontal cube slice) as a CPU job.
# Tracking counterpart of rcube_pmap.sh (the 3D lambda2 cube animation) and rplot_pmap.sh.
# usage: ./rtrack_pmap.sh SIMULATION [CUBE] [notest]
#        e.g. ./rtrack_pmap.sh darwin_240718_400m_r1 0 notest
#   CUBE   = cube index -> cube_<CUBE>.nc in the run's scratch dir (default 0)
#   notest = render the full animation; omit (or any other 3rd arg) = single test frame

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
    echo "Usage: $0 SIMULATION [CUBE] [notest]" >&2
    exit 1
fi
SIM=$1
CUBE=${2:-0}
NOTEST=${3:-}

mkdir -p "$LOGDIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=track_${SIM}_${CUBE}
#SBATCH --partition=$SB_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --account=$SB_ACCOUNT
#SBATCH --output=$LOGDIR/track_${SIM}_${CUBE}_%j.out
#SBATCH --error=$LOGDIR/track_${SIM}_${CUBE}_%j.out

set -e
$SB_MODULES
cd $MAPY_SRC
$PLOT_VENV/bin/python cube_track.py $SIM $NOTEST --cube $CUBE
EOF
