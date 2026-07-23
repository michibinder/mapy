#!/bin/bash
# Submit cube_track_profile.py (vertical-profile vortex-feature tracking animation) as a CPU job.
# Profile companion of rtrack_pmap.sh: tracks w features on every ~1-km cube level and animates
# the median feature-speed profile vs the ambient / box-mean wind profiles.
# usage: ./rprof_pmap.sh SIMULATION [CUBE] [notest] [FIELD]
#        e.g. ./rprof_pmap.sh darwin_240718_400m_coralT_ifs_wcoast 0 notest t
#   CUBE   = cube index -> cube_<CUBE>.nc in the run's scratch dir (default 0)
#   notest = render the full animation; omit (or any other 3rd arg) = single test frame
#   FIELD  = tracked field: w (|w|, default), t (|T'|) or tpv (signed T' peaks & valleys
#            + peak-valley distance panel)

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
    echo "Usage: $0 SIMULATION [CUBE] [notest] [FIELD]" >&2
    exit 1
fi
SIM=$1
CUBE=${2:-0}
NOTEST=${3:-}
FIELD=${4:-w}

mkdir -p "$LOGDIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=trackprof_${SIM}_${CUBE}_${FIELD}
#SBATCH --partition=$SB_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --account=$SB_ACCOUNT
#SBATCH --output=$LOGDIR/trackprof_${SIM}_${CUBE}_${FIELD}_%j.out
#SBATCH --error=$LOGDIR/trackprof_${SIM}_${CUBE}_${FIELD}_%j.out

set -e
$SB_MODULES
cd $MAPY_SRC
$PLOT_VENV/bin/python cube_track_profile.py $SIM $NOTEST --cube $CUBE --field $FIELD
EOF
