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
USER=b309199
MAPY_SRC=/work/bd0620/$USER/mapy/src
PLOT_VENV=/home/b/$USER/venvs/post-venv

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 SIMULATION [CUBE] [notest] [FIELD]" >&2
    exit 1
fi
SIM=$1
CUBE=${2:-0}
NOTEST=${3:-}
FIELD=${4:-w}

LOGDIR=/work/bd0620/$USER/mapy/plot_logs
mkdir -p "$LOGDIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=trackprof_${SIM}_${CUBE}_${FIELD}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --account=bd0620
#SBATCH --output=$LOGDIR/trackprof_${SIM}_${CUBE}_${FIELD}_%j.out
#SBATCH --error=$LOGDIR/trackprof_${SIM}_${CUBE}_${FIELD}_%j.out

set -e
cd $MAPY_SRC
$PLOT_VENV/bin/python cube_track_profile.py $SIM $NOTEST --cube $CUBE --field $FIELD
EOF
