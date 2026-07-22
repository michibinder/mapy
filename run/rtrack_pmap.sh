#!/bin/bash
# Submit cube_track.py (vortex-feature tracking animation of a horizontal cube slice) as a CPU job.
# Tracking counterpart of rcube_pmap.sh (the 3D lambda2 cube animation) and rplot_pmap.sh.
# usage: ./rtrack_pmap.sh SIMULATION [CUBE] [notest]
#        e.g. ./rtrack_pmap.sh darwin_240718_400m_r1 0 notest
#   CUBE   = cube index -> cube_<CUBE>.nc in the run's scratch dir (default 0)
#   notest = render the full animation; omit (or any other 3rd arg) = single test frame
USER=b309199
MAPY_SRC=/work/bd0620/$USER/mapy/src
PLOT_VENV=/home/b/$USER/venvs/post-venv

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 SIMULATION [CUBE] [notest]" >&2
    exit 1
fi
SIM=$1
CUBE=${2:-0}
NOTEST=${3:-}

LOGDIR=/work/bd0620/$USER/mapy/plot_logs
mkdir -p "$LOGDIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=track_${SIM}_${CUBE}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --account=bd0620
#SBATCH --output=$LOGDIR/track_${SIM}_${CUBE}_%j.out
#SBATCH --error=$LOGDIR/track_${SIM}_${CUBE}_%j.out

set -e
cd $MAPY_SRC
$PLOT_VENV/bin/python cube_track.py $SIM $NOTEST --cube $CUBE
EOF
