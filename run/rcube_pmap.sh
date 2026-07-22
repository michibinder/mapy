#!/bin/bash
# Submit cube_lid.py (3D lambda2 multiview + virtual-lidar cube animation) as a CPU batch job.
# Cube counterpart of rplot_pmap.sh (which runs the slice animation pmap_slc_lid.py).
# usage: ./rcube_pmap.sh SIMULATION [CUBE] [notest]
#        e.g. ./rcube_pmap.sh darwin_240718_400m_r1 0 notest
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
#SBATCH --job-name=cube_${SIM}_${CUBE}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --account=bd0620
#SBATCH --output=$LOGDIR/cube_${SIM}_${CUBE}_%j.out
#SBATCH --error=$LOGDIR/cube_${SIM}_${CUBE}_%j.out

set -e
cd $MAPY_SRC
$PLOT_VENV/bin/python cube_lid.py $SIM $NOTEST --cube $CUBE
EOF
