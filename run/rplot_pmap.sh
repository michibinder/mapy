#!/bin/bash
# Submit the pmap_slc_lid.py plotting/animation script as a CPU batch job.
# usage: ./rplot_pmap.sh VAR SIMULATION [notest] [ZKM]
#        e.g. ./rplot_pmap.sh vortex linhydro_lowN notest
#        e.g. ./rplot_pmap.sh over darwin_240718_400m notest 58   (pin horizontal slice to ~58 km)
#   ZKM = target altitude [km] for the horizontal (z) slice; picks the nearest output level,
#         overriding the region default zslice index (runs with different output_slices.z grids
#         map the same km to different indices).
USER=b309199
MAPY_SRC=/work/bd0620/$USER/mapy/src
PLOT_VENV=/home/b/$USER/venvs/post-venv

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

LOGDIR=/work/bd0620/$USER/mapy/plot_logs
mkdir -p "$LOGDIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=plot_${SIM}_${VAR}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --account=bd0620
#SBATCH --output=$LOGDIR/plot_${SIM}_${VAR}_%j.out
#SBATCH --error=$LOGDIR/plot_${SIM}_${VAR}_%j.out

set -e
cd $MAPY_SRC
$PLOT_VENV/bin/python pmap_slc_lid.py $VAR $SIM $NOTEST $ZKM_ARG
EOF
