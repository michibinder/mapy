#!/bin/bash
# Submit the dyn_overview.py dynamics-overview animation as a CPU batch job.
# usage: ./rdyn_pmap.sh SIMULATION [notest] [w|t] [wind|tprime]
USER=b309199
MAPY_SRC=/work/bd0620/$USER/mapy/src
PLOT_VENV=/home/b/$USER/venvs/post-venv

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 SIMULATION [notest]" >&2
    exit 1
fi
SIM=$1
NOTEST=${2:-}
VAR=${3:-t}
MODE=${4:-wind}

LOGDIR=/work/bd0620/$USER/mapy/plot_logs
mkdir -p "$LOGDIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=dyn_${SIM}_${VAR}_${MODE}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --account=bd0620
#SBATCH --output=$LOGDIR/dyn_${SIM}_${VAR}_${MODE}_%j.out
#SBATCH --error=$LOGDIR/dyn_${SIM}_${VAR}_${MODE}_%j.out

set -e
cd $MAPY_SRC
$PLOT_VENV/bin/python dyn_overview.py $SIM $NOTEST $VAR $MODE
EOF
