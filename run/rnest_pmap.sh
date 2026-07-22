#!/bin/bash
# Submit the pmaputils nested-BC generator (cube -> input_<N>.nc) as a CPU batch job.
# usage: ./rnest_pmap.sh BB_SIM GRID_NC OUTDIR [CUBE] [DT_S] [LB_CONFIG] [extra nesting args...]
#        e.g. ./rnest_pmap.sh darwin_240718_era5_4km_fullorog \
#                 /scratch/b/b309199/darwin_240718_800m_dummy/data_0.nc \
#                 /work/bd0620/b309199/data/nests/darwin_240718_800m_input_fullorog \
#                 0 600 PMAP-Snapshots/config/darwin_240718_800m_nest.yml --skip-existing
#   BB_SIM    = big-brother simulation jobname (its /scratch cube_<CUBE>.nc is the source)
#   GRID_NC   = little-brother dummy data_0.nc (grid template)
#   OUTDIR    = destination for input_<N>.nc files
#   CUBE      = cube index (default 0)
#   DT_S      = uniform output cadence [s] (default 600)
#   LB_CONFIG = little-brother config.yml -> interior zeroing (omit or "" = full fields)
USER=b309199
NEST_VENV=/home/b/$USER/venvs/post-venv

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 BB_SIM GRID_NC OUTDIR [CUBE] [DT_S] [LB_CONFIG] [extra args...]" >&2
    exit 1
fi
BB_SIM=$1
GRID_NC=$2
OUTDIR=$3
CUBE=${4:-0}
DT_S=${5:-600}
LB_CONFIG=${6:-}
shift $(( $# > 6 ? 6 : $# ))
EXTRA_ARGS="$@"

CUBE_NC=/scratch/b/$USER/$BB_SIM/cube_${CUBE}.nc
CONFIG_ARG=""
if [ -n "$LB_CONFIG" ]; then CONFIG_ARG="--config $LB_CONFIG"; fi

LOGDIR=/work/bd0620/$USER/mapy/plot_logs
mkdir -p "$LOGDIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=nest_${BB_SIM}_c${CUBE}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --account=bd0620
#SBATCH --output=$LOGDIR/nest_${BB_SIM}_c${CUBE}_%j.out
#SBATCH --error=$LOGDIR/nest_${BB_SIM}_c${CUBE}_%j.out

set -e
$NEST_VENV/bin/python -m pmaputils.preprocessing.nesting \\
    $CUBE_NC $GRID_NC $OUTDIR \\
    --dt $DT_S --workers 8 $CONFIG_ARG $EXTRA_ARGS
EOF
