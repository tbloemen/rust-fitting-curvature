#!/bin/sh
#SBATCH --job-name="mnist-all_off-hyperbolic"
#SBATCH --partition=compute
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-msc-cs

set -eu

MODE="pareto"
DATASET="mnist"
EXPERIMENT="all_off"
GEOMETRY="hyperbolic"

module load 2025
module load compiler
module load rust

OUT=/scratch/"$USER"/fitting/results/${MODE}_${EXPERIMENT}_${DATASET}_${GEOMETRY}.jsonl
mkdir -p "$(dirname "$OUT")"

srun ./target/release/optimizer \
  --mode "$MODE" --dataset "$DATASET" --experiment "$EXPERIMENT" \
  --n-trials 500 --n-seeds 3 \
  --geometry "$GEOMETRY" \
  --threads "$SLURM_CPUS_PER_TASK" \
  --data-path /scratch/"$USER"/fitting/data \
  --output "$OUT"
