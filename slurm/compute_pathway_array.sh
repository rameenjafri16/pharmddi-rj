#!/bin/bash
#SBATCH --job-name=pathway_array
#SBATCH --account=def-cottenie
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-19
#SBATCH --output=/scratch/rjafri/rj_subset_exp1/outputs/logs/pathway_array_%A_%a.out
#SBATCH --error=/scratch/rjafri/rj_subset_exp1/outputs/logs/pathway_array_%A_%a.err

echo "Task $SLURM_ARRAY_TASK_ID started: $(date)"

source /home/rjafri/ddi_venv/bin/activate
export PYTHONPATH=/scratch/rjafri/rj_subset_exp1:$PYTHONPATH

python3 /scratch/rjafri/rj_subset_exp1/scripts/pathway_worker.py

echo "Task $SLURM_ARRAY_TASK_ID finished: $(date)"
