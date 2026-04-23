#!/bin/bash
#SBATCH --job-name=ablation_no_fix2
#SBATCH --account=def-cottenie_gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256G
#SBATCH --output=/scratch/rjafri/rj_subset_exp1/outputs/logs/ablation_no_fix2_%j.out
#SBATCH --error=/scratch/rjafri/rj_subset_exp1/outputs/logs/ablation_no_fix2_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rjafri@uoguelph.ca

# ── Environment ────────────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

source /home/rjafri/ddi_venv/bin/activate

# Install packages that may be missing on compute nodes
pip install typing_extensions pandas numpy scipy tqdm packaging psutil pillow pyzmq sympy platformdirs "huggingface-hub>=0.34.0,<1.0" --quiet

export PYTHONPATH=/scratch/rjafri/rj_subset_exp1:$PYTHONPATH
export HF_HOME=/scratch/rjafri/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/rjafri/.cache/huggingface
export HF_HUB_OFFLINE=1
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /scratch/rjafri/rj_subset_exp1

echo "Python: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"

# ── Verify packages ────────────────────────────────────────────────────────────
python -c "
import typing_extensions, pandas, numpy, torch
print(f'typing_extensions: OK')
print(f'pandas: {pandas.__version__}')
print(f'torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB)')
"

# ── Verify model is cached ─────────────────────────────────────────────────────
python -c "
import os
cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
model_dir = os.path.join(cache, 'hub', 'models--meta-llama--Llama-3.3-70B-Instruct')
if os.path.exists(model_dir):
    print(f'Model found: {model_dir}')
else:
    print(f'ERROR: Model not found at {model_dir}')
    exit(1)
"

# ── Run pilot experiment ───────────────────────────────────────────────────────
echo ""
echo "Starting subset pilot experiment..."
echo "Comparing Tanimoto vs Pathway retrieval on 4000 stratified pairs"
echo "Teacher: Llama-3.3-70B-Instruct"
echo ""

python scripts/run_subset_pilot.py \
    --config configs/config_ablation_no_fix2.yaml \
    --n-pairs 4000 \
    --seed 42

EXIT_CODE=$?

echo ""
echo "Job finished with exit code: $EXIT_CODE"
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== RESULTS ==="
    REPORT="/scratch/rjafri/rj_subset_exp1/outputs/ablation_no_fix2/comparison_report.txt"
    if [ -f "$REPORT" ]; then
        cat "$REPORT"
    fi
fi

exit $EXIT_CODE
