#!/bin/bash
#SBATCH --job-name=rj_full_dataset_a
#SBATCH --account=def-cottenie_gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256G
#SBATCH --output=/scratch/rjafri/CoT_DDI/outputs/logs/rj_full_%j.out
#SBATCH --error=/scratch/rjafri/CoT_DDI/outputs/logs/rj_full_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rjafri@uoguelph.ca

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

source /home/rjafri/ddi_venv/bin/activate
pip install typing_extensions pandas numpy scipy tqdm packaging psutil pillow \
    pyzmq sympy platformdirs "huggingface-hub>=0.34.0,<1.0" --quiet

export PYTHONPATH=/scratch/rjafri/CoT_DDI:$PYTHONPATH
export HF_HOME=/scratch/rjafri/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/rjafri/.cache/huggingface
export HF_HUB_OFFLINE=1
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /scratch/rjafri/CoT_DDI

echo "Python: $(which python)"

python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

python -c "
import os
cache = os.environ.get('HF_HOME', '')
model_dir = os.path.join(cache, 'hub', 'models--meta-llama--Llama-3.3-70B-Instruct')
if os.path.exists(model_dir):
    print(f'Model found: {model_dir}')
else:
    print(f'ERROR: Model not found')
    exit(1)
"

mkdir -p outputs/logs
mkdir -p outputs/rj_dataset_a/teacher_traces

echo ""
echo "Starting full RJ teacher generation..."
echo "Dataset A: 236K pairs, 129 classes"
echo "Config: pathway retrieval, fixes 3+4+5, no fix1/fix2"
echo ""

python -m src.teacher_generation --config configs/config_full_rj.yaml

EXIT_CODE=$?
echo ""
echo "Job finished with exit code: $EXIT_CODE"
echo "End time: $(date)"
exit $EXIT_CODE
