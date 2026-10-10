#!/bin/bash
#SBATCH --job-name=video-compression
#SBATCH --time=48:00:00
#SBATCH --account=def-sadafsk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/home/tariq1/scratch/video_experiments/logs/pipeline_%j.out
#SBATCH --error=/home/tariq1/scratch/video_experiments/logs/pipeline_%j.err

set -e

# ============================================================
# Configurable parameters
# ============================================================
LAMBDA_CLS="${1:-0.00}"          # Classification loss weight (pass as argument or default 0.1)
LAMBDA_MSE=1.0                  # Distortion loss weight
BS=64                           # Batch size
TOTAL_EPOCHS_IFRAME=80        # Epochs for I-frame compressor
TOTAL_EPOCHS_PFRAME=80         # Epochs for P-frame compressor
TOTAL_EPOCHS_UNIVERSAL=80      # Epochs for universal representation
TOTAL_EPOCHS_3FRAME=80         # Epochs for 3-frame model
Z_DIM_IFRAME=4                  # I-frame latent dim
L_IFRAME=16                     # I-frame quantization levels
Z_DIM_PFRAME=1                  # P-frame latent dim
L_PFRAME=2                      # P-frame quantization levels
DATASET=MNIST
SAVE_BASE="./saved_models"

# ============================================================
# Environment setup
# ============================================================
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Lambda CLS: $LAMBDA_CLS"

module spider arch/sse3
module spider gcc/14.3
module spider cuda/13.2
module spider python/3.14.2

source /home/tariq1/scratch/envs/video_env/bin/activate

echo "Python:"
which python
python --version

echo "NVIDIA:"
nvidia-smi

echo "PyTorch:"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

cd "$SLURM_SUBMIT_DIR"

# ============================================================
# Folder names (must match each script's naming convention)
# ============================================================
IFRAME_FOLDER="Iframe_dim_128_z_dim_${Z_DIM_IFRAME}_L_${L_IFRAME}_single_bit_False_bs_${BS}_lambdaMSE_${LAMBDA_MSE}_lambdaCLS_${LAMBDA_CLS}"
PFRAME_FOLDER="Pframe_dim_128_z_dim_${Z_DIM_PFRAME}_L_${L_PFRAME}_single_bit_False_bs_${BS}_lambdaMSE_${LAMBDA_MSE}_lambdaCLS_${LAMBDA_CLS}"
UNIVERSAL_FOLDER="Universal_dim_128_z_dim_${Z_DIM_PFRAME}_L_${L_PFRAME}_single_bit_False_bs_${BS}_lambdaMSE_${LAMBDA_MSE}_lambdaCLS_${LAMBDA_CLS}"
THREEFRAME_FOLDER="Universal_R1inf_3frames_dim_128_z_dim_${Z_DIM_PFRAME}_L_${L_PFRAME}_single_bit_False_bs_${BS}_lambdaMSE_${LAMBDA_MSE}_lambdaCLS_${LAMBDA_CLS}"

IFRAME_PATH="${SAVE_BASE}/${IFRAME_FOLDER}"
PFRAME_PATH="${SAVE_BASE}/${PFRAME_FOLDER}"
UNIVERSAL_PATH="${SAVE_BASE}/${UNIVERSAL_FOLDER}"

# ============================================================
# Step 1: P-frame compressor (lambda_cls=0.0, only MSE)
# ============================================================
echo ""
echo "============================================"
echo "Step 1/4: P-frame compressor"
echo "============================================"
python train.py \
    --total_epochs ${TOTAL_EPOCHS_PFRAME} \
   --bs ${BS} \
    --z_dim ${Z_DIM_PFRAME} \
    --L ${L_PFRAME} \
    --lambda_MSE ${LAMBDA_MSE} \
    --lambda_cls ${LAMDBA_CLS} \
    --dataset ${DATASET}
#
echo "Step 1 complete: ${PFRAME_PATH}"
echo "Results saved in: ${SAVE_BASE}/"
echo "Lambda CLS: ${LAMBDA_CLS}"
echo "Finished at $(date)"
echo "============================================"

