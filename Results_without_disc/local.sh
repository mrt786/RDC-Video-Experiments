#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"


# ============================================================
# Configurable parameters
# ============================================================
LAMBDA_CLS="${1:-0.00}"          # Classification loss weight (pass as argument or default 0.1)
LAMBDA_MSE=1.0                  # Distortion loss weight
BS=64                           # Batch size
TOTAL_EPOCHS_IFRAME=40        # Epochs for I-frame compressor
TOTAL_EPOCHS_PFRAME=40         # Epochs for P-frame compressor
TOTAL_EPOCHS_UNIVERSAL=40      # Epochs for universal representation
TOTAL_EPOCHS_3FRAME=40         # Epochs for 3-frame model
Z_DIM_IFRAME= 12                # I-frame latent dim
L_IFRAME=2                     # I-frame quantization levels
Z_DIM_PFRAME=6                  # P-frame latent dim
L_PFRAME=2                      # P-frame quantization levels
DATASET=MNIST


# ============================================================
# Environment setup
# ============================================================


echo "Python:"
which python
python --version

echo "PyTorch:"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"



# ============================================================
# Step 1: I-frame compressor (lambda_cls=different)
# ============================================================
echo ""
echo "============================================"
echo "Step 1/5: I-frame compressor"
echo "============================================"
python I-frame.py \
   --total_epochs ${TOTAL_EPOCHS_IFRAME} \
  --bs ${BS} \
   --z_dim ${Z_DIM_IFRAME} \
    --L ${L_IFRAME} \
   --lambda_MSE ${LAMBDA_MSE} \
   --lambda_cls ${LAMBDA_CLS} \
    --dataset ${DATASET}
echo "Step 1 Completed"
# echo "Finished at $(date)"
# echo "============================================"

# ============================================================
# Step 2: P-frame compressor (lambda_cls=different)
# ============================================================
echo ""
echo "============================================"
echo "Step 2/5: P-frame compressor"
echo "============================================"
python train_f2_stream1.py \
   --total_epochs ${TOTAL_EPOCHS_PFRAME} \
  --bs ${BS} \
   --z_dim ${Z_DIM_PFRAME} \
   --L ${L_PFRAME} \
   --lambda_MSE ${LAMBDA_MSE} \
   --lambda_cls ${LAMBDA_CLS} \
   --dataset ${DATASET}

echo "Step 2 complete"
echo "============================================"

# ============================================================
# Step 3: Universa Architecture for 2nd frame
# ============================================================
echo ""
echo "============================================"
echo "Step 3/5: Universal Architecture"
echo "============================================"

python train_f2_universal_stream1.py \
    --total_epochs ${TOTAL_EPOCHS_UNIVERSAL} \
   --bs ${BS} \
    --z_dim ${Z_DIM_PFRAME} \
    --L ${L_PFRAME} \
    --lambda_MSE ${LAMBDA_MSE} \
    --lambda_cls ${LAMBDA_CLS} \
    --dataset ${DATASET}

echo "Step 3 completed"
echo "============================================"



# ============================================================
# Step 4: P-frame compressor (lambda_cls=different)
# ============================================================
echo ""
echo "============================================"
echo "Step 4/5: P-frame compressor for 3rd frame"
echo "============================================"
python train_f3_stream1.py \
   --total_epochs ${TOTAL_EPOCHS_3FRAME} \
  --bs ${BS} \
   --z_dim ${Z_DIM_PFRAME} \
   --L ${L_PFRAME} \
   --lambda_MSE ${LAMBDA_MSE} \
   --lambda_cls ${LAMBDA_CLS} \
   --dataset ${DATASET}

echo "Step 3 completed"
echo "============================================"

# ============================================================
# Step 3: Universal Architecture for 3rd frame
# ============================================================
echo ""
echo "============================================"
echo "Step 5/5: Universal Architecture"
echo "============================================"

python train_f3_universal_stream1.py \
    --total_epochs ${TOTAL_EPOCHS_3FRAME} \
   --bs ${BS} \
    --z_dim ${Z_DIM_PFRAME} \
    --L ${L_PFRAME} \
    --lambda_MSE ${LAMBDA_MSE} \
    --lambda_cls ${LAMBDA_CLS} \
    --dataset ${DATASET}

echo "Step 4 completed"
echo "============================================"