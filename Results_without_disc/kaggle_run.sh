#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/kaggle/working/results-without-disc/Results_without_disc"

cd "$PROJECT_DIR"

LAMBDA_CLS="${1:-0.00}" 
LAMBDA_MSE=1.0
BS=64
EPOCHS=2
Z_DIM=12
L=2
DATASET=MNIST

echo "Working directory: $(pwd)"
echo "Python version:"
python --version

echo "GPU information:"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"






python train_f2_stream2.py \
    --total_epochs "$EPOCHS" \
    --bs "$BS" \
    --z_dim "$Z_DIM" \
    --L "$L" \
    --lambda_MSE "$LAMBDA_MSE" \
    --lambda_cls "$LAMBDA_CLS" \
    --dataset "$DATASET" \
    --path ./data/



python train_f2_universal_stream2.py \
    --total_epochs "$EPOCHS" \
    --bs "$BS" \
    --z_dim "$Z_DIM" \
    --L "$L" \
    --lambda_MSE "$LAMBDA_MSE" \
    --lambda_cls "$LAMBDA_CLS" \
    --dataset "$DATASET" \
    --path ./data/


python train_f3_stream2.py \
    --total_epochs "$EPOCHS" \
    --bs "$BS" \
    --z_dim "$Z_DIM" \
    --L "$L" \
    --lambda_MSE "$LAMBDA_MSE" \
    --lambda_cls "$LAMBDA_CLS" \
    --dataset "$DATASET" \
    --path ./data/



python train_f3_universal_stream2.py \
    --total_epochs "$EPOCHS" \
    --bs "$BS" \
    --z_dim "$Z_DIM" \
    --L "$L" \
    --lambda_MSE "$LAMBDA_MSE" \
    --lambda_cls "$LAMBDA_CLS" \
    --dataset "$DATASET" \
    --path ./data/


echo "Training completed."