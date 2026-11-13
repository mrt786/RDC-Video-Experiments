# Stream2 Video Compression Pipeline

## Overview

`stream2` is the coding configuration in which:

- **Frame 1** is transmitted as an I-frame at a relatively high rate.
- **Frames 2 and 3** are reconstructed at a lower rate using P-frame models and previously reconstructed frames.

The frame dependency is:

```text
Frame 1
  └── high-rate I-frame reconstruction/reference
        └── ssf2: reconstruct Frame 2 at low rate
              └── reconstructed Frame 2 (x2_hat)
                    └── ssf3: reconstruct Frame 3 at low rate
```

The low-rate models use `ScaleSpaceFlow` with stochastic latent quantization. The latent dimension and quantization level are controlled by `z_dim` and `L`. The rate reported by the scripts is derived from these values:

```text
rate = z_dim × 2 × log2(L)
```

The current Slurm configuration uses:

- Batch size: `64`
- MSE distortion weight: `lambda_MSE=1.0`
- P-frame latent dimension: `z_dim=6`
- Quantization levels: `L=2`
- Dataset: `MNIST`
- Eight CPU data-loader workers

## Files and their roles

### `train_f2_stream2.py`

This is the ordinary second-frame P-frame trainer.

For each sequence, it selects:

```text
x_ref = Frame 1
x_cur = Frame 2
```

It then computes:

```text
x2_hat = ssf(x_cur, x_ref)
```

where `ssf` is a `ScaleSpaceFlow` model.

The loss is:

```text
loss_x2 =
    lambda_MSE × MSE(x2_hat, x2)
    + lambda_cls × classification_loss(x2_hat)
```

All active components of this model are trained:

- motion encoder;
- motion decoder;
- residual encoder;
- P encoder;
- residual decoder.

The trained weights are saved in the P-frame checkpoint directory as:

```text
m_enc.pth
m_dec.pth
p_enc.pth
r_enc.pth
r_dec.pth
```

The normal `train_f2_stream2.py` stage is used to create the initial low-rate P-frame model. In the current script configuration, the optional pretrained path is not used, so this stage starts from a newly initialized model.

### `train_f2_universal_stream2.py`

This stage creates the **universal second-frame model**.

It loads the previously trained P-frame checkpoint produced by `train_f2_stream2.py` and continues training with:

```text
Frame 1 -> universal ssf -> reconstructed Frame 2
```

The model is created with encoder freezing enabled. In the current implementation, this causes the motion encoder, motion decoder, and residual encoder path to run without gradients. The differentiable trainable path is mainly:

- P encoder;
- residual decoder.

Therefore, `freeze_enc=True` freezes more than only the two encoder modules; it also prevents the motion decoder from being updated. This behavior is part of the current implementation and should be considered when interpreting the checkpoints.

This stage saves the universal second-frame checkpoint under a path of the form:

```text
saved_models/stream2/Pframe_z_dim_<z_dim>_L_<L>/
    Universal_dim_128_z_dim_<z_dim>_L_<L>_bs_64_lambdaMSE_1.0_lambdaCLS_<lambda_cls>/
```

The later stages use the checkpoint generated with `lambda_cls=0.01` as their fixed universal initialization/reference model.

### `train_f3_stream2.py`

This is the ordinary third-frame trainer. It loads the universal second-frame checkpoint into two models:

```text
ssf2: reconstructs Frame 2 from Frame 1
ssf3: reconstructs Frame 3 from reconstructed Frame 2
```

For every training sequence:

```text
x1 = Frame 1
x2 = Frame 2
x3 = Frame 3

x2_hat = ssf2(x2, x1)
x3_hat = ssf3(x3, x2_hat)
```

The `ssf2` result is used as the reference for the third-frame model. `ssf2` is kept fixed by:

- loading its pretrained universal weights;
- not including it in the optimizer;
- evaluating it under `torch.no_grad()`.

The third-frame model `ssf3` is initialized from the universal second-frame checkpoint and is then fine-tuned using:

```text
loss_x3 =
    lambda_MSE × MSE(x3_hat, x3)
    + lambda_cls × classification_loss(x3_hat)
```

All `ssf3` components are trainable in this script:

- motion encoder;
- motion decoder;
- residual encoder;
- P encoder;
- residual decoder.

The script evaluates all three frames, but only the third-frame loss is used for the optimizer update. It saves the trained third-frame model components as `m_enc.pth`, `m_dec.pth`, `p_enc.pth`, `r_enc.pth`, and `r_dec.pth`.

### `train_f3_universal_stream2.py`

This stage creates the universal third-frame model.

It loads:

```text
ssf2 <- universal second-frame checkpoint
ssf3 <- third-frame checkpoint produced by train_f3_stream2.py
```

The forward chain remains:

```text
x2_hat = ssf2(x2, x1)
x3_hat = ssf3(x3, x2_hat)
```

The second-frame model `ssf2` remains fixed. The universal third-frame stage selectively trains `ssf3`:

| Component | Status |
|---|---|
| `motion_encoder` | Frozen |
| `motion_decoder` | Trainable |
| `res_encoder` | Frozen |
| `P_encoder` | Trainable |
| `res_decoder` | Trainable |

The optimizer is constructed from only the parameters marked trainable. This allows the motion decoder to adapt while keeping the motion encoder and residual encoder fixed.

## Training order

The complete dependency order is:

```text
1. train_f2_stream2.py
   Train the initial low-rate Frame 2 P-frame model.

2. train_f2_universal_stream2.py
   Load the Frame 2 P-frame model and create the universal Frame 2 model.

3. train_f3_stream2.py
   Load the universal Frame 2 model into ssf2 and ssf3.
   Keep ssf2 fixed and train ssf3 for Frame 3 reconstruction.

4. train_f3_universal_stream2.py
   Load the Frame 2 universal model as ssf2.
   Load the trained Frame 3 model as ssf3.
   Keep the selected ssf3 encoders fixed and train the selected decoder-side
   and motion-decoder components.
```

The scripts use frames `0`, `1`, and `2` from the loaded sequence as Frames 1, 2, and 3. Although the data loader requests sequences of length `8`, these stream2 training scripts use only the first three frames for the described reconstruction chain.

## Preparing the environment

Before submitting a training job, complete every TODO in `grex.sh`:

```text
Results_without_disc/grex.sh
```

The current TODOs include:

- updating the output and error-log paths;
- deciding whether the data-loader worker count should remain `8`;
- creating the required virtual environment;
- updating the virtual-environment/source path if necessary.

The required modules and Python environment must be available on the cluster. The job script currently loads:

```text
arch/avx2
gcc/13.2.0
cuda/12.4.1
python/3.11.11
```

It then activates the configured virtual environment and checks the Python and PyTorch installations.

## Running the experiments

The job script is submitted through Slurm. From the directory containing `grex.sh`, first complete the environment and module TODOs, install all required dependencies, and verify that the dataset and checkpoint directories are available.

### Required first run

Run the classification-loss value `0.01` first:

```bash
sbatch grex.sh 0.01
```

Wait for this job to finish successfully before submitting the other runs.

This order is required because the later stages use models trained with `lambda_cls=0.01` as fixed pretrained models. In particular, the scripts contain checkpoint paths corresponding to the `0.01` universal model. The `0.01` run must therefore create the required model files before the other classification-loss experiments begin.

### Remaining runs

After the `0.01` job has completed, submit:

```bash
sbatch grex.sh 0.0
sbatch grex.sh 0.03
sbatch grex.sh 0.07
sbatch grex.sh 0.1
```

Each argument is passed to `grex.sh` as `LAMBDA_CLS`. The MSE weight remains fixed at `1.0`.

Do not start dependent runs before their required checkpoint-producing jobs have completed. Also avoid running multiple jobs that write to the same output directory at the same time.

## What `grex.sh` runs

The job script currently executes four active stages in sequence:

```text
Step 2: train_f2_stream2.py
Step 3: train_f2_universal_stream2.py
Step 3: train_f3_stream2.py
Step 4: train_f3_universal_stream2.py
```

The I-frame script is present in `grex.sh`, but its commands are currently commented out. This means the first-frame model is not trained by the current job script. The first frame is conceptually the high-rate reference frame, while the stream2 scripts consume the frame sequence directly for their P-frame experiments.

`set -e` is enabled in `grex.sh`. If any stage exits with an error, the remaining stages in that job stop immediately.

## Outputs

The models and experiment results are written below:

```text
saved_models/stream2/Pframe_z_dim_<z_dim>_L_<L>/
```

Each experiment directory can contain:

- model checkpoints:
  - `m_enc.pth`
  - `m_dec.pth`
  - `p_enc.pth`
  - `r_enc.pth`
  - `r_dec.pth`
- an epoch-performance CSV file;
- reconstruction preview images under `recons/`;
- saved input batches and reconstructed outputs in compressed `.npz` files.

The CSV records frame-wise distortion, classification loss, accuracy, and the configured MSE/classification weights. For the third-frame scripts, the model update is based on the third-frame reconstruction loss, while the Frame 1 and Frame 2 metrics are reported for comparison.

## Summary

Stream2 uses a high-rate first frame and low-rate predictive coding for the next frames:

```text
High-rate Frame 1
        |
        v
Low-rate ssf2 -> Frame 2 reconstruction
        |
        v
Low-rate ssf3 -> Frame 3 reconstruction
```

The essential rule for running the experiments is to complete the `grex.sh` TODOs, run `sbatch grex.sh 0.01` first, wait for all four stages to finish, and only then run the remaining values `0.0`, `0.03`, `0.07`, and `0.1`.