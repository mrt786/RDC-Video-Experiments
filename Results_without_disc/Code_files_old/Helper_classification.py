

import torch.nn.functional as F
import torchvision
import torch
import os

def save_recon_preview(ssf, vis_batch, device, out_dir, epoch, num_show=8):
    """
    Save reference, original current frame, and reconstruction separately.

    Uses a fixed batch captured once before training so previews are
    directly comparable across epochs.
    """

    was_training = ssf.training
    ssf.eval()

    x, _ = vis_batch

    # [B, H, W, T, C] -> [B, C, H, W, T]
    x = x.permute(0, 4, 1, 2, 3).to(device).float()

    # Reference frame and current frame
    x_ref = x[:, :, 0, ...]
    x_cur = x[:, :, 1, ...]

    # Reconstruct current frame using reference frame
    with torch.no_grad():
        x_hat = ssf(x_cur, x_ref)

    n = min(num_show, x.size(0))

    # Keep only the images we want to visualize
    x_ref = x_ref[:n]
    x_cur = x_cur[:n]
    x_hat = x_hat[:n]

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Save reference images
    reference_grid = torchvision.utils.make_grid(
        x_ref,
        nrow=n,
        padding=2
    )

    reference_path = os.path.join(
        out_dir,
        f"reference_epoch_{epoch + 1:04d}.png"
    )

    torchvision.utils.save_image(
        reference_grid,
        reference_path
    )

    # Save original current images
    current_grid = torchvision.utils.make_grid(
        x_cur,
        nrow=n,
        padding=2
    )

    current_path = os.path.join(
        out_dir,
        f"current_epoch_{epoch + 1:04d}.png"
    )

    torchvision.utils.save_image(
        current_grid,
        current_path
    )

    # Save reconstructed images
    reconstruction_grid = torchvision.utils.make_grid(
        x_hat,
        nrow=n,
        padding=2
    )

    reconstruction_path = os.path.join(
        out_dir,
        f"reconstruction_epoch_{epoch + 1:04d}.png"
    )

    torchvision.utils.save_image(
        reconstruction_grid,
        reconstruction_path
    )

    print(f"Saved reference:      {reference_path}")
    print(f"Saved current:       {current_path}")
    print(f"Saved reconstruction:{reconstruction_path}")

    if was_training:
        ssf.train()
