# No discriminator, no adversarial loss. Only distortion loss and classification loss.

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
from models import *
from utils import *
import torch.nn.functional as F
from helper import *
import argparse
import os

from CNN_MNIST import *

cuda = True
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

parser = argparse.ArgumentParser('I-frame compressor training (Step 1)')
parser.add_argument('--total_epochs', type=int, default=2000, help='number of epochs of training')
parser.add_argument('--bs', type=int, default=64, help='size of the batch')
parser.add_argument('--dim', type=int, default=128, help='latent dimension')
parser.add_argument('--z_dim', type=int, default=4, help='z dim')
parser.add_argument('--L', type=int, default=16, help='quantization levels')
parser.add_argument('--skip_fq', type=int, default=5, help='loop frequency for generator update')
parser.add_argument('--lambda_MSE', type=float, default=1.0, help='Distortion loss weight')
parser.add_argument('--lambda_cls', type=float, default=0.0, help='Classification loss weight')
parser.add_argument('--path', type=str, default='./data/', help='Data Path')
parser.add_argument('--pre_path', type=str, default='None', help='Pretrained_Path')
parser.add_argument('--single_bit', type=int, default=0)
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--vis_every', type=int, default=20, help='Save reconstruction previews every N epochs')


def evaluate(
    encoder,
    decoder,
    model_classifier,
    cross_entropy_criterion,
    test_loader,
    device
):
    mse_loss = nn.MSELoss(reduction='sum')

    encoder.eval()
    decoder.eval()
    model_classifier.eval()

    total_dist = 0.0
    total_cls = 0.0
    total_correct = 0
    num_x = 0

    with torch.no_grad():

        for x, y in test_loader:

            x = x.permute(0, 4, 1, 2, 3)
            x = x.to(device).float()
            y = y.to(device)

            x1 = x[:, :, 0, ...]

            hx = encoder(x1)
            x1_hat = decoder(hx[0])

            batch_size = x1.size(0)
            channels = x1.size(1)
            height = x1.size(2)
            width = x1.size(3)

            cls_output = model_classifier(x1_hat)

            total_dist += mse_loss(
                x1,
                x1_hat
            ).item()

            total_cls += (
                cross_entropy_criterion(
                    cls_output,
                    y
                ).item() * batch_size
            )

            total_correct += (
                (cls_output.argmax(dim=1) == y)
                .sum()
                .item()
            )

            num_x += batch_size

    num_pixels = (
        channels *
        height *
        width *
        num_x
    )

    avg_dist = total_dist / num_pixels
    avg_cls = total_cls / num_x
    avg_acc = total_correct / num_x

    return avg_dist, avg_cls, avg_acc
def set_models_state(list_models, state):
    if state == 'train':
        for model in list_models:
            model.train()
    else:
        for model in list_models:
            model.eval()


def set_opt_zero(opts):
    for opt in opts:
        opt.zero_grad()



        
        
def preview_reconstructed(
    x_ref_recon,
    labels,
    out_dir,
    epoch
):
    """
    Save a PNG preview showing pixel values of ONLY x_ref_recon[0].

    The reconstructed I-frame is converted from [0,1]
    to integer pixel values [0,255].
    """

    # ---------------------------------------------------------
    # Use ONLY the zeroth reconstructed image
    # ---------------------------------------------------------
    img = x_ref_recon[0]

    # ---------------------------------------------------------
    # Image size
    # ---------------------------------------------------------
    height, width = img.shape[-2:]

    print(f"Image size: {height} x {width}")

    # ---------------------------------------------------------
    # Convert [1,H,W] -> [H,W]
    # ---------------------------------------------------------
    img = (
        img.detach()
        .cpu()
        .squeeze()
        .numpy()
    )

    # ---------------------------------------------------------
    # Convert [0,1] -> integer [0,255]
    # ---------------------------------------------------------
    pixel_values = np.clip(
        np.round(img * 255),
        0,
        255
    ).astype(np.uint8)

    # ---------------------------------------------------------
    # Create figure
    # ---------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    # ---------------------------------------------------------
    # Display reconstructed image
    # ---------------------------------------------------------
    ax.imshow(
        pixel_values,
        cmap='gray',
        vmin=0,
        vmax=255
    )

    ax.axis('off')

    # ---------------------------------------------------------
    # Write pixel values
    # ---------------------------------------------------------
    for row in range(height):
        for col in range(width):

            value = pixel_values[row, col]

            text_color = (
                'white'
                if value < 128
                else 'black'
            )

            ax.text(
                col,
                row,
                str(value),
                ha='center',
                va='center',
                fontsize=5,
                color=text_color
            )

    # ---------------------------------------------------------
    # Digit label
    # ---------------------------------------------------------
    digit_label = labels[0].item()

    ax.set_title(
        f"Frame 1 (I-frame Recon) | Digit {digit_label}",
        fontsize=12
    )

    # ---------------------------------------------------------
    # Figure title
    # ---------------------------------------------------------
    fig.suptitle(
        f"Epoch {epoch + 1} | "
        f"Reconstructed I-frame Pixel Values | "
        f"Image Size = {width}×{height} | ",
        fontsize=13,
        weight='bold'
    )

    plt.tight_layout()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    os.makedirs(
        out_dir,
        exist_ok=True
    )

    path = os.path.join(
        out_dir,
        f"preview_reconstructed_epoch_{epoch + 1:04d}.png"
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches='tight'
    )

    plt.close(fig)

    print(
        f"Saved reconstructed pixel preview: {path}"
    )



def save_pixel_value_preview_iframe(
    x1,
    x1_hat,
    labels,
    out_dir,
    epoch,
    lambda_cls
):
    """
    Save pixel values for one or more I-frame source/reconstruction pairs.

    x1:
        Original I-frame, shape [B, C, H, W]

    x1_hat:
        Reconstructed I-frame, shape [B, C, H, W]

    labels:
        MNIST labels, shape [B]
    """

    n = x1.size(0)

    # ---------------------------------------------------------
    # Image size
    # ---------------------------------------------------------
    height, width = x1.shape[-2:]

    print(f"Image size: {height} x {width}")

    # ---------------------------------------------------------
    # Create figure
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        n,
        2,
        figsize=(10, 4 * n)
    )

    # Important: when n == 1, axes is 1-dimensional
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        'Frame 1\n(Source)',
        'Frame 1\n(I-frame Recon)'
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(
            title,
            fontsize=11
        )

    # ---------------------------------------------------------
    # Draw source + reconstruction
    # ---------------------------------------------------------
    for idx in range(n):

        digit_label = labels[idx].item()

        imgs = [
            x1[idx],
            x1_hat[idx]
        ]

        for col_idx, img in enumerate(imgs):

            # [1,H,W] -> [H,W]
            img = img.detach().cpu().squeeze().numpy()

            # [0,1] -> [0,255]
            pixel_values = np.clip(
                np.round(img * 255),
                0,
                255
            ).astype(np.uint8)

            axes[idx, col_idx].imshow(
                pixel_values,
                cmap='gray',
                vmin=0,
                vmax=255
            )

            axes[idx, col_idx].axis('off')

            # -------------------------------------------------
            # Write pixel values
            # -------------------------------------------------
            for row in range(height):
                for col in range(width):

                    value = pixel_values[row, col]

                    text_color = (
                        'white'
                        if value < 128
                        else 'black'
                    )

                    axes[idx, col_idx].text(
                        col,
                        row,
                        str(value),
                        ha='center',
                        va='center',
                        fontsize=5,
                        color=text_color
                    )

        axes[idx, 0].set_ylabel(
            f"Digit {digit_label}",
            fontsize=11,
            rotation=0,
            labelpad=50,
            va='center'
        )

    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------
    fig.suptitle(
        f"Epoch {epoch + 1} | "
        f"I-frame Pixel Values | "
        f"Image Size = {width}×{height} | "
        f"Lambda_Cls = {lambda_cls}",
        fontsize=13,
        weight='bold'
    )

    plt.tight_layout()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(
        out_dir,
        f"pixel_epoch_{epoch + 1:04d}.png"
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches='tight'
    )

    plt.close(fig)

    print(f"Saved pixel-value preview: {path}")
    
    
    preview_reconstructed(
        x_ref_recon=x1_hat,
        labels=labels,
        out_dir=out_dir,
        epoch=epoch,
    )

    


def save_recon_preview_iframe(encoder, decoder, vis_batch, device, out_dir, epoch, lambda_cls, num_show=3):
    """Save a PNG grid: source x1 | reconstructed x1_hat."""
    was_training_encoder = encoder.training
    was_training_decoder = decoder.training

    encoder.eval()
    decoder.eval()
    

    x, y = vis_batch
    x = x.permute(0, 4, 1, 2, 3).to(device).float()
    y = y.to(device)
    x1 = x[:, :, 0, ...]

    with torch.no_grad():
        hx = encoder(x1)
        x1_hat = decoder(hx[0])

    n = min(num_show, x.size(0))

    fig, axes = plt.subplots(n, 2, figsize=(9, 3 * n))
    
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ['Frame 1\n(Source)', 'Frame 1\n(Recon)']
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=10)

    
    for idx in range(n):
        digit_label = y[idx].item()
        imgs = [x1[idx], x1_hat[idx]]

        for col_idx, img in enumerate(imgs):
            axes[idx, col_idx].imshow(
                img.detach().cpu().squeeze().numpy(),
                cmap='gray',
                vmin=0,
                vmax=1
            )
            axes[idx, col_idx].axis('off')

        axes[idx, 0].set_ylabel(
            f"Digit {digit_label}",
            fontsize=11,
            rotation=0,
            labelpad=50,
            va='center'
        )
        
    
    fig.suptitle(f"Epoch {epoch + 1} | I-Frame Reconstructions at Lambda_Cls {lambda_cls} ", fontsize=13, weight='bold')
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"recon_epoch_{epoch + 1:04d}.png")
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved reconstruction preview: {path}")
    
    save_pixel_value_preview_iframe(
        x1[0:1],x1_hat[0:1], y[0:1] , out_dir, epoch, lambda_cls
    )
    
    
    if was_training_encoder:
        encoder.train()

    if was_training_decoder:
        decoder.train()


def main():
    args = parser.parse_args()
    dim = args.dim
    z_dim = args.z_dim
    bs = args.bs
    skip_fq = args.skip_fq
    vis_every = args.vis_every
    total_epochs = args.total_epochs
    lambda_MSE = args.lambda_MSE
    lambda_cls = args.lambda_cls
    L = args.L
    path = args.path
    pre_path = args.pre_path
    single_bit = bool(args.single_bit)
    dataset = args.dataset

    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

    if dataset == 'MNIST':
        activation = torch.sigmoid
    else:
        activation = torch.tanh

    model_classifier = Net().to(device)
    trained_model_path = "./trained_classifier.pth"
    if os.path.exists(trained_model_path):
        model_classifier.load_state_dict(torch.load(trained_model_path, map_location=device))
        model_classifier.eval()
        print(f"Loaded trained classifier from {trained_model_path}")
    else:
        model_classifier = classifier()
    
    for param in model_classifier.parameters():
        param.requires_grad = False
        

    cross_entropy_criterion = nn.CrossEntropyLoss()

    encoder = Encoder(dim=z_dim, nc=1, stochastic=True, quantize_latents=True, L=L)
    decoder = Decoder_Iframe(dim=z_dim)

    folder_name = (
        f"Iframe_dim_{dim}"
        f"_z_dim_{z_dim}"
        f"_L_{L}"
        f"_single_bit_{single_bit}"
        f"_bs_{bs}"
        f"_lambdaMSE_{lambda_MSE}"
        f"_lambdaCLS_{lambda_cls}"
    )
    print("Settings: ", folder_name)

    save_dir = f'./saved_models/compressed/Iframe_z_dim_{z_dim}_L_{L}/' + folder_name
    os.makedirs(save_dir, exist_ok=True)

    if pre_path != 'None':
        encoder.load_state_dict(torch.load(pre_path + '/encoder.pth'))
        decoder.load_state_dict(torch.load(pre_path + '/decoder.pth'))

    list_models = [encoder, decoder]
    encoder.to(device)
    decoder.to(device)
    train_loader, test_loader = get_dataloader(data_root=path, seq_len=2, batch_size=bs, num_digits=1)
    mse = torch.nn.MSELoss()

    opt_e = torch.optim.RMSprop(encoder.parameters(), lr=1e-4)
    opt_g = torch.optim.RMSprop(decoder.parameters(), lr=1e-4)
    list_opt = [opt_e, opt_g]

    csv_path = os.path.join(save_dir, f"Rate_{z_dim}_CLS_{lambda_cls}_performance.csv")
    csv_file = open(csv_path, 'w')
    csv_file.write("Epoch, Accuracy, Distortion Loss, Classification Loss, Lambda_MSE, Lambda_CLS\n")

    recon_dir = os.path.join(save_dir, 'recons')
    vis_batch = next(iter(test_loader))

    batches_done = 0


    for epoch in range(total_epochs):
        set_models_state(list_models, 'train')

        for i, (x, y) in enumerate(iter(train_loader)):
            set_opt_zero(list_opt)

            x = x.permute(0, 4, 1, 2, 3)
            x = x.to(device).float()
            y = y.to(device)

            x1 = x[:, :, 0, ...]
            hx = encoder(x1)
            x1_hat = decoder(hx[0])

            if i % skip_fq == 0:
                distortion_loss = mse(x1_hat, x1)

                classifier_output = model_classifier(x1_hat)

                classification_loss = cross_entropy_criterion(
                    classifier_output,
                    y
                )

                loss = (
                    lambda_MSE * distortion_loss
                    + lambda_cls * classification_loss
                )

                loss.backward()

                opt_e.step()
                opt_g.step()

                if batches_done % 100 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{total_epochs}] "
                        f"Batch [{i}/{len(train_loader)}] "
                        f"Loss: {loss.item():.4f} "
                        f"Distortion: {distortion_loss.item():.4f} "
                        f"Classification: {classification_loss.item():.4f}"
                )
            batches_done += 1
            
            
        dist, cls, acc = evaluate(encoder, decoder, model_classifier, cross_entropy_criterion, test_loader, device)

        csv_file.write(
            f"{epoch + 1}, {acc:.4f},{dist:.4f}, "
            f"{cls:.4f},"
            f"{lambda_MSE},{lambda_cls}\n"
        )
        csv_file.flush()

        print(f"Epoch: {epoch + 1} | "
              f"Frame1 [Dist: {dist:.4f}, Cls: {cls:.4f}, Acc: {acc:.4f}] | "
              f"Lambda_MSE: {lambda_MSE} Lambda_CLS: {lambda_cls}")

        if (epoch + 1) % vis_every == 0 or epoch == 0:
            save_recon_preview_iframe(encoder, decoder, vis_batch, device, recon_dir, epoch, lambda_cls)

    csv_file.close()

    set_models_state(list_models, 'eval')

    torch.save(encoder.state_dict(), os.path.join(save_dir, 'encoder.pth'))
    torch.save(decoder.state_dict(), os.path.join(save_dir, 'decoder.pth'))

    for i, (x, y) in enumerate(iter(train_loader)):
        x = x.permute(0, 4, 1, 2, 3)
        x = x.to(device).float()
        break
    np.savez_compressed(os.path.join(save_dir, "x"), a=x.detach().cpu().numpy())

    for i in range(5):
        x1 = x[:, :, 0, ...]
        with torch.no_grad():
            hx = encoder(x1)
            x1_hat = decoder(hx[0])
        np.savez_compressed(os.path.join(save_dir, "x_hat" + str(i)), a=x1_hat.detach().cpu().numpy())


if __name__ == "__main__":
    main()


