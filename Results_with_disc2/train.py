# Compressed I-frame
# Version with distortion and classification for each of the frames
import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
from models import *
from trainR1inf_universal import set_models_state
from utils import *
import torch.nn.functional as F
from helper import *
from CNN_MNIST import *
import math
import torch.autograd as autograd

cuda = True
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

parser = argparse.ArgumentParser('P-frame compressor training (Step 2)')
parser.add_argument('--total_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--bs', type=int, default=64, help='size of the batch')
parser.add_argument('--dim', type=int, default=128, help='common_dim')
parser.add_argument('--z_dim', type=int, default=1, help='z dim')
parser.add_argument('--L', type=int, default=2, help='quantization levels')
parser.add_argument('--skip_fq', type=int, default=1, help='loop frequency for generator update')
parser.add_argument('--lambda_MSE', type=float, default=1.0, help='Distortion loss weight')
parser.add_argument('--lambda_cls', type=float, default=0.0, help='Classification loss weight')
parser.add_argument('--path', type=str, default='./data/', help='Data Path')
parser.add_argument('--pre_path', type=str, default='None', help='Pretrained_Path (for ssf P-frame)')
parser.add_argument('--pre_path_iframe', type=str, default='None', help='Pretrained_Path for I-frame compressor')
parser.add_argument('--single_bit', type=int, default=0)
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--vis_every', type=int, default=5, help='Save reconstruction previews every N epochs')
parser.add_argument('--x1_mode', type=str, default='recon', help='Mode for x1: recon or original')
parser.add_argument('--lambda_adv', type=float, default=0.015, help='Adversarial loss weight')
parser.add_argument('--lambda_gp', type=float, default=10.0, help='WGAN-GP gradient penalty weight')
parser.add_argument('--n_critic', type=int, default=1, help='Discriminator updates per generator update')
parser.add_argument('--lr_critic', type=float, default=2e-4, help='Discriminator learning rate')
parser.add_argument('--beta1_critic', type=float, default=0.5, help='Discriminator Adam beta1')
parser.add_argument('--beta2_critic', type=float, default=0.9, help='Discriminator Adam beta2')
parser.add_argument(
    '--iframe_discriminator_path',
    type=str,
    default='None',
    help='I-frame discriminator.pth checkpoint or directory containing it',
)


def assert_image_shape(real_images, reconstructed_images):
    if real_images.shape != reconstructed_images.shape:
        raise RuntimeError(
            f"Image shape changed: real={tuple(real_images.shape)}, "
            f"reconstructed={tuple(reconstructed_images.shape)}"
        )

def set_models_state(list_models, state):
    if state == 'train':
        for model in list_models:
            model.train()
    else:
        for model in list_models:
            model.eval()


def evaluate(ssf, model_classifier, cross_entropy_criterion, test_loader, device, encoder, decoder):
    mse_loss = nn.MSELoss(reduction='sum')

    ssf.eval()
    
    model_classifier.eval()
    encoder.eval()
    decoder.eval()

    total_dist_x2 = 0.0
    total_cls_x2 = 0.0
    total_acc_x2 = 0.0
    
    total_dist_x1 = 0.0
    total_cls_x1 = 0.0
    total_acc_x1 = 0.0
    
    num_x = 0

    for i, (x, y) in enumerate(iter(test_loader)):
        with torch.no_grad():
            x = x.permute(0, 4, 1, 2, 3)
            x = x.to(device).float()
            y = y.to(device)

            x_cur = x[:, :, 1, ...]
            x_ref = x[:, :, 0, ...]
            
            hx = encoder(x_ref)
            x_ref_recon = decoder(hx[0])
            x_hat = ssf(x_cur, x_ref_recon)

            batch_size = x.size(0)
            channels = x.size(1)
            height = x.size(3)
            width = x.size(4)
            
            cls_x1 = model_classifier(x_ref_recon)
            total_dist_x1 += mse_loss(x_ref, x_ref_recon).item()
            total_cls_x1 += cross_entropy_criterion(cls_x1, y).item() * batch_size
            total_acc_x1 += (cls_x1.argmax(dim=1) == y).sum().item()

            
            cls_x2 = model_classifier(x_hat)

            total_dist_x2 += mse_loss(x_cur, x_hat).item()
            total_cls_x2 += cross_entropy_criterion(cls_x2, y).item() * batch_size
            total_acc_x2 += (cls_x2.argmax(dim=1) == y).sum().item()
            num_x += len(x)

    n_batches = len(test_loader)
    
    avg_dist_x1 = total_dist_x1 / (channels * height * width * num_x)
    avg_cls_x1 = total_cls_x1 / num_x
    avg_acc_x1 = total_acc_x1 / num_x



    avg_dist_x2 = total_dist_x2 / (channels * height * width * num_x)

    avg_cls_x2 = total_cls_x2 / num_x
    avg_acc_x2 = total_acc_x2 / num_x

    return (avg_dist_x1,
            avg_cls_x1, 
            avg_acc_x1,
            
            
            avg_dist_x2,
            avg_cls_x2, 
            avg_acc_x2)



def set_opt_zero(opts):
    for opt in opts:
        opt.zero_grad()

def preview_reconstructed(
    x_ref_recon,
    labels,
    out_dir,
    epoch,
    rate
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
        f"Image Size = {width}×{height} | "
        f"Rate = {rate:.2f}",
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
    


def save_pixel_value_preview(
    x_ref,
    x_cur,
    x_ref_recon,
    x_hat,
    labels,
    out_dir,
    epoch,
    rate
):
    """
    Save pixel values for ONLY the zeroth sample.

    Pixel values are displayed as integers [0,255].
    """

    # ONLY zeroth image
    x_ref = x_ref[0:1]
    x_cur = x_cur[0:1]
    x_ref_recon = x_ref_recon[0:1]
    x_hat = x_hat[0:1]
    labels = labels[0:1]

    height, width = x_ref.shape[-2:]

    print(f"Image size: {height} x {width}")

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4)
    )

    axes = np.expand_dims(axes, axis=0)

    col_titles = [
        'Frame 1\n(Source)',
        'Frame 2\n(Source)',
        'Frame 1\n(I-frame Recon)',
        'Frame 2\n(P-frame Recon)'
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(
            title,
            fontsize=11
        )

    digit_label = labels[0].item()

    imgs = [
        x_ref[0],
        x_cur[0],
        x_ref_recon[0],
        x_hat[0]
    ]

    for col_idx, img in enumerate(imgs):

        img = img.detach().cpu().squeeze().numpy()

        # [0,1] -> [0,255]
        pixel_values = np.clip(
            np.round(img * 255),
            0,
            255
        ).astype(np.uint8)

        axes[0, col_idx].imshow(
            pixel_values,
            cmap='gray',
            vmin=0,
            vmax=255
        )

        axes[0, col_idx].axis('off')

        # Write pixel values
        for row in range(height):
            for col in range(width):

                value = pixel_values[row, col]

                text_color = (
                    'white'
                    if value < 128
                    else 'black'
                )

                axes[0, col_idx].text(
                    col,
                    row,
                    str(value),
                    ha='center',
                    va='center',
                    fontsize=5,
                    color=text_color
                )

    axes[0, 0].set_ylabel(
        f"Digit {digit_label}",
        fontsize=11,
        rotation=0,
        labelpad=50,
        va='center'
    )

    fig.suptitle(
        f"Epoch {epoch + 1} | "
        f"Universal Pixel Values | "
        f"Image Size = {width}×{height} | "
        f"Rate = {rate:.2f}",
        fontsize=13,
        weight='bold'
    )

    plt.tight_layout()

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

    print(
        f"Saved pixel-value preview: {path}"
    )
    
    preview_reconstructed(
    x_ref_recon=x_ref_recon,
    labels=labels,
    out_dir=out_dir,
    epoch=epoch,
    rate=rate
)
     
    

def save_recon_preview(
    ssf,
    vis_batch,
    device,
    out_dir,
    epoch,
    encoder,
    decoder,
    rate,
    num_show=4
):
    """Save a PNG grid: source/reconstructed I-frame and P-frame."""

    was_training = ssf.training

    ssf.eval()

    x, labels = vis_batch

    x = x.permute(0, 4, 1, 2, 3).to(device).float()

    labels = labels.to(device)

    x_ref = x[:, :, 0, ...]
    x_cur = x[:, :, 1, ...]

    # ---------------------------------------------------------
    # Compute reconstructions ONCE
    # ---------------------------------------------------------
    with torch.no_grad():

        hx = encoder(x_ref)

        x_ref_recon = decoder(hx[0])

        x_hat = ssf(
            x_cur,
            x_ref_recon
        )

    # ---------------------------------------------------------
    # Number of images for normal preview
    # ---------------------------------------------------------
    n = min(
        num_show,
        x.size(0)
    )

    fig, axes = plt.subplots(
        n,
        4,
        figsize=(12, 4 * n)
    )

    if n == 1:
        axes = np.expand_dims(
            axes,
            axis=0
        )

    col_titles = [
        'Frame 1\n(Reference)',
        'Frame 2\n(Source)',
        'Frame 1\n(Reconstructed)',
        'Frame 2\n(P-frame Recon)'
    ]

    for col_idx, title in enumerate(col_titles):

        axes[0, col_idx].set_title(
            title,
            fontsize=10
        )

    # ---------------------------------------------------------
    # Normal reconstruction preview
    # ---------------------------------------------------------
    for idx in range(n):

        digit_label = labels[idx].item()

        imgs = [
            x_ref[idx],
            x_cur[idx],
            x_ref_recon[idx],
            x_hat[idx]
        ]

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

    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------
    fig.suptitle(
        f"Epoch {epoch + 1} | "
        f"P-Frame Reconstructions "
        f"(ref=I-frame recon) & Rate={rate:.2f}",
        fontsize=13,
        weight='bold'
    )

    plt.tight_layout()

    # ---------------------------------------------------------
    # Save normal reconstruction preview
    # ---------------------------------------------------------
    os.makedirs(
        out_dir,
        exist_ok=True
    )

    path = os.path.join(
        out_dir,
        f"recon_epoch_{epoch + 1:04d}.png"
    )

    plt.savefig(
        path,
        dpi=120,
        bbox_inches='tight'
    )

    plt.close(fig)

    print(
        f"Saved reconstruction preview: {path}"
    )

    # ---------------------------------------------------------
    # Pixel-value preview
    # ONLY zeroth image
    #
    # Uses the already computed tensors above.
    # No second encoder/decoder/SSF pass.
    # ---------------------------------------------------------
    save_pixel_value_preview(
        x_ref=x_ref,
        x_cur=x_cur,
        x_ref_recon=x_ref_recon,
        x_hat=x_hat,
        labels=labels,
        out_dir=out_dir,
        epoch=epoch,
        rate=rate
    )

    # ---------------------------------------------------------
    # Restore SSF training state
    # ---------------------------------------------------------
    if was_training:
        ssf.train()

def main():
    args = parser.parse_args()
    dim = args.dim
    z_dim = args.z_dim
    bs = args.bs
    skip_fq = args.skip_fq + 1 # Reference Image paper Experiment
    vis_every = args.vis_every
    total_epochs = args.total_epochs
    lambda_MSE = args.lambda_MSE
    lambda_cls = args.lambda_cls
    lambda_adv = args.lambda_adv
    lambda_gp = args.lambda_gp
    L = args.L
    path = args.path
    pre_path = args.pre_path
    dataset = args.dataset
    x1_mode = args.x1_mode
    
    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

    if dataset == 'MNIST':
        activation = torch.sigmoid
    else:
        activation = torch.tanh

    model_classifier = Net().to(device)
    trained_model_path = "./trained_classifier.pth"
    
    if os.path.exists(trained_model_path):
        model_classifier.load_state_dict(
            torch.load(
                trained_model_path,
                map_location=device
            )
        )
        print(f"Loaded trained classifier from {trained_model_path}")
    else:
        model_classifier = classifier().to(device)
        print("Warning: trained classifier not found; using new classifier.")
        
    model_classifier.eval()

    for param in model_classifier.parameters():
        param.requires_grad = False
        
    cross_entropy_criterion = nn.CrossEntropyLoss()

    ssf = ScaleSpaceFlow(num_levels=1, dim=z_dim, stochastic=True, quantize_latents=True, L=L)
    discriminator = DiscriminatorMNIST(n_channel=1).to(device)
    # discriminator_path =  "./saved_models/Disc/compressed/Iframe_z_dim_12_L_2/Iframe_dim_128_z_dim_12_L_2_single_bit_False_bs_64_lambdaMSE_1.0_lambdaCLS_0.01_lambdaADV_1.0_lambdaGP_10.0/discriminator.pth"
    
    # load_iframe_discriminator(discriminator, discriminator_path, device)


    
    I_dim = 12
    I_L = 2
    folder_name = (
        f"Pframe_dim_{dim}"
        f"_z_dim_{z_dim}"
        f"_L_{L}"
        f"_bs_{bs}"
        f"_lambdaMSE_{lambda_MSE}"
        f"_lambdaCLS_{lambda_cls}"
    )
    
    
    print("Settings: ", folder_name)
    # change it to I_dim & L
    save_dir = f'./saved_models/Disc/compressed/Iframe_z_dim_{I_dim}_L_{I_L}/' + folder_name
    os.makedirs(save_dir, exist_ok=True)


    if pre_path != 'None':
        #prefix_path = 'z'+str(z_dim)+'l'+str(L)+'_MMSE'
        ssf.motion_encoder.load_state_dict(torch.load(pre_path + '/m_enc.pth'))
        ssf.motion_decoder.load_state_dict(torch.load(pre_path + '/m_dec.pth'))
        ssf.P_encoder.load_state_dict(torch.load(pre_path + '/p_enc.pth'))
        ssf.res_encoder.load_state_dict(torch.load(pre_path + '/r_enc.pth'))
        ssf.res_decoder.load_state_dict(torch.load(pre_path + '/r_dec.pth'))

    
    encoder = Encoder(dim=I_dim, nc=1, stochastic=True, quantize_latents=True, L=I_L) #Generator Side
    decoder = Decoder_Iframe(dim=I_dim)
    encoder.cuda()
    decoder.cuda()
    encoder.eval()
    decoder.eval()
    encoder.load_state_dict(torch.load(f'./saved_models/Disc/compressed/Iframe_z_dim_{I_dim}_L_{I_L}/Iframe_dim_128_z_dim_{I_dim}_L_{I_L}_single_bit_False_bs_64_lambdaMSE_1.0_lambdaCLS_0.01/encoder.pth'))
    decoder.load_state_dict(torch.load(f'./saved_models/Disc/compressed/Iframe_z_dim_{I_dim}_L_{I_L}/Iframe_dim_128_z_dim_{I_dim}_L_{I_L}_single_bit_False_bs_64_lambdaMSE_1.0_lambdaCLS_0.01/decoder.pth'))
    
    
    
    list_models = [ssf, discriminator]
    ssf.to(device)

    train_loader, test_loader = get_dataloader(data_root=path, seq_len=8, batch_size=bs, num_digits=1)
    mse = torch.nn.MSELoss()

    opt_ssf = torch.optim.RMSprop(ssf.parameters(), lr=5e-5)
    opt_discriminator = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr_critic,
        betas=(args.beta1_critic, args.beta2_critic),
    )
    list_opt = [opt_ssf]
    rate = math.log(z_dim * 2 * L, 2)
    csv_path = os.path.join(save_dir, f"Rate_{rate:.2f}_CLS_{lambda_cls}_performance.csv")
    csv_file = open(csv_path, 'w')
    csv_file.write("Epoch, Accuracy1, Distortion Loss1, Classification Loss1, Accuracy2, Distortion Loss2, Classification Loss2,    Lambda_MSE, Lambda_CLS\n")
    
    recon_dir = os.path.join(save_dir, 'recons')
    vis_batch = next(iter(test_loader))

    batches_done = 0
    
    discriminator_loss = 0
    
    for epoch in range(total_epochs):
        set_models_state(list_models, 'train')

        for i, (x, y) in enumerate(iter(train_loader)):
            set_opt_zero(list_opt)

            x = x.permute(0, 4, 1, 2, 3)
            x = x.to(device).float()
            y = y.to(device)

            x_cur = x[:, :, 1, ...]
            x_ref = x[:, :, 0, ...]
            
            
            if i % skip_fq  != 0:
                # ============================================================
                # 1. Train Discriminator
                # ============================================================    

                for parameter in ssf.parameters():
                    parameter.requires_grad = False
                for parameter in discriminator.parameters():
                    parameter.requires_grad = True
                
                
                with torch.no_grad():
                    hx = encoder(x_ref)
                    x_ref_recon = decoder(hx[0])
                    x_hat = ssf(x_cur, x_ref_recon)
                assert_image_shape(x_cur, x_hat)
                
                opt_discriminator.zero_grad()
                real_validity = discriminator(x_cur)
                fake_validity = discriminator(x_hat.detach())
                
                gradient_penalty = compute_gradient_penalty(
                    discriminator, x_cur, x_hat.detach()
                )
                
                discriminator_loss = (
                    -torch.mean(real_validity)
                    + torch.mean(fake_validity)
                    + lambda_gp * gradient_penalty
                )
                discriminator_loss.backward()
                opt_discriminator.step()

            else:
                for parameter in discriminator.parameters():
                    parameter.requires_grad = False
                    
                    
                for parameter in ssf.parameters():
                    parameter.requires_grad = True
                
                
                
                set_opt_zero([opt_ssf])
                # Recompute after enabling SSF gradients. The tensor used for
                # discriminator training was intentionally detached/frozen.
                
                with torch.no_grad():
                    hx = encoder(x_ref)
                    x_ref_recon = decoder(hx[0])
                
                x_hat = ssf(x_cur, x_ref_recon)
                assert_image_shape(x_cur, x_hat)
                
                
                fake_validity = discriminator(x_hat)
                perception_loss_x2 = -fake_validity.mean()
                
                distortion_loss_x1 = mse(x_ref, x_ref_recon)
                classifier_output_x1 = model_classifier(x_ref_recon)
                classification_loss_x1 = cross_entropy_criterion(classifier_output_x1, y)



                distortion_loss_x2 = mse(x_hat, x_cur)
                classifier_output_x2 = model_classifier(x_hat)
                classification_loss_x2 = cross_entropy_criterion(classifier_output_x2, y)
                
                
                loss_x2 = (
                    lambda_MSE * distortion_loss_x2
                    + lambda_cls * classification_loss_x2
                    + lambda_adv * perception_loss_x2
                )
                
                loss_x2.backward()
                opt_ssf.step()

                
                if batches_done % 100 == 0:
                    print(f"Epoch [{epoch}/{total_epochs}]  Batch [{i}/{len(train_loader)}] "
                                                        f"Discriminator: {discriminator_loss:.4f} "
                                                        f"Distortion: {distortion_loss_x1.item():.4f} "
                                                        f"Classification: {classification_loss_x1.item():.4f} "
                                                        f"Adversarial: {perception_loss_x2.item():.4f}"
                                                        
                                        f"Loss: {loss_x2.item():.4f} Distortion: {distortion_loss_x2.item():.4f} "
                                        f"Classification: {classification_loss_x2.item():.4f}")
            batches_done += 1        
                
        (dist_mse_x1, cls_mse_x1, acc_mse_x1,dist_mse_x2,
                 cls_mse_x2, 
                 acc_mse_x2) = evaluate(ssf, model_classifier, cross_entropy_criterion, test_loader, device, encoder, decoder)

        csv_file.write(
                   # epoch+1,  accuracy1, distortion loss1, classification loss1, accuracy2, distortion loss2, classification loss2, lambda_MSE, lambda_CLS
                   f"{epoch + 1},{acc_mse_x1:.4f},{dist_mse_x1:.4f}, "
                   f"{cls_mse_x1:.4f},{acc_mse_x2:.4f},{dist_mse_x2:.4f}, "
                   f"{cls_mse_x2:.4f},"
                   f"{lambda_MSE},{lambda_cls}\n"
               )
        
        
        csv_file.flush()

        print(f"Epoch: {epoch + 1} | "
                      f"MSE [Dist: {dist_mse_x1:.4f}, Cls: {cls_mse_x1:.4f}, Acc: {acc_mse_x1:.4f}] | "
                      f"MSE [Dist: {dist_mse_x2:.4f}, Cls: {cls_mse_x2:.4f}, Acc: {acc_mse_x2:.4f}] | "
                      f"Lambda_MSE: {lambda_MSE} Lambda_CLS: {lambda_cls}")
        
        
        

        if (epoch + 1) % vis_every == 0 or epoch == 0:
            save_recon_preview(ssf, vis_batch, device, recon_dir, epoch, encoder, decoder, rate)

    csv_file.close()

    set_models_state(list_models, 'eval')

    torch.save(ssf.motion_encoder.state_dict(), os.path.join(save_dir, 'm_enc.pth'))
    torch.save(ssf.motion_decoder.state_dict(), os.path.join(save_dir, 'm_dec.pth'))
    torch.save(ssf.P_encoder.state_dict(), os.path.join(save_dir, 'p_enc.pth'))
    torch.save(ssf.res_encoder.state_dict(), os.path.join(save_dir, 'r_enc.pth'))
    torch.save(ssf.res_decoder.state_dict(), os.path.join(save_dir, 'r_dec.pth'))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, 'discriminator.pth'))

    for i, (x, y) in enumerate(iter(train_loader)):
        x = x.permute(0, 4, 1, 2, 3)
        x = x.to(device).float()
        break
    np.savez_compressed(os.path.join(save_dir, "x"), a=x.detach().cpu().numpy())

    for i in range(5):
        x_cur = x[:, :, 1, ...]
        x_ref = x[:, :, 0, ...]
        with torch.no_grad():
            hx = encoder(x_ref)
            x_ref_recon = decoder(hx[0])
            
        x_hat = ssf(x_cur, x_ref_recon)
        np.savez_compressed(os.path.join(save_dir, "x_hat" + str(i)), a=x_hat.detach().cpu().numpy())


if __name__ == "__main__":
    main()
