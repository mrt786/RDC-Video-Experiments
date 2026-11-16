import argparse
import os
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
from CNN_MNIST import *
from Helper_classification import *
import math
import torch.autograd as autograd

cuda = True
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

parser = argparse.ArgumentParser('Universal representation training (Step 2)')
parser.add_argument('--total_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--bs', type=int, default=64, help='size of the batch')
parser.add_argument('--dim', type=int, default=128, help='common_dim')
parser.add_argument('--z_dim', type=int, default=1, help='z dim')
parser.add_argument('--L', type=int, default=2, help='quantization levels')
parser.add_argument('--skip_fq', type=int, default=1, help='loop frequency for generator update')
parser.add_argument('--lambda_MSE', type=float, default=1.0, help='Distortion loss weight')
parser.add_argument('--lambda_cls', type=float, default=0.1, help='Classification loss weight')
parser.add_argument('--path', type=str, default='./data/', help='Data Path')
parser.add_argument('--pre_path', type=str, default='./fixed_models/', help='Pretrained_Path (MSE checkpoint)')
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--vis_every', type=int, default=5, help='Save reconstruction previews every N epochs')
parser.add_argument('--lambda_adv', type=float, default=0.015, help='Adversarial loss weight')
parser.add_argument('--lambda_gp', type=float, default=10.0, help='WGAN-GP gradient penalty weight')
parser.add_argument('--n_critic', type=int, default=1, help='Discriminator updates per generator update')
parser.add_argument('--lr_critic', type=float, default=2e-4, help='Discriminator learning rate')
parser.add_argument('--beta1_critic', type=float, default=0.5, help='Discriminator Adam beta1')
parser.add_argument('--beta2_critic', type=float, default=0.9, help='Discriminator Adam beta2')

def assert_image_shape(real_images, reconstructed_images):
    if real_images.shape != reconstructed_images.shape:
        raise RuntimeError(
            f'Image shape changed: real={tuple(real_images.shape)}, '
            f'reconstructed={tuple(reconstructed_images.shape)}'
        )


def set_models_state(list_models, state):
    if state == 'train':
        for model in list_models:
            model.train()
    else:
        for model in list_models:
            model.eval()



"""

    Returns the average distortion, classification loss, and accuracy for both the reference frame and the current frame after evaluating the model on the test dataset.
"""

def evaluate(
    ssf2,ssf3,
    model_classifier,
    cross_entropy_criterion,
    test_loader,
    device, encoder , decoder
):
    mse_loss = nn.MSELoss(reduction='sum')

    set_models_state([ssf2, ssf3, model_classifier], 'eval')

    total_dist_x1 = 0.0
    total_cls_x1 = 0.0
    total_correct_x1 = 0

    total_dist_x2 = 0.0
    total_cls_x2 = 0.0
    total_correct_x2 = 0

    total_dist_x3 = 0.0
    total_cls_x3 = 0.0
    total_correct_x3 = 0
    
    num_x = 0

    with torch.no_grad():
        for x, y in test_loader:

            x = x.permute(0, 4, 1, 2, 3)
            x = x.to(device).float()
            y = y.to(device)

            x1 = x[:, :, 0, ...]
            x2 = x[:, :, 1, ...]
            x3 = x[:, :, 2, ...]
            hx1 = encoder(x1)
            x1_hat = decoder(hx1[0])
        
            x2_hat = ssf2(
                x2,
                x1_hat
            )

            x3_hat = ssf3(
                x3, x2_hat
            )
            batch_size = x1.size(0)

            channels = x1.size(1)
            height = x1.size(2)
            width = x1.size(3)

            # -------------------------------------------------
            # Frame 1 Evaluation
            # -------------------------------------------------
            cls_x1 = model_classifier(x1_hat)

            total_dist_x1 += mse_loss(
                x1,
                x1_hat
            ).item()

            total_cls_x1 += (
                cross_entropy_criterion(
                    cls_x1,
                    y
                ).item() * batch_size
            )

            total_correct_x1 += (
                (cls_x1.argmax(dim=1) == y)
                .sum()
                .item()
            )

            # -------------------------------------------------
            # Frame 2 Evaluation
            # -------------------------------------------------
            cls_x2 = model_classifier(x2_hat)

            total_dist_x2 += mse_loss(
                x2,
                x2_hat
            ).item()

            total_cls_x2 += (
                cross_entropy_criterion(
                    cls_x2,
                    y
                ).item() * batch_size
            )

            total_correct_x2 += (
                (cls_x2.argmax(dim=1) == y)
                .sum()
                .item()
            )
            # -------------------------------------------------
            # Frame 3 Evaluation
            # -------------------------------------------------
            cls_x3 = model_classifier(x3_hat)

            total_dist_x3 += mse_loss(
                x3,
                x3_hat
            ).item()

            total_cls_x3 += (
                cross_entropy_criterion(
                    cls_x3,
                    y
                ).item() * batch_size
            )

            total_correct_x3 += (
                (cls_x3.argmax(dim=1) == y)
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

    avg_dist_x1 = total_dist_x1 / num_pixels
    avg_cls_x1 = total_cls_x1 / num_x
    avg_acc_x1 = total_correct_x1 / num_x

    avg_dist_x2 = total_dist_x2 / num_pixels
    avg_cls_x2 = total_cls_x2 / num_x
    avg_acc_x2 = total_correct_x2 / num_x
    
    avg_dist_x3 = total_dist_x3 / num_pixels
    avg_cls_x3 = total_cls_x3 / num_x
    avg_acc_x3 = total_correct_x3 / num_x

    return (
        avg_dist_x1,
        avg_cls_x1,
        avg_acc_x1,
        avg_dist_x2,
        avg_cls_x2,
        avg_acc_x2,
        avg_dist_x3,
        avg_cls_x3,
        avg_acc_x3
        
    )

def set_opt_zero(opts):
    for opt in opts:
        opt.zero_grad()


        
def save_recon_preview_universal(
    ssf2, ssf3,
    vis_batch,
    device,
    out_dir,
    epoch,
    rate,encoder,
    decoder,
    num_show=4
):
    """Save a compact PNG grid of reference, source, and P-frame reconstructions."""

    was_training = ssf3.training
    ssf3.eval()

    x, y = vis_batch

    x = x.permute(0, 4, 1, 2, 3).to(device).float()
    y = y.to(device)

    x1 = x[:, :, 0, ...]
    x2 = x[:, :, 1, ...]
    x3 = x[:, :, 2, ...]

    # ---------------------------------------------------------
    # Compute reconstructions ONCE
    # ---------------------------------------------------------
    with torch.no_grad():
        hx = encoder(x1)
        x1_hat = decoder(hx[0])
        x2_hat = ssf2(
            x2,
            x1_hat
        )
        x3_hat = ssf3(
            x3, x2_hat
        )

    # ---------------------------------------------------------
    # Number of images for normal preview
    # ---------------------------------------------------------
    n = min(num_show, x.size(0))

    fig, axes = plt.subplots(
        n,
        6,
        figsize=(10,8),
        gridspec_kw={
            'wspace': 0.015,
            'hspace': 0.015,
        },
        facecolor='white'
    )

    if n == 1:
        axes = np.expand_dims(
            axes,
            axis=0
        )

    # col_titles = [
    #     'Frame 1\n(Reference)',
    #     'Frame 1\n(Recon)',
    #     'Frame 2\n(Recon)'
    # ]

    # for col_idx, title in enumerate(col_titles):

    #     axes[0, col_idx].set_title(
    #         title,
    #         fontsize=10
    #     )

    # ---------------------------------------------------------
    # Normal reconstruction preview
    # ---------------------------------------------------------
    for idx in range(n):

        imgs = [
            x1[idx],
            x2[idx],
            x3[idx],
            x1_hat[idx],
            x2_hat[idx],
            x3_hat[idx]
        ]

        for col_idx, img in enumerate(imgs):

            axes[idx, col_idx].imshow(
                img.detach().cpu().squeeze().numpy(),
                cmap='gray',
                vmin=0,
                vmax=1,
                interpolation='nearest'
            )

            axes[idx, col_idx].axis('off')

    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------
    # fig.suptitle(
    #     f"Epoch {epoch + 1} | "
    #     f"Universal Reconstructions "
    #     f"(ref=I-frame recon) & Rate={rate:.2f}",
    #     fontsize=13,
    #     weight='bold'
    # )

    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1,
        wspace=0.015,
        hspace=0.015
    )

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

    if was_training:
        ssf3.train()



def main():
    args = parser.parse_args()
    dim = args.dim
    z_dim = args.z_dim
    bs = args.bs
    vis_every = args.vis_every
    total_epochs = args.total_epochs
    lambda_MSE = args.lambda_MSE
    lambda_cls = args.lambda_cls
    lambda_adv = args.lambda_adv
    lambda_gp = args.lambda_gp
    skip_fq = args.skip_fq + 1
    L = args.L
    path = args.path
    pre_path = args.pre_path
    dataset = args.dataset
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
    discriminator = DiscriminatorMNIST(n_channel=1).to(device)
    
    
    # Loading model for first frame reconstruction
    I_dim = 12
    I_L = 2
    encoder = Encoder(dim=I_dim, nc=1, stochastic=True, quantize_latents=True, L=I_L) #Generator Side
    decoder = Decoder_Iframe(dim=I_dim)
    Iframe_path = f'./saved_models/Iframe_Models/Iframe_z_dim_{I_dim}_L_{I_L}/Iframe_dim_128_z_dim_{I_dim}_L_{I_L}_single_bit_False_bs_64_lambdaMSE_1.0_lambdaCLS_0.01'
    encoder.load_state_dict(torch.load(f'{Iframe_path}/encoder.pth'))
    decoder.load_state_dict(torch.load(f'{Iframe_path}/decoder.pth'))
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    ssf2 = ScaleSpaceFlow(num_levels=1, dim=z_dim, stochastic=True, quantize_latents=True, L=L, freeze_enc=True)
       
    folder_name = (
        f"Nframe_dim_{dim}"
        f"_z_dim_{z_dim}"
        f"_L_{L}"
        f"_bs_{bs}"
        f"_lambdaMSE_{lambda_MSE}"
        f"_lambdaCLS_{lambda_cls}"
    )
         
    save_dir = f'./saved_models/stream1/Iframe_z_dim_{I_dim}_L_{I_L}/Pframe_z_dim_{z_dim}_L_{L}/' + folder_name
    os.makedirs(save_dir, exist_ok=True)
    pre_path = f"./saved_models/stream1/Iframe_z_dim_{I_dim}_L_{I_L}/Pframe_z_dim_{z_dim}_L_{L}/Universal_dim_128_z_dim_{z_dim}_L_{L}_bs_64_lambdaMSE_1.0_lambdaCLS_0.01"
    # 2nd frame models are loaded and fixed
    if pre_path is not None:
        ssf2.motion_encoder.load_state_dict(torch.load(pre_path + '/m_enc.pth'))
        ssf2.motion_decoder.load_state_dict(torch.load(pre_path + '/m_dec.pth'))
        ssf2.P_encoder.load_state_dict(torch.load(pre_path + '/p_enc.pth'))
        ssf2.res_encoder.load_state_dict(torch.load(pre_path + '/r_enc.pth'))
        ssf2.res_decoder.load_state_dict(torch.load(pre_path + '/r_dec.pth'))
        print("Pre-Trained Models are loaded...")
    ssf2.eval()
    
    

    
    ssf3 = ScaleSpaceFlow(num_levels=1, dim=z_dim, stochastic=True, quantize_latents=True, L=L)
    # Load the pre-trained Universal model of frame 2
    pre_path2 = f"./saved_models/stream1/Iframe_z_dim_{I_dim}_L_{I_L}/Pframe_z_dim_{z_dim}_L_{L}/Universal_dim_128_z_dim_{z_dim}_L_{L}_bs_64_lambdaMSE_1.0_lambdaCLS_0.01"
    
    if pre_path2 is not None:
        ssf3.motion_encoder.load_state_dict(torch.load(pre_path2 + '/m_enc.pth'))
        ssf3.motion_decoder.load_state_dict(torch.load(pre_path2 + '/m_dec.pth'))
        ssf3.P_encoder.load_state_dict(torch.load(pre_path2 + '/p_enc.pth'))
        ssf3.res_encoder.load_state_dict(torch.load(pre_path2 + '/r_enc.pth'))
        ssf3.res_decoder.load_state_dict(torch.load(pre_path2 + '/r_dec.pth'))
        discriminator.load_state_dict(torch.load(pre_path2 + '/discriminator.pth'))
        print("Pre-Trained Models are loaded...")

    print("Settings: ", folder_name)
    # change it to I_dim & L
    
    list_models = [ssf3, discriminator]
    ssf3.to(device)
    ssf2.to(device)
    
    train_loader, test_loader = get_dataloader(data_root=path, seq_len=8, batch_size=bs, num_digits=1)
    mse = torch.nn.MSELoss()

    opt_ssf3 = torch.optim.RMSprop(ssf3.parameters(), lr=5e-5)
    opt_discriminator = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr_critic,
        betas=(args.beta1_critic, args.beta2_critic),
    )
    list_opt = [opt_ssf3]
    rate =  z_dim * 2 * math.log( L, 2)
    csv_path = os.path.join(save_dir, f"Rate_{rate}_CLS_{lambda_cls}_performance.csv")
    csv_file = open(csv_path, 'w')
    csv_file.write("Epoch, Accuracy1, Distortion Loss1, Classification Loss1, Accuracy2, Distortion Loss2, Classification Loss2, Accuracy3, Distortion Loss3, Classification Loss3, Lambda_MSE, Lambda_CLS\n")

    recon_dir = os.path.join(save_dir, 'recons')
    vis_batch = next(iter(test_loader))

    batches_done = 0
    discriminator_loss = torch.tensor(0.0, device=device)

    for epoch in range(total_epochs):
        set_models_state(list_models, 'train')

        for i, (x, y) in enumerate(iter(train_loader)):
            x = x.permute(0, 4, 1, 2, 3)
            x = x.to(device).float()
            y = y.to(device)

            x2 = x[:, :, 1, ...]
            x1 = x[:, :, 0, ...] 
            x3 = x[:, :, 2, ...] 
            if i % skip_fq != 0:
                for parameter in ssf3.parameters():
                    parameter.requires_grad = False
                for parameter in discriminator.parameters():
                    parameter.requires_grad = True

                with torch.no_grad():
                    hx = encoder(x1)
                    x1_hat = decoder(hx[0])
                    x2_hat = ssf2(x2, x1_hat)
                    x3_hat = ssf3(x3, x2_hat)
                assert_image_shape(x3, x3_hat)

                opt_discriminator.zero_grad()
                real_validity = discriminator(x3)
                fake_validity = discriminator(x3_hat.detach())
                gradient_penalty = compute_gradient_penalty(
                    discriminator, x3, x3_hat.detach()
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
                for parameter in ssf3.parameters():
                    parameter.requires_grad = True

                set_opt_zero([opt_ssf3])
                with torch.no_grad():
                    hx = encoder(x1)
                    x1_hat = decoder(hx[0])
                    x2_hat = ssf2(x2, x1_hat)
                x3_hat = ssf3(x3, x2_hat)
                assert_image_shape(x3, x3_hat)

                fake_validity = discriminator(x3_hat)
                perception_loss_x3 = -fake_validity.mean()
                distortion_loss_x1 = mse(x1, x1_hat)
                classifier_output_x1 = model_classifier(x1_hat)
                classification_loss_x1 = cross_entropy_criterion(
                    classifier_output_x1, y
                )
                distortion_loss_x2 = mse(x2_hat, x2)
                classifier_output_x2 = model_classifier(x2_hat)
                classification_loss_x2 = cross_entropy_criterion(
                    classifier_output_x2, y
                )
                distortion_loss_x3 = mse(x3_hat, x3)
                classifier_output_x3 = model_classifier(x3_hat)
                classification_loss_x3 = cross_entropy_criterion(
                    classifier_output_x3, y
                )

                loss_x3 = (
                    lambda_MSE * distortion_loss_x3
                    + lambda_cls * classification_loss_x3
                    + lambda_adv * perception_loss_x3
                )
                loss_x3.backward()
                opt_ssf3.step()

                if batches_done % 100 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{total_epochs}] "
                        f"Batch [{i}/{len(train_loader)}] "
                        f"Discriminator: {discriminator_loss.item():.4f} "
                        f"Loss: {loss_x3.item():.4f} "
                        f"Distortion: {distortion_loss_x3.item():.4f} "
                        f"Classification: {classification_loss_x3.item():.4f}"
                    )
            
            batches_done += 1
            
                
        (dist_mse_x1, cls_mse_x1, acc_mse_x1,dist_mse_x2,
                 cls_mse_x2, 
                 acc_mse_x2, dist_mse_x3, cls_mse_x3, acc_mse_x3 ) = evaluate(ssf2, ssf3, model_classifier, cross_entropy_criterion, test_loader, device, encoder, decoder)
        
        csv_file.write(
            # epoch+1,  accuracy1, distortion loss1, classification loss1, accuracy2, distortion loss2, classification loss2,accuracy3, distortion loss3, classification loss3, lambda_MSE, lambda_CLS
            f"{epoch + 1},{acc_mse_x1:.4f},{dist_mse_x1:.4f}, "
            f"{cls_mse_x1:.4f},{acc_mse_x2:.4f},{dist_mse_x2:.4f}, "
            f"{cls_mse_x2:.4f},"
            f"{acc_mse_x3:.4f},{dist_mse_x3:.4f}, "
            f"{cls_mse_x3:.4f},"
            f"{lambda_MSE},{lambda_cls}\n"
        )
        
        csv_file.flush()

        
        print(f"Epoch: {epoch + 1} | "
              f"MSE [Dist: {dist_mse_x1:.4f}, Cls: {cls_mse_x1:.4f}, Acc: {acc_mse_x1:.4f}] | "
              f"MSE [Dist: {dist_mse_x2:.4f}, Cls: {cls_mse_x2:.4f}, Acc: {acc_mse_x2:.4f}] | "
              f"MSE [Dist: {dist_mse_x3:.4f}, Cls: {cls_mse_x3:.4f}, Acc: {acc_mse_x3:.4f}] | "
              f"Lambda_MSE: {lambda_MSE} Lambda_CLS: {lambda_cls}")
        

        if (epoch + 1) % vis_every == 0 or epoch == 0:
                       save_recon_preview_universal(ssf2,ssf3, vis_batch, device, recon_dir, epoch,rate, encoder, decoder)
    csv_file.close()

    set_models_state(list_models, 'eval')

    torch.save(ssf3.motion_encoder.state_dict(), os.path.join(save_dir, 'm_enc.pth'))
    torch.save(ssf3.motion_decoder.state_dict(), os.path.join(save_dir, 'm_dec.pth'))
    torch.save(ssf3.P_encoder.state_dict(), os.path.join(save_dir, 'p_enc.pth'))
    torch.save(ssf3.res_encoder.state_dict(), os.path.join(save_dir, 'r_enc.pth'))
    torch.save(ssf3.res_decoder.state_dict(), os.path.join(save_dir, 'r_dec.pth'))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, 'discriminator.pth'))

    for i, (x, y) in enumerate(iter(train_loader)):
        x = x.permute(0, 4, 1, 2, 3)
        x = x.to(device).float()
        break
    np.savez_compressed(os.path.join(save_dir, "x"), a=x.detach().cpu().numpy())

    for i in range(5):
        with torch.no_grad():
            x2 = x[:, :, 1, ...]
            x1 = x[:, :, 0, ...]
            x3 = x[:, :, 2, ...]
            hx = encoder(x1)
            x1_hat = decoder(hx[0])
            x2_hat = ssf2(x2, x1_hat)
            x3_hat = ssf3(x3, x2_hat)
            np.savez_compressed(os.path.join(save_dir, "x3_hat" + str(i)), a=x3_hat.detach().cpu().numpy())


if __name__ == "__main__":
    main()
