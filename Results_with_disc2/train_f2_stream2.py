# Uncompressed first frame
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
import math

cuda = True
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

parser = argparse.ArgumentParser('P-frame compressor training (Step 2)')
parser.add_argument('--total_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--bs', type=int, default=64, help='size of the batch')
parser.add_argument('--dim', type=int, default=128, help='common_dim')
parser.add_argument('--z_dim', type=int, default=1, help='z dim')
parser.add_argument('--L', type=int, default=2, help='quantization levels')
parser.add_argument('--lambda_MSE', type=float, default=1.0, help='Distortion loss weight')
parser.add_argument('--lambda_cls', type=float, default=0.0, help='Classification loss weight')
parser.add_argument('--path', type=str, default='./data/', help='Data Path')
parser.add_argument('--pre_path', type=str, default='None', help='Pretrained_Path (for ssf P-frame)')
parser.add_argument('--single_bit', type=int, default=0)
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--vis_every', type=int, default=5, help='Save reconstruction previews every N epochs')
parser.add_argument('--x1_mode', type=str, default='recon', help='Mode for x1: recon or original')
parser.add_argument('--skip_fq', type=int, default=1, help='loop frequency for generator update')
parser.add_argument('--lambda_adv', type=float, default=0.015, help='Adversarial loss weight')
parser.add_argument('--lambda_gp', type=float, default=10.0, help='WGAN-GP gradient penalty weight')
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


def evaluate(ssf, model_classifier, cross_entropy_criterion, test_loader, device):
    mse_loss = nn.MSELoss(reduction='sum')

    ssf.eval()
    
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
            x_hat = ssf(x_cur, x_ref)

            batch_size = x.size(0)
            channels = x.size(1)
            height = x.size(3)
            width = x.size(4)
            
            cls_x1 = model_classifier(x_ref)
            total_dist_x1 += mse_loss(x_ref, x_ref).item()
            total_cls_x1 += cross_entropy_criterion(cls_x1, y).item() * batch_size
            total_acc_x1 += (cls_x1.argmax(dim=1) == y).sum().item()

            
            cls_x2 = model_classifier(x_hat)

            total_dist_x2 += mse_loss(x_cur, x_hat).item()
            total_cls_x2 += cross_entropy_criterion(cls_x2, y).item() * batch_size
            total_acc_x2 += (cls_x2.argmax(dim=1) == y).sum().item()
            num_x += len(x)
    
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


def save_recon_preview(
    ssf,
    vis_batch,
    device,
    out_dir,
    epoch,
    rate,
    num_show=4
):
    """Save a compact PNG grid of reference, source, and reconstructed frames."""

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

        x_hat = ssf(
            x_cur,
            x_ref
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
        3,
        figsize=(6, 2 * n),
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
    #     'Frame 2\n(Source)',
    #     'Frame 2\n(P-frame Recon)'
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
            x_ref[idx],
            x_cur[idx],
            x_hat[idx]
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
    #     f"P-Frame Reconstructions "
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
    vis_every = args.vis_every
    total_epochs = args.total_epochs
    lambda_MSE = args.lambda_MSE
    lambda_cls = args.lambda_cls
    lambda_adv = args.lambda_adv
    lambda_gp = args.lambda_gp
    skip_fq = args.skip_fq + 1
    L = args.L
    path = args.path
    # pre_path = args.pre_path
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

    ssf = ScaleSpaceFlow(num_levels=1, dim=z_dim, stochastic=True, quantize_latents=True, L=L)
    discriminator = DiscriminatorMNIST(n_channel=1).to(device)

    folder_name = (
        f"Pframe_dim_{dim}"
        f"_z_dim_{z_dim}"
        f"_L_{L}"
        f"_bs_{bs}"
        f"_lambdaMSE_{lambda_MSE}"
        f"_lambdaCLS_{lambda_cls}"
    )
    
    
    # change this to path of the models if you want to train for more epochs
    pre_path = None
    
    
    print("Settings: ", folder_name)
    # change it to I_dim & L
    save_dir = f'./saved_models/stream2/Pframe_z_dim_{z_dim}_L_{L}/' + folder_name
    os.makedirs(save_dir, exist_ok=True)


    if pre_path != None :
        #prefix_path = 'z'+str(z_dim)+'l'+str(L)+'_MMSE'
        ssf.motion_encoder.load_state_dict(torch.load(pre_path + '/m_enc.pth'))
        ssf.motion_decoder.load_state_dict(torch.load(pre_path + '/m_dec.pth'))
        ssf.P_encoder.load_state_dict(torch.load(pre_path + '/p_enc.pth'))
        ssf.res_encoder.load_state_dict(torch.load(pre_path + '/r_enc.pth'))
        ssf.res_decoder.load_state_dict(torch.load(pre_path + '/r_dec.pth'))

    
    
    
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
    rate =  z_dim * 2 * math.log( L, 2)
    csv_path = os.path.join(save_dir, f"Rate_{rate}_CLS_{lambda_cls}_performance.csv")
    csv_file = open(csv_path, 'w')
    csv_file.write("Epoch, Accuracy1, Distortion Loss1, Classification Loss1, Accuracy2, Distortion Loss2, Classification Loss2, Lambda_MSE, Lambda_CLS\n")
    
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

            x_cur = x[:, :, 1, ...]
            x_ref = x[:, :, 0, ...]
            
            if i % skip_fq != 0:
                for parameter in ssf.parameters():
                    parameter.requires_grad = False
                for parameter in discriminator.parameters():
                    parameter.requires_grad = True

                with torch.no_grad():
                    x_hat = ssf(x_cur, x_ref)
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
                x_hat = ssf(x_cur, x_ref)
                assert_image_shape(x_cur, x_hat)

                fake_validity = discriminator(x_hat)
                perception_loss_x2 = -fake_validity.mean()
                distortion_loss_x1 = mse(x_ref, x_ref)
                classifier_output_x1 = model_classifier(x_ref)
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
                    print(
                        f"Epoch [{epoch + 1}/{total_epochs}] "
                        f"Batch [{i}/{len(train_loader)}] "
                        f"Discriminator: {discriminator_loss.item():.4f} "
                        f"Loss: {loss_x2.item():.4f} "
                        f"Distortion: {distortion_loss_x2.item():.4f} "
                        f"Classification: {classification_loss_x2.item():.4f}"
                    )

            batches_done += 1
                    
                
        (dist_mse_x1, cls_mse_x1, acc_mse_x1,dist_mse_x2,
                 cls_mse_x2, 
                 acc_mse_x2) = evaluate(ssf, model_classifier, cross_entropy_criterion, test_loader, device)

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
            save_recon_preview(ssf, vis_batch, device, recon_dir, epoch, rate)

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
        x_hat = ssf(x_cur, x_ref)
        np.savez_compressed(os.path.join(save_dir, "x_hat" + str(i)), a=x_hat.detach().cpu().numpy())


if __name__ == "__main__":
    main()
