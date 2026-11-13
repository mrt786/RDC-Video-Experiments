import argparse
from ast import Load
import os
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
#from utils import *
import matplotlib.pyplot as plt
import torch.nn.utils.spectral_norm as spectralnorm
from torch.nn import init
import torchvision
import torch
import torch.nn as nn
from Results_without_disc.CNN_MNIST import Net
from models import *
from utils import *
import torch.nn.functional as F
from helper import *
from Helper_classification import *
from CNN_MNIST import *

cuda = True
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

import argparse

"""
dim = 128
z_dim = 2
lambda_gp = 50
bs =64
d_penalty = 0
skip_fq = 1
total_epochs = 200
lambda_P = 0
lambda_MSE = 10
"""
parser = argparse.ArgumentParser('training config of N frames. Currently, Im focusing only on 3')
parser.add_argument('--total_epochs', type=int, default=300, help='number of epochs of training')
parser.add_argument('--bs', type=int, default=64, help='size of the batch')
parser.add_argument('--dim', type=int, default=128, help='common_dim')
parser.add_argument('--z_dim', type=int, default=1, help='z dim')
parser.add_argument('--L', type=int, default=2, help='z dim')
parser.add_argument('--skip_fq', type=int, default=5, help='loop frequency for WGAN')
parser.add_argument('--lambda_MSE', type=float, default=1.0, help='Perceptual Penalty')
parser.add_argument('--lambda_cls', type=float, default=0.1, help='Classification loss weight')
parser.add_argument('--single_bit', type=int, default=0)
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--vis_every', type=int, default=5, help='Save reconstruction previews every N epochs')



def preview_reconstructed(
    x_ref_recon,
    labels,
    out_dir,
    epoch,
    rate
):
    """
    Save a PNG preview showing pixel values of ONLY x_ref_recon[0].

    The reconstructed frame is converted from [0,1]
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
   
def save_recon_preview_universal(
    unv_2frame_ssf,
    unv_nframe_ssf,
    vis_batch,
    device,
    out_dir,
    epoch,
    encoder,
    decoder,
    rate,
    num_show=4
):
    
    """
                        {Reconstruction at Epoch N Rate=R}
    Save a PNG grid:           | Source  | Reconstructions |
        Digit1                 | x1|x2|x3| x1 | x2 | x3    |
        Digit2                 | x1|x2|x3| x1 | x2 | x3    |
        Digit3                 | x1|x2|x3| x1 | x2 | x3    |                               
    """

    was_training = unv_nframe_ssf.training
    unv_nframe_ssf.eval()
    
    x, y = vis_batch

    x = x.permute(0, 4, 1, 2, 3).to(device).float()
    y = y.to(device)

    x_1 = x[:, :, 0, ...]
    x_2 = x[:, :, 1, ...]
    x_3 = x[:, :, 2, ...]
    # ---------------------------------------------------------
    # Compute reconstructions ONCE
    # ---------------------------------------------------------
    with torch.no_grad():

        hx = encoder(x_1)

        x1_recon = decoder(hx[0])

        x2_recon = unv_2frame_ssf(
            x_2,
            x1_recon
        )
# check this, do we need to pass x_hat_2 only or do we need other?
        x3_recon = unv_nframe_ssf(
            x_3,
            x2_recon
        )

    # ---------------------------------------------------------
    # Number of images for normal preview
    # ---------------------------------------------------------
    n = min(num_show, x.size(0))

    fig, axes = plt.subplots(
        n,
        6,
        figsize=(12, 6 * n)
    )

    if n == 1:
        axes = np.expand_dims(
            axes,
            axis=0
        )
        
# Todo: Set column titles like mentioned above
#       No Need to add to much space between the images showing in the plot. Just draw a white line.
    col_titles = [
        'Frame 1\n(Reference)',
        'Frame 2\n(Source)',
        'Frame 3\n(Source)',
        'Frame 4\n(Source)'
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

        digit_label = y[idx].item()

        imgs = [
            x_1[idx],
            x_2[idx],
            x_3[idx],
            
            x1_recon[idx],
            x2_recon[idx],
            x3_recon[idx]
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
    # fig.suptitle(
    #     f"Epoch {epoch + 1} | "
    #     f"(ref=I-frame recon) & Rate={rate:.2f}",
    #     fontsize=13,
    #     weight='bold'
    # )

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
    
    if was_training:
        unv_nframe_ssf.train()




def set_models_state(list_models, state):
    if state =='train':
        for model in list_models:
            model.train()
    else:
        for model in list_models:
            model.eval()

def set_opt_zero(opts):
    for opt in opts:
        opt.zero_grad()


def main():
    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    #Params
    dim = args.dim#128
    z_dim = args.z_dim #2
    lambda_gp = args.lambda_gp #50
    bs = args.bs #64
    # d_penalty = args.d_penalty #0
    skip_fq = args.skip_fq #10
    total_epochs = args.total_epochs #200
    # lambda_P = args.lambda_P
    # lambda_PM = args.lambda_PM
    lambda_MSE = args.lambda_MSE
    L = args.L
    path = args.path
    pre_path = args.pre_path
    single_bit = bool(args.single_bit)
    dataset=args.dataset
    
    #Set the folder name
    folder_name='R1inf_3frames_dim_'+str(dim)+'|z_dim_'+str(z_dim)+'|L_'+str(L)+'|_single_bit'+str(single_bit)+'|bs_'+str(bs)+ '|lambdaMSE_' + str(lambda_MSE)
    print ("Settings: ", folder_name)

    os.makedirs('./saved_models/compressed/' + folder_name, exist_ok=True)
    f = open('./saved_models/compressed/' + folder_name + "/performance.txt", "a")

    #Define Models

    if dataset == 'MNIST':
        activation = torch.sigmoid
    else:
        activation = torch.tanh
        
    
    # Todo:
    """ 
        3. Remove the ssf1_hat as it is for the discriminator
    """
    ssf1_MMSE = ScaleSpaceFlow(num_levels=1, dim=z_dim, stochastic=True, quantize_latents=True, L=L,single_bit=single_bit)
    ssf_nfames = ScaleSpaceFlow_R1eps_e2e_3frames(num_levels=1, dim=z_dim, stochastic=True, quantize_latents=True, L=L,single_bit=single_bit,num_c=1, activation=activation)

    model_classifier = Net().to(torch.device)

    trained_model_path = "./trained_classifier.pth"

    if os.path.exists(trained_model_path):
        model_classifier.load_state_dict(
            torch.load(
                trained_model_path,
                map_location=device
            )
        )
        print(
            f"Loaded trained classifier from "
            f"{trained_model_path}"
        )
    else:
        model_classifier = classifier().to(device)

    model_classifier.eval()

    for param in model_classifier.parameters():
        param.requires_grad = False

    cross_entropy_criterion = nn.CrossEntropyLoss()

    I_dim = 12
    I_L = 2
    pre_path_frame2_models= './saved_models/compressed/I_frame_z_dim_12_L_2/Universal_dim_128_z_dim_12_L_2_bs_64_lambdaMSE_1.0_lambdaCLS_0.01_Iframe_pretrained'
    ssf1_MMSE.motion_encoder.load_state_dict(torch.load(pre_path_frame2_models+'/m_enc.pth'))
    ssf1_MMSE.motion_decoder.load_state_dict(torch.load(pre_path_frame2_models+'/m_dec.pth'))
    ssf1_MMSE.P_encoder.load_state_dict(torch.load(pre_path_frame2_models+'/p_enc.pth'))
    ssf1_MMSE.res_encoder.load_state_dict(torch.load(pre_path_frame2_models+'/r_enc.pth'))
    ssf1_MMSE.res_decoder.load_state_dict(torch.load(pre_path_frame2_models+'/r_dec.pth'))


    list_models = [ssf_nfames]

    ssf1_MMSE.cuda()
    ssf1_MMSE.eval()

    ssf_nfames.cuda()
    
    # Do we have to Load the model which has been trained on I-frame and P-frame or train from the scratch?
    iframe_pretrained_path = './saved_models/compressed/Iframe_z_dim_{I_dim}_L_{I_L}/Iframe_dim_128_z_dim_{I_dim}_L_{I_L}_single_bit_False_bs_64_lambdaMSE_1.0_lambdaCLS_0.01'
    encoder = Encoder(dim=I_dim, nc=1, stochastic=True, quantize_latents=True, L=I_L) #Generator Side
    decoder = Decoder_Iframe(dim=I_dim)

    encoder.load_state_dict(torch.load(f'{iframe_pretrained_path}/encoder.pth'))
    decoder.load_state_dict(torch.load(f'{iframe_pretrained_path}/decoder.pth'))
    encoder.cuda()
    decoder.cuda()
    encoder.eval()
    decoder.eval()
    

    #Define Data Loader
    train_loader, test_loader = get_dataloader(data_root=path, seq_len=8, batch_size=bs, num_digits=1)
    mse = torch.nn.MSELoss()

    #discriminator.train()
    opt_ssf= torch.optim.RMSprop(ssf_nfames.parameters(), lr=5e-5)
    

    list_opt = [opt_ssf]
    
    
    # Make sure that X1_hat and X2_hat are used for predicting X3_hat
    for epoch in range(total_epochs):
        set_models_state(list_models, 'train')
        for i,x in enumerate(iter(train_loader)):
            #Set 0 gradient
            set_opt_zero(list_opt)

            #Get the data
            x = x.permute(0, 4, 1, 2, 3)
            x = x.cuda().float()

            x3 = x[:,:,2,...]
            x2 = x[:,:,1,...]
            x1 = x[:,:,0,...]
            with torch.no_grad():
                x2_bar = ssf1_MMSE(x2, x1)
                x2_hat = ssf1_hat(x2, x1)
            x_hat = ssf_nfames(x3, x2_bar, x1, x2_hat )
            
            # Modify the loss function according to the classification 
            loss = lambda_MSE*mse(x_hat, x3) 
            loss.backward()

            opt_ssf.step()
            print ('done epoch: ', epoch)

        """
            TODO:
            1. Change the printing statements
            2. Call the neccessary functions for reconstructions
            3. Modify the reconstruction functions so that they can take the same input of digits and have a better comparison 
        """
        if epoch %10 == 0:
            show_str= "Epoch: "+ str(epoch) + "l_PM, l_P, l_MSE " + str(lambda_PM) + str(lambda_P)+ " " \
            +str(lambda_MSE) + " " + " P loss: " + str(cal_W1(ssf, ssf1_hat, ssf1_MMSE, discriminator, discriminator_M, test_loader))
            print (show_str)
            f.write(show_str+"\n")

    show_str= "Epoch: "+ str(epoch) + "l_PM, l_P, l_MSE " + str(lambda_PM) + str(lambda_P)+ " " \
    +str(lambda_MSE) + " " + " P loss: " + str(cal_W1(ssf, ssf1_hat, ssf1_MMSE, discriminator, discriminator_M, test_loader))
    print (show_str)
    f.write(show_str+"\n")

    set_models_state(list_models, 'eval')

    torch.save(ssf_nfames.motion_encoder.state_dict(), os.path.join("./saved_models/compressed/" + folder_name, 'm_enc.pth'))
    torch.save(ssf_nfames.motion_decoder.state_dict(), os.path.join("./saved_models/compressed/" + folder_name, 'm_dec.pth'))
    torch.save(ssf_nfames.P_encoder.state_dict(), os.path.join("./saved_models/compressed/" + folder_name, 'p_enc.pth'))
    torch.save(ssf_nfames.res_encoder.state_dict(), os.path.join("./saved_models/compressed/" + folder_name, 'r_enc.pth'))
    torch.save(ssf_nfames.res_decoder.state_dict(), os.path.join("./saved_models/compressed/" + folder_name, 'r_dec.pth' ))

    f.close()

    #save some figures
    for i,x in enumerate(iter(train_loader)):
        x = x.permute(0, 4, 1, 2, 3)
        x = x.cuda().float()
        break
    np.savez_compressed("./saved_models/" + folder_name+"/x", a=x.detach().cpu().numpy())

    for i in range(5): #generate same figure 5 times
        x_cur = x[:,:,1,...]
        x_ref = x[:,:,0,...]
        x_hat = ssf(x_cur, x_ref)
        np.savez_compressed("./saved_models/" + folder_name+"/x_hat"+str(i), a=x_hat.detach().cpu().numpy())

if __name__ == "__main__":
    main()
