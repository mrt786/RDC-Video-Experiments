import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import re
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image

def get_rate(folder_name):
    z_dim = int(re.search(r"z_dim_(\d+)", folder_name).group(1))
    L = int(re.search(r"_L_(\d+)", folder_name).group(1))
    return 2*z_dim * math.log(L, 2), z_dim




def plot_accuracy_over_frames(path, title, z_dim):

    path_to_folders = path
    list_of_folders = os.listdir(path_to_folders)
    
    acc_folder = None
    for folder in list_of_folders:
        if f"Nframe_Universal_dim_128_z_dim_{z_dim}_L_2_bs_64_lambdaMSE_1.0_lambdaCLS_0.01" in folder:
            acc_folder = folder

    if acc_folder is None:
        print("No path found to plot accuracies")
        return
    rate, _ = get_rate(acc_folder)
    path_to_unv_csv = (
                path_to_folders
                + "/"
                + acc_folder
                + f"/Rate_{float(rate)}0_CLS_{0.01}_performance.csv"
            )

    df_unv = pd.read_csv(path_to_unv_csv)

        # Use the final evaluation rows
    df_eval = df_unv.tail(5)

    accuracy_columns = [" Accuracy1", " Accuracy2", " Accuracy3"]

    means = df_eval[accuracy_columns].mean()
    print(means)
    stds = df_eval[accuracy_columns].std()

    frames = [1, 2, 3]

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        frames,
        means.values,
        yerr=stds.values,
        marker="o",
        markersize=8,
        linewidth=2,
        capsize=5,
    )

    plt.xticks(frames)
    plt.ylim(0, 1)  # Use (0, 100) if your accuracies are percentages
    plt.xlabel("Frame number")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"Results/{title}.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


    print("RDC Results are saved in the Results folder")
   

def plot_rdc_curve_single_rate_multiple_frames(path, title):

    path_to_folders = path
    list_of_folders = os.listdir(path_to_folders)
    plt.figure(figsize=(8,6))
    

    updated_folders = {
        "Universal": [],
        "Nframe": []
    }

    # ---------------------------------------------------------
    # Separate Universal and P-Frame folders
    # ---------------------------------------------------------
    for folder in list_of_folders:
        if "Universal" in folder.split("_")[1]:
            updated_folders["Universal"].append(folder)
        elif "Nframe" in folder.split("_")[0]:
            updated_folders["Nframe"].append(folder) 

    print(updated_folders)
    updated_folders["Universal"].sort()
    updated_folders["Nframe"].sort()

    # ---------------------------------------------------------
    # Get rate from Universal folder
    # ---------------------------------------------------------
    rate, _ = get_rate(updated_folders["Universal"][0])

    # ---------------------------------------------------------
    # Plot Universal and Nframe
    # ---------------------------------------------------------
    i = 0
    colors = ['b','g','r','c','m','y']


    for p_path, unv_path in zip(
        updated_folders["Nframe"],
        updated_folders["Universal"]
    ):

        if i > 3:
            break

        # -----------------------------------------------------
        # Get lambda_cls values
        # -----------------------------------------------------
        lambda_cls = unv_path.split("lambdaCLS_")[1].split("_")[0]
        lambda_cls_p = p_path.split("lambdaCLS_")[1].split("_")[0]

        # -----------------------------------------------------
        # CSV paths
        # -----------------------------------------------------
        path_to_unv_csv = (
            path_to_folders
            + "/"
            + unv_path
            + f"/Rate_{float(rate)}0_CLS_{lambda_cls}_performance.csv"
        )

        path_to_Nframe_csv = (
            path_to_folders
            + "/"
            + p_path
            + f"/Rate_{float(rate)}_CLS_{lambda_cls_p}_performance.csv"
        )
        
        # continue
        # -----------------------------------------------------
        # Read Universal CSV
        # -----------------------------------------------------
        df_unv = pd.read_csv(path_to_unv_csv)

        # Take last rows
        df_unv = df_unv.tail()

        # Calculate mean
        df_unv_mean = df_unv.mean()

        # -----------------------------------------------------
        # Read Nframe CSV
        # -----------------------------------------------------
        df_Nframe = pd.read_csv(path_to_Nframe_csv)

        df_Nframe = df_Nframe.tail()

        # Calculate mean
        df_Nframe_mean = df_Nframe.mean()          
        plt.scatter(
                    df_unv_mean[" Distortion Loss3"],
                    df_unv_mean[" Accuracy3"],
                    marker="o",
                    edgecolors="black",
                    c= colors[i],
                    s=60,
                    label=f"λ_cls = {lambda_cls}"
                )
                
        plt.scatter(
            df_Nframe_mean[" Distortion Loss3"],
            df_Nframe_mean[" Accuracy3"],
            marker="o",
            c= colors[i],
            s=80,
            label=f"λ_cls = {lambda_cls_p}"
                )
                  
      
        

        i += 1

    # ---------------------------------------------------------
    # Nframe subplot formatting
    # ---------------------------------------------------------
    plt.title(title)
    plt.xlabel("Distortion Loss")
    plt.ylabel("Classification Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    plt.savefig(
        f"Results/{title}.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    print("RDC Results are saved in the Results folder")

def main():
    z_dim = 1
    plot_accuracy_over_frames(f"./saved_models3/stream2/Pframe_z_dim_{z_dim}_L_2", title = f"Frame Accuracies over Rate={z_dim * 2}", z_dim=z_dim)
    # plot_rdc_curve_single_rate_multiple_frames("./saved_models3/stream2/Pframe_z_dim_1_L_2", title ="RDC Curve Over Rate=2")
    pass

if __name__ == "__main__":
    main()
    