


#  I want to plot different subplots to show the results of the analysis. The results are present in the saved_models/directories/performance.csv


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
    return int(math.log2(2 * z_dim * L)), z_dim

def plot_reconstructed_images():
    
    path_to_folders = os.path.join(
        os.getcwd(),
        "saved_models"
    )

    # Lambda CLS values we want to plot
    lambda_cls_values = [
        "0.0",
        "0.01",
        "0.05",
        "0.1",
        # "0.15",
        # "0.33",
        # "0.5"
    ]

    # The three rates / z_dims
    z_dims = [1, 2, 4]

    # Check saved_models exists
    if not os.path.exists(path_to_folders):
        print(f"ERROR: saved_models folder not found:")
        print(path_to_folders)
        return

    # Get only Universal folders
    universal_folders = [
        folder
        for folder in os.listdir(path_to_folders)
        if "Universal" in folder
    ]

    # Output directory
    # Since we are already inside Results,
    # this creates:
    # ...\Experiments4\Results\Reconstructed_Images
    results_dir = os.path.join(
        os.getcwd(),"Results",
        "Reconstructed_Images"
    )

    os.makedirs(results_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Loop over each lambdaCLS
    # ---------------------------------------------------------

    for lambda_cls in lambda_cls_values:

        print("\n" + "=" * 70)
        print(f"lambdaCLS = {lambda_cls}")
        print("=" * 70)

        selected_folders = []

        # -----------------------------------------------------
        # Find EXACTLY one folder for each z_dim
        # -----------------------------------------------------

        for z_dim in z_dims:

            matching_folders = [
                folder
                for folder in universal_folders
                if f"z_dim_{z_dim}_" in folder
                and f"lambdaCLS_{lambda_cls}" in folder
            ]

            if len(matching_folders) == 0:

                print(
                    f"Missing folder: "
                    f"z_dim={z_dim}, lambdaCLS={lambda_cls}"
                )

                continue

            if len(matching_folders) > 1:

                print(
                    f"WARNING: Multiple folders found for "
                    f"z_dim={z_dim}, lambdaCLS={lambda_cls}:"
                )

                for folder in matching_folders:
                    print("   ", folder)

            # We expect exactly one
            folder = matching_folders[0]

            rate, z_dim_found = get_rate(folder)

            print({
                "folder": folder,
                "rate": rate,
                "z_dim": z_dim_found
            })

            selected_folders.append({
                "folder": folder,
                "rate": rate,
                "z_dim": z_dim_found
            })

        # -----------------------------------------------------
        # If nothing was found, move to next lambda
        # -----------------------------------------------------

        if len(selected_folders) == 0:
            print(
                f"No folders found for lambdaCLS={lambda_cls}"
            )
            continue

        # -----------------------------------------------------
        # Create figure
        # -----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            len(selected_folders),
            figsize=(15, 5)
        )

        # If there is only one image
        if len(selected_folders) == 1:
            axes = [axes]

        # -----------------------------------------------------
        # Plot images
        # -----------------------------------------------------

        for ax, item in zip(axes, selected_folders):

            folder = item["folder"]
            rate = item["rate"]
            z_dim = item["z_dim"]

            recon_path = os.path.join(
                path_to_folders,
                folder,
                "recons",
                "recon_epoch_0080.png"
            )

            # ---------------------------------------------
            # Check image exists
            # ---------------------------------------------

            if not os.path.exists(recon_path):

                print(f"Missing reconstruction:")
                print(recon_path)

                ax.axis("off")

                ax.set_title(
                    f"z_dim = {z_dim}\n"
                    f"R = {rate}\n"
                    f"Missing",
                    fontsize=12
                )

                continue

            # ---------------------------------------------
            # Load image
            # ---------------------------------------------

            image = Image.open(recon_path)

            ax.imshow(image)

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # ---------------------------------------------
            # Boundary around image
            # ---------------------------------------------

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_edgecolor("black")

            # ---------------------------------------------
            # Title
            # ---------------------------------------------

            ax.set_title(
                f" Rate = {rate}",
                fontsize=13
            )

        # -----------------------------------------------------
        # Overall title
        # -----------------------------------------------------

        fig.suptitle(
            f"Universal Reconstruction "
            f"(λCLS = {lambda_cls})",
            fontsize=16
        )

        plt.tight_layout()

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        save_path = os.path.join(
            results_dir,
            f"Universal_Reconstruction_lambdaCLS_{lambda_cls}_R1_Compressed_.png"
        )

        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.show()

        plt.close(fig)

        print(f"Saved: {save_path}")

      
      
def plot_universal_rdc_curve(title):

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------

    path_to_folders = os.path.join(
        os.getcwd(),
        "saved_models",
        "R1_Compressed"
    )

    # ---------------------------------------------------------
    # Lambda CLS values
    # ---------------------------------------------------------

    lambda_cls_values = [
        "0.0",
        "0.01",
        "0.05",
        "0.1",
        # "0.15",
        # "0.33",
        # "0.5"
    ]

    # ---------------------------------------------------------
    # Rate -> z_dim mapping
    #
    # R = log2(2 * z_dim * L)
    # L = 2
    #
    # z_dim = 1 -> R = 2
    # z_dim = 2 -> R = 3
    # z_dim = 4 -> R = 4
    # ---------------------------------------------------------

    rate_to_zdim = {
        2: 1,
        3: 2,
        4: 4
    }

    # ---------------------------------------------------------
    # Get all folders
    # ---------------------------------------------------------

    list_of_folders = os.listdir(path_to_folders)

    universal_folders = []
    pframe_folders = []
    rate_counter = {
        "Universal":{},
        "P-Frame": {}
    }
    for folder in list_of_folders:
    
        if "Iframe" in folder:
            continue

        if "Universal" in folder:
            rate, z_dim = get_rate(folder)
            rate_counter["Universal"][z_dim] = rate_counter["Universal"].get(z_dim, 0) + 1
            if rate_counter["Universal"][z_dim] > 4:
                continue
            universal_folders.append({"folder": folder, "rate": rate, "z_dim": z_dim})
        else:
            rate, z_dim = get_rate(folder)
            rate_counter["P-Frame"][z_dim] = rate_counter["P-Frame"].get(z_dim, 0) + 1
            if rate_counter["P-Frame"][z_dim] > 4:
                continue
            pframe_folders.append({"folder": folder, "rate": rate, "z_dim": z_dim})
    # ---------------------------------------------------------
    # Results dictionary
    #
    # results[model][rate] =
    # [
    #     (lambda_cls, distortion, accuracy),
    #     ...
    # ]
    # ---------------------------------------------------------

    results = {
        "Universal": {
            2: [],
            3: [],
            4: []
        },
        "P-Frame": {
            2: [],
            3: [],
            4: []
        }
    }

    # =========================================================
    # UNIVERSAL
    # =========================================================

    for item in universal_folders:

        folder = item["folder"]
        rate = item["rate"]
        z_dim = item["z_dim"]

        if rate not in rate_to_zdim:
            continue

        # Get lambdaCLS from folder name
        lambda_cls = folder.split("lambdaCLS_")[1]

        # CSV filename
        csv_path = os.path.join(
            path_to_folders,
            folder,
            f"Rate_{z_dim}_CLS_{lambda_cls}_performance.csv"
        )

        if not os.path.exists(csv_path):

            print(f"Missing: {csv_path}")
            continue

        df = pd.read_csv(csv_path).tail()

        df_mean = df.mean(numeric_only=True)

        distortion = df_mean[" Distortion Loss"]
        accuracy = df_mean[" Accuracy"]

        results["Universal"][rate].append(
            (
                lambda_cls,
                distortion,
                accuracy
            )
        )

    # =========================================================
    # P-FRAME
    # =========================================================

    for item in pframe_folders:

        folder = item["folder"]
        rate = item["rate"]
        z_dim = item["z_dim"]

        if rate not in rate_to_zdim:
            continue

        # Get lambdaCLS from folder name
        lambda_cls = folder.split("lambdaCLS_")[1]

        # CSV filename
        csv_path = os.path.join(
            path_to_folders,
            folder,
            f"Rate_{z_dim}_CLS_{lambda_cls}_performance.csv"
        )

        if not os.path.exists(csv_path):

            print(f"Missing: {csv_path}")
            continue

        df = pd.read_csv(csv_path).tail()

        df_mean = df.mean(numeric_only=True)

        distortion = df_mean[" Distortion Loss"]
        accuracy = df_mean[" Accuracy"]

        results["P-Frame"][rate].append(
            (
                lambda_cls,
                distortion,
                accuracy
            )
        )

    # =========================================================
    # PLOT
    # =========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    rates = [2, 3, 4]

    # ---------------------------------------------------------
    # Plot each rate in its own subplot
    # ---------------------------------------------------------

    for ax, rate in zip(axes, rates):

        z_dim = rate_to_zdim[rate]

        # -----------------------------------------------------
        # Universal
        # -----------------------------------------------------

        universal_points = sorted(
            results["Universal"][rate],
            key=lambda x: float(x[0])
        )

        if universal_points:

            x_universal = [
                point[1]
                for point in universal_points
            ]

            y_universal = [
                point[2]
                for point in universal_points
            ]

            ax.plot(
                x_universal,
                y_universal,
                marker="o",
                linewidth=0,
                markersize=7,
                label="Universal"
            )

            # Add lambdaCLS labels to points
            for lambda_cls, x, y in universal_points:

                ax.annotate(
                    f"λ={lambda_cls}",
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8
                )

        # -----------------------------------------------------
        # P-Frame
        # -----------------------------------------------------

        pframe_points = sorted(
            results["P-Frame"][rate],
            key=lambda x: float(x[0])
        )

        if pframe_points:

            x_pframe = [
                point[1]
                for point in pframe_points
            ]

            y_pframe = [
                point[2]
                for point in pframe_points
            ]

            ax.plot(
                x_pframe,
                y_pframe,
                marker="o",
                linewidth=0,
                markersize=7,
                label="P-Frame"
            )

            # Add lambdaCLS labels to points
            for lambda_cls, x, y in pframe_points:

                ax.annotate(
                    f"λ={lambda_cls}",
                    (x, y),
                    xytext=(5, -12),
                    textcoords="offset points",
                    fontsize=8
                )

        # -----------------------------------------------------
        # Subplot formatting
        # -----------------------------------------------------

        ax.set_title(
            f"Rate = {rate} ",
            fontsize=14
        )

        ax.set_xlabel(
            "Distortion Loss",
            fontsize=12
        )

        ax.set_ylabel(
            "Classification Accuracy",
            fontsize=12
        )

        ax.grid(
            True,
            alpha=0.3
        )

        ax.legend()

        # Boundary around subplot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)

    # ---------------------------------------------------------
    # Overall title
    # ---------------------------------------------------------

    fig.suptitle(
        title,
        fontsize=18
    )

    plt.tight_layout()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    results_dir = os.path.join(
        os.getcwd(),
        "Results"
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    save_path = os.path.join(
        results_dir,
        {title} +".png"
    )

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    plt.close(fig)

    print(
        f"RDC curves saved to:\n{save_path}"
    )
  

def plot_rdc_curve_single_rate(path, title):
    path_to_folders = path
    list_of_folders = os.listdir(path_to_folders)
    plt.figure(figsize=(10, 10))
    updated_folders = {
        "Universal":[],
        "P-Frame":[]
    }
    df = []
    for i in range(len(list_of_folders)):
        if "Iframe" in list_of_folders[i].split("_")[0]:
            continue
        elif "Universal" in list_of_folders[i]:
            updated_folders['Universal'].append((list_of_folders[i]))
            
        else:
            updated_folders['P-Frame'].append((list_of_folders[i]))
    
    

    i = 0
    for  p_path,unv_path in zip(updated_folders["P-Frame"], updated_folders["Universal"]):
        if i > 3:
            break
        lambda_cls = unv_path.split("lambdaCLS_")[1].split("_")[0]
        lambda_cls_p = p_path.split("lambdaCLS_")[1].split("_")[0]
        
        path_to_unv_csv = path_to_folders + "/" + unv_path + f"/Rate_{5.58}_CLS_{lambda_cls}_performance.csv"
        path_to_pframe_csv = path_to_folders + "/" + p_path + f"/Rate_{5.584962500721157}_CLS_{lambda_cls}_performance.csv"
        df_unv = pd.read_csv(path_to_unv_csv)
        df_unv = df_unv.tail()
        df_unv_mean = df_unv.mean()
        

        
        df_pframe = pd.read_csv(path_to_pframe_csv)
        df_pframe = df_pframe.tail()
        df_pframe_mean = df_pframe.mean()
        df_unv_mean["Model"] = "U"
        df_pframe_mean["Model"] = "P"
        df.append(df_unv_mean)
        df.append(df_pframe_mean)
        
        i += 1
        plt.scatter(df_unv_mean[' Distortion Loss2'], df_unv_mean[' Accuracy2'],  marker = 'o', label='lambda_cls = ' + str(lambda_cls) + ' Universal')
        plt.scatter(df_pframe_mean[' Distortion Loss2'], df_pframe_mean[' Accuracy2'], marker = 'o',edgecolors='black', s = 60, label='lambda_cls = ' + str(lambda_cls_p) + ' P-Frame')
        plt.title(f"{title}")
        
        plt.xlabel('Distortion Loss')
        plt.ylabel('Classification Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"Results/{title}.png", dpi=200,bbox_inches="tight")
    plt.show()
    df = pd.DataFrame(df)
    print(df)
    df = df[['Model', ' Distortion Loss2', ' Accuracy2', ' Lambda_CLS', ' Accuracy1',' Distortion Loss1']]
    df.to_csv(f"Results/{title}.csv", index=False)

    print("RDC Results are saved in the Results folder")


def plot_rdc_curve_single_rate_multiple_frames(path, title):

    path_to_folders = path
    list_of_folders = os.listdir(path_to_folders)

    # Create 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    updated_folders = {
        "Universal": [],
        "P-Frame": []
    }

    # ---------------------------------------------------------
    # Separate Universal and P-Frame folders
    # ---------------------------------------------------------
    for folder in list_of_folders:

        # Skip I-frame folders
        if "Iframe" in folder.split("_")[0]:
            continue

        elif "Universal" in folder:
            updated_folders["Universal"].append(folder)

        else:
            updated_folders["P-Frame"].append(folder)

    # Optional: sort so that Universal/P-Frame lambda values
    # are processed in a consistent order
    updated_folders["Universal"].sort()
    updated_folders["P-Frame"].sort()

    # ---------------------------------------------------------
    # Get rate from Universal folder
    # ---------------------------------------------------------
    rate, _ = get_rate(updated_folders["Universal"][0])

    # ---------------------------------------------------------
    # Plot Universal and P-Frame
    # ---------------------------------------------------------
    i = 0


    for p_path, unv_path in zip(
        updated_folders["P-Frame"],
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

        path_to_pframe_csv = (
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
        # Read P-Frame CSV
        # -----------------------------------------------------
        df_pframe = pd.read_csv(path_to_pframe_csv)

        df_pframe = df_pframe.tail()

        # Calculate mean
        df_pframe_mean = df_pframe.mean()

        # -----------------------------------------------------
        # UNIVERSAL
        #
        # Frame 1:
        #     X = Distortion Loss1
        #     Y = Accuracy1
        # -----------------------------------------------------
        axes[0].scatter(
            df_unv_mean[" Distortion Loss1"],
            df_unv_mean[" Accuracy1"],
            marker="o",
            c="deeppink",
            s=60,
            label=f"λ_cls = {lambda_cls}"
        )
        axes[0].scatter(
            df_pframe_mean[" Distortion Loss1"],
            df_pframe_mean[" Accuracy1"],
            marker="o",
            c="deeppink",
            edgecolors="black",
            s=80,
            label=f"λ_cls = {lambda_cls_p}"
        )
        
        

        # -----------------------------------------------------
        # P-FRAME
        #
        # Frame 2:
        #     X = Distortion Loss2
        #     Y = Accuracy2
        # -----------------------------------------------------

        axes[1].scatter(
            df_unv_mean[" Distortion Loss2"],
            df_unv_mean[" Accuracy2"],
            marker="o",
            c="deeppink",
            s=60,
            label=f"λ_cls = {lambda_cls}"
        )
        
        axes[1].scatter(
            df_pframe_mean[" Distortion Loss2"],
            df_pframe_mean[" Accuracy2"],
            marker="o",
            edgecolors="black",
            c="deeppink",
            s=80,
            label=f"λ_cls = {lambda_cls_p}"
        )
          

      
        

        i += 1

    # ---------------------------------------------------------
    # Universal subplot formatting
    # ---------------------------------------------------------
    axes[0].set_title("Frame 1")
    axes[0].set_xlabel("Distortion Loss")
    axes[0].set_ylabel("Classification Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # P-Frame subplot formatting
    # ---------------------------------------------------------
    axes[1].set_title("Frame 2")
    axes[1].set_xlabel("Distortion Loss")
    axes[1].set_ylabel("Classification Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # Overall title
    # ---------------------------------------------------------
    fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    # Save
    plt.savefig(
        f"Results/{title}.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print("RDC Results are saved in the Results folder")

def plot_rdc_curve_single_rate_multiple_frames_universal_only(path, title):

    path_to_folders = path
    list_of_folders = os.listdir(path_to_folders)
    fig , ax = plt.subplots(figsize=(6, 5))
    
    
    universal_folders = [folder for folder in list_of_folders if "Universal" in folder]
    universal_folders.sort()
    i = 0
    rate, _ = get_rate(universal_folders[0])
    for path in universal_folders:
        if i > 3:
            break
        
        i += 1
        lambda_cls = path.split("lambdaCLS_")[1].split("_")[0]
        # path_to_csv = os.path.join(path_to_folders, path, f"Rate_{float(rate)}0_CLS_{lambda_cls}_performance.csv")
        if "Pframe" in path:
            
            path_to_csv = path_to_folders+ '/' + path  + '/' +  f"Rate_5.584962500721157_CLS_{lambda_cls}_performance.csv"
        else:
            path_to_csv = path_to_folders+ '/' + path  + '/' +  f"Rate_5.58_CLS_{lambda_cls}_performance.csv"
            
        if not os.path.exists(path_to_csv):
            print(f"Missing: {path_to_csv}")
            continue
        
        df = pd.read_csv(path_to_csv).tail()
        df_mean = df.mean()
        
        ax.scatter(df_mean[" Distortion Loss1"], df_mean[" Accuracy1"], marker="o",c='deeppink', s=60, label=f"λ_cls = {lambda_cls} Frame 1")
        ax.scatter(df_mean[" Distortion Loss2"], df_mean[" Accuracy2"], marker="o",c='deeppink', edgecolors="black", s=80,label=f"λ_cls = {lambda_cls} Frame 2")
    
    ax.set_title(title)
    ax.set_xlabel("Distortion Loss")
    ax.set_ylabel("Classification Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"Results/{title}.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.show()
    print("RDC Results are saved in the Results folder")
    


# def plot_rdc_curve_single_rate(path, title):
#     path_to_folders = path
#     list_of_folders = os.listdir(path_to_folders)
#     plt.figure(figsize=(10, 10))
#     updated_folders = {
#         "Universal":[],
#         "P-Frame":[]
#     }
#     z_dim, L = 'z_dim_4','2'
    
#     for i in range(len(list_of_folders)):
#         if z_dim in list_of_folders[i] and L in list_of_folders[i]:
            
#             if "Iframe" in list_of_folders[i].split("_")[0]:
#                 continue
#             elif "Universal" in list_of_folders[i]:
#                 updated_folders['Universal'].append((list_of_folders[i]))
                
#             else:
#                 updated_folders['P-Frame'].append((list_of_folders[i]))
    
    

#     i = 0
#     for  p_path,unv_path in zip(updated_folders["P-Frame"], updated_folders["Universal"]):
#         if i > 3:
#             break
#         lambda_cls = unv_path.split("lambdaCLS_")[1].split("_")[0]
#         lambda_cls_p = p_path.split("lambdaCLS_")[1].split("_")[0]
        
#         path_to_unv_csv = path_to_folders + "/" + unv_path + f"/Rate_1_CLS_{lambda_cls}_performance.csv"
#         path_to_pframe_csv = path_to_folders + "/" + p_path + f"/Rate_1_CLS_{lambda_cls}_performance.csv"
#         df_unv = pd.read_csv(path_to_unv_csv)
#         df_unv = df_unv.tail()
#         df_unv_mean = df_unv.mean()
        
        
#         df_pframe = pd.read_csv(path_to_pframe_csv)
#         df_pframe = df_pframe.tail()
#         df_pframe_mean = df_pframe.mean()
        
#         i += 1
#         plt.scatter(df_unv_mean[' Distortion Loss'], df_unv_mean[' Accuracy'],  marker = 'o', label='lambda_cls = ' + str(lambda_cls) + ' Universal')
#         plt.scatter(df_pframe_mean[' Distortion Loss'], df_pframe_mean[' Accuracy'], marker = 'o',edgecolors='black', s = 60, label='lambda_cls = ' + str(lambda_cls_p) + ' P-Frame')
#         plt.title(f"{title}")
#         plt.xlabel('Distortion Loss')
#         plt.ylabel('Classification Accuracy')
#         plt.legend()

#     plt.tight_layout()
#     plt.savefig(f"Results/{title}.png", dpi=200,bbox_inches="tight")
#     plt.show()
#     print("RDC Results are saved in the Results folder")




def compare_rdc_curve(path1, path2, title):
    fig , axes = plt.subplots(1, 2, figsize=(12, 6))
    images = [path1, path2]
    for ax, path in zip(axes, images):
        img = Image.open(path)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    results_dir = os.path.join(
        os.getcwd(),
        "Results"
    )
    
    save_path = os.path.join(
                results_dir,
                f"{title}.png"
    )
    
    plt.savefig(
        save_path,
        dpi=200,
        # bbox_inches="tight"
    )

    plt.show()

    plt.close(fig)

    print(f"Saved: {save_path}")
    
    
    
                
def plot_rdc_curve():

    path_to_folders = os.path.join(os.getcwd(), "saved_models")
    list_of_folders = os.listdir(path_to_folders)

    updated_folders = {
        "Universal": [],
        "P-Frame": []
    }
    rate_counter = {
        "Universal":{},
        "P-Frame": {}
    }
    # Separate Universal and P-frame folders
    for folder in list_of_folders:

        if "Iframe" in folder:
            continue

        if "Universal" in folder:
            rate, z_dim = get_rate(folder)
            rate_counter["Universal"][z_dim] = rate_counter["Universal"].get(z_dim, 0) + 1
            if rate_counter["Universal"][z_dim] > 4:
                continue
            updated_folders["Universal"].append(folder)
        else:
            rate, z_dim = get_rate(folder)
            rate_counter["P-Frame"][z_dim] = rate_counter["P-Frame"].get(z_dim, 0) + 1
            if rate_counter["P-Frame"][z_dim] > 4:
                continue
            updated_folders["P-Frame"].append(folder)
            
    # Sort by bitrate
    updated_folders["Universal"].sort(key=get_rate)
    updated_folders["P-Frame"].sort(key=get_rate)

    results = {
        "Universal": [],
        "P-Frame": []
    }
    for p_path, unv_path in zip(updated_folders["P-Frame"],
                                updated_folders["Universal"]):

        rate, z_dim= get_rate(unv_path)

        lambda_cls_unv = unv_path.split("lambdaCLS_")[1]
        lambda_cls_p = p_path.split("lambdaCLS_")[1]

        path_to_unv_csv = os.path.join(
            path_to_folders,
            unv_path,
            f"Rate_{z_dim}_CLS_{lambda_cls_unv}_performance.csv"
        )

        path_to_pframe_csv = os.path.join(
            path_to_folders,
            p_path,
            f"Rate_{z_dim}_CLS_{lambda_cls_p}_performance.csv"
        )

        if not os.path.exists(path_to_unv_csv):
            print(f"Missing: {path_to_unv_csv}")
            continue

        if not os.path.exists(path_to_pframe_csv):
            print(f"Missing: {path_to_pframe_csv}")
            continue

        df_unv = pd.read_csv(path_to_unv_csv).tail()
        df_pframe = pd.read_csv(path_to_pframe_csv).tail()

        df_unv_mean = df_unv.mean(numeric_only=True)
        df_pframe_mean = df_pframe.mean(numeric_only=True)

        results["Universal"].append((
            rate,
            df_unv_mean[" Distortion Loss"],
            df_unv_mean[" Accuracy"]
        ))

        results["P-Frame"].append((
            rate,
            df_pframe_mean[" Distortion Loss"],
            df_pframe_mean[" Accuracy"]
        ))

    # Define the rates present
    rates = [2, 3, 4]

    # Colormap and normalization
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=min(rates), vmax=max(rates))
    
    ax = plt.gca()

    for model in ["Universal", "P-Frame"]:

        pts = sorted(results[model], key=lambda x: x[0])

        x = [p[1] for p in pts]      # Distortion
        y = [p[2] for p in pts]      # Accuracy
        r = [p[0] for p in pts]      # Bitrate

        # # Colored line
        # ax.plot(
        #     x,
        #     y,
        #     color='gray',
        #     linewidth=1.5,
        #     alpha=0.7
        # )

        # Colored scatter points
        if model == "P-Frame":
            ax.scatter(
                x,
                y,
                c=r,
                cmap=cmap,
                norm=norm,
                s=90,
                edgecolors='black',
                label=model
            )
            
        else:
            sc = ax.scatter(
                x,
                y,
                c=r,
                cmap=cmap,
                norm=norm,
                s=50,
                edgecolors='black',
                label=model
            )

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Bitrate (R)")

    plt.title("Rate-Distortion-Classification (RDC) Curve")
    plt.xlabel("Distortion Loss")
    plt.ylabel("Classification Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("Results/Rate_Distortion_Classification_Curve.png",
                dpi=200, bbox_inches="tight")
    plt.show()


    print("RDC curve saved to Results/Rate_Distortion_Classification_Curve.png")
    
    
    
def convert_csv_to_updated_csv():
    path_to_folders = os.getcwd() + "/saved_models"
    list_of_folders = os.listdir(path_to_folders)
    

    plt.figure(figsize=(10, 10))
    for i in range(len(list_of_folders)):
            data = []
            path_to_csv = path_to_folders + "/" + list_of_folders[i] + "/performance.csv"
            with open(f"{path_to_csv}", "r") as f:
                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    values = {}

                    for item in line.split(","):
                        key, value = item.split(":", 1)
                        values[key.strip()] = float(value.strip())

                    data.append(values)

            df = pd.DataFrame(data)
            print(f"{path_to_folders}/{list_of_folders[i]}/updated_performance.csv")
            df.to_csv(f"{path_to_folders}/{list_of_folders[i]}/updated_performance.csv", index=False)


def plot_loss_and_accuracy():
                
    path_to_folders = os.getcwd() + "/saved_models"
    list_of_folders = os.listdir(path_to_folders)
    plt.figure(figsize=(10, 10))
    for i in range(len(list_of_folders)):
        path_to_csv = path_to_folders + "/" + list_of_folders[i] + "/updated_performance.csv"
        df = pd.read_csv(path_to_csv)

        plt.subplot(2, 2, i + 1)
        plt.plot(df['Epoch'], df['Accuracy'], label='Accuracy')
        plt.plot(df['Epoch'], df['Test Loss'], label='Test Loss')
        Lamda_CLS = df['Lambda_CLS'].iloc[0]
        # plt.plot(df['Distortion Loss'], df['Classification Loss'], label=f'Distortion vs Classification Loss ')
        plt.title(f"Lambda Classification" + f" = {Lamda_CLS}")
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
    plt.savefig(f"./Results/Accuracy Vs Test Loss.png")
    plt.show()

def plot_distortion_and_classification_loss():
    # how to set the figure size and the number of subplots based on the number of folders in saved_models
    path_to_folders = os.getcwd() + "/saved_models"
    list_of_folders = os.listdir(path_to_folders)
    plt.figure(figsize=(10, 10))
    for i in range(len(list_of_folders)):

        path_to_csv = path_to_folders + "/" + list_of_folders[i] + "/updated_performance.csv"
        df = pd.read_csv(path_to_csv)
        plt.subplot(2, 2, i + 1)
        plt.plot(df['Epoch'], df['Distortion Loss'], label='Distortion Loss')
        plt.plot(df['Epoch'], df['Classification Loss'], label='Classification Loss')
        Lamda_CLS = df['Lambda_CLS'].iloc[0]
        plt.title(f"Lambda Classification" + f" = {Lamda_CLS}")
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
    plt.savefig(f"./Results/Classification Loss Vs Distortion Loss.png")
    plt.show()



def reconstruct_and_plot_images():

    path_to_folders = os.getcwd() + "/saved_models"
    list_of_folders = os.listdir(path_to_folders)
    # Skip those folders containing IFrame
    

    lambda_values = [
        folder.split("lambda_cls_")[1].split("_")[0]
        for folder in list_of_folders
    ]

    fig, axes = plt.subplots(
        len(list_of_folders),
        3,
        figsize=(12, 4 * len(list_of_folders))
    )

    # If there is only one folder, make axes 2D
    if len(list_of_folders) == 1:
        axes = axes.reshape(1, 3)

    for i, folder in enumerate(list_of_folders[:]):

        path_to_img = (
            path_to_folders
            + "/"
            + folder
            + "/recons/recon_epoch_0050.png"
        )

        img = plt.imread(path_to_img)

        height = img.shape[0]

        # Split image into 3 rows
        reference = img[:height // 3]
        current = img[height // 3:2 * height // 3]
        reconstruction = img[2 * height // 3:]

        # Reference
        axes[i, 0].imshow(reference)
        axes[i, 0].set_title(
            f"Reference"
        )
        axes[i, 0].axis("off")

        # Current
        axes[i, 1].imshow(current)
        axes[i, 1].set_title(f" λ_cls = {lambda_values[i]}\nOriginal Current")
        axes[i, 1].axis("off")

        # Reconstruction
        axes[i, 2].imshow(reconstruction)
        axes[i, 2].set_title("Reconstruction")
        axes[i, 2].axis("off")

    plt.tight_layout()

    os.makedirs("Results", exist_ok=True)

    plt.savefig(
        "Results/Reconstructed Images.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    



def main():
    # import pandas as pd
    # plot_loss_and_accuracy()
    # reconstruct_and_plot_images()
    # plot_distortion_and_classification_loss()
    # plot_rdc_curve()
    # plot_universal_rdc_curve()
    # plot_reconstructed_images()
    # path = os.getcwd() + "/saved_models" + "/R1_UnCompressed"

    # plot_rdc_curve_single_rate(path = path, title = "RDC Curve for Uncompressed I-Frame Z=4 L=2")
    
    # path = os.getcwd() + "/saved_models" + "/R1_Compressed/Iframe_z_dim_2_L_2"
    
    # plot_rdc_curve_single_rate(path = path, title = "RDC Curve for Compressed I-Frame Z=2 L=2")
    
    
    # compare_rdc_curve(
    #     path1 = "Results/RDC Curve for Compressed I-Frame Z=4 L=2.png",
    #     path2 = "Results/RDC Curve for Compressed I-Frame Z=2 L=2.png",
    #     title = "Comparison of RDC Curves for Compressed I-Frame Z=4 L=2 and Z=2 L=2"
    # )
    
    # compare_rdc_curve(
    #     path1 = "saved_models/R1_Uncompressed/Universal_dim_128_z_dim_2_L_2_bs_64_lambdaMSE_1.0_lambdaCLS_0.01/recons/recon_epoch_0080.png",
    #     path2 = "saved_models/R1_Compressed/Iframe_z_dim_4_L_2/Universal_dim_128_z_dim_1_L_2_bs_64_lambdaMSE_1.0_lambdaCLS_0.01_Iframe_pretrained/recons/recon_epoch_0080.png",
    #     title = "Universal Constructions Comparison for Uncompressed & Compressed I-Frame Z=4 L=2 and Z=2 L=2"
    # )
    
    
    # compare_rdc_curve(
    #     path1 = "saved_models/R1_Compressed/Iframe_z_dim_2_L_2/Universal_dim_128_z_dim_1_L_2_bs_64_lambdaMSE_1.0_lambdaCLS_0.01_Iframe_pretrained/recons/recon_epoch_0080.png",
    #     path2 = "saved_models/R1_Compressed/Iframe_z_dim_4_L_2/Universal_dim_128_z_dim_1_L_2_bs_64_lambdaMSE_1.0_lambdaCLS_0.01_Iframe_pretrained/recons/recon_epoch_0080.png",
    #     title = "Universal Constructions Comparison for Compressed I-Frame Z=2 L=2 and Z=4 L=2"
    # )
    # plot_rdc_curve_single_rate(path = path, title = "RDC Curve for Compressed I-Frame Z=2 L=2")

    
    # plot_rdc_curve_single_rate_multiple_frames("./saved_models/compressed/Iframe_z_dim_4_L_2/Pframe_z_dim_1_L_2", title = "RDC Comparison over R1 = 3 & R2 = 2")
    # plot_rdc_curve_single_rate_multiple_frames("./saved_models/compressed/Iframe_z_dim_4_L_2/Pframe_z_dim_2_L_2", title = "RDC Comparison over R1 = 3 & R2 = 3")
    # plot_rdc_curve_single_rate_multiple_frames("./saved_models/compressed/Iframe_z_dim_12_L_2/Pframe_z_dim_2_L_2", title = "RDC Comparison over R1 = 4.585 & R2 = 3")
    # plot_rdc_curve_single_rate_multiple_frames("./saved_models/compressed/Iframe_z_dim_12_L_2/Pframe_z_dim_4_L_2", title = "RDC Comparison over R1 = 4.585 & R2 = 4")
        
    # plot_rdc_curve_single_rate_multiple_frames_universal_only("./saved_models/compressed/Iframe_z_dim_4_L_2/Pframe_z_dim_1_L_2", title = "RDC Comparison for Universal Encoder over R1 = 3 & R2 = 2")
    # plot_rdc_curve_single_rate_multiple_frames_universal_only("./saved_models/compressed/Iframe_z_dim_4_L_2/Pframe_z_dim_2_L_2", title = "RDC Comparison for Universal Encoder over R1 = 3 & R2 = 3")
    # plot_rdc_curve_single_rate_multiple_frames_universal_only("./saved_models/compressed/Iframe_z_dim_12_L_2/Pframe_z_dim_2_L_2", title = "RDC Comparison for Universal Encoder over R1 = 4.585 & R2 = 3")
    # plot_rdc_curve_single_rate_multiple_frames_universal_only("./saved_models/compressed/Iframe_z_dim_12_L_2/Pframe_z_dim_4_L_2", title = "RDC Comparison for Universal Encoder over R1 = 4.585 & R2 = 4")
    
    
    # plot_rdc_curve_single_rate_multiple_frames_universal_only("./saved_models/compressed/I_frame", title = "RDC Comparison for Universal Encoder over R = 12")
    plot_rdc_curve_single_rate(path = "./saved_models2/compressed/Iframe_z_dim_12_L_2", title = "RDC Curve over R=12 Final")
    
if __name__ == "__main__":
    main()
    