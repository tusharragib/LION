# %% This example shows how to train FBPConvNet for full angle, noisy measurements.

# %% Imports

# Standard imports
import matplotlib.pyplot as plt
import pathlib
from skimage.metrics import structural_similarity as ssim

# Torch imports
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

# Lion imports
# *** MODIFICATION 1: Import FBPConvNet ***
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
from LION.optimizers.SupervisedSolver import SupervisedSolver
from LION.optimizers.RISINGSolver import RISINGSolver

from LION.models.post_processing.RISINGConvNet import RISINGConvNet
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
import LION.CTtools.ct_geometry as ctgeo
import LION.CTtools.ct_utils as ct
from LION.classical_algorithms.tv_min import tv_min

import pandas as pd


def my_ssim(x, y):
    # x = x.cpu().numpy().squeeze()
    # y = y.cpu().numpy().squeeze()
    x = x.detach().cpu().numpy().squeeze()
    y = y.detach().cpu().numpy().squeeze()
    return ssim(x, y, data_range=x.max() - x.min())


# %%
# % Chose device:
device = torch.device("cuda:3")
torch.cuda.set_device(device)

# # Define your data paths
# savefolder = pathlib.Path("/store/LION/smrsi2/trained_models/test_debbuging_FBPConvNet/")
# final_result_fname = "FBPConvNet.pt"
# checkpoint_fname = "FBPConvNet_check_*.pt"
# validation_fname = "FBPConvNet_min_val.pt"

# %% Define experiment

# experiment = ct_experiments.LowDoseCTRecon(dataset="LIDC-IDRI")
# # experiment = ct_experiments.ExtremeLowDoseCTRecon(dataset="LIDC-IDRI")
# # %% Dataset
# lidc_dataset = experiment.get_training_dataset()
# lidc_dataset_val = experiment.get_validation_dataset()
# lidc_dataset_test = experiment.get_testing_dataset()
# # smaller dataset for example. Remove this for full dataset
# indices = torch.arange(10)
# lidc_dataset = data_utils.Subset(lidc_dataset, indices)
# lidc_dataset_val = data_utils.Subset(lidc_dataset_val, indices)

# get one sample


# %% Define DataLoader
# Use the same amount of training


# batch_size = 1
# lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
# lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=False)
# lidc_test = DataLoader(lidc_dataset_test, batch_size, shuffle=False)


# %% Model

# *** MODIFICATION 2: Initialize FBPConvNet with its parameters ***
default_parameters = RISINGConvNet.default_parameters()
# FBPConvNet does not use 'learned_step' or 'n_iters' as it's a non-iterative model.
# The network structure and parameters are configured within its default_parameters.
model, options, data = RISINGConvNet.load("/store/LION/smrsi2/trained_models/test_debbuging_RISINGConvNet/RISINGConvNet.pt")
model.cite()
model.cite("bib")

geo = ctgeo.Geometry.default_parameters()
dataset = LIDC_IDRI(mode='train', geometry_parameters = geo)

# smaller dataset for example. Remove this for full dataset
indices = torch.arange(10)
dataset = data_utils.Subset(dataset, indices)
#lidc_dataset_val = data_utils.Subset(lidc_dataset_val, indices)

LR = 1e-4
RIS_ITERS = 10                # "tv_min" iterations for RIS (the RIS part)
RIS_ITERS_high = 300          # "tv_min" iterations for ING (the RIS part)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
geo = ctgeo.Geometry.default_parameters()
lam = 0.00001


op = ct.make_operator(geo)
################################## [The following is for a single datapoint] ###################
################################## [Starts here] ###############################################
# #dataset = LIDC_IDRI(mode='train', geometry_parameters = geo)
# first_sample = dataset[2]
# sinogram, ground_truth_unsqueezed = first_sample # ground_truth_unsqueezed.shape = [1, 512, 512]

# ground_truth = torch.squeeze(ground_truth_unsqueezed)

# # sinogram = torch.from_numpy(sinogram_np)

# sinogram = torch.unsqueeze(sinogram, 0)

# op = ct.make_operator(geo)
# #sino = op(torch.from_numpy(first_sample)) # The dataset alread has the projection set!

# recon = tv_min(sinogram, op, lam=lam, num_iterations=RIS_ITERS)
# #recon_fdk = fdk(sinogram, op)

# # recon_test = torch.squeeze(recon)

# highRecon = model(recon)
# recon_test = torch.squeeze(highRecon)


# # Create a new figure and axes
# numpy_image = recon_test.detach().cpu().numpy()
# fig, ax = plt.subplots(1, 1)
# ax.imshow(numpy_image, cmap='gray')
# ax.set_title('Reconstructed Image')
# fig.colorbar(ax.get_images()[0], ax=ax, label='Intensity')
# fig.savefig("RISINGRecon.png")
# plt.show()

# ssim_score = my_ssim(recon_test, ground_truth)
###################################### [Ends here] ###########################################


################################## [The following is for a single datapoint] ###################
################################## [Starts here] ###############################################

results = []

for i in range (0, len(dataset)): #enumerate(len(dataset)) # range(0,3)
    sample = dataset[i]
    
    sinogram, ground_truth_unsqueezed = sample # ground_truth_unsqueezed.shape = [1, 512, 512]
    
    ground_truth = torch.squeeze(ground_truth_unsqueezed)
    sinogram = torch.unsqueeze(sinogram, 0)

    recon = tv_min(sinogram, op, lam=lam, num_iterations=RIS_ITERS)
    recon_high = tv_min(sinogram, op, lam=lam, num_iterations=RIS_ITERS_high)
    recon_high = torch.squeeze(recon_high)
   
    highRecon = model(recon)
    recon_test = torch.squeeze(highRecon)

    ssim_score_high = my_ssim(recon_test, recon_high)
    ssim_score_GT = my_ssim(recon_test, ground_truth)

    filename = f"image_{i}.png"

    results.append([filename, ssim_score_high, ssim_score_GT])


    print(f"Processed {filename}, SSIM: {ssim_score_high:.4f}, SSIM: {ssim_score_GT:.4f}")

    # # Create a new figure and axes
    # numpy_image = recon_test.detach().cpu().numpy()
    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(numpy_image, cmap='gray')
    # ax.set_title(filename)
    # fig.colorbar(ax.get_images()[0], ax=ax, label='Intensity')
    # fig.savefig(filename)
    # plt.show()

    # --- 1. Load/Generate the Three Image Arrays ---
    recon = torch.squeeze(recon)
    recon = recon.detach().cpu().numpy()
    recon_test = recon_test.detach().cpu().numpy()
    recon_high = recon_high.detach().cpu().numpy()
    ground_truth = ground_truth.detach().cpu().numpy()

    # --- 2. Create Figure and Axes (1 Row, 3 Columns) ---

    # fig: the overall container
    # ax: a list/array of 3 individual axes (ax[0], ax[1], ax[2])
    fig, ax = plt.subplots(1, 4, figsize=(16, 4)) # Set figsize for better viewing


    # --- 3. Plot Each Image Explicitly ---

    # --- Plot 1: recon_test (on ax[0]) ---
    im0 = ax[0].imshow(recon, cmap='gray')
    ax[0].set_title("recon_10iter")
    ax[0].axis('off')
    fig.colorbar(im0, ax=ax[0], label='Intensity', fraction=0.046, pad=0.04)
    
    im1 = ax[1].imshow(recon_test, cmap='gray')
    ax[1].set_title("recon_test")
    ax[1].axis('off')
    fig.colorbar(im1, ax=ax[1], label='Intensity', fraction=0.046, pad=0.04)

    # --- Plot 2: recon_high (on ax[1]) ---
    im2 = ax[2].imshow(recon_high, cmap='gray')
    ax[2].set_title("recon_300iter")
    ax[2].axis('off')
    fig.colorbar(im2, ax=ax[2], label='Intensity', fraction=0.046, pad=0.04)

    # --- Plot 3: ground_truth (on ax[2]) ---
    im3 = ax[3].imshow(ground_truth, cmap='gray')
    ax[3].set_title("ground_truth")
    ax[3].axis('off')
    fig.colorbar(im3, ax=ax[3], label='Intensity', fraction=0.046, pad=0.04)


    # --- 4. Final adjustments and Display ---
    fig.suptitle(filename, fontsize=16, fontweight='bold')

    # Adjust layout to prevent titles/labels from overlapping
    fig.tight_layout()

    # Save the combined figure (optional)
    fig.savefig(filename)

    # Display the figure
    plt.show()

# 1. Create a DataFrame (N rows x 2 columns)
df = pd.DataFrame(results, columns=['Image_Name', 'SSIM_Score_high', 'SSIM_Score_GT'])

# 2. Save the DataFrame as a CSV file
output_csv_filename = 'ssim_results.csv'
df.to_csv(output_csv_filename, index=False) 

#print(f"\nSuccessfully saved {N} results to {output_csv_filename}")