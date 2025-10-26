import os
import time
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim




# LION imports
from LION.utils.paths import LIDC_IDRI_PATH
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI

import LION.CTtools.ct_geometry as ctgeo
import LION.CTtools.ct_utils as ct
import LION.experiments.ct_experiments as ct_experiments

from LION.classical_algorithms.tv_min import tv_min
from LION.classical_algorithms.fdk import fdk

from LION.models.post_processing.FBPConvNet import FBPConvNet
#from LION.utils.parameter import LIONParameter

# NUM_PROJECTIONS =           
# IMAGE_SIZE =               
# BATCH_SIZE = 
# NUM_EPOCHS = 
LR = 1e-4
RIS_ITERS = 10                # "tv_min" iterations for RIS (the RIS part)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
geo = ctgeo.Geometry.default_parameters()
lam = 0.00001

dataset = LIDC_IDRI(mode='train', geometry_parameters = geo)

first_sample = dataset[0]

sinogram, ground_truth_np = first_sample

# sinogram_np = torch.from_numpy(sinogram))
# sinogram_np = np.expand_dims(sinogram_np, 0)  # has to be 3D, even for 2D images

# sinogram = torch.from_numpy(sinogram_np)

sinogram = torch.unsqueeze(sinogram, 0)

op = ct.make_operator(geo)
#sino = op(torch.from_numpy(first_sample)) # The dataset alread has the projection set!

recon = tv_min(sinogram, op, lam=lam, num_iterations=RIS_ITERS)
#recon_fdk = fdk(sinogram, op)

recon_test = torch.squeeze(recon)

# Display the 2D reconstructed image
numpy_image = recon_test.detach().cpu().numpy()
plt.imshow(numpy_image, cmap='gray')
plt.title('ReconstructedImage')
plt.savefig("ReconstructedImage.png")
plt.show()

# Create a new figure and axes
numpy_image = recon_test.detach().cpu().numpy()
fig, ax = plt.subplots(1, 1)
ax.imshow(numpy_image, cmap='gray')
ax.set_title('Reconstructed Image')
fig.colorbar(ax.get_images()[0], ax=ax, label='Intensity')
fig.savefig("ReconstructedImage_10.png")
plt.show()

experiment = ct_experiments.SparseAngleCTRecon()
experiment_parameters = ct_experiments.SparseAngleCTRecon.default_parameters()
experiment = ct_experiments.SparseAngleCTRecon(experiment_params=experiment_parameters)

experiment = ct_experiments.SparseAngleCTRecon(dataset="LIDC-IDRI")

print(experiment)

experiment.geometry