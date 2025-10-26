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

# from LION.models.post_processing.FBPConvNet import FBPConvNet
from LION.models.post_processing.RISINGConvNet import RISINGConvNet


def my_ssim(x, y):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    return ssim(x, y, data_range=x.max() - x.min())


# %%
# % Chose device:
device = torch.device("cuda:3")
torch.cuda.set_device(device)

# Define your data paths
savefolder = pathlib.Path("/store/LION/smrsi2/trained_models/test_debbuging_RISINGConvNet/")
final_result_fname = "RISINGConvNet.pt"
checkpoint_fname = "RISINGConvNet_check_*.pt"
validation_fname = "RISINGConvNet_min_val.pt"

# %% Define experiment

experiment = ct_experiments.LowDoseCTRecon(dataset="LIDC-IDRI")
# experiment = ct_experiments.ExtremeLowDoseCTRecon(dataset="LIDC-IDRI")
# %% Dataset
lidc_dataset = experiment.get_training_dataset()
lidc_dataset_val = experiment.get_validation_dataset()
lidc_dataset_test = experiment.get_testing_dataset()
# smaller dataset for example. Remove this for full dataset
indices = torch.arange(10)
lidc_dataset = data_utils.Subset(lidc_dataset, indices)
lidc_dataset_val = data_utils.Subset(lidc_dataset_val, indices)

# get one sample


# %% Define DataLoader
# Use the same amount of training


batch_size = 1
lidc_dataloader = DataLoader(lidc_dataset, batch_size, shuffle=True)
lidc_validation = DataLoader(lidc_dataset_val, batch_size, shuffle=False)
lidc_test = DataLoader(lidc_dataset_test, batch_size, shuffle=False)


# %% Model

# *** MODIFICATION 2: Initialize FBPConvNet with its parameters ***
default_parameters = RISINGConvNet.default_parameters()
# FBPConvNet does not use 'learned_step' or 'n_iters' as it's a non-iterative model.
# The network structure and parameters are configured within its default_parameters.
model = RISINGConvNet(experiment.geometry, default_parameters)
model.cite()
model.cite("bib")

# %% Optimizer
train_param = LIONParameter()

# loss fn
loss_fcn = torch.nn.MSELoss()
train_param.optimiser = "adam"

# optimizer
train_param.epochs = 200
train_param.learning_rate = 1e-4
train_param.betas = (0.9, 0.99)
train_param.loss = "MSELoss"
optimiser = torch.optim.Adam(
    model.parameters(), lr=train_param.learning_rate, betas=train_param.betas
)

# %% Train
# create solver
'''
solver = SupervisedSolver(
    model, optimiser, loss_fcn, verbose=True, save_folder=savefolder
)
'''
solver = RISINGSolver(
    model, optimiser, loss_fcn, verbose=True, save_folder=savefolder
)

# YOU CAN IGNORE THIS. You can 100% just write your own pytorch training loop.
# LIONSover is just a convinience class that does some stuff for you, no need to use it.

# set data
solver.set_training(lidc_dataloader)
solver.set_validation(lidc_validation, 10, validation_fname=validation_fname)
solver.set_testing(lidc_test, my_ssim)

# set checkpointing procedure
solver.set_checkpointing(
    checkpoint_fname, 10, load_checkpoint_if_exists=False, save_folder=savefolder
)
# train
solver.train(train_param.epochs)
# delete checkpoints if finished
solver.clean_checkpoints()
# save final result
solver.save_final_results(final_result_fname, savefolder)

# test

#solver.test()

plt.figure()
plt.semilogy(solver.train_loss[1:])
plt.savefig("loss.png")