import os
import pathlib
import numpy as np

# Set target CUDA device index before importing PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless plotting
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn as nn

# LION Framework Imports
from LION.utils.parameter import LIONParameter
import LION.experiments.ct_experiments as ct_experiments
import LION.CTtools.ct_geometry as ctgeo
from LION.optimizers.Z_RISINGSolver import RISINGSolver
from LION.data_loaders.Z_ImageDataset3D import create_3d_dataloaders
from LION.models.post_processing.Z_RISINGConvNet_3D import RISINGConvNet_3D as RISINGConvNet

# Enforce deterministic behavior and disable TF32 precision
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def my_ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    """Calculates 3D/2D Structural Similarity Index (SSIM) between target and prediction."""
    x_np = x.cpu().numpy().squeeze()
    y_np = y.cpu().numpy().squeeze()
    return ssim(x_np, y_np, data_range=x_np.max() - x_np.min())


# =====================================================================
# CUSTOM LOSS FUNCTIONS
# =====================================================================

class L1LossOnly(nn.Module):
    """Standard L1 Mean Absolute Error Loss."""
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(pred, target)


class L1_MSE_Loss(nn.Module):
    """Hybrid loss combining L1 and MSE weighted by alpha."""
    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.l1(pred, target) + (1.0 - self.alpha) * self.mse(pred, target)


class L1_Gradient_Loss(nn.Module):
    """Composite loss measuring absolute voxel intensity error and 3D spatial gradient differences."""
    def __init__(self, lambda_grad: float = 0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.lambda_grad = lambda_grad

    def gradient(self, x: torch.Tensor):
        """Computes finite differences along X, Y, and Z spatial dimensions for 5D tensors (B, C, D, H, W)."""
        dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        return dx, dy, dz

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1_loss = self.l1(pred, target)

        dx_p, dy_p, dz_p = self.gradient(pred)
        dx_t, dy_t, dz_t = self.gradient(target)

        grad_loss = (
            torch.mean(torch.abs(dx_p - dx_t)) +
            torch.mean(torch.abs(dy_p - dy_t)) +
            torch.mean(torch.abs(dz_p - dz_t))
        )

        return l1_loss + self.lambda_grad * grad_loss


# =====================================================================
# CUSTOM CALLBACK CLASS WRAPPING THE VALIDATION LOSS FUNCTION
# =====================================================================

class EpochCallbackLoss(nn.Module):
    """Wraps validation loss evaluation to execute mid-training checkpoints, model cleaning, 
    and persistent history tracking without altering internal solver routines.
    """
    def __init__(self, base_loss_fn: nn.Module, model: nn.Module, experiment, train_param: LIONParameter, savefolder: pathlib.Path):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.model = model
        self.experiment = experiment
        self.train_param = train_param
        self.savefolder = savefolder
        
        self.solver = None  # Bound externally post-solver creation
        self.previous_model_path = None
        
        # Disk storage for persistent metric tracking across session restarts
        self.history_file = self.savefolder / "validation_history.txt"
        self.persistent_val_history = []
        
        # Restore historical metrics if resuming training from disk
        if self.history_file.exists():
            try:
                with open(self.history_file, "r") as f:
                    self.persistent_val_history = [float(line.strip()) for line in f if line.strip()]
                print(f"[Callback] Loaded {len(self.persistent_val_history)} historical validation points from disk.")
            except Exception as e:
                print(f"[Callback] Could not load validation history file: {e}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_val = self.base_loss_fn(pred, target)
        
        if self.solver is not None:
            current_epoch = getattr(self.solver, "current_epoch", None)
            if current_epoch is None:
                return loss_val
            
            print(f"\n--- [Callback] Mid-Training Checkpoint Logic for Epoch {current_epoch} ---")
            
            # 1. Save checkpoint for current epoch
            epoch_result_fname = f"RISINGConvNet_3D_epoch_{current_epoch}.pt"
            current_model_path = self.savefolder / epoch_result_fname
            
            self.model.save(
                current_model_path,
                geometry=self.experiment.geometry,
                dataset=None,
                training=self.train_param,
            )
            print(f"Epoch {current_epoch} model saved to {current_model_path}")
            
            # 2. Cleanup preceding epoch checkpoint to conserve disk space
            if self.previous_model_path and self.previous_model_path.exists() and self.previous_model_path != current_model_path:
                try:
                    self.previous_model_path.unlink()
                    print(f"Deleted previous model: {self.previous_model_path.name}")
                except Exception as e:
                    print(f"Could not delete previous model file: {e}")
                    
            self.previous_model_path = current_model_path

            # 3. Append metric logs and regenerate training curves
            try:
                current_val_loss = loss_val.item()
                train_losses = [x for x in self.solver.train_loss if x is not None and x > 0]
                true_epoch_idx = len(train_losses)
                
                # Append validation loss and write to persistent file
                if len(self.persistent_val_history) < true_epoch_idx:
                    self.persistent_val_history.append(current_val_loss)
                    with open(self.history_file, "a") as f:
                        f.write(f"{current_val_loss}\n")

                # Generate loss plot
                if true_epoch_idx > 0:
                    plt.figure(figsize=(10, 6))
                    
                    train_epochs = list(range(1, true_epoch_idx + 1))
                    plt.plot(train_epochs, train_losses, "b-", label="Training Loss", linewidth=2)

                    if len(self.persistent_val_history) > 0:
                        val_epochs = list(range(true_epoch_idx - len(self.persistent_val_history) + 1, true_epoch_idx + 1))
                        
                        if len(val_epochs) != len(self.persistent_val_history) or val_epochs[0] < 1:
                            val_epochs = list(range(1, len(self.persistent_val_history) + 1))

                        plt.plot(val_epochs, self.persistent_val_history, "r-", label="Validation Loss", linewidth=2)

                    plt.xlabel("Epoch", fontsize=12)
                    plt.ylabel("Loss", fontsize=12)
                    plt.title(f"RISINGConvNet 3D Training Curves (Epoch {current_epoch})", fontsize=14, fontweight="bold")
                    plt.legend(fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

                    plot_path = pathlib.Path.cwd() / "training_curves_HUMUCorrected.png"
                    pdf_path = pathlib.Path.cwd() / "training_curves_HUMUCorrected.pdf"

                    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
                    plt.close('all')
                    
                    print(f"Updated training curves perfectly aligned to epoch {true_epoch_idx}")
            except Exception as e:
                print(f"Could not update curves at epoch {current_epoch}: {e}")
                
        return loss_val


# =====================================================================
# MAIN EXECUTION & PIPELINE CONFIGURATION
# =====================================================================

if __name__ == "__main__":
    # User Controls
    # Set number of training patches, and patch size based on GPU memory constraints.
    gpu_id = 0
    num_train_patch = 8
    patch_size = 128

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print("Selected physical GPU:", gpu_id)
    print("PyTorch CUDA device:", torch.cuda.current_device(), torch.cuda.get_device_name(0))
    print("Number of training patches per volume:", num_train_patch)
    print("Patch size:", patch_size)

    # Output directory setup
    # Adjust the path below to your desired save location
    savefolder = pathlib.Path("/store/cia-lion/smrsi2/trained_models/test_debbuging_RISINGConvNet/")
    savefolder.mkdir(parents=True, exist_ok=True)

    final_result_fname = "RISINGConvNet_3D.pt"
    checkpoint_fname = "RISINGConvNet_check_3D.pt"
    validation_fname = "RISINGConvNet_min_val_3D.pt"

    # CT Experiment & Geometry Setup
    experiment = ct_experiments.LimitedAngleCTRecon()
    experiment.geometry = ctgeo.Geometry.parallel_default_parameters()

    DEBUG_MODE = False
    max_samples_debug = 10

    # Data Loaders
    # Adjust the paths below to point to your actual dataset locations
    lidc_dataloader, lidc_validation, lidc_test = create_3d_dataloaders(
        data_path="/store/cia-lion/smrsi2/TIGRE-LIDC-IDRI_data_recon5",
        target_path="/store/cia-lion/smrsi2/TIGRE-LIDC-IDRI_target_recon50",
        batch_size=1,
        train_val_test_ratio=(0.70, 0.15, 0.15),
        file_pattern="*.npy",
        random_seed=42,
        max_samples=max_samples_debug if DEBUG_MODE else None,
    )

    # Model & Parameter Initialization
    default_parameters = RISINGConvNet.default_parameters()
    model = RISINGConvNet(default_parameters, experiment.geometry)
    model = model.to(device)
    model.cite()
    model.cite("bib")

    train_param = LIONParameter()
    loss_fcn = torch.nn.MSELoss() 
    train_param.optimiser = "adam"
    train_param.epochs = 30
    train_param.learning_rate = 1e-4
    train_param.betas = (0.9, 0.99)
    train_param.loss = "MSELoss"

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=train_param.learning_rate,
        betas=train_param.betas
    )

    # Callback & Solver Setup
    validation_callback_loss = EpochCallbackLoss(
        base_loss_fn=loss_fcn,
        model=model,
        experiment=experiment,
        train_param=train_param,
        savefolder=savefolder
    )

    solver = RISINGSolver(
        model=model,
        optimizer=optimiser,
        loss_fn=loss_fcn,
        geometry=experiment.geometry,
        verbose=True,
        device=device,
        save_folder=savefolder,
        num_train_patch=num_train_patch,
        patch_size=patch_size,
    )

    # Link solver instance into callback
    validation_callback_loss.solver = solver

    solver.set_training(lidc_dataloader, loss_fcn)
    solver.set_validation(lidc_validation, validation_freq=1, validation_fn=validation_callback_loss)
    solver.set_testing(lidc_test, testing_fn=loss_fcn)

    solver.set_saving(savefolder, final_result_fname=final_result_fname)
    solver.set_checkpointing(checkpoint_fname=checkpoint_fname, checkpoint_freq=1)

    # Run Training Pipeline
    print(f"Starting training for {train_param.epochs} epochs...")
    solver.train(train_param.epochs)

    # Final Model Saving
    model.save(
        savefolder / final_result_fname,
        geometry=experiment.geometry,
        dataset=None,
        training=train_param,
    )
    print(f"Final model saved to {savefolder / final_result_fname}")

    # Load Best Validation Weights & Evaluate
    try:
        model.load_state_dict(torch.load(savefolder / validation_fname, map_location=device))
        print("Loaded best validation model")
    except FileNotFoundError:
        print("Validation model not found, using final model")

    test_loss = solver.test()
    print(f"Test loss mean: {np.mean(test_loss):.6f}" if len(test_loss) > 0 else "Test loss unavailable")
    print("Training completed successfully!")