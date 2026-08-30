import os
import pathlib
import numpy as np
import torch
from tqdm import tqdm

# Set target CUDA device index before importing PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import LION.experiments.ct_experiments as ct_experiments
import LION.CTtools.ct_geometry as ctgeo
from LION.data_loaders.Z_ImageDataset3D import create_3d_dataloaders
from LION.models.post_processing.Z_RISINGConvNet_3D import RISINGConvNet_3D

# Enforce deterministic PyTorch execution settings
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


# =====================================================================
# USER CONTROLS & PATH CONFIGURATIONS
# =====================================================================
gpu_id = 0
batch_size = 1
patch_size = 128
stride = 32                  # Overlap stride (32 = 75% overlap, 64 = 50% overlap)
use_tiled_inference = True

model_path = "/store/cia-lion/smrsi2/trained_models/test_debbuging_RISINGConvNet/RISINGConvNet_3D.pt"
data_path = "/store/cia-lion/smrsi2/TIGRE-LIDC-IDRI_data_recon5"
target_path = "/store/cia-lion/smrsi2/TIGRE-LIDC-IDRI_target_recon50"
output_dir = "/store/cia-lion/smrsi2/trained_models/test_debbuging_RISINGConvNet/test_denoised_outputs"

train_val_test_ratio = (0.70, 0.15, 0.15)
random_seed = 42

DEBUG_MODE = False
max_samples_debug = 5


# =====================================================================
# HELPER FUNCTIONS & INFERENCE UTILITIES
# =====================================================================

def ensure_5d(x: torch.Tensor) -> torch.Tensor:
    """Ensures input tensor is formatted as [B, C, D, H, W].
    
    Supports:
        - [D, H, W]
        - [B, D, H, W]
        - [B, C, D, H, W]
    """
    if x.ndim == 3:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 4:
        x = x.unsqueeze(1)
    elif x.ndim == 5:
        pass
    else:
        raise ValueError(f"Unsupported tensor shape: {x.shape}")
    return x


def make_weight_window(tile_size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Creates a smooth 3D Hann window for seamless overlapping patch blending."""
    w = np.hanning(tile_size)
    w = np.maximum(w, 1e-6)
    w3d = w[:, None, None] * w[None, :, None] * w[None, None, :]
    return torch.from_numpy(w3d).to(dtype=dtype)


def infer_full_volume(model: torch.nn.Module, volume: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Performs direct full-volume inference."""
    with torch.no_grad():
        out = model(volume.to(device))
    return out


def infer_by_tiles_overlap(
    model: torch.nn.Module,
    volume: torch.Tensor,
    device: torch.device,
    tile: int = 128,
    stride: int = 32
) -> torch.Tensor:
    """Executes 3D sliding-window inference with spatial reflection padding and smooth window blending.
    
    Args:
        model: Trained 3D PyTorch model.
        volume: Input 5D Tensor [1, 1, D, H, W].
        device: Target CUDA device.
        tile: Cubic patch dimension.
        stride: Step size for patch sliding.
        
    Returns:
        Denoised output 5D Tensor [1, 1, D, H, W].
    """
    _, _, D, H, W = volume.shape

    # Apply reflection padding to diminish border artifacts
    pad = tile // 4
    volume_padded = torch.nn.functional.pad(
        volume,
        (pad, pad, pad, pad, pad, pad),
        mode="reflect"
    )

    _, _, Dp, Hp, Wp = volume_padded.shape

    # Accumulate patches directly on GPU memory to eliminate CUDA-Host transfer overhead
    output_padded = torch.zeros_like(volume_padded, device=device)
    weight_sum = torch.zeros_like(volume_padded, device=device)

    # Initialize blending kernel on GPU
    weight_patch = make_weight_window(tile, dtype=volume.dtype).unsqueeze(0).unsqueeze(0).to(device)

    # Compute coordinate step ranges
    dz_list = list(range(0, Dp - tile + 1, stride))
    hy_list = list(range(0, Hp - tile + 1, stride))
    wx_list = list(range(0, Wp - tile + 1, stride))

    # Force coverage for remaining edge pixels
    if dz_list[-1] != Dp - tile:
        dz_list.append(Dp - tile)
    if hy_list[-1] != Hp - tile:
        hy_list.append(Hp - tile)
    if wx_list[-1] != Wp - tile:
        wx_list.append(Wp - tile)

    with torch.no_grad():
        for dz in dz_list:
            for hy in hy_list:
                for wx in wx_list:
                    patch = volume_padded[:, :, dz:dz+tile, hy:hy+tile, wx:wx+tile].to(device)
                    patch_out = model(patch)
                    
                    # Accumulate weighted prediction and window weights
                    output_padded[:, :, dz:dz+tile, hy:hy+tile, wx:wx+tile] += patch_out * weight_patch
                    weight_sum[:, :, dz:dz+tile, hy:hy+tile, wx:wx+tile] += weight_patch

    # Normalize by accumulated weight map
    output_padded = output_padded / torch.clamp(weight_sum, min=1e-8)

    # Crop padded boundaries back to original dimensions
    output = output_padded[:, :, pad:pad+D, pad:pad+H, pad:pad+W]
    return output.cpu()


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main():
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)

    print("Selected physical GPU:", os.environ["CUDA_VISIBLE_DEVICES"])
    print("PyTorch CUDA device:", torch.cuda.current_device(), torch.cuda.get_device_name(gpu_id))

    save_path = pathlib.Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Instantiate CT experiment and geometric layout
    experiment = ct_experiments.LimitedAngleCTRecon()
    experiment.geometry = ctgeo.Geometry.parallel_default_parameters()

    # Create dataloaders and isolate test split
    _, _, test_loader = create_3d_dataloaders(
        data_path=data_path,
        target_path=target_path,
        batch_size=batch_size,
        train_val_test_ratio=train_val_test_ratio,
        file_pattern="*.npy",
        random_seed=random_seed,
        max_samples=max_samples_debug if DEBUG_MODE else None,
    )

    # Instantiate and load model weights
    params = RISINGConvNet_3D.default_parameters()
    model = RISINGConvNet_3D(params, experiment.geometry).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Loaded model state dict from: {model_path}")
    print("Starting inference on the test split...")

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_loader, desc="Inference Progress")):
            data = ensure_5d(data)
            target = ensure_5d(target)

            if data.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, but received batch size {data.shape[0]}")

            if use_tiled_inference:
                output = infer_by_tiles_overlap(
                    model=model,
                    volume=data,
                    device=device,
                    tile=patch_size,
                    stride=stride
                )
            else:
                output = infer_full_volume(model, data, device).cpu()

            output_vol = output.squeeze().numpy()
            file_name = test_loader.dataset.get_file_name(i)

            # Persist denoised volume arrays to disk
            np.save(save_path / f"denoised_{file_name}", output_vol)

    print(f"Inference complete! Saved denoised outputs to: {save_path}")


if __name__ == "__main__":
    main()