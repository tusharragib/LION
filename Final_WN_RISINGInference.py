import os
import pathlib
import numpy as np
import torch
from tqdm import tqdm

# Target GPU device allocation
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import LION.CTtools.ct_geometry as ctgeo
import LION.experiments.ct_experiments as ct_experiments
from LION.data_loaders.Z_WN_ImageDataset3D import create_3d_dataloaders
from LION.models.post_processing.Z_RISINGConvNet_3D import RISINGConvNet_3D

# =========================
# USER CONTROLS
# =========================
gpu_id = 0
batch_size = 1
patch_size = 128
stride = 32  # Overlap stride (e.g., 64 = 50% overlap, 32 = 75% overlap)
use_tiled_inference = True

model_path = "/store/cia-lion/smrsi2/trained_models/test_WN_RealRISING_debbuging_RISINGConvNet/RISINGConvNet_3D.pt"
data_path = "/store/cia-lion/smrsi2/WNHospitalData/3D_recon/dataRegistered"
target_path = "/store/cia-lion/smrsi2/WNHospitalData/3D_recon/target"
output_dir = "/store/cia-lion/smrsi2/trained_models/test_WN_RealRISING_debbuging_RISINGConvNet/test_denoised_outputs"

train_val_test_ratio = (0.70, 0.15, 0.15)
random_seed = 42

DEBUG_MODE = False
max_samples_debug = 5
# =========================

# Determinism Controls
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def ensure_5d(x: torch.Tensor) -> torch.Tensor:
    """Standardize input tensor shape to 5D [B, C, D, H, W]."""
    if x.ndim == 3:
        return x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 4:
        return x.unsqueeze(1)
    elif x.ndim == 5:
        return x
    else:
        raise ValueError(f"Unsupported tensor shape: {x.shape}")


def make_weight_window(tile_size: int) -> torch.Tensor:
    """Generates a smooth 3D Hann window for seamless overlapping tile blending."""
    w = np.hanning(tile_size)
    w = np.maximum(w, 1e-6)
    w3d = w[:, None, None] * w[None, :, None] * w[None, None, :]
    return torch.from_numpy(w3d.astype(np.float32))


def infer_full_volume(model: torch.nn.Module, volume: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Direct full-volume pass."""
    with torch.no_grad():
        return model(volume.to(device)).cpu()


def infer_by_tiles_overlap(
    model: torch.nn.Module,
    volume: torch.Tensor,
    device: torch.device,
    tile: int = 128,
    stride: int = 32
) -> torch.Tensor:
    """Runs 3D sliding-window tiled inference with reflection padding and Hann blending."""
    _, _, D, H, W = volume.shape
    pad = tile // 4

    volume_padded = torch.nn.functional.pad(
        volume,
        (pad, pad, pad, pad, pad, pad),
        mode="reflect"
    )

    _, _, Dp, Hp, Wp = volume_padded.shape
    output_padded = torch.zeros_like(volume_padded, device="cpu")
    weight_sum = torch.zeros_like(volume_padded, device="cpu")

    weight_patch_gpu = make_weight_window(tile).unsqueeze(0).unsqueeze(0).to(device)
    weight_patch_cpu = weight_patch_gpu.cpu()

    dz_list = list(range(0, Dp - tile + 1, stride))
    hy_list = list(range(0, Hp - tile + 1, stride))
    wx_list = list(range(0, Wp - tile + 1, stride))

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
                    weighted_patch = patch_out * weight_patch_gpu

                    output_padded[:, :, dz:dz+tile, hy:hy+tile, wx:wx+tile] += weighted_patch.cpu()
                    weight_sum[:, :, dz:dz+tile, hy:hy+tile, wx:wx+tile] += weight_patch_cpu

    output_padded = output_padded / torch.clamp(weight_sum, min=1e-8)
    return output_padded[:, :, pad:pad+D, pad:pad+H, pad:pad+W]


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print("Selected physical GPU:", gpu_id)
    print("PyTorch CUDA device:", torch.cuda.current_device(), torch.cuda.get_device_name(0))

    save_path = pathlib.Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Setup CT Geometry Configuration
    experiment = ct_experiments.LimitedAngleCTRecon()
    experiment.geometry = ctgeo.Geometry.parallel_default_parameters()

    # Create Test DataLoader
    _, _, test_loader = create_3d_dataloaders(
        data_path=data_path,
        target_path=target_path,
        batch_size=batch_size,
        train_val_test_ratio=train_val_test_ratio,
        file_pattern="*.npy",
        random_seed=random_seed,
        max_samples=max_samples_debug if DEBUG_MODE else None,
    )

    # Initialize RISING Model
    params = RISINGConvNet_3D.default_parameters()
    model = RISINGConvNet_3D(params, experiment.geometry).to(device)

    # Load Weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Loaded model from: {model_path}")
    print("Running inference on test set only...")

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_loader, desc="Processing Volumes")):
            data = ensure_5d(data)
            target = ensure_5d(target)

            if data.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got {data.shape[0]}")

            if use_tiled_inference:
                output = infer_by_tiles_overlap(
                    model,
                    data,
                    device,
                    tile=patch_size,
                    stride=stride
                )
            else:
                output = infer_full_volume(model, data, device)

            output_vol = output.squeeze().cpu().numpy()

            # Retrieve sample file name safely across standard datasets and Subsets
            dataset_ref = test_loader.dataset
            if hasattr(dataset_ref, "get_file_name"):
                file_name = dataset_ref.get_file_name(i)
            elif hasattr(dataset_ref, "dataset") and hasattr(dataset_ref.dataset, "get_file_name"):
                real_idx = dataset_ref.indices[i] if hasattr(dataset_ref, "indices") else i
                file_name = dataset_ref.dataset.get_file_name(real_idx)
            else:
                file_name = f"volume_{i:04d}.npy"

            # Save Denoised Volume
            np.save(save_path / f"denoised_{file_name}", output_vol)

    print(f"Inference complete. Saved outputs to: {save_path}")


if __name__ == "__main__":
    main()