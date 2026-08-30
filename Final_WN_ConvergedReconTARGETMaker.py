import os
from pathlib import Path
import numpy as np
import pydicom
from scipy import ndimage
from tqdm import tqdm

# Set target CUDA device before importing TIGRE
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tigre
import tigre.algorithms as algs

# Initialize TIGRE GPU Configuration
gpuids = tigre.utilities.gpu.GpuIds()
gpuids.devices = [0]


# =====================================================================
# DATA PREPROCESSING & TRANSFORMATIONS
# =====================================================================

def pre_flip(projection: np.ndarray) -> np.ndarray:
    """Flips projection data along the first two spatial axes."""
    return np.flip(projection, axis=(0, 1))


def log_transform(projection: np.ndarray) -> np.ndarray:
    """Applies Beer-Lambert log transformation to raw attenuation projections.
    
    Formula: -log((I + 1) / (I_max + 1))
    """
    projection = projection.astype(np.float32)
    max_intensities = projection.max(axis=(1, 2), keepdims=True)
    return -np.log((projection + 1.0) / (max_intensities + 1.0))


def denoise_projections(projection: np.ndarray, method: str = "Gaussian") -> np.ndarray:
    """Applies 2D spatial filtering frame-by-frame across projection series."""
    projection = projection.astype(np.float32)
    filtered = np.empty_like(projection)

    if method == "Median":
        footprint = np.ones((3, 3), dtype=bool)
        for i in range(projection.shape[0]):
            filtered[i] = ndimage.median_filter(projection[i], footprint=footprint)
    else:  # Default: Gaussian
        for i in range(projection.shape[0]):
            filtered[i] = ndimage.gaussian_filter(projection[i], sigma=0.5)

    return filtered


def apply_circular_mask(reconstruction: np.ndarray, radius: int = 175) -> np.ndarray:
    """Applies a 2D cylindrical mask to suppress out-of-FOV reconstruction artifacts."""
    _, H, W = reconstruction.shape

    # Off-center coordinates for circular mask
    cy, cx = H // 2 + 5, W // 2 - 5

    # Generate coordinate grid
    y, x = np.ogrid[:H, :W]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

    return reconstruction * mask


def post_flip(reconstruction: np.ndarray) -> np.ndarray:
    """Rotates reconstructed volume 90 degrees counter-clockwise along spatial axes."""
    return np.rot90(reconstruction, k=-1, axes=(1, 2))


def normalize_3d_image(image: np.ndarray) -> np.ndarray:
    """Clips extreme intensity outliers (1st to 99.9th percentile) and scales values to [0, 1]."""
    p_min, p_max = np.percentile(image, [1.0, 99.9])
    clipped = np.clip(image, p_min, p_max)
    return (clipped - p_min) / (p_max - p_min + 1e-8)


def pad_to_target_shape(image: np.ndarray, target_shape: tuple = (512, 512, 512)) -> np.ndarray:
    """Symmetrically zero-pads a 3D volume to the target shape."""
    pad_width = []
    for cur, tgt in zip(image.shape, target_shape):
        total_pad = max(0, tgt - cur)
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_width.append((pad_before, pad_after))

    return np.pad(image, tuple(pad_width), mode="constant", constant_values=0)


# =====================================================================
# TIGRE GEOMETRY CONFIGURATION
# =====================================================================

def create_base_geometry() -> tigre.geometry:
    """Initializes standard Cone-Beam CT system geometry parameters."""
    geo = tigre.geometry_default(high_resolution=False)
    geo.gpuids = gpuids

    # Volume parameters
    geo.nVoxel = np.array([384, 384, 384], dtype=np.int32)
    geo.dVoxel = np.array([0.65390851619246, 0.7142857142, 0.7142857142], dtype=np.float32)
    geo.sVoxel = geo.nVoxel * geo.dVoxel

    # Source and Detector Distances (mm)
    geo.DSD = np.array([1194.9], dtype=np.float32)
    geo.DSO = np.array([810.0], dtype=np.float32)

    # Offsets and Alignment
    geo.offOrigin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    geo.offDetector = np.array([0.0, 0.0], dtype=np.float32)
    geo.rotDetector = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Auxiliary parameters
    geo.accuracy = 1.0
    geo.mode = "cone"
    geo.COR = 0.0

    return geo


def update_detector_geometry(geo: tigre.geometry, num_angles: int) -> None:
    """Updates detector resolution and pixel dimensions based on dosage scanning protocol."""
    if num_angles == 238:
        # Low Dose Protocol
        geo.nDetector = np.array([396, 512], dtype=np.int32)
        geo.dDetector = np.array([0.74, 0.741], dtype=np.float64)
    elif num_angles == 617:
        # High Dose Protocol
        geo.nDetector = np.array([742, 960], dtype=np.int32)
        geo.dDetector = np.array([0.391, 0.392], dtype=np.float64)
    else:
        raise ValueError(
            f"Invalid projection count ({num_angles}). Expected 238 (Low Dose) or 617 (High Dose)."
        )

    geo.sDetector = geo.nDetector * geo.dDetector


# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================

def main():

    # Define input and output directories as required.
    input_dir = Path("/store/cia-lion/smrsi2/WNHospitalData/2D_Projections/LOW")
    output_dir = Path("/store/cia-lion/smrsi2/WNHospitalData/3D_recon/target_LOW")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch and filter target DICOM files in index range [1, 1045]
    all_files = sorted(input_dir.glob("Projection_2D_LOW_*.dcm"))
    filtered_files = []

    for fp in all_files:
        try:
            file_num = int(fp.stem.split("_")[3])
            if 1 <= file_num <= 1045:
                filtered_files.append(fp)
        except (IndexError, ValueError):
            continue

    print(f"Found {len(filtered_files)} DICOM files for processing.")
    geo_TIGRE = create_base_geometry()

    # Iterative OS-ASD-POCS Target Reconstruction Settings (Full Resolution)
    blocksize = 17
    num_iter = 50
    alpha = 0.0001

    for fp in tqdm(filtered_files, desc="Processing Target Volumes"):
        try:
            # 1. Read DICOM file and metadata
            full_dicom = pydicom.dcmread(fp)
            sinogram = full_dicom.pixel_array
            dicom_metadata = full_dicom

            # 2. Preprocess projections
            sinogram = pre_flip(sinogram)
            sinogram = log_transform(sinogram)
            sinogram = denoise_projections(sinogram, method="Gaussian")

            # 3. Compute projection angles
            angles_deg = dicom_metadata.PositionerPrimaryAngleIncrement
            angles = np.deg2rad(angles_deg) + np.deg2rad(dicom_metadata.PositionerPrimaryAngle)

            # 4. Update detector geometry dimensions
            update_detector_geometry(geo_TIGRE, len(angles))

            # 5. OS-ASD-POCS Full Iterative Reconstruction
            recon = algs.os_asd_pocs(
                sinogram,
                geo_TIGRE,
                angles,
                num_iter,
                alpha=alpha,
                gpuids=gpuids,
                blocksize=blocksize,
            )

            # 6. Post-processing and normalization
            recon = normalize_3d_image(recon)
            recon = apply_circular_mask(recon)
            recon = post_flip(recon)
            recon = normalize_3d_image(recon)
            recon_padded = pad_to_target_shape(recon, target_shape=(512, 512, 512))

            # 7. Save reconstructed target array
            suffix_parts = fp.stem.split("_")[2:]
            new_filename = f"ReconVolume_3D_{'_'.join(suffix_parts)}_target.npy"
            out_path = output_dir / new_filename

            np.save(out_path, recon_padded)
            print(f"Successfully saved: {out_path.name} | Shape: {recon_padded.shape}")

        except Exception as e:
            print(f"Error processing {fp.name}: {e}")
            continue


if __name__ == "__main__":
    main()