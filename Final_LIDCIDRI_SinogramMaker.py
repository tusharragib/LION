"""
TIGRE 3D CT Sinogram Generation Pipeline.

Loads 3D CT volumes (.npy), converts HU values to linear attenuation coefficients (mu),
normalizes intensity, projects them into sinograms using TIGRE forward projection (Ax),
and saves the result to disk.
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import tigre
from LION.CTtools.ct_utils import from_HU_to_mu
from tqdm import tqdm

# Target GPU setup
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # If you only have one GPU, this can be omitted or just put "0". Adjust as needed.


def setup_tigre_geometry() -> Tuple[tigre.geometry, np.ndarray]:
    """Configure TIGRE CT geometry parameters and projection angles."""
    gpuids = tigre.utilities.gpu.GpuIds()
    gpuids.devices = [0]

    geo = tigre.geometry_default(high_resolution=False)
    geo.gpuids = gpuids

    # Volume & Detector Configuration
    geo.nVoxel = np.array([512, 512, 512])
    geo.dVoxel = np.array([0.75, 0.75, 0.75])
    geo.sVoxel = geo.nVoxel * geo.dVoxel

    geo.nDetector = np.array([512, 512])
    geo.dDetector = np.array([0.88, 0.88])
    geo.sDetector = geo.nDetector * geo.dDetector

    geo.DSD = np.array([4096.0])  # Distance Source-Detector
    geo.DSO = np.array([3500.0])  # Distance Source-Origin

    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    return geo, angles


def preprocess_volume(image: np.ndarray) -> np.ndarray:
    """Threshold, convert HU to mu, and apply percentile min-max normalization."""
    # Threshold background air/sub-zero values
    image = np.maximum(image, 0)

    # Convert HU units to attenuation values (mu)
    image_mu = from_HU_to_mu(image)

    # Robust Min-Max scaling via 1st and 99.9th percentiles
    p_min, p_max = np.percentile(image_mu, [1, 99.9])
    clipped = np.clip(image_mu, p_min, p_max)
    normalized = (clipped - p_min) / (p_max - p_min)

    return normalized.astype(np.float32)


def process_dataset(input_dir: Path, output_dir: Path) -> None:
    """Generate and save sinograms for all 3D numpy volumes in input directory."""
    geo, angles = setup_tigre_geometry()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.npy"))
    if not files:
        print(f"No .npy files found in {input_dir}")
        return

    for fp in tqdm(files, desc="Processing volumes"):
        try:
            # 1. Load & Preprocess Volume
            volume = np.load(fp)
            prep_volume = preprocess_volume(volume)

            # 2. Forward Projection (Sinogram Generation)
            sinogram = tigre.Ax(prep_volume, geo, angles)

            # 3. Save Output
            out_path = output_dir / f"{fp.stem}_sinogram.npy"
            np.save(out_path, sinogram)

        except Exception as e:
            print(f"Error processing {fp.name}: {e}")
            continue


if __name__ == "__main__":
    
    # Adjust these paths as needed. 
    # INPUT_DIR = Path("/store/cia-lion/smrsi2/LIDC-IDRI_3D")
    # OUTPUT_DIR = Path("/store/cia-lion/smrsi2/TIGRE-LIDC-IDRI_3D_sinogram")
    INPUT_DIR = Path("/store/cia-lion/smrsi2/test")
    OUTPUT_DIR = Path("/store/cia-lion/smrsi2/test2")

    process_dataset(INPUT_DIR, OUTPUT_DIR)