"""
TIGRE CT Image Reconstruction Pipeline (OS-ASD-POCS).

Loads 3D CT sinograms (.npy), runs OS-ASD-POCS converged iterative reconstructions (for 50 iterations) to 
reconstruct 3D volumes, normalizes the intensity, and saves output numpy arrays.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tigre
import tigre.algorithms as algs
from tqdm import tqdm

# Target GPU setup
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # You may adjust this to "0" or any other GPU index as needed.


def setup_tigre_geometry() -> Tuple[tigre.geometry, np.ndarray, tigre.utilities.gpu.GpuIds]:
    """Configure TIGRE CT geometry parameters, projection angles, and GPU settings."""
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
    return geo, angles, gpuids


def normalize_3d_image(image: np.ndarray) -> np.ndarray:
    """Normalize 3D volume using 1st and 99.9th percentile clipping to [0, 1]."""
    p_min, p_max = np.percentile(image, [1, 99.9])
    clipped = np.clip(image, p_min, p_max)
    scaled = (clipped - p_min) / (p_max - p_min)
    return scaled.astype(np.float32)


def get_filtered_sinograms(input_dir: Path, min_id: int = 1, max_id: int = 1012) -> List[Path]:
    """Filter sinogram files matching the LIDC-IDRI range [min_id, max_id]."""
    files = sorted(input_dir.glob("LIDC-IDRI-*_sinogram.npy"))
    filtered_files = []

    for fp in files:
        # Regex to safely extract numeric patient ID
        match = re.search(r"LIDC-IDRI-(\d+)", fp.name)
        if match:
            patient_num = int(match.group(1))
            if min_id <= patient_num <= max_id:
                filtered_files.append(fp)

    return filtered_files


def run_reconstruction_pipeline(
    input_dir: Path, 
    output_dir: Path, 
    num_iter: int = 50, 
    blocksize: int = 17
) -> None:
    """Process sinograms into reconstructed 3D volumes using OS-ASD-POCS."""
    geo, angles, gpuids = setup_tigre_geometry()
    output_dir.mkdir(parents=True, exist_ok=True)

    sinogram_files = get_filtered_sinograms(input_dir, min_id=1, max_id=1012)
    if not sinogram_files:
        print(f"No matching sinograms found in {input_dir}")
        return

    for fp in tqdm(sinogram_files, desc="Reconstructing volumes"):
        try:
            # 1. Load Sinogram
            sinogram = np.load(fp)

            # 2. Run OS-ASD-POCS Reconstruction
            recon = algs.os_asd_pocs(
                sinogram,
                geo,
                angles,
                niter=num_iter,
                alpha=0.0001,
                gpuids=gpuids,
                blocksize=blocksize,
            )

            # 3. Post-process & Normalize
            recon_normalized = normalize_3d_image(recon)

            # 4. Save Reconstructed Volume
            out_filename = fp.stem.replace("_sinogram", "_recon50") + ".npy"
            out_path = output_dir / out_filename
            np.save(out_path, recon_normalized)

        except Exception as err:
            print(f"Error processing {fp.name}: {err}")
            continue


if __name__ == "__main__":

    # Adjust this path to your actual input and output directories
    INPUT_DIR = Path("/store/cia-lion/smrsi2/TIGRE-LIDC-IDRI_3D_sinogram")
    OUTPUT_DIR = Path("/store/cia-lion/smrsi2/TIGRE-LIDC-IDRI_data_recon50")


    run_reconstruction_pipeline(INPUT_DIR, OUTPUT_DIR, num_iter=50)