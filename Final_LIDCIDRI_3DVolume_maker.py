"""
LIDC-IDRI DICOM Preprocessing Pipeline.

Loads 2D DICOM slice series for patients defined in a CSV, resizes slices to 512x512,
pads the Z-axis to reach a target physical length (400mm), linearly interpolates
the volume to a uniform slice count (512), and saves the resulting 3D volume as .npy files.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
from scipy.ndimage import zoom

# --- Configuration & Constants --- (Please adjust paths as needed)

CSV_PATH = "slice_thicknesses_all_patients.csv" 
ROOT_DIR = "/store/LION/datasets/raw/LIDC-IDRI/LIDC-IDRI/"
OUTPUT_DIR = "/store/LION/smrsi2/LIDC-IDRI_3D/"


TARGET_Z_AXIS_MM = 400.0
TARGET_Z_SLICES = 512
PADDING_VALUE = -1024.0  # CT Air HU equivalent
XY_SIZE = 512
DEFAULT_SLICE_THICKNESS = 2.5


def find_dicom_sequence(patient_path: str, min_files: int = 50) -> Optional[Tuple[str, List[str]]]:
    """Search patient directory tree for a folder containing at least `min_files` DICOM files."""
    for dirpath, _, filenames in os.walk(patient_path):
        dcm_files = [f for f in filenames if f.lower().endswith('.dcm')]
        if len(dcm_files) > min_files:
            return dirpath, sorted(dcm_files)
    return None


def read_and_resize_slices(
    seq_dir: str, 
    dcm_files: List[str]
) -> Tuple[List[np.ndarray], List[float]]:
    """Read DICOM files, resize spatial axes to XY_SIZE x XY_SIZE, and extract slice thicknesses."""
    slices_data = []
    slice_thicknesses = []

    for fname in dcm_files:
        fp = os.path.join(seq_dir, fname)
        try:
            ds = pydicom.dcmread(fp)
            pixel_array = ds.pixel_array.astype(np.float32)

            # Resize spatial dimensions if necessary
            if pixel_array.shape != (XY_SIZE, XY_SIZE):
                scale_factor = XY_SIZE / pixel_array.shape[0]
                pixel_array = zoom(pixel_array, (scale_factor, scale_factor), order=1)

            slices_data.append(pixel_array)

            # Extract slice thickness metadata
            st = ds.get('SliceThickness', None)
            slice_thicknesses.append(float(st) if st is not None else np.nan)

        except Exception as err:
            print(f"Could not read {fp}: {err}")
            continue

    return slices_data, slice_thicknesses


def get_patient_thickness(
    df: pd.DataFrame, 
    patient_name: str, 
    dcm_thicknesses: List[float]
) -> float:
    """Determine slice thickness prioritizing CSV values, fallback to DICOM headers, then default."""
    if patient_name in df.columns:
        csv_thicknesses = df[patient_name].iloc[1:].dropna().values
        if len(csv_thicknesses) > 0:
            return float(np.mean(csv_thicknesses))

    valid_dcm_thicknesses = [t for t in dcm_thicknesses if not np.isnan(t)]
    if len(valid_dcm_thicknesses) > 0:
        return float(np.mean(valid_dcm_thicknesses))

    print(f"Warning: Using default slice thickness of {DEFAULT_SLICE_THICKNESS}mm for {patient_name}")
    return DEFAULT_SLICE_THICKNESS


def pad_volume_z(
    volume: np.ndarray, 
    slice_thickness: float, 
    target_mm: float = TARGET_Z_AXIS_MM
) -> np.ndarray:
    """Pad volume equally on top/bottom with air HU (-1024) if total Z-height is below target_mm."""
    current_z_mm = volume.shape[0] * slice_thickness
    if current_z_mm >= target_mm:
        print(f"  No padding needed ({current_z_mm:.1f}mm >= {target_mm}mm)")
        return volume

    needed_mm = target_mm - current_z_mm
    needed_slices = int(np.ceil(needed_mm / slice_thickness))

    pad_top = needed_slices // 2
    pad_bottom = needed_slices - pad_top

    top_array = np.full((pad_top, XY_SIZE, XY_SIZE), PADDING_VALUE, dtype=np.float32)
    bottom_array = np.full((pad_bottom, XY_SIZE, XY_SIZE), PADDING_VALUE, dtype=np.float32)

    padded_volume = np.concatenate([top_array, volume, bottom_array], axis=0)
    print(f"  Padded: +{pad_top} top, +{pad_bottom} bottom -> {padded_volume.shape[0]} slices")
    return padded_volume


def resample_z(volume: np.ndarray, target_slices: int = TARGET_Z_SLICES) -> np.ndarray:
    """Resample 3D volume along Z-axis using 1D linear interpolation."""
    z_orig, y, x = volume.shape
    z_old = np.linspace(0, 1, z_orig)
    z_new = np.linspace(0, 1, target_slices)

    # Vectorized 1D interpolation across the Z axis for speed
    volume_flat = volume.reshape(z_orig, -1)
    interp_flat = np.zeros((target_slices, y * x), dtype=np.float32)

    for i in range(y * x):
        interp_flat[:, i] = np.interp(z_new, z_old, volume_flat[:, i])

    return interp_flat.reshape(target_slices, y, x)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    for patient_name in df.columns:
        patient_path = os.path.join(ROOT_DIR, patient_name)
        if not os.path.isdir(patient_path):
            print(f"Patient directory not found: {patient_path}")
            continue

        # 1. Locate DICOM sequence
        seq_info = find_dicom_sequence(patient_path)
        if not seq_info:
            print(f"No large DICOM sequence found for patient {patient_name}; skipping.")
            continue
        seq_dir, dcm_files = seq_info
        print(f"\nProcessing {patient_name}: {seq_dir} ({len(dcm_files)} DICOMs)")

        # 2. Read and resize DICOM slices
        slices_data, slice_thicknesses = read_and_resize_slices(seq_dir, dcm_files)
        if not slices_data:
            print(f"No valid DICOM slices loaded for {patient_name}; skipping.")
            continue

        # 3. Stack and calculate physical parameters
        avg_thickness = get_patient_thickness(df, patient_name, slice_thicknesses)
        volume = np.stack(slices_data, axis=0).astype(np.float32)

        # 4. Pad Z axis if needed
        volume = pad_volume_z(volume, avg_thickness, TARGET_Z_AXIS_MM)

        # 5. Resample Z-axis to fixed 512 slices
        volume = resample_z(volume, TARGET_Z_SLICES)
        print(f"  Final volume shape: {volume.shape}")

        # 6. Save array
        output_filepath = os.path.join(OUTPUT_DIR, f"{patient_name}.npy")
        np.save(output_filepath, volume)
        print(f"Saved → {output_filepath}")


if __name__ == "__main__":
    main()