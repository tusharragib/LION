# Z_ImageDataset3D.py

# =============================================================================
# This file is part of LION library
# License : BSD-3
#
# Custom 3D Image Dataset Loader
# =============================================================================

from typing import List, Tuple, Optional, Union
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageDataset3D(Dataset):
    """
    A flexible 3D image dataset loader that reads 3D images and supports
    train/validation/test splits.
    """

    def __init__(
        self,
        data_path: Union[str, pathlib.Path],
        split: str = 'train',
        train_val_test_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        file_pattern: str = '*.npy',
        random_seed: int = 42,
        target_path: Optional[Union[str, pathlib.Path]] = None,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        normalize: bool = False,
        max_samples: Optional[int] = None,
    ):
        assert split in ['train', 'validation', 'test'], \
            f"split must be 'train', 'validation', or 'test', got {split}"

        assert len(train_val_test_ratio) == 3, \
            "train_val_test_ratio must have 3 values"

        ratio = np.array(train_val_test_ratio, dtype=np.float64)

        if np.any(ratio < 0):
            raise ValueError("train_val_test_ratio values must be non-negative")

        if ratio.sum() <= 0:
            raise ValueError("train_val_test_ratio sum must be > 0")

        # Normalize automatically, but do NOT hardcode any split here
        ratio = ratio / ratio.sum()

        self.data_path = pathlib.Path(data_path)
        self.target_path = pathlib.Path(target_path) if target_path else None
        self.split = split
        self.train_val_test_ratio = tuple(ratio.tolist())
        self.file_pattern = file_pattern
        self.random_seed = random_seed
        self.transform = transform
        self.target_transform = target_transform
        self.normalize = normalize
        self.max_samples = max_samples

        if not self.data_path.is_dir():
            raise ValueError(f"data_path '{self.data_path}' is not a valid directory")

        if self.target_path and not self.target_path.is_dir():
            raise ValueError(f"target_path '{self.target_path}' is not a valid directory")

        self.all_files = sorted(list(self.data_path.glob(file_pattern)))

        if len(self.all_files) == 0:
            raise ValueError(f"No files found matching pattern '{file_pattern}' in {self.data_path}")

        print(f"Found {len(self.all_files)} images in {self.data_path}")
        print(f"Using split ratio (normalized if needed): {self.train_val_test_ratio}")

        self.indices = self._create_split()

    def _create_split(self) -> List[int]:
        np.random.seed(self.random_seed)
        n_samples = len(self.all_files)

        if self.max_samples is not None and self.max_samples > 0:
            n_samples = min(self.max_samples, n_samples)
            print(f"Using max_samples limit: {n_samples} total samples")

        shuffled_indices = np.random.permutation(len(self.all_files))[:n_samples]

        train_end = int(n_samples * self.train_val_test_ratio[0])
        val_end = train_end + int(n_samples * self.train_val_test_ratio[1])

        if self.split == 'train':
            indices = shuffled_indices[:train_end]
        elif self.split == 'validation':
            indices = shuffled_indices[train_end:val_end]
        else:
            indices = shuffled_indices[val_end:]

        print(f"{self.split.capitalize()} split: {len(indices)} samples")
        return indices.tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def _map_input_to_target_name(self, input_name: str) -> str:
        """
        Maps input filename to target filename.

        Supported mappings:
        1. *_sinogram.npy -> *.npy
        2. *_recon5.npy -> *_recon50.npy

        If no rule matches, fallback to identical filename.
        """
        # if "_sinogram" in input_name:
        #     return input_name.replace("_sinogram", "")
        # if "recon5" in input_name:
        #     return input_name.replace("recon5", "recon50")
        # # if "LOW" in input_name:
        # #     return input_name.replace("LOW", "HIGH")
        # if "_data" in input_name:
        #     return input_name.replace("_data", "_target")
        if "LOW" in input_name and "_data" in input_name:
            return input_name.replace("LOW", "HIGH").replace("_data", "")
        return input_name

    def __getitem__(self, idx: int):
        file_idx = self.indices[idx]
        file_path = self.all_files[file_idx]

        image = self._load_image(file_path)
        image_tensor = torch.from_numpy(image).float()

        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        if self.target_path:
            target_name = self._map_input_to_target_name(file_path.name)
            target_file_path = self.target_path / target_name

            if not target_file_path.exists():
                raise FileNotFoundError(f"Target file not found: {target_file_path}")

            target = self._load_image(target_file_path)
            target_tensor = torch.from_numpy(target).float()

            if target_tensor.ndim == 3:
                target_tensor = target_tensor.unsqueeze(0)

            if self.target_transform:
                target_tensor = self.target_transform(target_tensor)

            # Joint normalization for paired input/target
            if self.normalize:
                joint_min = torch.minimum(image_tensor.min(), target_tensor.min())
                joint_max = torch.maximum(image_tensor.max(), target_tensor.max())
                scale = joint_max - joint_min

                if scale > 0:
                    image_tensor = (image_tensor - joint_min) / (scale + 1e-8)
                    target_tensor = (target_tensor - joint_min) / (scale + 1e-8)

            return image_tensor, target_tensor

        if self.normalize:
            image_min = image_tensor.min()
            image_max = image_tensor.max()
            scale = image_max - image_min
            if scale > 0:
                image_tensor = (image_tensor - image_min) / (scale + 1e-8)

        return image_tensor, image_tensor

    @staticmethod
    def _load_image(file_path: pathlib.Path) -> np.ndarray:
        file_path = pathlib.Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix == '.npy':
            return np.load(file_path)

        elif suffix == '.h5' or suffix == '.hdf5':
            try:
                import h5py
                with h5py.File(file_path, 'r') as f:
                    key = list(f.keys())[0]
                    return np.array(f[key])
            except ImportError:
                raise ImportError("h5py is required to load .h5 files")

        elif suffix == '.gz' or (suffix == '.nii' or file_path.suffixes == ['.nii', '.gz']):
            try:
                import nibabel as nib
                img = nib.load(file_path)
                return np.array(img.dataobj)
            except ImportError:
                raise ImportError("nibabel is required to load .nii/.nii.gz files")

        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def get_file_name(self, idx: int) -> str:
        file_idx = self.indices[idx]
        return self.all_files[file_idx].name


def create_3d_dataloaders(
    data_path: Union[str, pathlib.Path],
    batch_size: int = 1,
    train_val_test_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    file_pattern: str = '*.npy',
    random_seed: int = 42,
    target_path: Optional[Union[str, pathlib.Path]] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    max_samples: Optional[int] = None,
):
    from torch.utils.data import DataLoader

    train_dataset = ImageDataset3D(
        data_path=data_path,
        split='train',
        train_val_test_ratio=train_val_test_ratio,
        file_pattern=file_pattern,
        random_seed=random_seed,
        target_path=target_path,
        normalize=True,
        max_samples=max_samples,
    )

    val_dataset = ImageDataset3D(
        data_path=data_path,
        split='validation',
        train_val_test_ratio=train_val_test_ratio,
        file_pattern=file_pattern,
        random_seed=random_seed,
        target_path=target_path,
        normalize=True,
        max_samples=max_samples,
    )

    test_dataset = ImageDataset3D(
        data_path=data_path,
        split='test',
        train_val_test_ratio=train_val_test_ratio,
        file_pattern=file_pattern,
        random_seed=random_seed,
        target_path=target_path,
        normalize=True,
        max_samples=max_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader