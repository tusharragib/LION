# Z_RISINGSolver.py

import warnings
import torch
from torch.optim.optimizer import Optimizer
import numpy as np

from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from LION.models.LIONmodel import LIONmodel
from LION.optimizers.LIONsolver import LIONsolver, SolverParams
from tqdm import tqdm


class RISINGSolver(LIONsolver):
    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        loss_fn,
        geometry: Geometry = None,
        verbose: bool = False,
        model_regularization=None,
        device: torch.device = None,
        save_folder: str = None,
        num_train_patch: int = 1,
        patch_size: int = 128,
    ):
        super().__init__(
            model,
            optimizer,
            loss_fn,
            geometry=geometry,
            verbose=verbose,
            device=device,
            solver_params=SolverParams(),
            save_folder=save_folder,
        )
        if verbose:
            print("Supervised solver training on device:", device)

        self.op = make_operator(self.geometry)
        self.num_train_patch = num_train_patch
        self.patch_size = patch_size

    def _ensure_5d(self, x):
        """
        Convert tensor to shape [B, C, D, H, W].
        Supports:
            [D, H, W]
            [B, D, H, W]
            [B, C, D, H, W]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0).unsqueeze(0)   # [1,1,D,H,W]
        elif x.ndim == 4:
            x = x.unsqueeze(1)                # [B,1,D,H,W]
        elif x.ndim == 5:
            pass
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")
        return x

    def _sample_random_patches(self, data, target):
        """
        data, target expected shape: [B, C, D, H, W]
        Returns:
            data_patches, target_patches of shape [N, C, p, p, p]
        """
        data = self._ensure_5d(data)
        target = self._ensure_5d(target)

        assert data.shape == target.shape, f"data and target shape mismatch: {data.shape} vs {target.shape}"

        B, C, D, H, W = data.shape
        p = self.patch_size

        if B != 1:
            raise ValueError(f"Expected batch size 1 from dataloader, got B={B}")

        if D < p or H < p or W < p:
            raise ValueError(f"Patch size {p} is larger than volume shape {(D, H, W)}")

        data_patches = []
        target_patches = []

        for i in range(self.num_train_patch):
            d_s = np.random.randint(0, D - p + 1)
            h_s = np.random.randint(0, H - p + 1)
            w_s = np.random.randint(0, W - p + 1)

            print(f"Random patch {i}: d={d_s}, h={h_s}, w={w_s}")

            data_patch = data[:, :, d_s:d_s+p, h_s:h_s+p, w_s:w_s+p]
            target_patch = target[:, :, d_s:d_s+p, h_s:h_s+p, w_s:w_s+p]

            data_patches.append(data_patch.squeeze(0))     # [C,p,p,p]
            target_patches.append(target_patch.squeeze(0)) # [C,p,p,p]

        data_patches = torch.stack(data_patches, dim=0).contiguous()     # [N,C,p,p,p]
        target_patches = torch.stack(target_patches, dim=0).contiguous() # [N,C,p,p,p]

        return data_patches, target_patches

    def mini_batch_step(self, data, target):
        print(f"DEBUG: data raw shape: {data.shape}")
        print(f"DEBUG: target raw shape: {target.shape}")

        device = next(self.model.parameters()).device

        data = data.to(device)
        target = target.to(device)

        data = self._ensure_5d(data)
        target = self._ensure_5d(target)

        if not torch.isfinite(data).all():
            raise ValueError("Input data contains NaN/Inf")
        if not torch.isfinite(target).all():
            raise ValueError("Target data contains NaN/Inf")

        data_patches, target_patches = self._sample_random_patches(data, target)

        print("patch batch shape:", data_patches.shape)
        print("patch batch dtype:", data_patches.dtype)
        print("patch batch device:", data_patches.device)

        output = self.model(data_patches)

        print("output shape:", output.shape)
        print("target shape:", target_patches.shape)

        loss = self.loss_fn(output, target_patches)

        del data_patches, target_patches, output
        torch.cuda.empty_cache()

        return loss

    def _validation_loop(self):
        self.model.eval()
        total_val_loss = []

        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(self.validation_loader)):
                data = data.to(self.device)
                target = target.to(self.device)

                data = self._ensure_5d(data)
                target = self._ensure_5d(target)

                if not torch.isfinite(data).all():
                    raise ValueError("Validation data contains NaN/Inf")
                if not torch.isfinite(target).all():
                    raise ValueError("Validation target contains NaN/Inf")

                data_patches, target_patches = self._sample_random_patches(data, target)

                outputs = self.model(data_patches)
                loss = self.validation_fn(outputs, target_patches)
                total_val_loss.append(loss.item())

        return np.array(total_val_loss)

    def test(self):
        self.model.eval()
        if self.check_testing_ready() != 0:
            warnings.warn("Solver not setup to test. Please call set_testing.")
            return np.array([])

        test_loss = np.array([])

        print("Running Test Loop...")
        with torch.no_grad():
            for data, target in tqdm(self.test_loader):
                data = data.to(self.device)
                target = target.to(self.device)

                data = self._ensure_5d(data)
                target = self._ensure_5d(target)

                if not torch.isfinite(data).all():
                    raise ValueError("Test data contains NaN/Inf")
                if not torch.isfinite(target).all():
                    raise ValueError("Test target contains NaN/Inf")

                data_patches, target_patches = self._sample_random_patches(data, target)

                output = self.model(data_patches)
                loss = self.testing_fn(output, target_patches)
                test_loss = np.append(test_loss, loss.detach().cpu().numpy())

        if self.verbose and len(test_loss) > 0:
            print(f"Final Test loss: {test_loss.mean()} - Std: {test_loss.std()}")

        return test_loss

    @staticmethod
    def default_parameters() -> SolverParams:
        return SolverParams()