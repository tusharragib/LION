# Taken from the supervised solver.

# numerical imports
import torch
from torch.optim.optimizer import Optimizer
import numpy as np

# Import base class
from LION.CTtools.ct_geometry import Geometry
from LION.CTtools.ct_utils import make_operator
from LION.exceptions.exceptions import LIONSolverException
from LION.models.LIONmodel import LIONmodel
from LION.optimizers.LIONsolver import LIONsolver, SolverParams
from LION.classical_algorithms.fdk import fdk
from LION.models.LIONmodel import ModelInputType

from LION.classical_algorithms.tv_min import tv_min

# standard imports
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
            print("Supervised solver training on device: ", device)
        self.op = make_operator(self.geometry)

    def mini_batch_step(self, sino,target):
        """
        This function isresponsible for performing a single mini-batch step of the optimization.
        returns the loss of the mini-batch
        """
        lam = 0.00001
        RIS_ITERS = 10
        RIS_ITERS_GT = 300
        target = tv_min(sino , self.op, lam=lam, num_iterations=RIS_ITERS_GT)
        # Forward pass
        if self.model.get_input_type() == ModelInputType.IMAGE:
            #data = fdk(sino, self.op)
            data = tv_min(sino , self.op, lam=lam, num_iterations=RIS_ITERS)
            if self.do_normalise:
                data = self.model.normalise.normalise(data)
                target = self.model.normalise.normalise(target)
        else:
            data = sino
        output = self.model(data)
        return self.loss_fn(output, target)

    @staticmethod
    def default_parameters() -> SolverParams:
        return SolverParams()
