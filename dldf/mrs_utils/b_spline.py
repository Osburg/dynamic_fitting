from typing import List

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch_cubic_spline_grids import CubicBSplineGrid1d


class CubicBSpline:

    def __init__(
        self,
        device: torch.device,
        n_splines: int,
        n_points: int,
        add_boundary_knots: bool = False,
        extend_interval_by: List[int] = None,
    ):
        """A class for evaluating cubic B-splines on an equidistant grid with n_splines basis functions and
        evaluated at n_points equidistant points.

        Args:
            device (torch.device): The device on which to perform calculations.
            n_splines (int): The number of basis functions.
            n_points (int): The number of points at which to evaluate the basis functions.
            add_boundary_knots (bool): If set to True, two equidistant knots wiull be outside the evaluated interval.
            extend_interval_by (List[int]): Evaluation along an extended interval is possible if the result still fits.
                into the interval [0, 1]. The number of additional equidistant knots on each side of the interval is
                specified by this input. If None is provided, the interval is extended by [0,0].
        """

        self.device = device
        self.n_splines = n_splines
        self.n_points = n_points

        # temporarily switch to CPU to generate the basis functions
        current_default_device = torch.get_default_device()
        torch.set_default_device("cpu")

        grid = CubicBSplineGrid1d(resolution=n_splines)
        grid.eval()

        self.basis_functions = torch.zeros((n_points, n_splines), requires_grad=False)
        if extend_interval_by is None:
            extend_interval_by = [0, 0]
        self.extended_basis_functions = torch.zeros(
            (n_points + sum(extend_interval_by), n_splines), requires_grad=False
        )

        # handle special cases for the interval where the basis functions are evaluated
        if add_boundary_knots:
            if n_splines <= 2:
                interval = torch.linspace(0, 1, n_points)
            elif n_splines == 3:
                interval = torch.linspace(0.25, 0.75, n_points)
            else:
                interval = torch.linspace(
                    1.0 / (n_splines - 1), (n_splines - 2) / (n_splines - 1), n_points
                )
        else:
            interval = torch.linspace(0, 1, n_points)

        lb = interval[0] - extend_interval_by[0] * (interval[1] - interval[0])
        ub = interval[-1] + extend_interval_by[1] * (interval[1] - interval[0])
        if lb < 0 or ub > 1:
            raise ValueError(
                "The chosen extension of the interval to be evaluated is above the supported range."
            )
        extended_interval = torch.linspace(lb, ub, n_points + sum(extend_interval_by))

        with torch.no_grad():
            for i in range(n_splines):
                data = torch.zeros_like(grid.data)
                data[0, i] = 1.0
                grid.data = data
                self.basis_functions[:, i] = grid(interval)
                self.extended_basis_functions[:, i] = grid(extended_interval)
        self.basis_functions = self.basis_functions.to(device)
        self.extended_basis_functions = self.extended_basis_functions.to(device)
        torch.set_default_device(current_default_device)

    def __call__(
        self, beta: Tensor, evaluate_extended_interval: bool = False
    ) -> Tensor:
        """Evaluate the cubic B-splines at the given coefficients.

        Args:
            beta (Tensor): The coefficients of the cubic B-splines. Expects a tensor of shape (n_splines, ...).
            evaluate_extended_interval (bool): If set to True, the basis functions are evaluated on an extended
                interval.

        Returns:
            Tensor: The evaluated cubic B-splines.
        """
        if evaluate_extended_interval:
            return torch.tensordot(self.extended_basis_functions, beta, dims=([1], [0]))
        else:
            return torch.tensordot(self.basis_functions, beta, dims=([1], [0]))

    def plot_basis(self):
        """Plot the cubic B-splines."""
        plt.plot(self.basis_functions)
        plt.show()
