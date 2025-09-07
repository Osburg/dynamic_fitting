from abc import ABC, abstractmethod
from typing import Tuple, Type, Union

import torch
from torch import Tensor
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs

from dldf.mrs_utils.b_spline import CubicBSpline


class Dynamics(ABC):
    """Base class for dynamics models of a single parameter."""

    kinetic_axis: Tensor
    repetition_axis: Tensor
    _n_parameters: int

    @abstractmethod
    def __call__(self, params: Tensor) -> Tuple[Tensor, Tensor]:
        """Applies the dynamics model to the input data.

        Args:
            params (Tensor): Input data. First dimension is reserved for the batch dimension

        Returns:
            Tensor: Output data.
        """
        pass

    @property
    def n_parameters(self) -> int:
        """Number of parameters of the dynamics model."""
        return self._n_parameters


class Linear(Dynamics):

    def __init__(
        self,
        repetition_axis: Tensor,
        kinetic_axis: Tensor = None,
        apply_softplus=False,
        **kwargs,
    ) -> None:
        """Linear dynamics model.

        Args:
            repetition_axis (Tensor): Axis of the repetition parameter.
            kinetic_axis (Tensor): Axis of the kinetic parameter, i.e. a more fine grained version of the repetition
                axis. If None is provided, a linspace with the same range as the repition axis and 100 points is used.
            **kwargs: Additional keyword arguments.
        """
        self.repetition_axis = repetition_axis
        if kinetic_axis is None:
            self.kinetic_axis = torch.linspace(
                repetition_axis.min(), repetition_axis.max(), 100
            )
        else:
            self.kinetic_axis = kinetic_axis
        self._n_parameters = 2
        self.apply_softplus = apply_softplus

    def __call__(self, params: Tensor) -> Tuple[Tensor, Tensor]:
        if params.shape[-1] != 2:
            raise ValueError("Input data must have length 2 along the last dimension.")
        _params = params.clone()
        if self.apply_softplus:
            _params[..., [1]] = torch.nn.functional.softplus(params[..., [1]])
        return (
            _params[..., [0]] * self.repetition_axis + _params[..., [1]],
            _params[..., [0]] * self.kinetic_axis + _params[..., [1]],
        )


class Sigmoid(Dynamics):
    def __init__(
        self,
        repetition_axis: Tensor,
        kinetic_axis: Tensor = None,
        apply_softplus=False,
        **kwargs,
    ) -> None:
        """Sigmoid dynamics model.

        Args:
            repetition_axis (Tensor): Axis of the repetition parameter.
            kinetic_axis (Tensor): Axis of the kinetic parameter, i.e. a more fine grained version of the repetition
                axis. If None is provided, a linspace with the same range as the repition axis and 100 points is used.
            **kwargs: Additional keyword arguments.
        """
        self.repetition_axis = repetition_axis
        if kinetic_axis is None:
            self.kinetic_axis = torch.linspace(
                repetition_axis.min(), repetition_axis.max(), 100
            )
        else:
            self.kinetic_axis = kinetic_axis
        self._n_parameters = 3
        self.apply_softplus = apply_softplus

    def __call__(self, params: Tensor) -> Tuple[Tensor, Tensor]:
        if params.shape[-1] != 3:
            raise ValueError("Input data must have length 3 along the last dimension.")
        _params = params.clone()
        if self.apply_softplus:
            _params[..., [0, 1]] = torch.nn.functional.softplus(params[..., [0, 1]])
        return (
            _params[..., [0]]
            * torch.sigmoid(
                10.0 * _params[..., [1]] * (self.repetition_axis - _params[..., [2]])
            ),
            _params[..., [0]]
            * torch.sigmoid(
                10.0 * _params[..., [1]] * (self.kinetic_axis - _params[..., [2]])
            ),
        )


class Exponential(Dynamics):
    def __init__(
        self, repetition_axis: Tensor, kinetic_axis: Tensor = None, **kwargs
    ) -> None:
        """Exponential dynamics model.

        Args:
            repetition_axis (Tensor): Axis of the repetition parameter.
            kinetic_axis (Tensor): Axis of the kinetic parameter, i.e. a more fine grained version of the repetition
                axis. If None is provided, a linspace with the same range as the repition axis and 100 points is used.
            **kwargs: Additional keyword arguments.
        """
        self.repetition_axis = repetition_axis
        if kinetic_axis is None:
            self.kinetic_axis = torch.linspace(
                repetition_axis.min(), repetition_axis.max(), 100
            )
        else:
            self.kinetic_axis = kinetic_axis
        self._n_parameters = 2

    def __call__(self, params: Tensor) -> Tuple[Tensor, Tensor]:
        """Applies the dynamics model to the input data.

        Args:
            params (Tensor): Input data. Expects a tensor with a length of 2 along the last axis.
        """
        if params.shape[-1] != 2:
            raise ValueError("Input data must have length 2 along the last dimension.")
        return params[..., [0]] * torch.exp(
            -params[..., [1]] * self.repetition_axis
        ), params[..., [0]] * torch.exp(self.kinetic_axis * params[..., [1]])


class NaturalSpline(Dynamics):
    def __init__(
        self,
        repetition_axis: Tensor,
        kinetic_axis: Tensor = None,
        knot_positions: Tensor = None,
        apply_softplus: bool = False,
        **kwargs,
    ) -> None:
        """Natural Spline dynamics model.

        Args:
            repetition_axis (Tensor): Axis of the repetition parameter.
            kinetic_axis (Tensor): Axis of the kinetic parameter, i.e. a more fine grained version of the repetition
                axis. If None is provided, a linspace with the same range as the repition axis and 100 points is used.
            knot_positions (Tensor): Positions of the knots for the spline along the repetition axis. If None is
                provided, the spline will be interpolated between the minimum and maximum of the repetition axis
                according to the input length.
            apply_softplus (bool): If True, the input data is passed through a softplus function before being used as
                spline coefficients.
            **kwargs: Additional keyword arguments.
        """
        self.repetition_axis = repetition_axis
        if kinetic_axis is None:
            self.kinetic_axis = torch.linspace(
                repetition_axis.min(), repetition_axis.max(), 100
            )
        else:
            self.kinetic_axis = kinetic_axis
        self.knot_positions = knot_positions
        self._n_parameters = len(knot_positions) if knot_positions is not None else None
        self.apply_softplus = apply_softplus

    def __call__(self, params: Tensor) -> Tuple[Tensor, Tensor]:
        """Applies the dynamics model to the input data.

        Args:
            params (Tensor): Input data. Expects a 2D tensor with shape (batch, length).

        Returns:
            Tuple[Tensor, Tensor]: Output data.
        """

        params = params.unsqueeze(-1)
        batch, length, channels = params.shape

        if self.knot_positions is not None:
            if length != len(self.knot_positions):
                raise ValueError(
                    "Input data must have length equal to the number of parameters."
                )
        else:
            self.knot_positions = torch.linspace(
                self.repetition_axis.min(), self.repetition_axis.max(), length
            )
            self._n_parameters = length

        if self.apply_softplus:
            params = torch.nn.functional.softplus(params)
        coefficients = natural_cubic_spline_coeffs(self.knot_positions, params)
        spline = NaturalCubicSpline(coefficients)
        return (
            spline.evaluate(self.repetition_axis)[..., 0],
            spline.evaluate(self.kinetic_axis)[..., 0],
        )


class Constant(Dynamics):
    def __init__(
        self,
        repetition_axis: Tensor,
        kinetic_axis: Tensor = None,
        apply_softplus: bool = False,
        **kwargs,
    ) -> None:
        """Ćonstant dynamics model.

        Args:
            repetition_axis (Tensor): Axis of the repetition parameter.
            kinetic_axis (Tensor): Axis of the kinetic parameter, i.e. a more fine grained version of the repetition
                axis. If None is provided, a linspace with the same range as the repition axis and 100 points is used.
            apply_softplus (bool): If True, the input data is passed through a softplus function.
        """
        self.repetition_axis = repetition_axis
        if kinetic_axis is None:
            self.kinetic_axis = torch.linspace(
                repetition_axis.min(), repetition_axis.max(), 100
            )
        else:
            self.kinetic_axis = kinetic_axis
        self._n_parameters = 1
        self.apply_softplus = apply_softplus

    def __call__(self, params: Tensor) -> Tuple[Tensor, Tensor]:
        if params.shape[-1] != 1:
            raise ValueError("Input data must have length 1 along the last dimension.")
        if self.apply_softplus:
            params = torch.nn.functional.softplus(params)
        return params[..., [0]] * torch.ones_like(self.repetition_axis), params[
            ..., [0]
        ] * torch.ones_like(self.kinetic_axis)


class ExponentialSaturation(Dynamics):
    def __init__(
        self,
        repetition_axis: Tensor,
        kinetic_axis: Tensor = None,
        apply_softplus: bool = False,
        **kwargs,
    ) -> None:
        """Exponential saturation dynamics model.

        Args:
            repetition_axis (Tensor): Axis of the repetition parameter.
            kinetic_axis (Tensor): Axis of the kinetic parameter, i.e. a more fine grained version of the repetition
                axis. If None is provided, a linspace with the same range as the repition axis and 100 points is used.
        """
        self.repetition_axis = repetition_axis
        if kinetic_axis is None:
            self.kinetic_axis = torch.linspace(
                repetition_axis.min(), repetition_axis.max(), 100
            )
        else:
            self.kinetic_axis = kinetic_axis
        self._n_parameters = 2
        self.apply_softplus = apply_softplus

    def __call__(self, params: Tensor) -> Tuple[Tensor, Tensor]:
        if params.shape[-1] != 2:
            raise ValueError("Input data must have length 2 along the last dimension.")
        if self.apply_softplus:
            params = torch.nn.functional.softplus(params)
        return (
            params[..., [0]]
            * (1 - torch.exp(-params[..., [1]] * self.repetition_axis)),
            params[..., [0]] * (1 - torch.exp(-params[..., [1]] * self.kinetic_axis)),
        )


class BSpline(Dynamics):
    def __init__(
        self,
        repetition_axis: Tensor,
        kinetic_axis: Tensor = None,
        knot_positions: Tensor = None,
        apply_softplus: bool = False,
        **kwargs,
    ):
        """Cubic B-Spline dynamics model.
        Info: The BSplines assume equidistant knots and a (sub-)interval of [0,1]. Only the length of the knot positions
        are used


        Args:
            repetition_axis (Tensor): Axis of the repetitions.
            kinetic_axis (Tensor): Axis of the kinetic parameter, i.e. a more fine grained version of the repetition
                axis. If None is provided, a linspace with the same range as the repition axis and 100 points is used.
            knot_positions (Tensor): Positions of the knots for the spline along the repetition axis. If None is
                provided, the spline will be interpolated between the minimum and maximum of the repetition axis
                according to the input length.
            apply_softplus (bool): If True, the input data is passed through a softplus function AFTER(!) the
                calculation of the spline values.
            **kwargs: Additional keyword arguments.
        """
        if kinetic_axis is None:
            self.kinetic_axis = torch.linspace(
                repetition_axis.min(), repetition_axis.max(), 100
            )
        else:
            self.kinetic_axis = kinetic_axis
        self._n_parameters = len(knot_positions)
        self.apply_softplus = apply_softplus

        # scale repetition axis and kinetic axis so that they fit into [0,1]
        maximum = max([torch.max(repetition_axis), torch.max(kinetic_axis)])
        minimum = min([torch.min(repetition_axis), torch.min(kinetic_axis)])
        self.repetition_axis = (repetition_axis - minimum) / (maximum - minimum)
        self.kinetic_axis = (kinetic_axis - minimum) / (maximum - minimum)

        # check if the scaled axes meet the assumptions of this implementation about the positions of evaluation
        assert torch.allclose(
            self.repetition_axis, torch.linspace(0.0, 1.0, len(repetition_axis))
        )
        assert torch.allclose(
            self.kinetic_axis, torch.linspace(0.0, 1.0, len(kinetic_axis))
        )

        self.spline_repetition_axis = CubicBSpline(
            device=self.repetition_axis.device,
            n_splines=self._n_parameters,
            n_points=len(self.repetition_axis),
            add_boundary_knots=False,
        )
        self.spline_kinetic_axis = CubicBSpline(
            device=self.kinetic_axis.device,
            n_splines=self._n_parameters,
            n_points=len(self.kinetic_axis),
            add_boundary_knots=False,
        )

    def __call__(self, params: Tensor) -> Tuple[Tensor, Tensor]:
        """Applies the dynamics model to the input data.

        Args:
            params (Tensor): Input data. Expects a 2D tensor with shape (batch, length).

        Returns:
            Tuple[Tensor, Tensor]: Output data.
        """

        values_repetition_axis = self.spline_repetition_axis(params.T).T
        values_kinetic_axis = self.spline_kinetic_axis(params.T).T

        if self.apply_softplus:
            values_repetition_axis = torch.nn.functional.softplus(
                values_repetition_axis
            )
            values_kinetic_axis = torch.nn.functional.softplus(values_kinetic_axis)

        return values_repetition_axis, values_kinetic_axis


class AgnosticDynamics(Dynamics):
    def __init__(
        self,
        repetition_axis: Tensor,
        kinetic_axis: Tensor = None,
        apply_softplus: bool = False,
        **kwargs,
    ) -> None:
        """Agnostic dynamics model, i.e. the identitiy function or a single softplus function. Equivalent to a natural
        spline with knots exactly at the time points along the repetition axis.

        Args:
            repetition_axis (Tensor): Axis of the repetition parameter.
            kinetic_axis (Tensor): Axis of the kinetic parameter, i.e. a more finegrained version of the repetition
                axis. If None is provided, a linspace with the same range as the repetition axis and 100 points is used.
            apply_softplus (bool): If True, the input data is passed through a softplus function.
        """
        self.repetition_axis = repetition_axis
        if kinetic_axis is None:
            self.kinetic_axis = torch.linspace(
                repetition_axis.min(), repetition_axis.max(), 100
            )
        else:
            self.kinetic_axis = kinetic_axis
        self._n_parameters = len(repetition_axis)
        self.apply_softplus = apply_softplus

    def __call__(self, params: Tensor) -> Tuple[Tensor, Tensor]:
        """
        The kinetic axis is filled using linear interpolation of the values along the repetition axis.
        """

        if params.shape[-1] != self._n_parameters:
            raise ValueError(
                f"Input data must have length {self._n_parameters} along the last dimension."
            )
        if self.apply_softplus:
            params = torch.nn.functional.softplus(params)

        # Do linear interpolation along the kinetic axis
        # Get the indices of the intervals
        indices = torch.searchsorted(
            self.repetition_axis, self.kinetic_axis, right=True
        )
        indices = torch.clamp(indices, 1, len(self.repetition_axis) - 1)

        # Get the time points for the intervals
        t0 = self.repetition_axis[indices - 1]
        t1 = self.repetition_axis[indices]

        # Get the measurements for the intervals
        x0 = params[..., indices - 1]
        x1 = params[..., indices]

        return (
            params,
            x0 + (x1 - x0) * (self.kinetic_axis - t0) / (t1 - t0),
        )


def get_dynamics_class(dynamics_type: str) -> Union[
    Type[Linear],
    Type[NaturalSpline],
    Type[BSpline],
    Type[Exponential],
    Type[Constant],
    Type[ExponentialSaturation],
]:
    """Get the dynamics class based on the string representation.

    Args:
        dynamics_type (str): The dynamics type. Can be "Linear", "Exponential", "Constant" or
            "NaturalSpline", "BSpline" or "AgnosticDynamics" or "Sigmoid".

    Returns:
        class: The dynamics class.
    """
    if dynamics_type == "Linear":
        return Linear
    elif dynamics_type == "Exponential":
        return Exponential
    elif dynamics_type == "Constant":
        return Constant
    elif dynamics_type == "NaturalSpline":
        return NaturalSpline
    elif dynamics_type == "BSpline":
        return BSpline
    elif dynamics_type == "AgnosticDynamics":
        return AgnosticDynamics
    elif dynamics_type == "Sigmoid":
        return Sigmoid
    elif dynamics_type == "ExponentialSaturation":
        return ExponentialSaturation
    else:
        raise ValueError(f"Unknown dynamics type: {dynamics_type}")


def dynamics_wrapper(dynamics: Dynamics):
    """A wrapper to turn a Dynamics object into a function operating on numpy arrays."""

    def dynamics_func(t, *args, **kwargs):
        p = torch.tensor(args, dtype=torch.float32)
        dynamics.repetition_axis = torch.tensor(t, dtype=torch.float32)
        return dynamics(p.unsqueeze(0))[0].squeeze().detach().cpu().numpy()

    return dynamics_func
