"""Fluid properties in SI units."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NewtonianFluid:
    """Incompressible Newtonian fluid.

    Attributes
    ----------
    rho : float
        Density [kg/m^3].
    mu : float
        Dynamic viscosity [Pa*s].
    """

    rho: float
    mu: float

    def __post_init__(self) -> None:
        if self.rho <= 0 or self.mu <= 0:
            raise ValueError("rho and mu must be positive.")

    @property
    def nu(self) -> float:
        """Kinematic viscosity [m^2/s]."""
        return self.mu / self.rho