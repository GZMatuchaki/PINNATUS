"""Rectangular geometry and point sampling (SI units).

The geometry describes shape only: dimensions, bounds, outward normals and
point sampling in the domain and on the boundary. It has no knowledge of
boundary conditions; deciding which edge is an inlet, a wall or an outlet is
the responsibility of the problem definition (``problems/``).
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import qmc

EDGES = ("bottom", "right", "top", "left")


class DomainSampling(str, Enum):
    """Sampling strategies for interior (collocation) points."""

    PSEUDO = "pseudo"  # independent uniform random points
    LHS = "lhs"        # Latin hypercube
    SOBOL = "sobol"    # scrambled Sobol sequence; requires n = 2^m
    GRID = "grid"      # cell-centred Cartesian grid


class BoundarySampling(str, Enum):
    """Sampling strategies for boundary points (applied edge by edge)."""

    LHS = "lhs"    # Latin hypercube along each edge
    GRID = "grid"  # cell-centred, uniformly spaced along each edge


@dataclass(frozen=True)
class Rectangle:
    """Axis-aligned rectangle defined by its lower-left corner and its size.

    Attributes
    ----------
    L : float
        Length along x [m].
    H : float
        Height along y [m].
    x_origin, y_origin : float
        Coordinates of the lower-left corner [m]. A rectangle centred at the
        origin is obtained with ``x_origin=-L/2`` and ``y_origin=-H/2``.
    """

    L: float
    H: float
    x_origin: float = 0.0
    y_origin: float = 0.0

    def __post_init__(self) -> None:
        if self.L <= 0 or self.H <= 0:
            raise ValueError("L and H must be positive.")

    # ------------------------------------------------------------------
    # Geometric quantities
    # ------------------------------------------------------------------
    @property
    def area(self) -> float:
        return self.L * self.H

    @property
    def perimeter(self) -> float:
        return 2.0 * (self.L + self.H)

    @property
    def x_min(self) -> float:
        return self.x_origin

    @property
    def x_max(self) -> float:
        return self.x_origin + self.L

    @property
    def y_min(self) -> float:
        return self.y_origin

    @property
    def y_max(self) -> float:
        return self.y_origin + self.H

    @property
    def corners(self) -> np.ndarray:
        """Corner coordinates, shape (4, 2), counter-clockwise from lower-left."""
        return np.array([
            [self.x_min, self.y_min],
            [self.x_max, self.y_min],
            [self.x_max, self.y_max],
            [self.x_min, self.y_max],
        ])

    @property
    def normals(self) -> dict[str, np.ndarray]:
        """Outward unit normal of each edge."""
        return {
            "bottom": np.array([0.0, -1.0]),
            "right": np.array([1.0, 0.0]),
            "top": np.array([0.0, 1.0]),
            "left": np.array([-1.0, 0.0]),
        }

    # ------------------------------------------------------------------
    # Domain sampling
    # ------------------------------------------------------------------
    def sample_domain(
        self,
        n_samples: int,
        seed: int | None = 42,
        method: DomainSampling | str = DomainSampling.LHS,
    ) -> np.ndarray:
        """Sample points strictly inside the rectangle.

        Parameters
        ----------
        n_samples : int
            Number of points.
        seed : int or None
            Seed of the random generator. Ignored by ``GRID``.
        method : DomainSampling or str
            Sampling strategy. Strings are accepted and validated
            (e.g. ``"lhs"``).

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 2) with columns (x, y).
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        method = DomainSampling(method)  # raises ValueError for unknown strings

        if method is DomainSampling.PSEUDO:
            unit = np.random.default_rng(seed).random((n_samples, 2))
        elif method is DomainSampling.LHS:
            unit = qmc.LatinHypercube(d=2, rng=seed).random(n_samples)
        elif method is DomainSampling.SOBOL:
            unit = self._sobol_unit_square(n_samples, seed)
        else:  # DomainSampling.GRID
            return self._domain_grid(n_samples)

        return qmc.scale(unit, [self.x_min, self.y_min], [self.x_max, self.y_max])

    @staticmethod
    def _sobol_unit_square(n_samples: int, seed: int | None) -> np.ndarray:
        """Scrambled Sobol points in [0, 1)^2. Requires n_samples = 2^m."""
        if n_samples & (n_samples - 1) != 0:
            raise ValueError("Sobol sampling requires n_samples = 2^m (e.g. 8192, 16384).")
        m = n_samples.bit_length() - 1
        return qmc.Sobol(d=2, scramble=True, rng=seed).random_base2(m=m)

    def _domain_grid(self, n_samples: int) -> np.ndarray:
        """Cell-centred nx-by-ny grid with nx * ny = n_samples.

        Among all factorisations of n_samples, the one whose cells are closest
        to square (dx ~ dy) is chosen. No point lies on the boundary.
        """
        factor_pairs = [
            (nx, n_samples // nx)
            for nx in range(1, n_samples + 1)
            if n_samples % nx == 0
        ]
        if n_samples > 1:
            factor_pairs = [(nx, ny) for nx, ny in factor_pairs if nx > 1 and ny > 1]
        if not factor_pairs:
            raise ValueError("Grid sampling requires n_samples = 1 or a composite number.")

        def cell_aspect_mismatch(pair: tuple[int, int]) -> float:
            nx, ny = pair
            return abs(np.log((self.L / nx) / (self.H / ny)))

        nx, ny = min(factor_pairs, key=cell_aspect_mismatch)
        x = self.x_min + (np.arange(nx) + 0.5) * self.L / nx
        y = self.y_min + (np.arange(ny) + 0.5) * self.H / ny
        grid_x, grid_y = np.meshgrid(x, y)
        return np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # ------------------------------------------------------------------
    # Boundary sampling
    # ------------------------------------------------------------------
    def boundary_counts(self, n_samples: int) -> dict[str, int]:
        """Split n_samples among the four edges in proportion to their length.

        Uses the largest-remainder (Hamilton) method, with at least one point
        per edge. The returned counts always sum to n_samples.
        """
        if n_samples < 4:
            raise ValueError("At least 4 boundary points are required.")

        ideal = {
            "bottom": n_samples * self.L / self.perimeter,
            "right": n_samples * self.H / self.perimeter,
            "top": n_samples * self.L / self.perimeter,
            "left": n_samples * self.H / self.perimeter,
        }
        counts = {edge: max(1, int(value)) for edge, value in ideal.items()}

        def deficit(edge: str) -> float:
            """Positive when the edge has fewer points than its ideal share."""
            return ideal[edge] - counts[edge]

        # Too few points: give one to the edge furthest below its share.
        while sum(counts.values()) < n_samples:
            counts[max(counts, key=deficit)] += 1

        # Too many points (caused by the one-point minimum): take one from the
        # edge furthest above its share, never dropping an edge below one point.
        while sum(counts.values()) > n_samples:
            removable = [edge for edge in counts if counts[edge] > 1]
            counts[min(removable, key=deficit)] -= 1

        assert sum(counts.values()) == n_samples
        return counts

    def sample_boundary(
        self,
        n_samples: int,
        seed: int | None = 42,
        method: BoundarySampling | str = BoundarySampling.LHS,
    ) -> dict[str, np.ndarray]:
        """Sample points on the four edges.

        Corners are never sampled: the relative position along each edge lies
        in the open interval (0, 1).

        Parameters
        ----------
        n_samples : int
            Total number of boundary points (at least 4).
        seed : int or None
            Seed of the random generator. Ignored by ``GRID``.
        method : BoundarySampling or str
            Sampling strategy along each edge.

        Returns
        -------
        dict[str, np.ndarray]
            ``{edge: points}`` with edges ``"bottom"``, ``"right"``, ``"top"``,
            ``"left"``; each array has shape (n_edge, 2).
        """
        method = BoundarySampling(method)
        counts = self.boundary_counts(n_samples)
        rng = np.random.default_rng(seed)  # shared by all edges: no repeated patterns

        t = {edge: self._unit_interval(n, rng, method) for edge, n in counts.items()}

        return {
            "bottom": np.column_stack((self.x_min + t["bottom"] * self.L,
                                       np.full(counts["bottom"], self.y_min))),
            "right": np.column_stack((np.full(counts["right"], self.x_max),
                                      self.y_min + t["right"] * self.H)),
            "top": np.column_stack((self.x_min + t["top"] * self.L,
                                    np.full(counts["top"], self.y_max))),
            "left": np.column_stack((np.full(counts["left"], self.x_min),
                                     self.y_min + t["left"] * self.H)),
        }

    @staticmethod
    def _unit_interval(n: int, rng: np.random.Generator, method: BoundarySampling) -> np.ndarray:
        """n relative positions in the open interval (0, 1)."""
        if method is BoundarySampling.GRID:
            return (np.arange(n) + 0.5) / n
        return qmc.LatinHypercube(d=1, rng=rng).random(n)[:, 0]