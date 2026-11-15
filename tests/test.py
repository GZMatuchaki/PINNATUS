"""Tests for physics.geometry."""

import numpy as np
import pytest

from physics.geometry import BoundarySampling, DomainSampling, Rectangle


@pytest.fixture
def channel():
    """5 m x 1 m channel centred at the origin."""
    return Rectangle(L=5.0, H=1.0, x_origin=-2.5, y_origin=-0.5)


# ---------------------------------------------------------------------------
# Geometric quantities
# ---------------------------------------------------------------------------
def test_bounds_area_and_perimeter(channel):
    assert (channel.x_min, channel.x_max) == (-2.5, 2.5)
    assert (channel.y_min, channel.y_max) == (-0.5, 0.5)
    assert channel.area == 5.0
    assert channel.perimeter == 12.0


def test_default_origin_is_lower_left_corner():
    rect = Rectangle(L=5.0, H=1.0)
    assert (rect.x_min, rect.x_max, rect.y_min, rect.y_max) == (0.0, 5.0, 0.0, 1.0)


def test_invalid_dimensions_are_rejected():
    with pytest.raises(ValueError):
        Rectangle(L=0.0, H=1.0)


def test_normals_are_unit_and_outward(channel):
    centre = channel.corners.mean(axis=0)
    midpoints = {
        "bottom": [centre[0], channel.y_min],
        "right": [channel.x_max, centre[1]],
        "top": [centre[0], channel.y_max],
        "left": [channel.x_min, centre[1]],
    }
    for edge, normal in channel.normals.items():
        assert np.isclose(np.linalg.norm(normal), 1.0)
        assert np.dot(np.array(midpoints[edge]) - centre, normal) > 0


# ---------------------------------------------------------------------------
# Domain sampling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method, n", [
    (DomainSampling.PSEUDO, 1000),
    (DomainSampling.LHS, 1000),
    (DomainSampling.SOBOL, 1024),
    (DomainSampling.GRID, 1000),
])
def test_domain_points_are_strictly_inside(channel, method, n):
    points = channel.sample_domain(n, method=method)
    assert points.shape == (n, 2)
    assert np.all((points[:, 0] > channel.x_min) & (points[:, 0] < channel.x_max))
    assert np.all((points[:, 1] > channel.y_min) & (points[:, 1] < channel.y_max))


def test_method_accepts_string_and_rejects_typos(channel):
    from_string = channel.sample_domain(100, method="lhs")
    from_enum = channel.sample_domain(100, method=DomainSampling.LHS)
    np.testing.assert_array_equal(from_string, from_enum)
    with pytest.raises(ValueError):
        channel.sample_domain(100, method="lsh")


def test_sobol_requires_power_of_two(channel):
    with pytest.raises(ValueError):
        channel.sample_domain(1000, method=DomainSampling.SOBOL)


def test_domain_sampling_is_reproducible(channel):
    np.testing.assert_array_equal(channel.sample_domain(500), channel.sample_domain(500))


# ---------------------------------------------------------------------------
# Boundary sampling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L, H", [(5.0, 1.0), (1.0, 5.0), (1.0, 1.0), (100.0, 1.0)])
def test_boundary_counts_sum_and_minimum(L, H):
    rect = Rectangle(L=L, H=H)
    for n in list(range(4, 300)) + [4000, 4001, 10007]:
        counts = rect.boundary_counts(n)
        assert sum(counts.values()) == n
        assert min(counts.values()) >= 1


def test_boundary_counts_are_proportional(channel):
    assert channel.boundary_counts(4000) == {"bottom": 1667, "right": 333, "top": 1667, "left": 333}


@pytest.mark.parametrize("method", list(BoundarySampling))
def test_boundary_points_lie_on_their_edge(channel, method):
    boundary = channel.sample_boundary(4000, method=method)
    assert np.allclose(boundary["bottom"][:, 1], channel.y_min)
    assert np.allclose(boundary["top"][:, 1], channel.y_max)
    assert np.allclose(boundary["left"][:, 0], channel.x_min)
    assert np.allclose(boundary["right"][:, 0], channel.x_max)
    assert sum(len(points) for points in boundary.values()) == 4000


@pytest.mark.parametrize("method", list(BoundarySampling))
def test_no_corner_and_no_duplicate_points(channel, method):
    points = np.vstack(list(channel.sample_boundary(4000, method=method).values()))
    assert len(np.unique(points, axis=0)) == len(points)
    corner_distance = np.linalg.norm(points[:, None, :] - channel.corners[None], axis=2)
    assert corner_distance.min() > 0.0


def test_no_corner_points_after_float32_cast(channel):
    """The network trains in float32; corners must stay excluded after the cast."""
    corners32 = channel.corners.astype(np.float32)
    for method in BoundarySampling:
        points = np.vstack(list(channel.sample_boundary(4000, method=method).values()))
        points32 = points.astype(np.float32)
        assert not any((points32 == corner).all(axis=1).any() for corner in corners32)


def test_grid_boundary_spacing_at_corners(channel):
    points = np.vstack(list(channel.sample_boundary(4000, method="grid").values()))
    corner_distance = np.linalg.norm(points[:, None, :] - channel.corners[None], axis=2)
    assert np.isclose(corner_distance.min(), 0.0015, atol=1e-5)


def test_boundary_sampling_is_reproducible(channel):
    first = channel.sample_boundary(4000)
    second = channel.sample_boundary(4000)
    assert all(np.array_equal(first[edge], second[edge]) for edge in first)



"""Plot the training points of the Poiseuille channel."""

from pathlib import Path

from physics.geometry import BoundarySampling, DomainSampling, Rectangle
from utils.plots import plot_points

OUTPUT_DIR = Path("outputs")


def main() -> None:
    channel = Rectangle(L=5.0, H=1.0, x_origin=-2.5, y_origin=-0.5)
    domain = channel.sample_domain(10_000, method=DomainSampling.LHS)
    boundary = channel.sample_boundary(4_000, method=BoundarySampling.GRID)

    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_points(channel, domain, boundary,
                title="Poiseuille channel: training points",
                save_path=OUTPUT_DIR / "training_points.png")


if __name__ == "__main__":
    main()