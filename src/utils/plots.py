"""Plotting utilities for training points."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle as RectanglePatch

# Colour-vision-deficiency-safe pair (validated) and neutral inks.
DOMAIN_COLOR = "#2a78d6"
BOUNDARY_COLOR = "#e34948"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
OUTLINE = "#c3c2b7"
BACKGROUND = "#fcfcfb"


def plot_points(
    geom,
    domain: np.ndarray,
    boundary: dict[str, np.ndarray],
    title: str = "Training points",
    zoom: float = 0.06,
    save_path: str | None = None,
    show: bool = True,
):
    """Plot domain points (blue) and boundary points (red).

    The top panel shows the whole rectangle; the bottom panels zoom into the
    two left-hand corners, where inlet and wall conditions usually meet.

    Parameters
    ----------
    geom
        Any object exposing ``x_min``, ``x_max``, ``y_min`` and ``y_max``.
    domain : np.ndarray
        Domain points, shape (n, 2).
    boundary : dict[str, np.ndarray]
        Boundary points per edge, as returned by ``Rectangle.sample_boundary``.
    title : str
        Figure title.
    zoom : float
        Side length of the corner zoom windows [m].
    save_path : str or None
        If given, the figure is saved to this path.
    show : bool
        Whether to display the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    boundary_all = np.vstack(list(boundary.values()))
    length = geom.x_max - geom.x_min
    height = geom.y_max - geom.y_min

    fig = plt.figure(figsize=(12, 6.2), facecolor=BACKGROUND)
    layout = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.6, wspace=0.18)
    ax_main = fig.add_subplot(layout[0, :])
    ax_lower = fig.add_subplot(layout[1, 0])
    ax_upper = fig.add_subplot(layout[1, 1])

    # Full rectangle
    _draw_panel(ax_main, geom, domain, boundary_all, domain_size=0.6, boundary_size=1.2)
    margin = 0.04 * height
    ax_main.set_xlim(geom.x_min - margin, geom.x_max + margin)
    ax_main.set_ylim(geom.y_min - margin, geom.y_max + margin)
    ax_main.set_aspect("equal")
    ax_main.set_title(title, loc="left", color=INK, fontsize=13, pad=30)

    edge_counts = " · ".join(f"{edge} {len(pts)}" for edge, pts in boundary.items())
    ax_main.text(0.0, 1.04, f"{length:g} m × {height:g} m   |   boundary points per edge: {edge_counts}",
                 transform=ax_main.transAxes, color=INK_SECONDARY, fontsize=9.5, va="bottom")

    ax_main.scatter([], [], s=30, color=DOMAIN_COLOR, label=f"domain ({len(domain)})")
    ax_main.scatter([], [], s=30, color=BOUNDARY_COLOR, label=f"boundary ({len(boundary_all)})")
    ax_main.legend(loc="lower right", bbox_to_anchor=(1.0, 1.13), ncol=2, frameon=False,
                   fontsize=10, labelcolor=INK_SECONDARY, handletextpad=0.2,
                   columnspacing=1.4, borderaxespad=0)

    # Corner zooms
    zoom_panels = {
        ax_lower: (geom.x_min, geom.y_min, "bottom-left corner"),
        ax_upper: (geom.x_min, geom.y_max, "top-left corner"),
    }
    for ax, (x_corner, y_corner, name) in zoom_panels.items():
        _draw_panel(ax, geom, domain, boundary_all, domain_size=9, boundary_size=16)

        x0 = x_corner - 0.15 * zoom
        y0 = y_corner - 0.15 * zoom if y_corner == geom.y_min else y_corner - 0.85 * zoom
        ax.set_xlim(x0, x0 + zoom)
        ax.set_ylim(y0, y0 + zoom)
        ax.set_aspect("equal")

        nearest = np.hypot(boundary_all[:, 0] - x_corner, boundary_all[:, 1] - y_corner).min()
        ax.set_title(f"zoom: {name}\nnearest boundary point to the corner: {nearest * 1e3:.2f} mm",
                     loc="left", color=INK_SECONDARY, fontsize=9.5, linespacing=1.5)
        ax.plot(x_corner, y_corner, marker="x", ms=7, mew=1.4, color=INK, zorder=4)

        # Zoom window outline on the main panel
        ax_main.add_patch(RectanglePatch((x0, y0), zoom, zoom, fill=False,
                                         ec=INK_SECONDARY, lw=0.9, zorder=5))

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BACKGROUND)
    if show:
        plt.show()
    return fig


def _draw_panel(ax, geom, domain, boundary_all, domain_size, boundary_size):
    """Draw the rectangle outline, domain points and boundary points on one axis."""
    ax.set_facecolor(BACKGROUND)
    ax.add_patch(RectanglePatch((geom.x_min, geom.y_min), geom.x_max - geom.x_min,
                                geom.y_max - geom.y_min, fill=False,
                                ec=OUTLINE, lw=1.0, zorder=1))
    ax.scatter(domain[:, 0], domain[:, 1], s=domain_size, color=DOMAIN_COLOR,
               linewidths=0, alpha=0.75, zorder=2, rasterized=True)
    ax.scatter(boundary_all[:, 0], boundary_all[:, 1], s=boundary_size, color=BOUNDARY_COLOR,
               linewidths=0, zorder=3, rasterized=True)
    ax.set_xlabel("x [m]", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("y [m]", color=INK_MUTED, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)
    for spine in ax.spines.values():
        spine.set_visible(False)