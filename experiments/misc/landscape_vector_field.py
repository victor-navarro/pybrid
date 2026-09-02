"""A figure showing attractor points in a loss landscape with a 2D vector field."""

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# Grid
N = 100
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)

# Landscapes
np.random.seed(43)
base_noise = np.random.randn(N, N) * 0.05


def make_landscape(base, attractor_points, sigma=12):
    """Return a smoothed landscape with attractor spikes added to *base*."""
    landscape_grid = base.copy()
    for px, py, val in attractor_points:
        ix = np.argmin(np.abs(x - px))
        iy = np.argmin(np.abs(y - py))
        landscape_grid[iy, ix] += val
    return gaussian_filter(landscape_grid, sigma=sigma)


# Landscape 1 – single attractor at (-0.5, 0.5)
Z1 = make_landscape(base_noise, [(-0.5, 0.5, -15)])

# Landscape 2 – old attractor shallower + new deep attractor at (0.5, -0.5)
Z2 = make_landscape(base_noise, [(-0.5, 0.5, -7.5), (0.5, -0.5, -15)])


def gradient_descent(loss_surface, start=(0.0, 0.0), steps=10, lr=0.12):
    """Take *steps* gradient steps on *loss_surface* starting from *start*."""
    positions = [start]
    cx, cy = start
    for _ in range(steps):
        ix = int(np.clip(np.argmin(np.abs(x - cx)), 1, N - 2))
        iy = int(np.clip(np.argmin(np.abs(y - cy)), 1, N - 2))
        gx = (loss_surface[iy, ix + 1] - loss_surface[iy, ix - 1]) / (
            x[ix + 1] - x[ix - 1]
        )
        gy = (loss_surface[iy + 1, ix] - loss_surface[iy - 1, ix]) / (
            y[iy + 1] - y[iy - 1]
        )
        cx = float(np.clip(cx - lr * gx, -1, 1))
        cy = float(np.clip(cy - lr * gy, -1, 1))
        positions.append((cx, cy))
    return positions


START = (0.65, 0.65)
STEPS = 1000
LR = 0.25
traj1 = gradient_descent(Z1, start=START, steps=STEPS, lr=LR)
traj2 = gradient_descent(Z2, start=START, steps=STEPS, lr=LR)


def loss_vector_field(loss_surface):
    """Return the negative gradient field for *loss_surface*."""
    dloss_dy, dloss_dx = np.gradient(loss_surface, y, x)
    return -dloss_dx, -dloss_dy


def draw_attractors(panel_ax, attractors):
    """Draw the attractor markers and their labels."""
    for attx, atty, label in attractors:
        txt = panel_ax.text(
            attx,
            atty,
            label,
            color="white",
            fontsize=24,
            weight="bold",
            ha="center",
            va="center",
            zorder=8,
        )
        txt.set_path_effects(
            [
                path_effects.Stroke(linewidth=3, foreground="black"),
                path_effects.Normal(),
            ]
        )


def z_at(loss_surface, px, py):
    ix = np.argmin(np.abs(x - px))
    iy = np.argmin(np.abs(y - py))
    return loss_surface[iy, ix]


def draw_walk(panel_ax, loss_surface, trajectory_points):
    """Draw the original walk through the landscape on the 2D panel."""
    tx = [p[0] for p in trajectory_points]
    ty = [p[1] for p in trajectory_points]

    panel_ax.plot(tx, ty, color="tomato", linewidth=3.5, zorder=7)
    panel_ax.plot(
        [tx[0]],
        [ty[0]],
        "o",
        color="tomato",
        markersize=10,
        zorder=8,
    )
    panel_ax.plot(
        [tx[-1]],
        [ty[-1]],
        "o",
        color="tomato",
        markersize=10,
        zorder=8,
    )


def plot_panel(
    panel_ax, loss_surface, attractor_specs, trajectory_points, arrow_length
):
    """Plot a loss landscape and its vector field on a single axis."""
    levels = np.linspace(loss_surface.min(), loss_surface.max(), 40)
    surface = panel_ax.contourf(X, Y, loss_surface, levels=levels, cmap="viridis_r")
    panel_ax.contour(
        X,
        Y,
        loss_surface,
        levels=levels[::4],
        colors="white",
        alpha=0.35,
        linewidths=0.6,
        linestyles="solid",
    )

    field_x, field_y = loss_vector_field(loss_surface)
    field_magnitude = np.hypot(field_x, field_y)
    field_magnitude[field_magnitude == 0] = 1.0
    unit_x = field_x / field_magnitude
    unit_y = field_y / field_magnitude
    arrow_lengths = np.maximum(field_magnitude * arrow_length, MIN_ARROW_LENGTH)
    skip = 11
    panel_ax.quiver(
        X[::skip, ::skip],
        Y[::skip, ::skip],
        unit_x[::skip, ::skip] * arrow_lengths[::skip, ::skip],
        unit_y[::skip, ::skip] * arrow_lengths[::skip, ::skip],
        color="white",
        alpha=0.8,
        pivot="mid",
        scale=1,
        scale_units="xy",
        width=0.006,
        headwidth=4.5,
        headlength=4.5,
        headaxislength=4.5,
        zorder=5,
    )

    draw_walk(panel_ax, loss_surface, trajectory_points)
    draw_attractors(panel_ax, attractor_specs)
    panel_ax.set_xlim(-1, 1)
    panel_ax.set_ylim(-1, 1)
    panel_ax.set_aspect("equal", adjustable="box")
    panel_ax.set_xticks([])
    panel_ax.set_yticks([])
    return surface


# Plot
fig, plot_axes = plt.subplots(2, 1, figsize=(4.0, 6.0), constrained_layout=True)

loss_surfaces = [Z1, Z2]
attractor_specs_list = [
    [(-0.5, 0.5, "4"), (0.5, -0.5, "A")],
    [(-0.5, 0.5, "4"), (0.5, -0.5, "A")],
]
trajectories = [traj1, traj2]
VECTOR_LENGTH = 3.0
MIN_ARROW_LENGTH = 0.07

surfaces = []
for panel_ax, loss_surface, attractor_specs, trajectory_points in zip(
    plot_axes, loss_surfaces, attractor_specs_list, trajectories
):
    surfaces.append(
        plot_panel(
            panel_ax, loss_surface, attractor_specs, trajectory_points, VECTOR_LENGTH
        )
    )

plt.savefig("plots/extra/landscape.svg", dpi=350)
plt.close()
