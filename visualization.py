import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_quiver_colored(u, v, weights, grid,
                    px_to_m: float | None = None,
                    min_weight: float = 0.0,
                    cmap: str = "viridis",
                    save_path: Optional[str] = None):
    Hc, Wc = u.shape
    X, Y = np.meshgrid(np.arange(Wc)*grid + grid/2.0,
                        np.arange(Hc)*grid + grid/2.0)

    speed_px_s = np.hypot(u, v)
    C = speed_px_s * px_to_m if (px_to_m and px_to_m > 0) else speed_px_s
    units = "m/s" if (px_to_m and px_to_m > 0) else "px/s"

    mask = weights >= float(min_weight)
    Xp, Yp, Up, Vp, Cp = X[mask], Y[mask], u[mask], v[mask], C[mask]

    plt.figure(figsize=(9,7))
    q = plt.quiver(Xp, Yp, Up, -Vp, Cp, angles="xy",
                    scale_units="xy", scale=1.0, cmap=cmap)
    plt.gca().invert_yaxis()
    cbar = plt.colorbar(q); cbar.set_label(f"Mean speed ({units})")
    plt.title("Aggregated Motion Vector Field (color = mean speed)")
    plt.xlabel("x (px)"); plt.ylabel("y (px)")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200)
    plt.show()


def plot_streamplot_colored(u, v, grid,
                            px_to_m: float | None = None,
                            density: float = 1.2,
                            linewidth: float = 1.2,
                            cmap: str = "viridis",
                            save_path: Optional[str] = None):
    Hc, Wc = u.shape
    X, Y = np.meshgrid(np.arange(Wc)*grid + grid/2.0,
                       np.arange(Hc)*grid + grid/2.0)

    speed_px_s = np.hypot(u, v)
    C = speed_px_s * px_to_m if (px_to_m and px_to_m > 0) else speed_px_s
    units = "m/s" if (px_to_m and px_to_m > 0) else "px/s"

    plt.figure(figsize=(9,7))
    sp = plt.streamplot(X, Y, u, -v, color=C, density=density,
                        linewidth=linewidth, cmap=cmap)
    plt.gca().invert_yaxis()
    cbar = plt.colorbar(sp.lines); cbar.set_label(f"Mean speed ({units})")
    plt.title("Dominant Trajectories (streamlines, color = mean speed)")
    plt.xlabel("x (px)"); plt.ylabel("y (px)")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=200)
    plt.show()
