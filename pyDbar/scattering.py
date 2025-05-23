import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from numpy.typing import NDArray
from typing import Tuple


def approx_scatter(k_grid: dict, pointwise_solver, tol: float = 1e-1, **kwargs):
    grid = k_grid["grid"]
    t_approx = np.zeros_like(grid)

    for m in range(grid.shape[0]):
        for n in range(grid.shape[1]):
            k = grid[m, n]

            # Only compute over support
            if abs(k) <= k_grid["r"] + tol:
                t_approx[m, n] = pointwise_solver(k=k, **kwargs)

    return t_approx


def pointwise_approx(current_mx: NDArray,
                     delta_theta: float,
                     delta_DN: NDArray,
                     k: complex):
    """
    Pointwise t^{exp} approximation based on "A Real-time D-bar Algorithm"

    Input:
    current_mx: NDArray - L x N matrix of current patterns
                          (L electrodes, N linearly independent patterns)
    delta_theta: float  - angular distance between electrodes
                          typically (2 * pi) / L
    delta_DN: NDArray   - Difference between DN maps
                          (e.g., delta_DN = DN_sigma - DN_ref)
    k: complex          - point in the k-grid. If this function is being called
                          from approx_scatter(), don't pass in k.
    """
    N = current_mx.shape[1]

    # Use angular coords z_l = exp((2 * pi * l) / L)
    z = np.exp(1j*np.arange(0, 2 * math.pi, delta_theta))

    c = current_mx.T @ np.exp(1j * k * z)
    d = current_mx.T @ np.exp(1j * k.conjugate() * z.conjugate())

    t_exp = complex(0, 0)
    for j in range(N):
        for m in range(N):
            t_exp += c[m] * d[j] * delta_DN[j, m]
    return t_exp


def pointwise_approx2(L: int,
                      body_radius: float,
                      d_theta: float,
                      electrode_area: float,
                      delta_DN: NDArray,
                      k: complex):
    """
    t^{exp} approximation based on "Reconstructions of Chest Phantoms"
    """

    def fourier_coef(n, k):
        """
        Compute Fourier coefficient defined as a_n(k) in paper
        """
        return ((1j * k) ** n) / math.factorial(n)

    # Note that m and n are summation indices that start at 1
    # and go to L/2 - 1. We define m_idx, n_idx for matrix indices.

    a_halfL_kbar = fourier_coef(L//2, k.conjugate())
    a_halfL_k = fourier_coef(L//2, k)

    second = complex(0, 0)
    for n in range(1, int(L//2)):
        n_idx = n - 1
        a_n_k = fourier_coef(n, k)
        second += a_halfL_kbar * a_n_k * (delta_DN[L//2, n_idx] + 1j * delta_DN[L//2, L//2 + n_idx])
    second *= math.sqrt(2)

    third = complex(0, 0)
    for m in range(1, int(L//2)):
        m_idx = m - 1
        a_m_kbar = fourier_coef(m, k.conjugate())
        third += a_m_kbar * a_halfL_k * (delta_DN[m_idx, L//2] - 1j * delta_DN[L//2 + m_idx, L//2])
    third *= math.sqrt(2)

    fourth = 2 * a_halfL_kbar * a_halfL_k * delta_DN[L//2, L//2]

    t_exp = complex(0, 0)
    for m in range(1, L//2):
        m_idx = m - 1

        a_m_kbar = fourier_coef(m, k.conjugate())

        for n in range(1, L//2):
            n_idx = n - 1
            a_n_k = fourier_coef(n, k)
            first_inside = (delta_DN[m_idx, n_idx] +
                            delta_DN[L//2 + m_idx, L//2 + n_idx] +
                            1j * (delta_DN[m_idx, L//2 + n_idx] - delta_DN[L//2 + m_idx, n_idx]))

            t_exp += a_m_kbar * a_n_k * first_inside + second + third + fourth

    t_exp *= (L * body_radius * d_theta) / electrode_area

    return t_exp


def plot_scatter(k_grid: dict,
                 t_exp: NDArray | None = None,
                 fig_size: Tuple[float, float] = (12, 6),
                 plot_points: bool = False,
                 out_file: str = ""):
    fig, axs = plt.subplots(1, 2, figsize=fig_size)

    # Extract values
    x = np.real(k_grid["grid"])
    y = np.imag(k_grid["grid"])

    if t_exp is not None:
        c1 = np.real(t_exp)
        c2 = np.imag(t_exp)

        # Create a mask where t_exp is zero
        mask = np.abs(t_exp) == 0

        # Mask the data (keep 2D structure)
        c1 = np.ma.masked_where(mask, c1)
        c2 = np.ma.masked_where(mask, c2)

        pcm1 = axs[0].pcolormesh(x, y, c1, shading='auto')
        pcm2 = axs[1].pcolormesh(x, y, c2, shading='auto')

    for i in range(len(axs)):
        # Draw a circle when not passed in t_exp (potential image for presentations)
        if t_exp is None:
            circle = Circle((0, 0), k_grid["r"], fill=False, edgecolor="red")
            axs[i].add_patch(circle)

        axs[i].set_aspect("equal")
        axs[i].grid(linestyle="--", linewidth=0.5, alpha=0.75)
        if plot_points:
            axs[i].scatter(x, y, marker=".", c="black", alpha=0.25)

    axs[0].set_title(rf"Scatter: Real Part ($r = ${k_grid['r']}, $M = 2^{({k_grid['m']})} = {k_grid["grid"].shape[0]}$)")
    axs[1].set_title(rf"Scatter: Imaginary Part ($r = ${k_grid['r']}, $M = 2^{({k_grid['m']})} = {k_grid["grid"].shape[0]}$)")
    # fig.colorbar(pcm1, ax=axs[0])
    # fig.colorbar(pcm2, ax=axs[1])
    fig.tight_layout()

    if out_file:
        fig.savefig(out_file)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    from pyDbar.k_grid import generate_kgrid
    from pyDbar.Mapper import Mapper, generate_base_DN
    from pyDbar.Simulation import Simulation
    from pyeit.mesh.wrapper import PyEITAnomaly_Circle

    L = 16
    delta_theta = 2 * math.pi / L
    electrode_area = 0.05

    anomaly = [PyEITAnomaly_Circle(center=[0.5, 0.], r=0.2, perm=1.5),
               PyEITAnomaly_Circle(center=[-0.5, 0.], r=0.2, perm=0.5)]

    body = Simulation(L=L, anomaly=anomaly)
    base = Simulation(L=L)
    body.simulate()
    base.simulate()

    body_map = Mapper(body.current,
                      body.voltage,
                      electrode_area=electrode_area)

    base_map = Mapper(base.current,
                      base.voltage,
                      electrode_area=electrode_area)

    body_DN = body_map.DN
    base_DN = base_map.DN
    delta_DN = body_DN - base_DN

    k_grid = generate_kgrid(r=3, m=5)
    t_exp = approx_scatter(k_grid,
                           pointwise_approx,
                           current_mx=body_map.j_mx,
                           delta_theta=delta_theta,
                           delta_DN=delta_DN)

    plot_scatter(k_grid, t_exp, plot_points=True)