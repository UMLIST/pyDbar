import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from numpy.typing import NDArray
from typing import Tuple
from pyDbar.k_grid import KGrid
from collections.abc import Callable


def approx_scatter(k_grid: KGrid,
                   pointwise_solver: Callable,
                   tol: float = 1e-1,
                   **kwargs) -> NDArray:
    """
    Approximate the scattering transform over specified grid,
    using the specified pointwise_solver(). The tolerance is
    used to define the support of the solution.

    We require that pointwise_solver() takes in k_row and k_col,
    the row and col indices of point k. These are passed in by
    approx_scatter().
    """
    coord = k_grid.grid["coords"]

    t_approx = np.zeros_like(coord)

    for m in range(coord.shape[0]):
        for n in range(coord.shape[1]):
            k = coord[m, n]

            # Only compute over support (disk)
            if abs(k) <= k_grid.r + tol:
                t_approx[m, n] = pointwise_solver(k_row=m, k_col=n, **kwargs)

    return t_approx


def get_coef_vecs(current_mx, k_grid, z):
    """
    Get \vec{c}(k) and \vec{d}(k) tensors, with indices [m, i, j]
    refering to the m^th \vec{c} and \vec{d}, and the [i, j]
    element of the k_grid; i.e., c_m(k) = c_m([i, j]), d_n(k) = d_n([i, j]).

    Derivation of \vec{c}(k) and \vec{d}(k) in "A Real-time D-bar Algorithm"
    """
    N = current_mx.shape[1]
    coords = k_grid.grid["coords"]

    c_vec = np.zeros((N, coords.shape[0], coords.shape[1]), dtype=complex)
    d_vec = np.zeros((N, coords.shape[0], coords.shape[1]), dtype=complex)

    for i in range(coords.shape[0]): # row
        for j in range(coords.shape[1]): # col
            k = coords[i, j]

            c_vec[:, i, j] = current_mx.T @ np.exp(1j * k * z)
            d_vec[:, i, j] = current_mx.T @ np.exp(1j * k.conjugate() * z.conjugate())

    return c_vec, d_vec


def pointwise_approx(N: int,
                     delta_DN: NDArray,
                     c_vec: NDArray,
                     d_vec: NDArray,
                     k_row: int,
                     k_col: int) -> complex:
    """
    Pointwise t^{exp} approximation based on "A Real-time D-bar Algorithm"

    Input:
    N: int            - Number of linearly-independent current patterns
    delta_DN: NDArray - Difference between DN maps
                        (e.g., delta_DN = DN_sigma - DN_ref)
    k_row: int        - Row index of grid point k
    k_col: int        - Col index of grid point k
    """
    t_exp = complex(0, 0)
    for j in range(N):
        for m in range(N):
            t_exp += c_vec[m, k_row, k_col] * d_vec[j, k_row, k_col] * delta_DN[j, m]
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


def plot_scatter(k_grid: KGrid,
                 t_exp: NDArray | None = None,
                 fig_size: Tuple[float, float] = (12, 6),
                 plot_points: bool = False,
                 out_file: str = ""):
    fig, axs = plt.subplots(1, 2, figsize=fig_size)

    # Extract values
    x = np.real(k_grid.grid["coords"])
    y = np.imag(k_grid.grid["coords"])

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
            circle = Circle((0, 0), k_grid.r, fill=False, edgecolor="red")
            axs[i].add_patch(circle)

        axs[i].set_aspect("equal")
        axs[i].grid(linestyle="--", linewidth=0.5, alpha=0.75)
        if plot_points:
            axs[i].scatter(x, y, marker=".", c="black", alpha=0.2, s=4)

    axs[0].set_title(rf"$\mathbf{{t}}^{{\text{{exp}}}}$: Real Part ($r = {k_grid.r}, M = 2^{{{k_grid.m}}} = {k_grid.M}$)")
    axs[1].set_title(rf"$\mathbf{{t}}^{{\text{{exp}}}}$: Imaginary Part ($r = {k_grid.r}, M = 2^{{{k_grid.m}}} = {k_grid.M}$)")
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
    from pyDbar.Mapper import Mapper
    from pyDbar.Simulation import Simulation
    from pyeit.mesh.wrapper import PyEITAnomaly_Circle
    from time import perf_counter

    """
    Offline phase
    """
    start_offline = perf_counter()

    # Set up parameters
    L = 16
    delta_theta = 2 * math.pi / L
    electrode_area = 0.05

    # Set up reference
    ref = Simulation(L=L)
    ref.simulate()

    ref_map = Mapper(ref.current,
                     ref.voltage,
                     electrode_area=electrode_area)

    k_grid = generate_kgrid(r=3, m=5)

    # Use angular coords z_l = exp((2 * pi * l) / L)
    z_bdry = np.exp(1j*np.arange(0, 2 * math.pi, delta_theta))

    # Get \vec{c} and \vec{d} for boundary points z_bdry
    c_vec, d_vec = get_coef_vecs(current_mx=ref_map.j_mx,
                                 k_grid=k_grid,
                                 z=z_bdry)

    end_offline = perf_counter()

    """
    Online phase
    """
    start_online = perf_counter()

    anomaly = [PyEITAnomaly_Circle(center=[0.5, 0.], r=0.2, perm=1.5),
               PyEITAnomaly_Circle(center=[-0.5, 0.], r=0.2, perm=0.5)]

    body = Simulation(L=L, anomaly=anomaly)
    body.simulate()

    body_map = Mapper(body.current,
                      body.voltage,
                      electrode_area=electrode_area)

    delta_DN = body_map.DN - ref_map.DN

    k_grid.grid["texp"] = approx_scatter(k_grid,
                                         pointwise_approx,
                                         N=body_map.j_mx.shape[1],
                                         delta_DN=delta_DN,
                                         c_vec=c_vec,
                                         d_vec=d_vec)

    end_online = perf_counter()

    print(f"Offline time: {end_offline - start_offline}")
    print(f"Online time: {end_online - start_online}")

    plot_scatter(k_grid, k_grid.grid["texp"], plot_points=True)