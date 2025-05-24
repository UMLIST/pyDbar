import numpy as np
import math
from scipy.fft import fft2, ifft2
from scipy.sparse.linalg import LinearOperator
from pyamg.krylov import gmres
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from typing import Tuple, Callable
from pyDbar.k_grid import KGrid

# mu = mu - vecrl(h*h * ifft(fft(G_dbar) * fft(T mu)))

def vecrl(f: NDArray) -> NDArray:
    """
    Generate array vecrl as described in "Demystified".
    """
    flat = f.flatten("F")
    N = len(flat)
    vecrl = np.zeros((N * 2))

    for i in range(N):
        vecrl[i] = flat[i].real
        vecrl[i + N] = flat[i].imag

    return vecrl

def get_G_dbar(k_grid: KGrid) -> NDArray:
    grid = k_grid.grid
    return np.where(grid["coords"] == 0, 0, 1 / (math.pi * grid["coords"]))


def fourier(k_grid: KGrid, mu: NDArray, G_dbar: NDArray, z: NDArray) -> NDArray:
    """
    Compute h^2 IFFT(FFT(G d_bar) * FFT(T mu_conj)) 
    """
    h = k_grid.h
    grid = k_grid.grid

    # Are we doing this only inside the disk??
    T_mu = get_T_mu(grid["coords"], grid["texp"], mu, z)

    return h**2 * ifft2(fft2(G_dbar) * fft2(T_mu)) # pyright: ignore[reportOperatorIssue]


def get_T_mu(coords: NDArray, t_exp: NDArray, mu: NDArray, z) -> NDArray:
    """
    Compute T mu = t^{exp} / (4pi * conj(k)) * e_{-z} * conj(mu),
    where e_{-z} = exp(-i * (kz + conj(kz))). The operations here
    use np's vectorized operations, so they're much faster than
    using nested for loops in Python.
    """
    # Flatten with column-major order. This should match vecrl().
    k = coords.flatten(order="F")
    t_exp_flat = t_exp.flatten(order="F")

    kz = k * z
    k_conj = np.conj(k)
    kz_conj = k_conj * np.conj(z)
    e_neg_z = np.exp(-1j * (kz + kz_conj))

    # Recall mu is [2K,], where real coeffs are in the
    # top half and imag coeffs are in the bottom half
    mu_complex = mu[:len(k)] + 1j * mu[len(k):]
    mu_conj = np.conj(mu_complex)

    T_mu = t_exp_flat / (4 * np.pi * k_conj) * e_neg_z * mu_conj

    # Don't forget to reshape back to [K x K]
    return T_mu.reshape((coords.shape[0], coords.shape[1]), order="F")


def solve_conductivity(domain: NDArray,
                       k_grid: KGrid,
                       domain_r: float = 1.) -> NDArray:
    K = k_grid.grid["coords"].shape[0] # K is the extended number of grid points
    sigmas = np.zeros(domain.shape[0])
    G_dbar = get_G_dbar(k_grid)

    # TODO: See if there's a more efficient way of doing this
    for i, dom_coords in enumerate(domain):
        x = dom_coords[0]
        y = dom_coords[1]
        z = x + 1j * y

        if np.abs(z) <= domain_r:
            def lin_op(mu):
                return mu - vecrl(fourier(k_grid, mu, G_dbar, z)) # pyright: ignore
            
            mu0 = vecrl(np.ones(K * K)) # Results in [K*K*2,] array
            A = LinearOperator((K * K * 2, K * K * 2), matvec=lin_op) # pyright: ignore[reportCallIssue]
            b = np.ones((K * K * 2))

            mu, errorcode = gmres(A, b, x0 = mu0, maxiter=5)

            # TODO: This isn't right (need to get mu at [0,0])
            mu_mx = mu[:(K * K)] + 1j * mu[(K * K):]
            sigmas[i] = np.abs(mu_mx[K//2])**2

    return sigmas


def plot_conductivity(domain, sigmas):
    fig, ax = plt.subplots(figsize=(6, 6))

    x = domain[:, 0]
    y = domain[:, 1]

    sc = ax.scatter(x, y, c=sigmas)
    fig.colorbar(sc)
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    from pyDbar.k_grid import generate_kgrid
    from pyDbar.scattering import approx_scatter, pointwise_approx, get_coef_vecs
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

    k_grid = generate_kgrid(r=3.8, m=4)

    # Use angular coords z_l = exp((2 * pi * l) / L)
    z_bdry = np.exp(1j*np.arange(0, 2 * math.pi, delta_theta))

    # Get \vec{c} and \vec{d} for boundary points z_bdry
    c_vec, d_vec = get_coef_vecs(current_mx=ref_map.j_mx,
                                 k_grid=k_grid,
                                 z=z_bdry)

    end_offline = perf_counter()
    print(f"Offline time: {end_offline - start_offline}")

    """
    Online phase
    """
    start_online = perf_counter()

    anomaly = [PyEITAnomaly_Circle(center=[0.5, 0.], r=0.2, perm=3),
               PyEITAnomaly_Circle(center=[-0.5, 0.], r=0.2, perm=0.1)]

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

    split_online = perf_counter()
    print(f"Online split time: {split_online - start_online}")

    n_px = 160
    x = np.linspace(start=-1, stop=1, num=n_px)
    y = np.linspace(start=-1, stop=1, num=n_px)
    xx, yy = np.meshgrid(x, y)

    domain = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    sigmas = solve_conductivity(domain, k_grid)

    end_online = perf_counter()
    print(f"Online time: {end_online - start_online}")

    plot_conductivity(domain, sigmas)
