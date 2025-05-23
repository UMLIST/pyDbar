import numpy as np
import math
from scipy.fft import fft2, ifft2
from scipy.sparse.linalg import LinearOperator, gmres

from numpy.typing import NDArray
from typing import Tuple

# mu = mu - vecrl(h*h * ifft(fft(G_dbar) * fft(T mu)))

def vecrl(f: NDArray) -> NDArray:
    """
    Generate vector vecrl with properties defined in "Demystified"
    """
    flat = f.flatten("F")
    L = flat.shape[0]
    vecrl = np.zeros((2 * L))

    for i in range(L):
        vecrl[i] = flat[i].real
        vecrl[i + L] = flat[i].imag

    return vecrl


def conj_mu(mu: NDArray) -> NDArray:
    """
    Helper function to conjugate mu.

    Note that the bottom half of mu contains the imaginary coefficients,
    so we only need to negate those to obtain the conjugate
    """
    N = mu.shape[0]
    conj_mu = np.copy(mu)
    conj_mu[N//2:] = -conj_mu[N//2:]

    return conj_mu


def get_lin_op(k_grid, t_exp, z):
    def lin_op(mu):
        f = fourier(k_grid, t_exp, mu, z)
        return mu - vecrl(f)
    return lin_op


def kz(k: complex, z: NDArray | Tuple) -> complex:
    return complex(k.real * z[0] - k.imag * z[1], k.real * z[1] + k.imag * z[0])


def kz_conj(k, z) -> complex:
    return kz(k, z).conjugate()


def T_mu(grid: NDArray, t_exp: NDArray, mu: NDArray, z: NDArray | Tuple) -> NDArray:
    """
    Compute T mu_conj
    """
    T_mu = np.zeros_like(grid)
    for j in range(grid.shape[1]):
        for i in range(grid.shape[0]):
            k = grid[i, j]
            e_neg_z = np.exp(-1j * (kz(k, z) + kz_conj(k, z)))

            # Is that the right index for conj_mu?
            T_mu[i, j] = t_exp[i, j] / (4 * math.pi * k.conjugate()) * e_neg_z * conj_mu(mu)[i + j]

    return T_mu


def G_dbar(grid: NDArray) -> NDArray:
    # If element is 0, return 0, otherwise return 1/(pi * k)
    return np.where(grid == 0, 0, 1 / (math.pi * grid))


def fourier(k_grid, t_exp, mu, z):
    """
    Compute h^2 IFFT(FFT(G d_bar) * FFT(T mu_conj)) 
    """
    h = k_grid["h"]
    grid = k_grid["grid"]

    G = G_dbar(grid)
    T = T_mu(grid, t_exp, mu, z)

    return h**2 * ifft2(fft2(G) * fft2(T)) # pyright: ignore[reportOperatorIssue]


def solve(domain):
    # domain contains z points
    pass


if __name__ == "__main__":
    test_grid = np.ones((6, 6), dtype=complex)
    test_t_exp = 1.5 * np.ones((6, 6), dtype=complex)
    print(test_t_exp)

    test_k_grid = {"grid": test_grid, "h": 1}
    test_mu = vecrl(np.ones_like(test_grid))
    test_z = (1, 1)

    f = fourier(test_k_grid, test_t_exp, test_mu, test_z)
    print(f)
    print("f shape", f.shape)

    v_f = vecrl(f)
    print(v_f)
    print("v shape", v_f.shape)