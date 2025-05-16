import numpy as np
import cmath
import math

pi = math.pi


def coef(n, k):
    """
    Fourier coefficient defined as a_n(k) in "Reconstructions of Chest Phantoms"
    """
    return ((1j * k) ** n) / math.factorial(n)


def approx_scatter(L, body_radius, d_theta, electrode_area, delta_DN, k: complex):
    # Note that m and n are summation indices that start at 1
    # and go to L/2 - 1. We define m_idx, n_idx for matrix indices.

    a_halfL_kbar = coef(L//2, k.conjugate())
    a_halfL_k = coef(L//2, k)

    second = complex(0, 0)
    for n in range(1, int(L//2)):
        n_idx = n - 1
        a_n_k = coef(n, k)
        second += a_halfL_kbar * a_n_k * (delta_DN[L//2, n_idx] + 1j * delta_DN[L//2, L//2 + n_idx])
    second *= math.sqrt(2)

    third = complex(0, 0)
    for m in range(1, int(L//2)):
        m_idx = m - 1
        a_m_kbar = coef(m, k.conjugate())
        third += a_m_kbar * a_halfL_k * (delta_DN[m_idx, L//2] - 1j * delta_DN[L//2 + m_idx, L//2])
    third *= math.sqrt(2)

    fourth = 2 * a_halfL_kbar * a_halfL_k * delta_DN[L//2, L//2]

    t_exp = complex(0, 0)
    for m in range(1, L//2):
        m_idx = m - 1

        a_m_kbar = coef(m, k.conjugate())

        for n in range(1, L//2):
            n_idx = n - 1
            a_n_k = coef(n, k)
            first_inside = (delta_DN[m_idx, n_idx] +
                            delta_DN[L//2 + m_idx, L//2 + n_idx] +
                            1j * (delta_DN[m_idx, L//2 + n_idx] - delta_DN[L//2 + m_idx, n_idx]))

            t_exp += a_m_kbar * a_n_k * first_inside + second + third + fourth

    t_exp *= (L * body_radius * d_theta) / electrode_area

    return t_exp


if __name__ == "__main__":
    from pyDbar.k_grid import generate_kgrid
    from pyDbar.Mapper import Mapper, generate_base_DN
    from pyDbar.Simulation import Simulation
    from pyeit.mesh.wrapper import PyEITAnomaly_Circle
    import matplotlib.pyplot as plt

    L = 16
    delta_theta = 2 * pi / L
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
    # base_DN = generate_base_DN(L, electrode_area=electrode_area)
    delta_DN = body_DN - base_DN
    print(delta_DN.shape)

    k_grid = generate_kgrid(32, -1, 1)
    t_exp = np.zeros_like(k_grid)
    for m in range(k_grid.shape[0]):
        for n in range(k_grid.shape[1]):
            k = k_grid[m, n]
            if abs(k) <= 1:
                t_exp[m, n] = approx_scatter(L=L,
                                             body_radius=1,
                                             d_theta=delta_theta,
                                             electrode_area=0.1,
                                             k=k,
                                             delta_DN=delta_DN)

    x = np.real(k_grid)
    y = np.imag(k_grid)
    c1 = np.real(t_exp)
    c2 = np.imag(t_exp)

    # Create a mask where t_exp is zero
    mask = np.abs(t_exp) == 0

    # Mask the data (keep 2D structure)
    c1 = np.ma.masked_where(mask, c1)
    c2 = np.ma.masked_where(mask, c2)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    pcm1 = axs[0].pcolormesh(x, y, c1, shading='auto')
    pcm2 = axs[1].pcolormesh(x, y, c2, shading='auto')
    axs[0].set_aspect("equal")
    axs[1].set_aspect("equal")

    axs[0].set_title("Scatter: Real Part")
    axs[1].set_title("Scatter: Imaginary Part")
    # fig.colorbar(pcm1, ax=axs[0])
    # fig.colorbar(pcm2, ax=axs[1])
    fig.tight_layout()
    plt.show()
