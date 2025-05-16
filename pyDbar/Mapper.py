import numpy as np
from numpy import linalg
from math import isclose

from numpy.typing import NDArray

def generate_base_DN(L: int, N: int = 0, electrode_area:float = 1., body_radius:float = 1.):
    """
    Algorithm based on "Reconstructions of Chest Phantoms"
    """
    if N == 0:
        N = L - 1

    base_DN = np.zeros((N, N))
    for m in range(1, N + 1):
        m_idx = m - 1
        if m <= (L / 2):
            entry = m
        else:
            entry = m - (L / 2)

        base_DN[m_idx, m_idx] = electrode_area / body_radius * entry

    return base_DN


class Mapper:
    """
    This class generates DN and ND maps, contained in one Map object.
    Methods are private.

    Input:
    current_mx: np.array   - With shape L x N, N < L.
                             If data is recorded differently, it must be transformed first.
    voltage_mx: np.array   - With shape L x N, N < L.
                             If data is recorded differently, it must be transformed first.
    electrode_area: float  - Electrode area. Default is 1.
    body_radius: float     - Radius of body being observed. Default is 1.
    ortho: bool            - Whether incoming data is already orthogonal or not.
                             This determines whether we need to run QR decomp to orthogonalize
                             current patterns.

    Attributes:
    ND: np.array           - Neumann-to-Dirichlet map (current-to-voltage)
    DN: np.array           - Dirichlet-to-Neumann map (voltage-to-current)
    """

    def __init__(self,
                 current_mx: NDArray, 
                 voltage_mx: NDArray,
                 electrode_area: float = 1.,
                 radius: float  = 1.,
                 ortho: bool = True):
        # Assert they have the same shape
        assert current_mx.shape == voltage_mx.shape, "Matrices need to be the same shape."

        # Assert N < L
        assert current_mx.shape[1] < current_mx.shape[0], "Current matrix needs to be L x N, N < L"
        assert voltage_mx.shape[1] < voltage_mx.shape[0], "Voltage matrix needs to be L x N, N < L"

        self.j_mx = current_mx # Matrix L x N
        self.v_mx = voltage_mx # Matrix L x N

        self.L, self.N = current_mx.shape

        self.electrode_area = electrode_area
        self.radius = radius
        self.ortho = ortho

        self.DN = np.zeros((self.N, self.N))
        self.ND = np.zeros((self.N, self.N))

        self._compute_ND()
        self._compute_DN()

    
    def _zero_mean_voltages(self):
        """
        Center voltage patterns st. sum(v) == 0 (with tolerance)
        """
        for k in range(self.N):
            v = self.v_mx[:, k]

            if not isclose(np.sum(v), 0):
                self.v_mx[:, k] = v - np.mean(v)

    
    def _compute_ND(self):
        self._zero_mean_voltages()

        if not self.ortho:
            # Use QR decomp to obtain orthonormal current matrix (Q)
            self.c_mx, R = linalg.qr(self.j_mx)

            # Right apply R^-1 to v_mx to obtain u_mx
            # (i.e. same action on both matrices)
            self.v_mx = self.v_mx @ linalg.inv(R)

        else:
            # Iteratively normalize current and voltage vectors
            for k in range(self.N):
                jk_norm = linalg.norm(self.j_mx[:, k], 2)
                self.j_mx[:, k] = self.j_mx[:, k]/jk_norm
                self.v_mx[:, k] = self.v_mx[:, k]/jk_norm

        # Populate ND map (R matrix)
        for m in range(self.N):
            for n in range(self.N):
                t_m = self.j_mx[:, m]
                v_n = self.v_mx[:, n]
                self.ND[m, n] = self.radius / self.electrode_area * np.inner(t_m, v_n)


    def _compute_DN(self):
        self.DN = linalg.inv(self.ND)


if __name__ == "__main__":
    # Simple test
    current = np.genfromtxt("./tests/EIT_Data/Current.txt", delimiter=" ").transpose()
    voltage = np.genfromtxt("./tests/EIT_Data/Voltage_1.txt", delimiter=" ").transpose()

    print(f"Current and voltage data from {current.shape[0]} electrodes, with {current.shape[1]} patterns.")

    mapper = Mapper(current, voltage)
    print("DN is a matrix with shape", mapper.DN.shape)
    print("ND is a matrix with shape", mapper.ND.shape)

    base_DN = generate_base_DN(L=16, N=15)
    print(base_DN)