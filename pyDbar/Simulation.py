from math import pi
import numpy as np
import matplotlib.pyplot as plt

import pyeit.mesh as mesh
from pyeit.mesh import quality
from pyeit.eit.utils import eit_scan_lines
from pyeit.eit.fem import Forward

class Simulation:
    def __init__(self, L: int, N: int = 0, anomaly = []):
        self.anomaly = anomaly
        self.L = L

        # By default, use L-1 linearly independent patterns
        if N == 0:
            self.N = L - 1
        else:
            self.N = N

        self.current = np.zeros((self.L, self.N))
        self.voltage = np.zeros_like(self.current)

        # For plotting
        self._mesh_nodes = []
        self._mesh_els = []
        self._mesh_perm = []


    def _adj_pattern(self, M: float):
        """
        Adjacent current patterns
        """
        for k in range(self.N):
            self.current[k, k] = -M
            self.current[k + 1, k] = M

    
    def _trig_pattern(self, M: float):
        """
        Trigonometric current patterns as seen in Isaacson et al.,
        "Reconstructions of Chest Phantoms by the D-bar Method for Electrical Impedance Tomography"
        Everything is offset by 1 (paper starts indexing at 1)

        Input: 
        M: float - Scalar multiplier
        """
        for k in range(self.N):
            for l in range(self.L):
                theta_l = (2 * pi * (l + 1)) / self.L

                if k + 1 <= (self.L/2 - 1):
                    entry =  M * np.cos((k + 1) * theta_l)
                elif k + 1 == self.L/2:
                    entry = M * np.cos(pi * (l + 1))
                else:
                    entry = M * np.sin((k + 1 - (self.L / 2)) * theta_l)

                self.current[l, k] = entry


    def simulate(self, pattern: str = "trig", M: float = 1., h0: float = 0.1):
        # Apply selected current patterns
        if pattern == "adj":
            self._adj_pattern(M)
        if pattern == "trig":
            self._trig_pattern(M)
        else:
            raise NotImplementedError("pattern can only be 'adj' or 'trig' for now")


        """ 0. build mesh """
        # h0 is initial mesh size
        mesh_obj = mesh.create(n_el=self.L, h0=h0)
        mesh_obj = mesh.set_perm(mesh_obj, anomaly=self.anomaly, background=1.0)

        # Extract nodes, elements, permitivity
        self._mesh_nodes = mesh_obj.node
        self._mesh_els = mesh_obj.element
        self._mesh_perm = mesh_obj.perm

        """ 1. FEM forward simulations """
        # setup EIT scan conditions
        ex_mat = eit_scan_lines(self.L, 1)
        ex_mat = ex_mat[0:self.L-1]
        
        for i in range(self.L-1):
            ex_line = ex_mat[i]
            fwd = Forward(mesh_obj)
            f = fwd.solve(ex_line)
            self.voltage[:, i] = np.real(f[mesh_obj.el_pos])


    def plot_mesh(self, out_file: str = ""):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        tpc = ax.tripcolor(self._mesh_nodes[:, 0], # type: ignore
                           self._mesh_nodes[:, 1], # type: ignore
                           self._mesh_els,
                           facecolors=np.real(self._mesh_perm),
                           edgecolors="k",
                           shading="flat",
                           cmap="RdBu",
                           alpha=0.5)
        ax.set_aspect("equal")
        fig.colorbar(tpc)
        fig.tight_layout()

        if out_file:
            fig.savefig(out_file)
            plt.close()
        else:
            plt.show()

        return fig, ax
        

if __name__ == "__main__":
    from pyeit.mesh.wrapper import PyEITAnomaly_Circle

    sim1 = Simulation(6)
    sim1.simulate()
    # sim1.plot_mesh()
    print("sim1 current\n", sim1.current)
    print()
    print("sim1 voltage\n", sim1.voltage)
    print()


    anom = PyEITAnomaly_Circle(center=[0, 0.5], r=0.3, perm=0.4)
    sim2 = Simulation(6, anomaly=anom)
    sim2.simulate(h0=0.08)
    # sim2.plot_mesh()

    print("sim2 current\n", sim2.current)
    print()
    print("sim2 voltage\n", sim2.voltage)
