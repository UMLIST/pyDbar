import numpy as np
import cmath
import math
from scipy.fft import fft2, ifft2, fftshift

pi = math.pi 


def generate_kgrid(r: float, p: int) -> dict:
    """
    Generate square k-grid in z-space.
    Given a radius of r, the kgrid lies within [-r, r]^2,
    with m = 2^p grid lines. Then the step size is h = (2*r)/(m-1).
    The returned object is a dictionary with keys listed below.

    Input:
    r: float - Radius of support
    p: int   - Value for 2^p, the number of grids generated

    Output dict parameters:
    grid: NDArray      - [m x m] grid points
    step_size: float   - Step size: h = (2*r)/(m-1)
    num_gridlines: int - Number of grid lines: m = 2^p
    radius: float      - Radius of support
    """
    num_gridlines = 2**p
    step_size = (2 * r) / (num_gridlines - 1)

    grid = np.zeros((num_gridlines, num_gridlines), dtype=complex)
    x = np.linspace(-r, r, num=num_gridlines)
    print(x)

    for m in range(num_gridlines):
        for n in range(num_gridlines):
            grid[m, n] = complex(x[m], x[n])

    k_grid = {
        "grid": grid,
        "step_size": step_size,
        "num_gridlines": num_gridlines,
        "radius": r,
        "p": p
    }

    return k_grid


class k_grid:
    
    pos_x = []
    pos_y = []
    
    def __init__(self, R, m):
        self.R = R
        self.m = m
        self.s = 2.3*R
        self.h = 2*(2.3*R)/(2**m)
        self.N = 2**m
        self.index = -1
        self.k = np.zeros((self.N, self.N), dtype=complex)
        self.generate()
        self.FG = self.fund_sol()
        
    def generate(self):
        
        for j in range(self.N):
            for jj in range(self.N):
                
                self.k[j, jj] = complex(-self.s + j*self.h, -self.s + jj*self.h)
        
                if(abs(self.k[j, jj]) < self.R):
                    self.pos_x.append(j)
                    self.pos_y.append(jj)
                
                if(abs(self.k[j, jj]) < 1e-7):
                    self.index = len(self.pos_x)-1
                    
      
    
    def fund_sol(self):
        
        eps = self.s/10
        RR = (self.s-eps)/2
        i0 = 1e-7
        
        G = np.zeros((self.N, self.N), dtype=complex)
        
        for j in range(self.N):
            for jj in range(self.N):
                
                abs_k = abs(self.k[j, jj])
                
                if (abs_k < self.s and abs_k > i0):
                    G[j, jj] = 1/(self.k[j, jj]*pi)
                    
                    if(abs_k >= 2*RR):
                        G[j, jj] = G[j, jj]*(1 - (abs_k - 2*RR)/eps)
        
        return fft2(fftshift(G))
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    k = generate_kgrid(32, -1, 1)

    x = np.real(k)
    y = np.imag(k)

    x = x[x != 0]
    y = y[y != 0]
    plt.plot(x, y, marker="o", linestyle="", c="blue")
    plt.show()