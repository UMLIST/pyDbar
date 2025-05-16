import numpy as np
import cmath
import math

pi = math.pi

# Slightly tweaked version of the old scattering class
class scattering:
    
    def __init__(self, k_grid, mapper, DN_0, scat_type = "exp"):
        self.tK = np.zeros((k_grid.N, k_grid.N), dtype=complex)

        self.k_grid = k_grid
        self.mapper = mapper
        self.DN_0 = DN_0
        self.scat_type = scat_type
        
        self.load_scattering()
        
        
    def load_scattering(self):
        
        if self.scat_type=="partial":
            self.partial_scattering()
        elif self.scat_type=="exp":
            self.exp_scattering()
        else:
            print("Does not exist or we need to implement that version!")
            
            
    def exp_scattering(self):
        
        dt = (2*pi)/self.mapper.L
        zt = np.exp(1j*np.arange(0, 2*pi, dt))
        ind0 = 1e-7
        
        for j in range(self.k_grid.N):
            for jj in range(self.k_grid.N):

                if ind0 < abs(self.k_grid.k[j, jj]) < self.k_grid.R:
                
                    Ez = np.exp(1j*self.k_grid.k[j, jj]*zt)
                    conj_Ez = np.exp(1j * self.k_grid.k[j, jj].conjugate() * np.conjugate(zt))
                
                    ck, residuals_c, rank_c, s_c = np.linalg.lstsq(self.mapper.c_mx, Ez, rcond=None)
                    dk, residuals_d, rank_d, s_d = np.linalg.lstsq(self.mapper.c_mx, conj_Ez, rcond=None)
                
                    for l in range(self.mapper.N):
                        for ll in range(self.mapper.N):
                            self.tK[j, jj] = self.tK[j, jj] + (ck[l]*(self.mapper.DN[ll, l]-self.DN_0[ll, l])*dk[ll])


                    self.tK[j, jj] = self.tK[j, jj]/(4*pi*(self.k_grid.k[j, jj].conjugate()))    
        
    
    
    def partial_scattering(self, Now, Ref, k_grid):
    
        G0 = np.zeros((Now.L, Now.L), dtype=complex)


        dt = (2*pi)/Now.L
        zt = np.exp(1j*np.arange(0, 2*pi, 2*pi/Now.L))

        for l in range(Now.L):
            for ll in range(Now.L):
                if l != ll:
                    G0[l, ll] = -(1/(2*pi))*cmath.log( abs( zt[l] - zt[ll] ) )

        

        dL = Now.DNmap - Ref.DNmap
        Phi = np.matmul(Now.Current.transpose(), Now.Current)
        PhidL = np.matmul(Now.Current, dL)

        ind0 = 1e-7

        M = Phi + np.matmul(Now.Current.transpose(),np.matmul(G0, PhidL))

        for j in range(k_grid.N):
            for jj in range(k_grid.N):
                if( abs(k_grid.k[j ,jj]) < k_grid.R and abs(k_grid.k[j, jj]) > ind0):

                    Ez = np.exp(1j*k_grid.k[j, jj]*zt)

                    psi_b, residuals, rank, s  = np.linalg.lstsq(M, np.matmul(Now.Current.transpose(),Ez), rcond=None)


                    for l in range(Now.L):
                        c = cmath.exp(1j*((k_grid.k[j, jj]*zt[l]).conjugate()))
                        for ll in range(Now.L-1):

                            self.tK[j, jj] = self.tK[j, jj] + c*PhidL[l, ll]*psi_b[ll]

                    self.tK[j, jj] = self.tK[j, jj]/(4*pi*(k_grid.k[j, jj].conjugate()))

    

if __name__ == "__main__":
    from pyDbar.k_grid import k_grid
    from pyDbar.Mapper import Mapper

    kp = k_grid(3.8, 4)

    current1 = np.genfromtxt("./tests/EIT_Data/Object1/Current.txt", delimiter=" ").transpose()
    voltage1 = np.genfromtxt("./tests/EIT_Data/Object1/Voltage.txt", delimiter=" ").transpose()

    current2 = np.genfromtxt("./tests/EIT_Data/Object2/Current.txt", delimiter=" ").transpose()
    voltage2 = np.genfromtxt("./tests/EIT_Data/Object2/Voltage.txt", delimiter=" ").transpose()

    mapper = Mapper(current1, voltage1)
    mapper_ref = Mapper(current2, voltage2)

    tK = scattering(kp, mapper, mapper_ref)