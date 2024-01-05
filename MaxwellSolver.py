import cupy as cp

import Constants
import ParticleDestributionValues
"""-------------------constants----------------------------------------------------------------------"""
pi = cp.pi

N = Constants.N # Number of particles
dx_p = Constants.dx_p # Particle destribution size in x
dy_p = Constants.dy_p # Particle destribution size in x
dz_p = Constants.dz_p # Particle destribution size in x

Nx = Constants.Nx # number of cells in x
Ny = Constants.Ny # number of cells in x
Nz = Constants.Nz # number of cells in x
dx = Constants.dx # stepsize x
dy = Constants.dy # stepsize y
dz = Constants.dz # stepsize z

dt = Constants.dt

N_e = ParticleDestributionValues.N_e # number of negative particles per macroparticle
N_p = ParticleDestributionValues.N_p # number of positive particles per macroparticle
e_ = Constants.e_ # elemtary charge

e_0 = Constants.e_0 # electric field constant
u_0 = Constants.u_0 # magnetic field constant
c_0 = Constants.c_0 # speed of light
"""-------------------constants----------------------------------------------------------------------"""

_1_ = cp.ones(N)


# create fft frequencies for FDTD solver
K1 = 2*pi*cp.fft.fftfreq(Nx, d = dx) # note that the methods of transposition are applied in order to get correct dimensions in the FTDT solver
K2 = 2*pi*cp.fft.fftfreq(Ny, d = dy) # (the dimensions must be adjusted as we are performing matrix multiplication)
K3 = 2*pi*cp.fft.fftfreq(Nz, d = dz)
K = cp.meshgrid(K1, K2, K3)

def MaxwellSolver(R_e, V_e, R_p, V_p, E, B, J):
    
    RINT_e = cp.array([cp.floor((cp.array(R_e[0])/dx).astype(int)).astype(int), cp.floor((cp.array(R_e[1])/dy).astype(int)).astype(int), cp.floor((cp.array(R_e[2])/dz).astype(int)).astype(int)])
    RINT_p = cp.array([cp.floor((cp.array(R_p[0])/dx).astype(int)).astype(int), cp.floor((cp.array(R_p[1])/dy).astype(int)).astype(int), cp.floor((cp.array(R_p[2])/dz).astype(int)).astype(int)])
    
    # FDTD method for solving Maxwell's Equations
    FE = [cp.fft.fftn(E[0]), cp.fft.fftn(E[1]), cp.fft.fftn(E[2])]
    FB = [cp.fft.fftn(B[0]), cp.fft.fftn(B[1]), cp.fft.fftn(B[2])]                
    FJ = [cp.fft.fftn(J[0]), cp.fft.fftn(J[1]), cp.fft.fftn(J[2])]
    
    FE[0] += c_0**2*1j*dt*(K[1]*FB[2] - K[2]*FB[1]) - dt/e_0*FJ[0]    
    FE[1] += c_0**2*1j*dt*(K[2]*FB[0] - K[0]*FB[2]) - dt/e_0*FJ[1]    
    FE[2] += c_0**2*1j*dt*(K[0]*FB[1] - K[1]*FB[0]) - dt/e_0*FJ[2]    
    
    FB[0] -= 1j*dt*(K[1]*FE[2] - K[2]*FE[1])
    FB[1] -= 1j*dt*(K[2]*FE[0] - K[0]*FE[2])
    FB[2] -= 1j*dt*(K[0]*FE[1] - K[1]*FE[0])
    
    E = [cp.fft.ifftn(FE[0]).real, cp.fft.ifftn(FE[1]).real, cp.fft.ifftn(FE[2]).real]            
    del(FE)
    B = [cp.fft.ifftn(FB[0]).real, cp.fft.ifftn(FB[1]).real, cp.fft.ifftn(FB[2]).real]
    del(FB)                
    J = [cp.fft.ifftn(FJ[0]).real, cp.fft.ifftn(FJ[1]).real, cp.fft.ifftn(FJ[2]).real]
    del(FJ)
    
    # allign as suitable array for particles
    E_e = cp.array([E[0][RINT_e[0], RINT_e[1], RINT_e[2]], E[1][RINT_e[0], RINT_e[1], RINT_e[2]], E[2][RINT_e[0], RINT_e[1], RINT_e[2]]])
    B_e = cp.array([B[0][RINT_e[0], RINT_e[1], RINT_e[2]], B[1][RINT_e[0], RINT_e[1], RINT_e[2]], B[2][RINT_e[0], RINT_e[1], RINT_e[2]]])
    del(RINT_e)
    E_p = cp.array([E[0][RINT_p[0], RINT_p[1], RINT_p[2]], E[1][RINT_p[0], RINT_p[1], RINT_p[2]], E[2][RINT_p[0], RINT_p[1], RINT_p[2]]])
    B_p = cp.array([B[0][RINT_p[0], RINT_p[1], RINT_p[2]], B[1][RINT_p[0], RINT_p[1], RINT_p[2]], B[2][RINT_p[0], RINT_p[1], RINT_p[2]]])
    del(RINT_p)
    
    
    return [E, B, E_e, B_e, E_p, B_p]

    
