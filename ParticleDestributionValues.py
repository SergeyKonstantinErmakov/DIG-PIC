import cupy as cp

import Constants
from scipy.stats import maxwell

Nx = Constants.Nx # number of cells in x
Ny = Constants.Ny # number of cells in x
Nz = Constants.Nz # number of cells in x
dx = Constants.dx # stepsize x
dy = Constants.dy # stepsize y
dz = Constants.dz # stepsize z

N = Constants.N # Number of particles

T_i = Constants.T_i # ion temperature
T_e = Constants.T_e # electron temperature

kB = Constants.kB

m_e = Constants.m_e # mass of negative particles 
m_p = Constants.m_p # mass of positive particles 
n_e = Constants.n_e
n_p = Constants.n_p


x = cp.arange(0, Nx*dx, dx)
y = cp.arange(0, Ny*dy, dy)
z = cp.arange(0, Nz*dz, dz)
x, y, z = cp.meshgrid(x, y, z)
r = cp.array([x.ravel(), y.ravel(), z.ravel()])
Dire = cp.array([cp.random.choice(cp.array([-1, 1]), len(r[0])), cp.random.choice(cp.array([-1, 1]), len(r[0])), cp.random.choice(cp.array([-1, 1]), len(r[0]))])
Dirp = cp.array([cp.random.choice(cp.array([-1, 1]), len(r[0])), cp.random.choice(cp.array([-1, 1]), len(r[0])), cp.random.choice(cp.array([-1, 1]), len(r[0]))])
R_e = r
V_e = 3**-0.5*cp.array(maxwell.rvs((kB*T_i/m_p)**0.5, size=len(r[0])))*Dire
#V_e = cp.array([cp.zeros(len(r[0])), cp.zeros(len(r[1])), cp.ones(len(r[2]))*3*10**4])
#V_p = cp.array([cp.zeros(len(r[0])), cp.zeros(len(r[1])), cp.ones(len(r[2]))*3*10**2])
R_p = R_e.copy()
V_p = 3**-0.5*cp.array(maxwell.rvs((kB*T_i/m_e)**0.5, size=len(r[0])))*Dirp # velocity maxwellian

sig = 0.1*Nx*dx
N_e = 0.85*n_e/(sig**2*2*cp.pi)**0.5*cp.e**(-0.5*((R_e[0] - 0.5*Nx*dx)**2 + (R_e[1] - 0.5*Ny*dy)**2 + (R_e[2] - 0.5*Nz*dz)**2)/sig**2) + 0.15*n_e
N_p = 0.85*n_p/(sig**2*2*cp.pi)**0.5*cp.e**(-0.5*((R_p[0] - 0.5*Nx*dx)**2 + (R_p[1] - 0.5*Ny*dy)**2 + (R_p[2] - 0.5*Nz*dz)**2)/sig**2) + 0.15*n_p

cp.save('R_e', R_e)
cp.save('V_e', V_e)
cp.save('R_p', R_p)
cp.save('V_p', V_p)

