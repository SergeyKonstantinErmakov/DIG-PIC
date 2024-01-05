import cupy as cp

import Constants
"""-------------------constants----------------------------------------------------------------------"""
pi = cp.pi

Nx = Constants.Nx # number of cells in x
Ny = Constants.Ny # number of cells in x
Nz = Constants.Nz # number of cells in x
dx = Constants.dx # stepsize x
dy = Constants.dy # stepsize y
dz = Constants.dz # stepsize z

N = Constants.N # Number of particles

T_i = Constants.T_i # ion temperature
T_e = Constants.T_e # electron temperature

u_0 = Constants.u_0
kB = Constants.kB

m_e = Constants.m_e # mass of negative particles 
m_p = Constants.m_p # mass of positive particles 
"""-------------------constants----------------------------------------------------------------------"""

_0_ = cp.zeros(N)
_1_ = cp.ones(N)


# define initial particle-/velocity destribution
R_p = cp.array(cp.load('R_p.npy'))    
V_p = cp.array(cp.load('V_p.npy'))    
R_e = cp.array(cp.load('R_e.npy'))    
V_e = cp.array(cp.load('V_e.npy'))    


# define initial fields
def EInitial(X, Y, Z):
    return cp.array([0.*X, 0.*Y, 0.*Z])
    


#Lz = Nz/2*dz
B_ex = cp.array(cp.load('Quad.npy'))    
def BInitial(X, Y, Z):
    #Bxp = -cp.pi*cp.exp( -(X - 0.5*Nx*dx)**2/Lz**2 + 0.5)*cp.sin(cp.pi*Z/(2*Lz))
    #Bzp = cp.pi*cp.exp( -(X - 0.5*Nx*dx)**2/Lz**2 + 0.5)*cp.cos(cp.pi*Z/(2*Lz))
    #B = 50.*cp.array([cp.tanh((Z - 0.5*Nz*dz)/Lz) + Bxp, 1/cp.cosh((Z - 0.5*Nz*dz)/Lz), Bzp])
    return B_ex
    #B = cp.array([X*0., 2.*cp.tanh(X/(Nx*dx) - 0.5), Z*0.])
    #return B
    

def JInitial(X, Y, Z):
    return cp.array([0.*X, 0.*Y, 0.*Z])
    

# define external Fields
def E_external(R):
    E = cp.array([R[0]*0., R[1]*0., R[2]*0.]) # external field values saved as a vector in order to immediately use with particle array
    return E


def B_external(R):
    IntX = cp.floor((cp.array(R[0])/dx).astype(int)).astype(int)
    IntY = cp.floor((cp.array(R[1])/dy).astype(int)).astype(int)
    IntZ = cp.floor((cp.array(R[2])/dz).astype(int)).astype(int)
    B = cp.array([B_ex[0][IntX, IntY, IntZ], B_ex[1][IntX, IntY, IntZ], B_ex[2][IntX, IntY, IntZ]])
    #B = cp.array([cp.tanh((R[2] - 0.5*Nz*dz)/Lz), 1/cp.cosh((R[2] - 0.5*Nz*dz)/Lz), 0.*R[2]])
    #B = cp.array([R[0]*0., R[1]*0., R[2]*0.])
    return B



