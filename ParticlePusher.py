import cupy as cp

import Fields
import Constants
"""-------------------constants----------------------------------------------------------------------"""
c_0 = Constants.c_0 # speed of light

N = Constants.N

Nx = Constants.Nx
Ny = Constants.Ny
Nz = Constants.Nz
dx = Constants.dx
dy = Constants.dy
dz = Constants.dz

dt = Constants.dt
"""-------------------constants----------------------------------------------------------------------"""

_1_ = cp.ones(N)



# for relativistic velocities
def Gamma(beta):
    return cp.reciprocal((cp.array(1 - beta))**0.5)




def Particle(q, m, n_0, R, V, E_0, B_0):
    E = E_0 + Fields.E_external(R)
    B = B_0 + Fields.B_external(R)
   
    #move through phase space
    _V_ = (V[0]**2 + V[1]**2 + V[2]**2)**0.5
    U = Gamma(_V_/c_0)*V
    
    # velocity update 1/2
    U[0] = U[0] + q*dt/(2*m)*(E[0] + V[1]*B[2] - V[2]*B[1])
    U[1] = U[1] + q*dt/(2*m)*(E[1] + V[2]*B[0] - V[0]*B[2])
    U[2] = U[2] + q*dt/(2*m)*(E[2] + V[0]*B[1] - V[1]*B[0])
                              
    U_ = U + q*dt/m*E
    
    tau = q*dt/(2*m)*B
    gamma_ = (1 + (U_[0]**2 + U_[1]**2 + U_[2]**2)/c_0**2)**0.5
    sig = gamma_**2 - (tau[0]**2 + tau[1]**2 + tau[2]**2)
    del(gamma_)
    U_s = (U_[0]*tau[0] + U_[1]*tau[1] + U_[2]*tau[2])/c_0
    gammai1 = (sig + (sig**2 + 4*(tau[0]**2 + tau[1]**2 + tau[2]**2 + U_s**2))**0.5)**0.5/2**0.5
    del(U_s)
    t = tau/gammai1
    s = 1/(1 + t[0]**2 + t[1]**2 + t[2]**2)
    
    # velocity update 1
    Ui1X = s*(U_[0] + (U_[0]*t[0] + U_[1]*t[1] + U_[2]*t[2])*t[0] + U_[1]*t[2] - U_[2]*t[1])
    Ui1Y = s*(U_[1] + (U_[0]*t[0] + U_[1]*t[1] + U_[2]*t[2])*t[1] + U_[2]*t[0] - U_[0]*t[2])
    Ui1Z = s*(U_[2] + (U_[0]*t[0] + U_[1]*t[1] + U_[2]*t[2])*t[2] + U_[0]*t[1] - U_[1]*t[0])
    Ui1 = cp.array([Ui1X, Ui1Y, Ui1Z])    
    
    del(U_)
    del(t)
    del(s)
    
    # first push
    V = (Gamma(_V_**2/c_0**2)*V + Ui1)/2
    R += V*dt/2  
    
    # second push
    U = Ui1 
    del(Ui1)
    V = U/gammai1
    R += V*dt/2
    
    V = cp.reciprocal(Gamma(_V_/c_0))*U 
    del(_V_)
    
    return [R, V]
    