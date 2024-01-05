import cupy as cp

import Constants
import ParticleDestributionValues

dx = Constants.dx
dy = Constants.dy
dz = Constants.dz

dx_p = Constants.dx_p
dy_p = Constants.dy_p
dz_p = Constants.dz_p

Nx = Constants.Nx
Ny = Constants.Ny
Nz = Constants.Nz

e_ = Constants.e_

N = Constants.N


def FieldGather(R_e, V_e, R_p, V_p, J, N_e, N_p):    
    A = cp.arange(0, Nx)
    B = cp.arange(0, Ny)
    C = cp.arange(0, Nz)
    
    A, B, C = cp.meshgrid(A, B, C)
    
    Ge = cp.array([A, B, C])*0.
    Gp = Ge.copy()
    
    # search for cells that are intersecting with front/back part of each particle
    RINTe0 = cp.array([cp.floor((R_e[0] - 0.5*dx_p)/dx).astype(int), cp.array(cp.floor((R_e[1] - 0.5*dy_p)/dy).astype(int)), cp.array(cp.floor((R_e[2] - 0.5*dz_p)/dz).astype(int))])
    RINTe1 = cp.array([cp.floor((R_e[0] + 0.5*dx_p)/dx).astype(int), cp.floor((R_e[1] + 0.5*dy_p)/dy).astype(int), cp.floor((R_e[2] + 0.5*dz_p)/dz).astype(int)])
    RINTp0 = cp.array([cp.floor((R_p[0] - 0.5*dx_p)/dx).astype(int), cp.floor((R_p[1] - 0.5*dy_p)/dy).astype(int), cp.floor((R_p[2] - 0.5*dz_p)/dz).astype(int)])
    RINTp1 = cp.array([cp.floor((R_p[0] + 0.5*dx_p)/dx).astype(int), cp.floor((R_p[1] + 0.5*dy_p)/dy).astype(int), cp.floor((R_p[2] + 0.5*dz_p)/dz).astype(int)])
    
    # sort out particles which are out of the numerical box --> create booleans
    BOOLe0 = cp.array([RINTe0[0] >= 0, RINTe0[1] >= 0, RINTe0[2] >= 0])
    BOOLe1 = cp.array([RINTe1[0] <= Nx, RINTe1[1] <= Ny, RINTe1[2] <= Nz])
    BOOLp0 = cp.array([RINTp0[0] >= 0, RINTp0[1] >= 0, RINTp0[2] >= 0])
    BOOLp1 = cp.array([RINTp1[0] <= Nx, RINTp1[1] <= Ny, RINTp1[2] <= Nz])
    
    # make one boolean for all dimensions (True*False = False)
    BOOLe0 = BOOLe0[0]*BOOLe0[1]*BOOLe0[2]
    BOOLe1 = BOOLe1[0]*BOOLe1[1]*BOOLe1[2]
    BOOLp0 = BOOLp0[0]*BOOLp0[1]*BOOLp0[2]
    BOOLp1 = BOOLp1[0]*BOOLp1[1]*BOOLp1[2]
    
    # make one bool for each particle type
    BOOLe = BOOLe0*BOOLe1
    BOOLp = BOOLp0*BOOLp1
    del(BOOLe0)
    del(BOOLe1)
    del(BOOLp0)
    del(BOOLp1)

    # sort out particles which are out of the numerical box --> throw them out of all the necessary arrays
    RINTe0 = cp.array([RINTe0[0][BOOLe], RINTe0[1][BOOLe], RINTe0[2][BOOLe]])
    RINTe1 = cp.array([RINTe1[0][BOOLe], RINTe1[1][BOOLe], RINTe1[2][BOOLe]])
    RINTp0 = cp.array([RINTp0[0][BOOLp], RINTp0[1][BOOLp], RINTp0[2][BOOLp]])
    RINTp1 = cp.array([RINTp1[0][BOOLp], RINTp1[1][BOOLp], RINTp1[2][BOOLp]])
    R_e = cp.array([R_e[0][BOOLe], R_e[1][BOOLe], R_e[2][BOOLe]])
    R_p = cp.array([R_p[0][BOOLp], R_p[1][BOOLp], R_p[2][BOOLp]])
    V_e = cp.array([V_e[0][BOOLe], V_e[1][BOOLe], V_e[2][BOOLe]])
    V_p = cp.array([V_p[0][BOOLp], V_p[1][BOOLp], V_p[2][BOOLp]])
    
    N_e_ = N_e[BOOLe]
    N_p_ = N_p[BOOLp]
    del(BOOLe)
    del(BOOLp)
    
    # calculate the overlap - point inside each macroparticle
    OVe = cp.array([1 - (R_e[0] + 0.5*dx_p)/dx + RINTe1[0], 1 - (R_e[1] + 0.5*dy_p)/dy + RINTe1[1], 1 - (R_e[2] + 0.5*dz_p)/dz + RINTe1[2]])
    OVp = cp.array([1 - (R_p[0] + 0.5*dx_p)/dx + RINTp1[0], 1 - (R_p[1] + 0.5*dy_p)/dy + RINTp1[1], 1 - (R_p[2] + 0.5*dz_p)/dz + RINTp1[2]])
    
    # calculate the contribution by each particle to each cell (shape-function) --> define integration limits _0ex, _0ey...
    _0e_ = cp.zeros(len(OVe[0]))
    _1e_ = cp.ones(len(OVe[0]))
    _0p_ = cp.zeros(len(OVp[0]))
    _1p_ = cp.ones(len(OVp[0]))
    _0ex = cp.array([_0e_, OVe[0]])
    _0ey = cp.array([_0e_, OVe[1]])
    _0ez = cp.array([_0e_, OVe[2]])
    _1ex = cp.array([OVe[0], _1e_])
    _1ey = cp.array([OVe[1], _1e_])
    _1ez = cp.array([OVe[2], _1e_])
    _0px = cp.array([_0p_, OVp[0]])
    _0py = cp.array([_0p_, OVp[1]])
    _0pz = cp.array([_0p_, OVp[2]])
    _1px = cp.array([OVp[0], _1p_])
    _1py = cp.array([OVp[1], _1p_])
    _1pz = cp.array([OVp[2], _1p_])
    del(OVe)
    del(OVp)
    del(_0e_)
    del(_1e_)
    del(_0p_)
    del(_1p_)
    RINT_E = [RINTe0, RINTe1]
    RINT_P = [RINTp0, RINTp1]
    del(RINTe0)
    del(RINTe1)
    del(RINTp0)
    del(RINTp1)
    
    # 0 means back, 1 means front
    for gx in [0, 1]:
        for gy in [0, 1]:
            for gz in [0, 1]:        
                a = 8
                C = 1 - 8/cp.pi**3*(cp.arctan(cp.e**(a/2)) - cp.arctan(cp.e**(-a/2)))**3
                We = -(8*(cp.arctan(cp.e**(a/2-a*_1ex[gx]))-cp.arctan(cp.e**(a/2-a*_0ex[gx])))*(cp.arctan(cp.e**(a/2-a*_1ey[gy]))-cp.arctan(cp.e**(a/2-a*_0ey[gy])))*(cp.arctan(cp.e**(a/2-a*_1ez[gz]))-cp.arctan(cp.e**(a/2-a*_0ez[gz]))))/cp.pi**3 + C*(_1ex[gx] - _0ex[gx])*(_1ey[gy] - _0ey[gy])*(_1ez[gz] - _0ez[gz])
                Wp = -(8*(cp.arctan(cp.e**(a/2-a*_1px[gx]))-cp.arctan(cp.e**(a/2-a*_0px[gx])))*(cp.arctan(cp.e**(a/2-a*_1py[gy]))-cp.arctan(cp.e**(a/2-a*_0py[gy])))*(cp.arctan(cp.e**(a/2-a*_1pz[gz]))-cp.arctan(cp.e**(a/2-a*_0pz[gz]))))/cp.pi**3 + C*(_1px[gx] - _0px[gx])*(_1py[gy] - _0py[gy])*(_1pz[gz] - _0pz[gz])
                
                Ge[0][RINT_E[gx][0], RINT_E[gy][1], RINT_E[gz][2]] += We*V_e[0]*N_e_*e_/(dx*dy*dz)
                Gp[0][RINT_P[gx][0], RINT_P[gy][1], RINT_P[gz][2]] += Wp*V_p[0]*N_p_*e_/(dx*dy*dz)
                Ge[1][RINT_E[gx][0], RINT_E[gy][1], RINT_E[gz][2]] += We*V_e[1]*N_e_*e_/(dx*dy*dz)
                Gp[1][RINT_P[gx][0], RINT_P[gy][1], RINT_P[gz][2]] += Wp*V_p[1]*N_p_*e_/(dx*dy*dz)
                Ge[2][RINT_E[gx][0], RINT_E[gy][1], RINT_E[gz][2]] += We*V_e[2]*N_e_*e_/(dx*dy*dz)
                Gp[2][RINT_P[gx][0], RINT_P[gy][1], RINT_P[gz][2]] += Wp*V_p[2]*N_p_*e_/(dx*dy*dz)
                
                J = Gp - Ge
                
                
                
    del(_0ex) 
    del(_0ey)
    del(_0ez) 
    del(_1ex) 
    del(_1ey) 
    del(_1ez) 
    del(_0px) 
    del(_0py) 
    del(_0pz) 
    del(_1px) 
    del(_1py) 
    del(_1pz)
    del(RINT_E)
    del(RINT_P)
    return [J, R_e, R_p, V_e, V_p, N_e_, N_p_, Ge, Gp]