import vtk
import concurrent.futures
import cupy as cp
import numpy as np
from vtk.util import numpy_support

import ParticlePusher
import MaxwellSolver
import Fields
import Constants
import FieldGather
import ParticleDestributionValues


"""-------------------constants----------------------------------------------------------------------"""
Nx = Constants.Nx # number of cells in x
Ny = Constants.Ny # number of cells in x
Nz = Constants.Nz # number of cells in x
dx = Constants.dx # stepsize x
dy = Constants.dy # stepsize y
dz = Constants.dz # stepsize z

t_i = Constants.t_i # initial t
t_f = Constants.t_f # final t
dt = Constants.dt # stepsize t

n_e = Constants.n_e # number of negative particles per macroparticle
n_p = Constants.n_p # number of positive particles per macroparticle
e_ = Constants.e_ # elementary charge
m_e = Constants.m_e # mass of negative particles 
m_p = Constants.m_p # mass of positive particles 
"""-------------------constants----------------------------------------------------------------------"""



# create a grid 
XX = cp.arange(0, Nx*dx, dx)
YY = cp.arange(0, Ny*dy, dy)
ZZ = cp.arange(0, Nz*dz, dz)

X, Y, Z = cp.meshgrid(XX, YY, ZZ)



R_e = Fields.R_e
V_e = Fields.V_e
R_p = Fields.R_p
V_p = Fields.V_p

E = Fields.EInitial(X, Y, Z)
B = Fields.BInitial(X, Y, Z)
J = Fields.JInitial(X, Y, Z)

E_e = []
B_e = []
E_p = []
B_p = []

N_e = ParticleDestributionValues.N_e
N_p = ParticleDestributionValues.N_p

t = t_i
while t <= t_f:
    
      
        
    FG = FieldGather.FieldGather(R_e, V_e, R_p, V_p, J*0., N_e, N_p)
    J = FG[0]
    R_e = FG[1]
    R_p = FG[2]
    V_e = FG[3]
    V_p = FG[4]
    N_e = FG[5]
    N_p = FG[6]
    Ge = cp.asnumpy(FG[7])
    Gp = cp.asnumpy(FG[8])

    EM = MaxwellSolver.MaxwellSolver(R_e, V_e, R_p, V_p, E, B, J)
    del(E)
    del(B)
    del(E_e)
    del(B_e)
    del(E_p)
    del(B_p)
    
    E = EM[0]
    B = EM[1]
    E_e = EM[2]
    B_e = EM[3]
    E_p = EM[4]
    B_p = EM[5]
        
    # electron and ion particle pusher on seperate threads
    with concurrent.futures.ThreadPoolExecutor() as executor:                            
        electron = executor.submit(ParticlePusher.Particle, -e_, m_e, n_e, R_e, V_e, E_e, B_e)
        proton = executor.submit(ParticlePusher.Particle, e_, m_p, n_p, R_p, V_p, E_p, B_p)
        
        R_e = electron.result()[0]
        V_e = electron.result()[1]
        R_p = proton.result()[0]
        V_p = proton.result()[1]
    
        
    # ---------------------------------------------------plotting---------------------------------------------------------------------------------
    if int(t/dt)%4==0:
        print("t=" + str(t))
        
        BDir = 'G:/Quad30Vel0/NUMPY/B/B_t=' + str(int(t/dt))
        BNP = cp.asnumpy(cp.array(B))
        cp.save(BDir, BNP)
        vtype = vtk.util.numpy_support.get_vtk_array_type(BNP[0].dtype)
        vcomponents = 3
        
        imageData = vtk.vtkImageData()
        imageData.SetSpacing(1.0, 1.0, 1.0)
        imageData.SetOrigin(0.0, 0.0, 0.0)
        imageData.SetDimensions(Nz, Ny, Nx)
        imageData.AllocateScalars(vtype, vcomponents)
        
        vtk_data_array = numpy_support.numpy_to_vtk(num_array=np.transpose(BNP).ravel(), deep=True, array_type=vtype)
        vtk_data_array.SetNumberOfComponents(3)
        vtk_data_array.SetName("Magnetic Field")
        imageData.GetPointData().SetScalars(vtk_data_array)
        
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetInputData(imageData)
        writer.SetFileName("G:/Quad30Vel0/VTK/B/B" + str(int(t/dt)) + ".vtk")
        writer.Write()
        
        
        EDir = 'G:/Quad30Vel0/NUMPY/E/E_t=' + str(int(t/dt))
        ENP = cp.asnumpy(cp.array(E))
        cp.save(EDir, ENP)
        vtype = vtk.util.numpy_support.get_vtk_array_type(ENP[0].dtype)
        vcomponents = 3
        
        imageData = vtk.vtkImageData()
        imageData.SetSpacing(1.0, 1.0, 1.0)
        imageData.SetOrigin(0.0, 0.0, 0.0)
        imageData.SetDimensions(Nz, Ny, Nx)
        imageData.AllocateScalars(vtype, vcomponents)
        
        vtk_data_array = numpy_support.numpy_to_vtk(num_array=np.transpose(ENP).ravel(), deep=True, array_type=vtype)
        vtk_data_array.SetNumberOfComponents(3)
        vtk_data_array.SetName("Electric Field")
        imageData.GetPointData().SetScalars(vtk_data_array)
        
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetInputData(imageData)
        writer.SetFileName("G:/Quad30Vel0/VTK/E/E" + str(int(t/dt)) + ".vtk")
        writer.Write()
        
        
        JDir = 'G:/Quad30Vel0/NUMPY/J/J_t=' + str(int(t/dt))
        JNP = cp.asnumpy(cp.array(J))
        cp.save(JDir, JNP)
        vtype = vtk.util.numpy_support.get_vtk_array_type(JNP[0].dtype)
        vcomponents = 3
        
        imageData = vtk.vtkImageData()
        imageData.SetSpacing(1.0, 1.0, 1.0)
        imageData.SetOrigin(0.0, 0.0, 0.0)
        imageData.SetDimensions(Nz, Ny, Nx)
        imageData.AllocateScalars(vtype, vcomponents)
        
        vtk_data_array = numpy_support.numpy_to_vtk(num_array=np.transpose(JNP).ravel(), deep=True, array_type=vtype)
        vtk_data_array.SetNumberOfComponents(3)
        vtk_data_array.SetName("J")
        imageData.GetPointData().SetScalars(vtk_data_array)
        
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetInputData(imageData)
        writer.SetFileName("G:/Quad30Vel0/VTK/J/J" + str(int(t/dt)) + ".vtk")
        writer.Write()
        
        
        vtype = vtk.util.numpy_support.get_vtk_array_type(Ge[0].dtype)
        vcomponents = 3
        
        imageData = vtk.vtkImageData()
        imageData.SetSpacing(1.0, 1.0, 1.0)
        imageData.SetOrigin(0.0, 0.0, 0.0)
        imageData.SetDimensions(Nz, Ny, Nx)
        imageData.AllocateScalars(vtype, vcomponents)
        
        vtk_data_array = numpy_support.numpy_to_vtk(num_array=np.transpose(Ge).ravel(), deep=True, array_type=vtype)
        vtk_data_array.SetNumberOfComponents(3)
        vtk_data_array.SetName("Ge")
        imageData.GetPointData().SetScalars(vtk_data_array)
        
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetInputData(imageData)
        writer.SetFileName("G:/Quad30Vel0/VTK/Ge/Ge" + str(int(t/dt)) + ".vtk")
        writer.Write()
        
        
        vtype = vtk.util.numpy_support.get_vtk_array_type(Gp[0].dtype)
        vcomponents = 3
        
        imageData = vtk.vtkImageData()
        imageData.SetSpacing(1.0, 1.0, 1.0)
        imageData.SetOrigin(0.0, 0.0, 0.0)
        imageData.SetDimensions(Nz, Ny, Nx)
        imageData.AllocateScalars(vtype, vcomponents)
        
        vtk_data_array = numpy_support.numpy_to_vtk(num_array=np.transpose(Gp).ravel(), deep=True, array_type=vtype)
        vtk_data_array.SetNumberOfComponents(3)
        vtk_data_array.SetName("Gp")
        imageData.GetPointData().SetScalars(vtk_data_array)
        
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetInputData(imageData)
        writer.SetFileName("G:/Quad30Vel0/VTK/Gp/Gp" + str(int(t/dt)) + ".vtk")
        writer.Write()
        
        
        
        R_eDir = 'G:/Quad30Vel0/NUMPY/R_e/R_e_t=' + str(int(t/dt)) 
        R_eNP = cp.asnumpy(R_e)
        cp.save(R_eDir, R_eNP)
        
        
        V_eDir = 'G:/Quad30Vel0/NUMPY/V_e/V_e_t=' + str(int(t/dt)) 
        V_eNP = cp.asnumpy(V_e)
        cp.save(V_eDir, V_eNP)
        
        
        R_pDir = 'G:/Quad30Vel0/NUMPY/R_p/R_p_t=' + str(int(t/dt))
        R_pNP = cp.asnumpy(R_p)
        cp.save(R_pDir, R_pNP)
        
        
        V_pDir = 'G:/Quad30Vel0/NUMPY/V_p/V_p_t=' + str(int(t/dt)) 
        V_pNP = cp.asnumpy(V_p)
        cp.save(V_pDir, V_pNP)
# ---------------------------------------------------plotting---------------------------------------------------------------------------------    
  
    
    t += dt
