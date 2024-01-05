# DIG-PIC
DIG-PIC stands for "Desktop Integrated GPU Particle-in-Cell Code". This is a 3D Relativistic PIC solver. It makes use of the Boris Particle Pusher as well as the PSTD method to solve Maxwell's equations. This allows for different kinds of plasma dynamics simulations.
In case of using this code, please cite this GitHub repository.

# Structure of the Code
The code is devided into 7 files. That is:
- **Constants.py** Here, natural constants and simulation parameters can be edited.
- **Fields.py** Initial electromagnetic fields can be adjusted here.
- **ParticleDestributionValues.py** Initial macropatricle distributions can be adjusted here.
- **MaxwellSolver.py** Implementation of PSTD maxwell solver.
- **ParticlePusher.py** Implementation of Boris particle pusher.
- **FieldGather.py** J-Field deposition of charged particles as well as assigning each particle a field.
- **Main.py** Main loop.

# Downloading and Editing Code
Feel free to download the code and make changes to it. Especially in case of improvements, changes or encountered errors please feel free to contact me so that the code can be improved!
- **sergey.k.ermakov_atSymbol_gmail.com**
