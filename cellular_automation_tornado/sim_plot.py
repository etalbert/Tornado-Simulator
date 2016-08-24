# A script to generate plots and/or animations for the cellular automata tornado

import numpy as np
import tables as tb
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time

PI = np.pi #Mathematical constant pi
DELTA_R = 20 #Change in radius across a parcel; in meters
DELTA_Z = 20 #Change in altitude across a parcel; in meters
DELTA_THETA = 2*PI/10 #Change in rotation across a parcel; in radians
DELTA_T = .1 #Change in time across a time step; in seconds
MAX_R = 1000 #Maximum radius in this model; in meters
MAX_Z = 1000 #Maximum altitude in this model; in meters
MAX_THETA = 2*PI #Maximum rotation in this model (det'd by geometry); in radians
MAX_T = 10 #Running time for the model; in seconds
NUM_R = int(MAX_R/DELTA_R) #Number of parcels in the radius direction
NUM_Z = int(MAX_Z/DELTA_Z) #Number of parcels in the altitude direction
NUM_THETA = int(MAX_THETA/DELTA_THETA) #Number of parcels in the theta direction
NUM_T = int(MAX_T/DELTA_T) #Number of timesteps

# get simulation data from table file
simulation_file = tb.open_file("simulation_data.h5", mode="r", title="Tornado Simulation Data")
table = simulation_file.root.parcel_data.readout


absolute_velocity = np.ndarray(shape=(NUM_R, NUM_Z, NUM_THETA, NUM_T+1))
for row in table.iterrows():
    radius = row['rIndex']
    height = row['zIndex']
    theta = row['thetaIndex']
    time_index = row['tIndex'] # time is already in the namespace, thus the different name style
    absolute_velocity[radius, height, theta, time_index] = math.sqrt(row['rVel']**2 + row['zVel']**2 + row['thetaVel']**2)

#Plotting
# plt.show()
# for t in range(NUM_T):
#    plt.pcolor(absolute_velocity[:,:,0,t])
#    plt.draw()
#    time.sleep(DELTA_T)

#plotting the change in absolute velocity for a single parcel
# timeList = [i*DELTA_T for i in range(NUM_T+1)]
# veloList = [i for i in absolute_velocity[0,0,0,:]]
# plt.plot(timeList, veloList)

R = np.arange(0, MAX_R, DELTA_R)
Z = np.arange(0, MAX_Z, DELTA_Z)
R, Z = np.meshgrid(R, Z)

fig = plt.figure(1)
ax = Axes3D(fig)
surf = ax.plot_surface(Z, R, absolute_velocity[:,:,0,0], rstride=1, cstride=1,
        cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(0, 10000)
ax.set_xlabel('Radius')
ax.set_ylabel('Height')
ax.set_title('0s, Total Velocity in and around a tornado as a function of radius and height')

fig = plt.figure(2)
ax = Axes3D(fig)
surf = ax.plot_surface(Z, R, absolute_velocity[:,:,0,49], rstride=1, cstride=1,
        cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(0, 10000)
ax.set_xlabel('Radius')
ax.set_ylabel('Height')
ax.set_title('5s, Total Velocity in and around a tornado as a function of radius and height')

fig = plt.figure(3)
ax = Axes3D(fig)
surf = ax.plot_surface(Z, R, absolute_velocity[:,:,0,-1], rstride=1, cstride=1,
        cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(0, 10000)
ax.set_xlabel('Radius')
ax.set_ylabel('Height')
ax.set_title('10s, Total Velocity in and around a tornado as a function of radius and height')

sincos = absolute_velocity[:,:,0,0] - absolute_velocity[:,:,0,0]
for r in range(NUM_R):
    for z in range(NUM_Z):
        sincos[r][z] = np.sin(r)+np.sin(z)
fig = plt.figure(4)
ax = Axes3D(fig)
surf = ax.plot_surface(Z, R, sincos, rstride=1, cstride=1,
        cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(-25, 25)
ax.set_xlabel('Radius')
ax.set_ylabel('Height')
ax.set_title('waves')

plt.show()

simulation_file.close()
