# Steady-State Cylindrical Tornado Initialization
# Code taken from previous steady-state tornado code
# It probably works. And probably could be made to work a lot better.

import numpy as np
import tables as tb

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

#constants, boundary condition variables and arrays
G_CONST = -9.81 # Acceleration of Earth's gravity; in meters / second^2,
IDEAL_GAS_CONST = 8314.4598 #gas costant R; in meters^3 * pascals / kelvin * kilogram-moles
MW_AIR = 29 #molecular weight of air; in kilograms / kilogram-moles
STP_PRESSURE = 101325 #standard atmospheric pressure; in pascals

DELTA_R = 20 # Change in radius across a parcel; in meters
DELTA_Z = 20 # Change in altitude across a parcel; in meters
MAX_R = 1000 # Maximum radius in this model; in meters
MAX_Z = 1000 # Maximum altitude in this model; in meters
NUM_R = int(MAX_R/DELTA_R) #Number of parcels in the radius direction
NUM_Z = int(MAX_Z/DELTA_Z) #Number of parcels in the altitude direction

DEW_TEMP = 18 #degrees celsius
def ambient_temp(height):
    return 30 - .0064*height

def velocity_max(height):#meters / second, velocity at edge of tornado for a given height
    return ANGULAR_VEL*tornado_radius(height)
ANGULAR_VEL = .4 #angular velocity, radians per second
TORNADO_SLOPE = .2 #meters r / meters z : this is backwards from how it looks on
                  #a graph, but r is dependent on z so that's how I'm doing it
#tornado_radius, as a function of height, with a value of 200 meters at z=0
def tornado_radius(height):
    return height*TORNADO_SLOPE + 200

#function to better use python loops. Which I really shouldn't be using at all
def drange(start, stop, step):
    while start < stop:
        yield start
        start += step

#Init a 2d (r,z) array of implicity defined structs to make up a tornado and
#surrounding area.  The way I'm currently initializing values, python looping
#is being used, but numpy implicit looping is far more efficient
#struct's members are, in order:
#   Velocity (rotational), meters / second
#   Temperature, degrees celsius
#   Density, kilograms / meters^3
#   Pressure, pascals
lattice = np.zeros((NUM_R, NUM_Z),
                   dtype = [('velocity', 'i4'),
                            ('temperature', 'i4'),
                            ('density', 'i4'),
                            ('pressure', 'i4'),
                            ('dvdr', 'i4'),
                            ('dtdr', 'i4'),
                            ('drhodr', 'i4'),
                            ('dpdr', 'i4'),
                            ('dvdz', 'i4'),
                            ('dtdz', 'i4'),
                            ('drhodz', 'i4'),
                            ('dpdz', 'i4')])

#These arrays point to the same memory locations as lattice does, so using them
#is essentially just a nicer-looking way of accessing lattice
velocity = lattice['velocity']
temperature = lattice['temperature']
density = lattice['density']
pressure = lattice['pressure']
dvdr = lattice['dvdr'] # dV/dr
dtdr = lattice['dtdr'] # dT/dr
drhodr = lattice['drhodr'] # d(rho)/dr
dpdr = lattice['dpdr'] # dP/dr
dvdz = lattice['dvdz'] # dV/dz
dtdz = lattice['dtdz'] # dT/dz
drhodz = lattice['drhodz'] # d(rho)/dz
dpdz = lattice['dpdz'] # dP/dz

for r in drange(0, NUM_R, 1):
    for z in drange(0, NUM_Z, 1):
        #r and z are indices, radius and height the actual values
        radius = r*DELTA_R
        height = z*DELTA_Z

        #Initialize velocity
        if radius <= tornado_radius(height):
            velocity[r][z] = radius*ANGULAR_VEL
        else:
            #tornado_radius(height) should be the radius of max windspeed
            velocity[r][z] = ANGULAR_VEL*(tornado_radius(height)**2)/radius

        #Initialize temperature
        temperature[r][z] = DEW_TEMP +\
                            (velocity_max(height) - velocity[r][z])/velocity_max(height) *\
                            (ambient_temp(height) - DEW_TEMP)

        #Initialize density to this estimated value so we can get an initial
        #value for pressure, and work from there
        density[r][z] = 1.15

        #Iteratively change pressure, density, and temperature
        for i in drange(0, 75, 1):#If this takes too long, decrease the middle number
            #Change pressure
            if radius <= tornado_radius(height):
                pressure[r][z] =\
                    .5*density[r][z]*radius*(ANGULAR_VEL**2) +\
                    density[r][z]*G_CONST*height +\
                    STP_PRESSURE -\
                    .5*density[r][z]*(tornado_radius(height)**2)*(ANGULAR_VEL**2) -\
                    .5*density[r][z]*tornado_radius(height)*(ANGULAR_VEL**2)

            else:
                pressure[r][z] =\
                    STP_PRESSURE -\
                    density[r][z]*(ANGULAR_VEL**2)*(tornado_radius(height)**4)/(2*(radius**2)) +\
                    density[r][z]*G_CONST*height

            #Change temperature
            #We're not doing this yet because, for now, we're just basing temp
            #off of velocity, which isn't iterated either

            #Change density
            density[r][z] = (pressure[r][z] * MW_AIR) /\
                            (IDEAL_GAS_CONST * (temperature[r][z]+273))

#Generate values for partial derivatives
dvdr, dvdz = np.gradient(velocity)
dtdr, dtdz = np.gradient(temperature)
drhodr, drhodz = np.gradient(density)
dpdr, dpdz = np.gradient(pressure)

# Push initialization data into a table, so it can be accessed from other files
# creating the table in which to store all the simulation data
class TableData(tb.IsDescription):
    """TableData: a class derived from tables.IsDescription, to store data from
    each parcel into a table cell.
    We probably don't need 64 bit precision, but perhaps it could prove useful
    or something.
    """
    r = tb.Float32Col()
    z = tb.Float32Col()
    rVel = tb.Float32Col()
    zVel = tb.Float32Col()
    thetaVel = tb.Float32Col()
    pressure = tb.Float32Col()
    temperature = tb.Float32Col()
    density = tb.Float32Col()
    viscocity = tb.Float32Col()

initialization_file = tb.open_file("initialization_data.h5", mode="w", title="Initialization Data")
group = initialization_file.create_group("/", "parcel_data")
table = initialization_file.create_table(group, 'readout', TableData, "Parcel Data Table")
parcel_table_row = table.row

for r in drange(0, NUM_R, 1):
    for z in drange(0, NUM_Z, 1):
        #r and z are indices; radius and height the actual values
        radius = r*DELTA_R
        height = z*DELTA_Z

        parcel_table_row['r'] = radius
        parcel_table_row['z'] = height
        parcel_table_row['rVel'] = 0
        parcel_table_row['zVel'] = 0
        parcel_table_row['thetaVel'] = velocity[r][z]
        parcel_table_row['pressure'] = pressure[r][z]
        parcel_table_row['temperature'] = temperature[r][z]
        parcel_table_row['density'] = 1 #density[r][z]
        parcel_table_row['viscocity'] = 1.85*(10**-5)
table.flush()

# #Plotting
# #I can probably write a function to do all this instead of repeating code like
# #I am, but sometimes Ctrl+C/Ctrl+V is a lot easier than actually thinking.
# #Plus this way, if changes need to be made to a single chart, it's much easier.
# R = np.arange(0, MAX_R, DELTA_R)
# Z = np.arange(0, MAX_Z, DELTA_Z)
# R, Z = np.meshgrid(R, Z)
#
# #plot pressure
# fig = plt.figure(1)
# ax = Axes3D(fig)#fig.gca(projection='3d') is what I had before, and it didn't work
# surf = ax.plot_surface(Z, R, pressure, rstride=1, cstride=1,
#         cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax.set_zlim(0, 175000)
# ax.set_xlabel('Radius')
# ax.set_ylabel('Height')
# ax.set_title('Pressure in and around a tornado as a function of radius and height')
#
# #plot velocity
# fig = plt.figure(2)
# ax = Axes3D(fig)
# surf = ax.plot_surface(Z, R, velocity, rstride=1, cstride=1,
#         cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax.set_zlim(0, 500)
# ax.set_xlabel('Radius')
# ax.set_ylabel('Height')
# ax.set_title('Rotational velocity in and around a tornado as a function of radius and height')
#
# #plot temperature
# fig = plt.figure(3)
# ax = Axes3D(fig)
# surf = ax.plot_surface(Z, R, temperature, rstride=1, cstride=1,
#         cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax.set_zlim(-100, 100)
# ax.set_xlabel('Radius')
# ax.set_ylabel('Height')
# ax.set_title('Temperature in and around a tornado as a function of radius and height')
#
# #plot density
# fig = plt.figure(4)
# ax = Axes3D(fig)
# surf = ax.plot_surface(Z, R, density, rstride=1, cstride=1,
#         cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax.set_zlim(0, 2)
# ax.set_xlabel('Radius')
# ax.set_ylabel('Height')
# ax.set_title('Density in and around a tornado as a function of radius and height')
#
# #plot partial of pressure in respect to radius
# fig = plt.figure(5)
# ax = Axes3D(fig)
# surf = ax.plot_surface(Z, R, dpdr, rstride=1, cstride=1,
#         cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax.set_zlim(-500, 500)
# ax.set_xlabel('Radius')
# ax.set_ylabel('Height')
# ax.set_title('Partial Derivative of Pressure in respect to Radius in and around a tornado as a function of radius and height')
#
# #plot partial of pressure in respect to height
# fig = plt.figure(6)
# ax = Axes3D(fig)
# surf = ax.plot_surface(Z, R, dpdz, rstride=1, cstride=1,
#         cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax.set_zlim(-500, 500)
# ax.set_xlabel('Radius')
# ax.set_ylabel('Height')
# ax.set_title('Partial Derivative of Pressure in respect to Height in and around a tornado as a function of radius and height')
#
# plt.show()

initialization_file.close()
