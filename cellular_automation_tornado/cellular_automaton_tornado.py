# 3-plus-time dimensional tornado model, using cylindrical cellular automata
# This will be initialized using properties found in the steady-state tornado
#   model; but can be initialized with any properties
# This model uses discrete parcels (the cells), each updating based on informa-
#   tion in neighboring parcels every time step
# For simplicity in regards to both getting and presenting information about
#   the tornado, as well as for a simpler update step, this uses cylindrical
#   coordinates rather than cartesian coordinates.  This, however, means that
#   parcels closer to the center will be larger than those further away, and
#   complicates how parcels on the radial edge will be handled.
#
# This will be my BEST CODE YET (this time I really mean it)
#
# If you want to assume that the tornado doesn't change with respect to theta,
#   all you need to do is set DELTA_THETA equal to MAX_THETA (and set both to
#   2*PI, if you're interested in being technically correct)
#
# TODO: Finish update(). Add plot/animation. Switch from C style code to PEP8.
#       Migrate to Git. Test.

import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import time
import tables as tb

# Global constants
# TODO: a single file from which all the R/Z/THETA info is pulled
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

class Parcel:
    """Parcel: a class to represent an air parcel, in cylindrical coordinates"""
    def __init__(self, r, z, theta,
                 rVel = None, zVel = None, thetaVel = None,
                 pressure = None, temperature = None, density = None,
                 viscocity = None):
        """Init: Create a Parcel object

        Required Parameters: r - radius, meters; z - altitude, meters; theta -
        rotation from an arbitrarily chosen plane, radians
        NOTE: Each of these parameters is the lower limit for its respective
        value in this parcel. For example, if a parcel spanned radius=4 meters
        to r=5 meters, r should be set to 4.

        Optional Parameters: rVel, zVel, thetaVel - velocities along each resp-
        ective coordinate, meters per second (even for thetaVel); pressure,
        Pascals; temperature, Kelvin; density, kilograms per cubic meter; visc-
        ocity, kilograms per meter per second
        NOTE: All of these optional parameters, save for density and viscocity,
        are deviations rather than absolute values; meaning they contain the
        amount by which that particular value in this parcel deviates from the
        average of that parameter

        TODO: Calculate pressure, temperature, or density, if the two other
        values are given, based on the ideal gas law (or whatever EOS)."""

        self.r = r
        self.z = z
        self.theta = theta

        self.rVel = rVel
        self.zVel = zVel
        self.thetaVel = thetaVel

        self.pressure = pressure
        self.temperature = temperature
        self.density = density
        self.viscocity = viscocity

        # Calculated values
        self.topArea = PI*(2*self.r*DELTA_R + DELTA_R*DELTA_R)*DELTA_THETA/MAX_THETA
        self.sideArea = DELTA_R*DELTA_Z
        self.innerArea = 2*PI*self.r*DELTA_Z*DELTA_THETA/MAX_THETA
        self.outerArea = 2*PI*(self.r+DELTA_R)*DELTA_Z*DELTA_THETA/MAX_THETA
        self.volume = self.topArea*DELTA_Z

        # Neighbor parcels - these will be defined by the setNeighbors method
        self.top = None
        self.bottom = None
        self.inner = None
        self.outer = None
        self.left = None
        self. right = None

    def setNeighbors(self, top = None, bottom = None, inner = None,
                     outer = None, left = None, right = None):
        """Set Neighbors: tell this parcel who its neighbors are

        I don't understand how python stores data or keeps track of objects or
        bindings, but if it works the way I think, these variables are actually
        bindings to the objects passed as parameters (the other parcels), so
        when those parcels are modified, these bindings reflect those modifica-
        tions.

        NOTE: 'left' and 'right' don't carry any meaning; those are just the
        two sides that aren't top, bottom, inner, or outer"""

        self.top = top
        self.bottom = bottom
        self.inner = inner
        self.outer = outer
        self.left = left
        self.right = right

    def update(self, newParcel):
        """Update: update this parcel and progress one time step, using inform-
        ation from neighboring parcels

        Since all parcels should be updated simultaneously, rather than updat-
        ing this parcel, which could affect the updating of its neighbors, this
        method returns the updated version of this parcel, to be dealt with by
        the user outside of this class

        Currently: This method only supports updating velocity. This method
        uses discrete change in values (deltas) in place of true differentials;
        since this model is discrete in nature, this approximation may be acce-
        ptable. This method changes velocity based on density, but only takes
        into account density of this parcel, and not that of neighbor parcels.
        This method handles boundary parcels by not doing any evaluations along
        the boundary faces.

        TODO: Support updating for all properties. Check velocity updates - I
        am less than confident that I did these correctly. Check if current
        way of handling boundary parcels is alright. Eliminate redundancy from
        multiple calculations across the same interface."""

        newRVel = self.rVel
        newZVel = self.zVel
        newThetaVel = self.thetaVel

        def deltaV(tau, area):
            """Delta V: a method to find the velocity change for a
            given tau and area.  Essentially, it's a simple equation that's
            repeated several times.

            TODO: use a better integrator for time"""

            return tau*area*DELTA_T / (self.density*self.volume)

        if self.top is not None:
            tauR = self.viscocity*(self.top.rVel-self.rVel)/DELTA_Z
            tauZ = (self.pressure-self.top.pressure) + 2*self.viscocity*(self.top.zVel-self.zVel)/DELTA_Z
            tauTheta = self.viscocity*(self.top.thetaVel-self.thetaVel)/DELTA_Z

            newRVel += deltaV(tauR, self.topArea)
            newZVel += deltaV(tauZ, self.topArea)
            newThetaVel += deltaV(tauTheta, self.topArea)

        if self.bottom is not None:
            tauR = self.viscocity*(self.bottom.rVel-self.rVel)/DELTA_Z
            tauZ = (self.bottom.pressure-self.pressure) + 2*self.viscocity*(self.bottom.zVel-self.zVel)/DELTA_Z
            tauTheta = self.viscocity*(self.bottom.thetaVel-self.thetaVel)/DELTA_Z

            newRVel += deltaV(tauR, self.topArea)
            newZVel += deltaV(tauZ, self.topArea)
            newThetaVel += deltaV(tauTheta, self.topArea)

        if self.inner is not None:
            tauR = (self.inner.pressure-self.pressure) + 2*self.viscocity*(self.inner.rVel-self.rVel)/DELTA_R
            tauZ = self.viscocity*(self.inner.zVel-self.zVel)/DELTA_R
            tauTheta = self.viscocity*self.r*((self.inner.thetaVel/self.inner.r)-(self.thetaVel/self.r))/DELTA_R

            newRVel += deltaV(tauR, self.innerArea)
            newZVel += deltaV(tauZ, self.innerArea)
            newThetaVel += deltaV(tauTheta, self.innerArea)

        if self.outer is not None:
            tauR = (self.pressure-self.outer.pressure) + 2*self.viscocity*(self.outer.rVel-self.rVel)/DELTA_R
            tauZ = self.viscocity*(self.outer.zVel-self.zVel)/DELTA_R
            tauTheta = self.viscocity*self.r*((self.outer.thetaVel/self.outer.r)-(self.thetaVel/self.r))/DELTA_R

            newRVel += deltaV(tauR, self.outerArea)
            newZVel += deltaV(tauZ, self.outerArea)
            newThetaVel += deltaV(tauTheta, self.outerArea)

        if self.left is not None:
            tauR = self.viscocity/self.r*(self.left.rVel-self.rVel)/DELTA_THETA
            tauZ = self.viscocity/self.r*(self.left.zVel-self.zVel)/DELTA_THETA
            tauTheta = (self.pressure-self.left.pressure) + 2*self.viscocity/self.r*((self.left.thetaVel-self.thetaVel)/DELTA_THETA + self.rVel)

            newRVel += deltaV(tauR, self.sideArea)
            newZVel += deltaV(tauZ, self.sideArea)
            newThetaVel += deltaV(tauTheta, self.sideArea)

        if self.right is not None:
            tauR = self.viscocity/self.r*(self.right.rVel-self.rVel)/DELTA_THETA
            tauZ = self.viscocity/self.r*(self.right.zVel-self.zVel)/DELTA_THETA
            tauTheta = (self.right.pressure-self.pressure) + 2*self.viscocity/self.r*((self.right.thetaVel-self.thetaVel)/DELTA_THETA + self.rVel)

            newRVel += deltaV(tauR, self.sideArea)
            newZVel += deltaV(tauZ, self.sideArea)
            newThetaVel += deltaV(tauTheta, self.sideArea)

        newParcel.r = self.r
        newParcel.z = self.z
        newParcel.theta = self.theta
        newParcel.rVel = newRVel
        newParcel.zVel = newZVel
        newParcel.thetaVel = newThetaVel
        newParcel.pressure = self.pressure
        newParcel.temperature = self.temperature
        newParcel.density = self.density
        newParcel.viscocity = self.viscocity

    @property
    def absoluteVelocity(self):
        """Absolute Velocity: return the absolute velocity of this parcel.
        """
        return np.sqrt(self.rVel**2 + self.zVel**2 + self.thetaVel**2)

def initParcelList(parcelList):
    # set coordinates
    for index, parcel in np.ndenumerate(parcelList):
        a = index[0]
        b = index[1]
        c = index[2]

        r = a*DELTA_R+DELTA_R/2
        z = b*DELTA_Z+DELTA_Z/2
        theta = c*DELTA_THETA+DELTA_THETA/2

        parcelList[index] = Parcel(r, z, theta)

    # set neighbors
    for index, parcel in np.ndenumerate(parcelList):
        a = index[0]
        b = index[1]
        c = index[2]

        if b < NUM_Z - 1:
            top = parcelList[a, b+1, c]
        else:
            top = None
        if b > 0:
            bottom = parcelList[a, b-1, c]
        else:
            bottom = None
        if a > 0:
            inner = parcelList[a-1, b, c]
        else:
            inner = None
        if a < NUM_R - 1:
           outer = parcelList[a+1, b, c]
        else:
            outer = None
        if c < NUM_THETA - 1:
            left = parcelList[a, b, c+1]
        else:
            left = parcelList[a, b, 0]
        if c > 0:
            right = parcelList[a, b, c-1]
        else:
            right = parcelList[a, b, NUM_THETA-1]

        parcelList[a, b, c].setNeighbors(
                top=top, bottom=bottom, inner=inner,
                outer=outer, left=left, right=right)

# ---------------- BEGIN the actual code ---------------------- #
#Initialize grid of parcels
parcelList = np.ndarray(shape=(NUM_R, NUM_Z, NUM_THETA), dtype=object)
tempList = np.ndarray(shape=(NUM_R, NUM_Z, NUM_THETA), dtype=object)

initParcelList(parcelList)
initParcelList(tempList)

# get initial values from steady-state model
initialization_file = tb.open_file("initialization_data.h5", mode="r", title="Initialization Data")
init_value_table = initialization_file.root.parcel_data.readout
init_values = np.ndarray(shape=(NUM_R, NUM_Z), dtype=(float,7))
for row in init_value_table.iterrows():
    init_values[row['r'], row['z']] = (row['rVel'], row['zVel'], row['thetaVel'],
                                       row['pressure'], row['temperature'],
                                       row['density'], row['viscocity'])

for index, parcel in np.ndenumerate(parcelList):
    rVel, zVel, thetaVel, pressure, temperature, density, viscocity = init_values[parcel.r/DELTA_R, parcel.z/DELTA_Z]

    parcelList[index].rVel = rVel
    parcelList[index].zVel = zVel
    parcelList[index].thetaVel = thetaVel
    parcelList[index].pressure = pressure
    parcelList[index].temperature = temperature
    parcelList[index].density = density
    parcelList[index].viscocity = viscocity

# creating the table in which to store all the simulation data
class TableData(tb.IsDescription):
    """TableData: a class derived from tables.IsDescription, to store data from
    each parcel in each timestep into a table cell.
    We probably don't need 64 bit precision, but perhaps it could prove useful
    or something.
    """
    r = tb.Float32Col()
    z = tb.Float32Col()
    theta = tb.Float32Col()
    t = tb.Float32Col()
    rVel = tb.Float32Col()
    zVel = tb.Float32Col()
    thetaVel = tb.Float32Col()

simulation_file = tb.open_file("simulation_data.h5", mode="w", title="Tornado Simulation Data")
group = simulation_file.create_group("/", "parcel_data")
table = simulation_file.create_table(group, 'readout', TableData, "Parcel Data Table")
parcel_table_row = table.row

step = 0
#Main update loop.
while step < NUM_T:
    for index, parcel in np.ndenumerate(parcelList):
        parcel_table_row['r'] = parcel.r
        parcel_table_row['z'] = parcel.z
        parcel_table_row['theta'] = parcel.theta
        parcel_table_row['t'] = step*DELTA_T
        parcel_table_row['rVel'] = parcel.rVel
        parcel_table_row['zVel'] = parcel.zVel
        parcel_table_row['thetaVel'] = parcel.thetaVel
        parcel_table_row.append()

        parcel.update(tempList[index])

    for index, parcel in np.ndenumerate(tempList):
        parcelList[index] = copy(parcel)

    step += 1
    table.flush()

# and one last update to the table, to get the last iteration
for index, parcel in np.ndenumerate(parcelList):
    parcel_table_row['r'] = parcel.r
    parcel_table_row['z'] = parcel.z
    parcel_table_row['theta'] = parcel.theta
    parcel_table_row['t'] = step*DELTA_T
    parcel_table_row['rVel'] = parcel.rVel
    parcel_table_row['zVel'] = parcel.zVel
    parcel_table_row['thetaVel'] = parcel.thetaVel
    parcel_table_row.append()
table.flush()

#Plotting
#for t in range(NUM_T):
#    plt.pcolor(absoluteVelocity[:,:,0,t])
#    plt.draw()
#    time.sleep(DELTA_T)

#plotting the change in absolute velocity for a single parcel
# timeList = [i*DELTA_T for i in range(NUM_T)]
# veloList = [i for i in absoluteVelocity[0,0,0,:]]
# plt.plot(timeList, veloList)
# plt.draw()

initialization_file.close()
simulation_file.close()
