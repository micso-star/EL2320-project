# EL2320 course project
# Author: Michaela Söderström 

"""
Initialization of parameters for the particle filter.
----------------------------------------------------
"""

# Load libraries
import numpy as np
import operator

def getParams(roi_coord, frame_size, init_hypotheses, M):
    '''
    Creates initial state of the particles.
    -------------------------------------------------
    INPUT
    roi_coord: tuple
        top left coordinates, width and height enclosing the region of interest
    frame_size: array
        size of the original frame
    init_hypotheses: bool
        decides how particles are initialized
    M: int
        no. of particles

    OUTPUT:
    S: array
        initial random particle set, 4xM [x; y; xdot; ydot; xdotdot; ydotdot]
    -------------------------------------------------
    '''
    np.random.seed(400)
    x =  roi_coord[0]
    y = roi_coord[1]
    w = roi_coord[2]
    h = roi_coord[3]
    x_max = tuple(map(operator.add, x, w))
    y_max = tuple(map(operator.add, y, h))

    if init_hypotheses:
        # Generate particle set from random uniform distribution (cumulative) for M no. of particles over the whole frame
        S = [[np.random.randint(0, frame_size[0], size=M)], 
            [np.random.randint(0, frame_size[1], size=M)],
            [np.zeros(M)], # xdot
            [np.zeros(M)], # ydot
            [np.zeros(M)], # xdotdot
            [np.zeros(M)]] # ydotdot
        S = np.reshape(S, (np.size(S, 0), M)) # reshape to correct dim

    else:
        # Generate particle set from random uniform distribution (cumulative) for M no. of particles in the region of interest
        S = [[np.random.randint(x, x_max, size=M)], 
            [np.random.randint(y, y_max, size=M)],
            [np.zeros(M)], # xdot
            [np.zeros(M)], # ydot
            [np.zeros(M)], # xdotdot
            [np.zeros(M)]] # ydotdot
        S = np.reshape(S, (np.size(S, 0), M)) # reshape to correct dim

    return S
