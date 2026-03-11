import numpy as np


class State:
    ''' represents the robot state at a given time '''
    def __init__(self, t=0.0, x=None, P=None):
        self.t = t
        self.x = np.array(x) if x is not None else np.zeros(3) # state (x, y, theta) at t (posterior)
        self.P = P if P is not None else np.zeros((3, 3)) # covariance matrix at t
        self.innovation = None
        self.Kinnovation = np.zeros(3)
        self.measurement = None
        self.z_mean = None


class Control:
    ''' represents the control input at a given time '''
    def __init__(self, t, v, omega, dt):
        self.t = t
        self.v = v
        self.omega = omega
        self.dt = dt


class Landmark:
    ''' represents a landmark '''
    def __init__(self, id, x=None, stddev=None):
        self.id = int(id)
        self.x = np.array(x) if x is not None else np.zeros(2) # position (x, y)
        self.stddev = np.array(stddev) if stddev is not None else np.zeros(2) # standard deviation (x, y)


class Measurement:
    ''' represents a landmark measurement, used for both actual measurements and estimated measurements '''
    def __init__(self, t, id, range, bearing):
        self.t = t
        self.z = np.array([range, bearing]) # measurement (range, bearing) at t
        self.id = int(id)
