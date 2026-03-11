import math

from models import State, Landmark, Measurement
from motion_model import normalize_angle


def measurement_model(state: State, landmark: Landmark) -> Measurement:
    ''' range and bearing measurement model, given a state and a landmark '''
    range = math.sqrt((landmark.x[0] - state.x[0])**2 + (landmark.x[1] - state.x[1])**2)
    bearing = math.atan2(landmark.x[1] - state.x[1], landmark.x[0] - state.x[0]) - state.x[2]
    bearing = normalize_angle(bearing)
    measurement = Measurement(state.t, landmark.id, range, bearing)
    return measurement


def get_xy_measurement(state: State, measurement: Measurement):
    ''' convert polar measurement to Cartesian coordinates '''
    x = state.x[0] + measurement.z[0] * math.cos(measurement.z[1] + state.x[2])
    y = state.x[1] + measurement.z[0] * math.sin(measurement.z[1] + state.x[2])
    return x, y
