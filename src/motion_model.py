import math

import numpy as np

from models import State, Control


def normalize_angle(angle): # shift to ]-pi, pi]
    return (angle + math.pi) % (2 * math.pi) - math.pi


def motion_model(prior: State, control: Control) -> State:
    ''' computes next state given a prior state and a control input '''
    x = np.zeros(prior.x.shape)
    if control.omega == 0:
        x[0] = prior.x[0] + control.v * math.cos(prior.x[2]) * control.dt
        x[1] = prior.x[1] + control.v * math.sin(prior.x[2]) * control.dt
        x[2] = prior.x[2]
    else:
        x[0] = prior.x[0] + (control.v / control.omega) * (math.sin(prior.x[2] + control.omega * control.dt) - math.sin(prior.x[2]))
        x[1] = prior.x[1] + (control.v / control.omega) * (math.cos(prior.x[2]) - math.cos(prior.x[2] + control.omega * control.dt))
        x[2] = prior.x[2] + control.omega * control.dt
    x[-1] = normalize_angle(x[-1])
    posterior = State(control.t, x=x)
    return posterior


def dead_reckoning(state_0: State, ds_Control: list) -> list:
    ''' dead reckoning implementation, loops motion model, returns list of states '''
    DR_State = []
    DR_State.append(state_0)
    for control in ds_Control:
        prior = DR_State[-1]
        posterior = motion_model(prior, control)
        DR_State.append(posterior)
    return DR_State
