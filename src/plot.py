import math

import numpy as np
import matplotlib.pyplot as plt

from models import State, Landmark
from motion_model import normalize_angle


def ds_Plot(ds_States, ds_Landmark_GroundTruth, labels, colors):
    ''' plot multiple datasets, ds_States is a list of states '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    for state, label, color in zip(ds_States, labels, colors):
        plot_state(fig, ax1, state, label, color)
        plot_state(fig, ax2, state, label, color, vectors=True, uncertainty=True)
    for ax in [ax1, ax2]:
        if len(ds_Landmark_GroundTruth) > 0:
            plot_landmarks(fig, ax, ds_Landmark_GroundTruth)
        ax.axis('equal')
        ax.grid(True)
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')

    ax1.set_title("Robot trajectories")
    ax2.set_title("Robot trajectories and orientations, zoomed in near start")
    ax1.legend(loc='lower left', bbox_to_anchor=(0.0, 1.05), ncol=len(labels))
    zoom = 1.0 # zoom in around start
    ax2.set_xlim(ds_States[0][0].x[0]-zoom, ds_States[0][0].x[0]+zoom)
    ax2.set_ylim(ds_States[0][0].x[1]-zoom, ds_States[0][0].x[1]+zoom)


def plot_state(fig, ax, ds_State, label, color, vectors=False, uncertainty=False):
    ''' plot robot trajectory and orientation for one dataset '''
    ds_x = [state.x[0] for state in ds_State]
    ds_y = [state.x[1] for state in ds_State]
    ax.plot(ds_x, ds_y, label=label, color=color)

    if vectors:
        ds_theta = [state.x[2] for state in ds_State]
        m = 20 # quiver spacing
        ds_x_q = ds_x[::m] if len(ds_x) > m else ds_x
        ds_y_q = ds_y[::m] if len(ds_y) > m else ds_y
        ds_theta_q = ds_theta[::m] if len(ds_theta) > m else ds_theta
        ax.scatter(ds_x[0], ds_y[0], marker='x', label=f'Start {label}', color=color)
        ax.scatter(ds_x[-1], ds_y[-1], marker='*', label=f'End {label}', color=color)
        ax.quiver(ds_x_q, ds_y_q, np.cos(ds_theta_q), np.sin(ds_theta_q),
                  scale=15, width=0.002, label=f'Orientation {label}', alpha=0.5, color=color)
    if uncertainty:
        xerr = [2*math.sqrt(state.P[0, 0]) for state in ds_State] # 2*stddev in x direction
        yerr = [2*math.sqrt(state.P[1, 1]) for state in ds_State] # 2*stddev in y direction
        ax.errorbar(ds_x, ds_y, xerr=xerr, yerr=yerr, alpha=0.1, color=color, elinewidth=1)

    ax.axis('equal')
    ax.grid(True)
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_xlim(min(ds_x)-0.05, max(ds_x)+0.05)
    ax.set_ylim(min(ds_y)-0.05, max(ds_y)+0.05)


def plot_landmarks(fig, ax, ds_Landmarks):
    ''' plot landmarks '''
    x = [landmark.x[0] for landmark in ds_Landmarks]
    y = [landmark.x[1] for landmark in ds_Landmarks]
    ax.scatter(x, y, marker='o', color='black')
    for landmark in ds_Landmarks:
        ax.text(landmark.x[0], landmark.x[1], f'LM{landmark.id}')


def plot_measurement_predictions(fig, axes, ds_State, ds_Landmark_GroundTruth):
    ''' plot landmark measurements predictions '''
    for i, state in enumerate(ds_State):
        ax = axes[i]
        ax.set_title(f"(x, y, theta) = ({state.x[0]:.2f} m, {state.x[1]:.2f} m, {state.x[2]:.2f} rad)")
        for landmark in state.LM_est:
            ax.scatter(landmark.x[0], landmark.x[1], marker='x', color='red')
            ax.text(landmark.x[0], landmark.x[1], f'LM prediction {landmark.id}')
            for landmark_gt in ds_Landmark_GroundTruth:
                if landmark.id == landmark_gt.id:
                    ax.scatter(landmark_gt.x[0], landmark_gt.x[1], marker='o', color='black')
                    ax.text(landmark_gt.x[0], landmark_gt.x[1], f'LM ground truth {landmark_gt.id}')
        ax.scatter(state.x[0], state.x[1], marker='*', color='green', label='Robot position')
        ax.quiver(state.x[0], state.x[1], math.cos(state.x[2]), math.sin(state.x[2]),
                  width=0.005, scale=20, color='green', label='Robot orientation')
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.legend()
        ax.axis('equal')
        ax.grid(True)


def plot_state_errors(ds_GroundTruth, list_States, labels, colors):
    ''' plot state, state error and correction (K*innovation) for a given list of states '''
    fig, ax = plt.subplots(3, 3, figsize=(20, 10), sharex=True)
    labels_state = ['X position (m)', 'Y position (m)', 'Orientation (rad)']
    labels_error = ['X position error (m)', 'Y position error (m)', 'Orientation error (rad)']
    labels_correction = ['Correction x (m)', 'Correction y (m)', 'Correction theta (rad)']

    alpha = 0.6 if len(list_States) > 1 else 1.0

    T = min(len(ds_GroundTruth), len(list_States[0]))
    ds_GroundTruth = ds_GroundTruth[0:T]
    list_States = [states[0:T] for states in list_States] # truncate all states to same length
    UKF_State = list_States[0]
    t = [UKF_State[i].t for i in range(len(UKF_State))] # time vector

    for i in range(3):
        state_gt = [ds_GroundTruth[j].x[i] for j in range(T)]
        ax[i, 0].plot(t, state_gt, label=labels[0], color=colors[0])

        for s, states in enumerate(list_States): # plot all states
            states_ = [states[j].x[i] for j in range(T)]
            ax[i, 0].plot(t, states_, label=labels[s+1], color=colors[s+1], alpha=alpha)

            errors = np.array([states[k].x - ds_GroundTruth[k].x for k in range(T)]) # T x 3, computed once per state
            errors[:, 2] = (errors[:, 2] + np.pi) % (2 * np.pi) - np.pi # normalize angle error (vectorized)
            ax[i, 1].plot(t, errors[:, i], label=labels[s+1], color=colors[s+1], alpha=alpha)
            ax[i, 2].plot([state.t for state in states if state.measurement is not None],
                          [state.Kinnovation[i] for state in states if state.measurement is not None],
                          label=labels[s+1], alpha=alpha, color=colors[s+1])

        ax[i, 0].set_ylabel(labels_state[i])
        ax[i, 0].grid(True)
        ax[i, 1].set_ylabel(labels_error[i])
        ax[i, 1].grid(True)
        ax[i, 2].set_ylabel(labels_correction[i])
        ax[i, 2].grid(True)

    ax[2, 0].set_xlabel('Time (s)')
    ax[2, 1].set_xlabel('Time (s)')
    ax[2, 2].set_xlabel('Time (s)')
    ax[0, 0].set_title("Robot states")
    ax[0, 1].set_title("State error to ground truth")
    ax[0, 2].set_title("State correction (K*Innovation)")
    ax[0, 0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.1), ncol=len(labels))


def plot_innovation(list_States, labels, colors):
    ''' plot measurement innovations for UKF '''
    fig, ax = plt.subplots(2, 2, figsize=(20, 10), sharex=True)
    labels_measurement = ['Range (m)', 'Bearing (rad)']
    labels_error = ['Range innovation (m)', 'Bearing innovation (rad)']
    labels = list(labels) # avoid mutating caller's list
    labels[0] = "Expected"

    alpha = 0.6 if len(list_States) > 1 else 1.0

    for i in range(2):
        for s, states in enumerate(list_States):
            measurements = [state.measurement.z for state in states if state.measurement is not None]
            predicted_measurements = [state.z_mean for state in states if state.measurement is not None]
            t = [state.t for state in states if state.measurement is not None]
            if s == 0: # only plot ground truth once
                ax[i, 0].plot(t, [m[i] for m in measurements],
                              label=labels[0], linestyle='None', marker='+', markersize=3, color=colors[0])
            ax[i, 0].plot(t, [p[i] for p in predicted_measurements],
                          label=labels[s+1], linestyle='None', marker='x', markersize=3, color=colors[s+1], alpha=alpha)

            ax[i, 1].plot(t, [m[i]-p[i] for m, p in zip(measurements, predicted_measurements)],
                          label=labels[s+1], alpha=alpha, color=colors[s+1])
        ax[i, 0].set_ylabel(labels_measurement[i])
        ax[i, 0].grid(True)
        ax[i, 1].set_ylabel(labels_error[i])
        ax[i, 1].grid(True)

    ax[1, 0].set_xlabel('Time (s)')
    ax[1, 1].set_xlabel('Time (s)')
    ax[0, 0].set_title("Landmark measurements")
    ax[0, 1].set_title("Innovations")
    ax[0, 0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.1), ncol=len(labels))
