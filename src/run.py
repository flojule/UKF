import math

import numpy as np
import matplotlib.pyplot as plt

from data import import_data
from models import State, Control
from motion_model import dead_reckoning, normalize_angle
from measurement_model import measurement_model, get_xy_measurement
from ukf import ukf, compute_weights
from plot import ds_Plot, plot_state, plot_state_errors, plot_innovation


def main():
    DEBUG = False

    i = 0 # dataset index
    partA = True
    partB = True

    export = False # export resampled data files
    use_resampled = False # use resampled data files
    dt = 1/20 # resampling timestep

    if use_resampled:
        dt = None
        export = False
        i = str(i) + "_RS"

    ds_Control, ds_GroundTruth, ds_Landmark_GroundTruth, ds_Measurement = import_data(i, dt=dt, export=export)
    colors = ["Blue", "Green", "Orange", "Cyan", "Pink", "Brown", "Purple", "Gray", "Red", "Magenta"] # for plotting multiple datasets

    if DEBUG:
        k = 2000 # reduce size for testing
        ds_Control = ds_Control[0:k]
        ds_GroundTruth = ds_GroundTruth[0:k+1]

    # ------------- Part A -------------
    if partA:
        # Q2:
        _Control = np.array([[0.5, 0, 1.0],
                            [0.0, -1/(2*math.pi), 1.0],
                            [0.5, 0, 1.0],
                            [0.0, 1/(2*math.pi), 1.0],
                            [0.5, 0, 1.0]]) # v, omega, dt
        _time = np.cumsum(_Control[:, 2]) - _Control[0, 2] # compute time stamps at the start of each control
        _Control = np.column_stack((_time, _Control)) # add time stamps to control (first column)
        _Control = [Control(row[0], row[1], row[2], row[3]) for row in _Control]

        DR_State = dead_reckoning(State(x=(0, 0, 0)), _Control) # start at (x, y, theta) = (0, 0, 0)
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_state(fig, ax, DR_State, label="Dead reckoning", color=colors[1], vectors=True)
        ax.set_title("Robot trajectory")
        ax.legend()
        ax.axis('equal')
        ax.grid(True)
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')

        # Q3:
        DR_State = dead_reckoning(ds_GroundTruth[0], ds_Control) # starting at first ground truth state
        labels = ["Ground truth", "Dead reckoning"]
        ds_Plot([ds_GroundTruth, DR_State], ds_Landmark_GroundTruth, labels, colors)
        plot_state_errors(ds_GroundTruth, [DR_State], labels, colors)

        # Q6:
        test_State = [State(x=[2.0, 3.0, 0.0]), State(x=[0.0, 3.0, 0.0]), State(x=[1.0, -2.0, 0.0])]
        test_LM_id = [6, 13, 17]

        for j, state in enumerate(test_State):
            landmark = [landmark_ for landmark_ in ds_Landmark_GroundTruth if landmark_.id == test_LM_id[j]][0] # extract single landmark from list
            measurement = measurement_model(state, landmark)
            x, y = get_xy_measurement(state, measurement)
            print(f"\nQuestion 6:\nRobot position: \n(x, y, theta) = ({state.x[0]:.3f} m, {state.x[1]:.3f} m, {state.x[2]:.3f} rad)")
            print(f"Landmark {measurement.id} predicted at: \n(range, bearing) = ({measurement.z[0]:.3f} m, {measurement.z[1]:.3f} rad)")
            print(f"(x, y) = ({x:.3f} m, {y:.3f} m)")
            print(f"Landmark {landmark.id} ground truth at: \n(x, y) = ({landmark.x[0]:.3f} m, {landmark.x[1]:.3f} m)")
            break

    # ------------- Part B -------------
    if partB:

        # q9: parameter exploration

        # ----- Noise parameters -----
        P_0 = np.diag([1e-6, 1e-6, 1e-6]) # initial covariance
        Q_0 = np.diag([1e-6, 1e-6, 3.6e-5]) # process noise covariance
        R_0 = np.diag([1e-2, 1e-2]) # measurement noise covariance
        # P_ = [P_0, P_0*1e2, P_0*1e4]
        P_ = [P_0, P_0, P_0]
        # Q_ = [Q_0, Q_0*1e-2, Q_0*1e2]
        Q_ = [Q_0, Q_0, Q_0]
        R_ = [R_0, R_0*1e-2, R_0*1e2]
        # R_ = [R_0, R_0, R_0]
        # ----- UKF parameters -----
        alpha_ = [0.1, 0.1, 0.1] # spread of sigma points
        kappa, beta = 0.0, 2.0

        # ----- Starting point assumptions -----
        state_00 = ds_GroundTruth[0] # initial state from ground truth
        stddev_01x0 = P_[1][0, 0]**0.5
        stddev_01x1 = P_[1][1, 1]**0.5
        stddev_01x2 = P_[1][2, 2]**0.5
        state_01 = State(x=[state_00.x[0] + stddev_01x0,
                            state_00.x[1] + stddev_01x1,
                            normalize_angle(state_00.x[2] + stddev_01x2)])
        stddev_02x0 = P_[2][0, 0]**0.5
        stddev_02x1 = P_[2][1, 1]**0.5
        stddev_02x2 = P_[2][2, 2]**0.5
        state_02 = State(x=[state_00.x[0] + stddev_02x0,
                            state_00.x[1] + stddev_02x1,
                            normalize_angle(state_00.x[2] + stddev_02x2)])
        state_0_ = [state_00, state_00, state_00] # update to [state_00, state_01, state_02] to test different starting points

        # Pre-index landmarks and measurements for O(1) lookup in the UKF loop
        landmark_dict = {lm.id: lm for lm in ds_Landmark_GroundTruth}
        meas_by_time = {}
        for m in ds_Measurement:
            meas_by_time.setdefault(round(m.t, 2), []).append(m)

        ds_Predicted = [] # add list of predicted states to this list
        labels = ["Ground truth"] # add labels with UKF ID to this list

        for (P, Q, R, alpha, state_0) in zip(P_, Q_, R_, alpha_, state_0_): # grid search over parameters
            init_state = State(t=state_0.t, x=state_0.x.copy(), P=P) # avoid mutating shared state_0 object
            weights_mean, weights_cov = compute_weights(3, alpha, kappa, beta) # n=3 for (x, y, theta)

            # q7 --- UKF loop ---
            UKF_State = []
            UKF_State.append(init_state)
            for control in ds_Control: # first measurement happens at t=11.12
                prior = UKF_State[-1]
                measurements = meas_by_time.get(round(control.t, 2), []) # O(1) lookup
                posterior = ukf(prior, control, measurements, landmark_dict, Q, R, weights_mean, weights_cov, alpha, kappa, beta) # accounts for no measurements if list is empty
                UKF_State.append(posterior)

            if DEBUG:
                print(f"\n Final state ground truth / estimate")
                print(f"(x, y, theta) = ({ds_GroundTruth[-1].x[0]:.3f} m, {ds_GroundTruth[-1].x[1]:.3f} m, {ds_GroundTruth[-1].x[2]:.3f} rad)")
                print(f"(x, y, theta) = ({UKF_State[-1].x[0]:.3f} m, {UKF_State[-1].x[1]:.3f} m, {UKF_State[-1].x[2]:.3f} rad)")

            # --- PLOTTING ---
            ds_Predicted.append(UKF_State)
            labels.append(f"UKF P:{P[0,0]/P_0[0,0]:.0e}, Q:{Q[0,0]/Q_0[0,0]:.0e}, R:{R[0,0]/R_0[0,0]:.0e}, alpha:{alpha}")

        # q8
        labels_UKF_DR = [labels[0], "UKF", "Dead reckoning"]
        DR_State = dead_reckoning(ds_GroundTruth[0], ds_Control)
        ds_Plot([ds_GroundTruth, ds_Predicted[0], DR_State],
                ds_Landmark_GroundTruth, labels_UKF_DR, colors)
        plot_state_errors(ds_GroundTruth, [ds_Predicted[0], DR_State], labels_UKF_DR, colors)

        # q9
        ds_States = [ds_GroundTruth]
        for state in ds_Predicted:
            ds_States.append(state)
        ds_Plot(ds_States, ds_Landmark_GroundTruth, labels, colors)
        plot_state_errors(ds_GroundTruth, ds_Predicted, labels, colors)
        plot_innovation(ds_Predicted, labels, colors)

        # compute distance and bearing average error for ds_Predicted[0]
        distance_errors = []
        bearing_errors = []
        myUKF = ds_Predicted[0]
        for t, state in enumerate(ds_GroundTruth):
            distance_errors.append(np.linalg.norm(np.array(state.x[:2]) - np.array(myUKF[t].x[:2])))
            bearing_errors.append(abs(normalize_angle(state.x[2] - myUKF[t].x[2])))

        avg_distance_error = np.mean(distance_errors)
        avg_bearing_error = np.mean(bearing_errors) # bearing_errors are absolute values in [0, pi], plain mean is correct

        print(f"Average distance error: {avg_distance_error:.3f} m")
        print(f"Average bearing error: {avg_bearing_error:.3f} rad")

    plt.show()


if __name__ == "__main__":
    main()
