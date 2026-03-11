import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    global DEBUG
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
        
        DR_State = dead_reckoning(State(x=(0,0,0)), _Control) # start at (x, y, theta) = (0, 0, 0)
        fig, ax = plt.subplots(figsize=(10,5))
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
        state_00 = ds_GroundTruth[0] # initial state from ground truth, add noise
        stddev_01x0 = P_[1][0,0]**0.5
        stddev_01x1 = P_[1][1,1]**0.5
        stddev_01x2 = P_[1][2,2]**0.5
        state_01 = State(x=[state_00.x[0] + stddev_01x0, 
                            state_00.x[1] + stddev_01x1, 
                            normalize_angle(state_00.x[2] + stddev_01x2)])
        stddev_02x0 = P_[2][0,0]**0.5
        stddev_02x1 = P_[2][1,1]**0.5
        stddev_02x2 = P_[2][2,2]**0.5
        state_02 = State(x=[state_00.x[0] + stddev_02x0, 
                            state_00.x[1] + stddev_02x1, 
                            normalize_angle(state_00.x[2] + stddev_02x2)])
        state_0_ = [state_00, state_00, state_00] # update to [state_00, state_01, state_02] to test different starting points

        ds_Predicted = [] # add list of predicted states to this list
        labels = ["Ground truth"] # add labels with UKF ID to this list

        for (P, Q, R, alpha, state_0) in zip(P_, Q_, R_, alpha_, state_0_): # grid search over parameters
            state_0.P = P
            weights_mean, weights_cov = compute_weights(3, alpha, kappa, beta) # n=3 for (x, y, theta)

            # q7 --- UKF loop ---
            UKF_State = []
            UKF_State.append(state_0)
            for control in ds_Control: # first measurement happens at t=11.12
                prior = UKF_State[-1]
                measurements = [measurement for measurement in ds_Measurement if abs(measurement.t - control.t) < 0.01] # find all measurements at this time step
                posterior = ukf(prior, control, measurements, ds_Landmark_GroundTruth, Q, R, weights_mean, weights_cov, alpha, kappa, beta) # accounts for no measurements if list is empty
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

        # compute distance and bearing average error for ds_Predicted[0]:
        distance_errors = []
        bearing_errors = []
        myUKF = ds_Predicted[0]
        for t, state in enumerate(ds_GroundTruth):
            distance_errors.append(np.linalg.norm(np.array(state.x[:2]) - np.array(myUKF[t].x[:2])))
            bearing_errors.append(abs(normalize_angle(state.x[2] - myUKF[t].x[2])))

        avg_distance_error = np.mean(distance_errors)
        avg_bearing_error = math.atan2(np.mean(np.sin(bearing_errors)), np.mean(np.cos(bearing_errors)))

        print(f"Average distance error: {avg_distance_error:.3f} m")
        print(f"Average bearing error: {avg_bearing_error:.3f} rad")

    plt.show()


### DATA IMPORT ###

def import_data(i, dt=None, export=False, robot_id=False):
    ds_Control_raw = import_dat(f'ds{i}/ds{i}_Control.dat')
    ds_GroundTruth_raw = import_dat(f'ds{i}/ds{i}_Groundtruth.dat')
    ds_Landmark_GroundTruth_raw = import_dat(f'ds{i}/ds{i}_Landmark_Groundtruth.dat') 
    ds_Measurement_raw = import_dat(f'ds{i}/ds{i}_Measurement.dat') 
    ds_Barcodes = import_dat(f'ds{i}/ds{i}_Barcodes.dat')  

    if dt is not None:
        ds_Control_raw, ds_Measurement_raw, ds_GroundTruth_raw = resample_data(dt, ds_Control_raw, ds_Measurement_raw, ds_GroundTruth_raw)
        ds_Control = [Control(control[0], control[1], control[2], dt) for control in ds_Control_raw]
    else:
        ds_Control = []
        for j, control in enumerate(ds_Control_raw): # convert into list of Control objects
            dt =  ds_Control_raw[j+1][0] - control[0] if j < len(ds_Control_raw) - 1 else control[0] - ds_Control_raw[j-1][0] # compute dt, exception for last element
            control = Control(control[0], control[1], control[2], dt)
            ds_Control.append(control)
    
    ds_Measurement = [Measurement(row[0], row[1], row[2], row[3]) for row in ds_Measurement_raw] # convert into list of Measurement objects
    ds_GroundTruth = [State(t=row[0], x=[row[1], row[2], row[3]]) for row in ds_GroundTruth_raw] # convert into list of State objects
    ds_Landmark_GroundTruth = [Landmark(row[0], [row[1], row[2]], [row[3], row[4]]) for row in ds_Landmark_GroundTruth_raw] # convert into list of Landmark objects
    
    for measurement in ds_Measurement:
        for j in range(len(ds_Barcodes)):
            if ds_Barcodes[j][1] == measurement.id:
                measurement.id = int(ds_Barcodes[j][0]) # replace barcode with id
                break

    if robot_id:
        robot_id = get_robot_id(ds_Measurement, ds_Barcodes)

    if export == True and dt is not None: # export resampled data files
        i = str(i) + "_RS" # resampled
        fn_control = f'ds{i}/ds{i}_Control.dat'
        fn_measurement = f'ds{i}/ds{i}_Measurement.dat'
        fn_groundtruth = f'ds{i}/ds{i}_Groundtruth.dat'
        export_dat(fn_control, ds_Control_raw)
        export_dat(fn_measurement, ds_Measurement_raw)
        export_dat(fn_groundtruth, ds_GroundTruth_raw)

        fn_barecodes = f'ds{i}/ds{i}_Barcodes.dat' # adding to have complete ds_RS folder
        fn_landmark = f'ds{i}/ds{i}_Landmark_Groundtruth.dat'
        export_dat(fn_barecodes, ds_Barcodes)
        export_dat(fn_landmark, ds_Landmark_GroundTruth_raw)


    return ds_Control, ds_GroundTruth, ds_Landmark_GroundTruth, ds_Measurement

def import_dat(filename):
    with open(filename, 'r') as file:
        return np.loadtxt(file)

def export_dat(filename, data):
    with open(f'{filename}', 'w') as file:
        np.savetxt(file, data, fmt='%.3f')

def resample_data(dt, ds_Control_raw, ds_Measurement_raw, ds_GroundTruth_raw):
    ''' resample data to fixed timestep dt 
    control and GroundTruth linearly interpolated to fixed timestep
    measurements rounded to nearest timestep
    '''

    min_time = ds_GroundTruth_raw[0, 0]
    ds_Control_raw[:, 0] -= min_time
    ds_GroundTruth_raw[:, 0] -= min_time
    ds_Measurement_raw[:, 0] -= min_time

    max_time = ds_GroundTruth_raw[-1, 0]
    t_grid = np.arange(0, max_time + dt/2, dt)

    ds_Control = np.zeros((t_grid.shape[0], ds_Control_raw.shape[1]))
    ds_Control[:, 0] = t_grid
    for i in range(1, ds_Control_raw.shape[1]):
        ds_Control[:, i] = np.interp(t_grid, ds_Control_raw[:, 0], ds_Control_raw[:, i])

    ds_GroundTruth = np.zeros((t_grid.shape[0], ds_GroundTruth_raw.shape[1]))
    ds_GroundTruth[:, 0] = t_grid
    for i in range(1, ds_GroundTruth_raw.shape[1]):
        ds_GroundTruth[:, i] = np.interp(t_grid, ds_GroundTruth_raw[:, 0], ds_GroundTruth_raw[:, i])

    ds_Measurement = ds_Measurement_raw.copy()
    ds_Measurement[:, 0] = np.round(ds_Measurement[:, 0] / dt) * dt
    ds_Measurement = ds_Measurement[ds_Measurement[:, 0] <= max_time]

    return ds_Control, ds_Measurement, ds_GroundTruth
    
def get_robot_id(ds_Measurement, ds_Barcodes):
    ''' find robot id by checking which barcode id is not in the measurements '''
    ids = set()
    ids.update(measurement.id for measurement in ds_Measurement)
    robot_id_possible = set()
    robot_id_possible.add([int(i) for i in ds_Barcodes[:, 0] if i not in ids][0]) # find the robot id in the barcodes
    if len(robot_id_possible) != 1:
        print(f"More than one possible robot: {robot_id_possible}")
        return None
    else:
        robot_id = int(robot_id_possible.pop())
        print(f"Robot id is {robot_id}")
        return robot_id

### MOTION MODEL ###

def motion_model(prior: "State", control: "Control"): 
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

def dead_reckoning(state_0: "State", ds_Control: list["Control"]):
    ''' dead reckoning implementation, loops motion model, returns list of states '''
    DR_State = []
    DR_State.append(state_0)
    for control in ds_Control:
        prior = DR_State[-1]
        posterior = motion_model(prior, control)
        DR_State.append(posterior)
    return DR_State

def normalize_angle(angle): # shift to ]-pi, pi]
    return (angle + math.pi) % (2 * math.pi) - math.pi 

### MEASUREMENT MODEL ###

def measurement_model(state: "State", landmark: "Landmark"):
    ''' range and bearing measurement model, given a state and a landmark '''
    range = math.sqrt((landmark.x[0] - state.x[0])**2 + (landmark.x[1] - state.x[1])**2)
    bearing = math.atan2(landmark.x[1] - state.x[1], landmark.x[0] - state.x[0]) - state.x[2]
    bearing = normalize_angle(bearing)
    measurement = Measurement(state.t, landmark.id, range, bearing)  
    return measurement

def get_xy_measurement(state: "State", measurement: "Measurement"):
    ''' convert polar measurement to Cartesian coordinates '''
    x = state.x[0] + measurement.z[0] * math.cos(measurement.z[1] + state.x[2])
    y = state.x[1] + measurement.z[0] * math.sin(measurement.z[1] + state.x[2])
    return x, y

### UKF ###

def ukf(prior: "State", control: "Control", measurements: list["Measurement"], ds_Landmark_GroundTruth: list["Landmark"], \
        Q, R, weights_mean, weights_cov, alpha, kappa, beta):
    ''' Unscented Kalman Filter implementation '''
    # Prediction step
    X_np = generate_sigma_points(prior, alpha, kappa, beta) # X are the sigma points, numpy array
    Y = [motion_model(State(x=sp), control) for sp in X_np] # Y are the propagated sigma points
    Y_np = np.array([sp.x for sp in Y]) # convert object to numpy array
    y_mean, Pyy = compute_mean_and_covariance(Y_np, Q, weights_mean, weights_cov) # same weights for mean and covariance
    
    # Correction step
    posterior = None
    if len(measurements) > 0: 
        for measurement in measurements:
            Y_np = generate_sigma_points(State(x=y_mean, P=Pyy), alpha, kappa, beta) # new sigma points around predicted mean
            Y = [State(x=sp, P=Pyy) for sp in Y_np]
            if measurement.id in [landmark_.id for landmark_ in ds_Landmark_GroundTruth]: # if measurement id not in landmark ground truth, return prediction as posterior
                landmark = [landmark_ for landmark_ in ds_Landmark_GroundTruth if landmark_.id == measurement.id][0] #  landmark corresponding to measurement
                Z = [measurement_model(sp, landmark) for sp in Y] # Z are the estimated measurement sigma points (list of measurementobjects)
                Z_np = np.array([m_est.z for m_est in Z]) # extract the measurement arrays from the Measurement objects
                
                z_mean, Pzz = compute_mean_and_covariance(Z_np, R, weights_mean, weights_cov) # same weights for mean and covariance

                Pyz = compute_cross_covariance(Y_np, y_mean, Z_np, z_mean, weights_cov) # cross covariance
                K = Pyz @ np.linalg.inv(Pzz) # Kalman gain
                innovation = measurement.z - z_mean # measurement innovation
                innovation[-1] = normalize_angle(innovation[-1])
                x = y_mean + K @ innovation
                x[-1] = normalize_angle(x[-1])
                P = Pyy - K @ Pzz @ K.T
                posterior = State(control.t, x=x, P=P)
                posterior.innovation = innovation # store innovation for plotting later
                posterior.Kinnovation += K @ innovation # store Kalman gain * innovation for plotting later
                posterior.measurement = measurement # store (last) measurement for plotting later
                posterior.z_mean = z_mean

                if DEBUG:
                    print(f"\n--- ukf loop ---")
                    print(f"\ntime: {control.t:.3f} s")
                    print(f"control * dt:               ({control.v*control.dt:.3f}m, {control.omega*control.dt:.3f} rad)")
                    print(f"prior (x, y, theta):        ({prior.x[0]:.4f} m, {prior.x[1]:.4f} m, {prior.x[2]:.4f} rad)")
                    print(f"motion model (x, y, theta): ({y_mean[0]:.4f} m, {y_mean[1]:.4f} m, {y_mean[2]:.4f} rad)")
                    print(f"posterior (x, y, theta):    ({posterior.x[0]:.4f} m, {posterior.x[1]:.4f} m, {posterior.x[2]:.4f} rad)")
                    print(f"Y_np: \n{Y_np}")
                    print(f"weights mean and cov: \n{weights_mean}\n{weights_cov}")
                    print(f"Pyy: \n{Pyy}")
                    print(f"Z_np: \n{Z_np}")
                    print(f"Pzz: \n{Pzz}")
                    print(f"weights sum and mean: {np.sum(weights_mean)}\n{weights_mean}")
                    print(f"landmark id: {landmark.id}")
                    print(f"landmark position                 (x, y): ({landmark.x[0]:.3f} m, {landmark.x[1]:.3f} m)")
                    x,y = get_xy_measurement(State(x=y_mean), measurement)
                    print(f"landmark estimated position       (x, y): ({x:.3f} m, {y:.3f} m)")
                    print(f"predicted measurement   (range, bearing): ({z_mean[0]:.3f} m, {z_mean[1]:.3f} rad)")
                    print(f"actual measurement      (range, bearing): ({measurement.z[0]:.3f} m, {measurement.z[1]:.3f} rad)")
                    print(f"innovation:                               ({innovation[0]:.3f} m, {innovation[1]:.3f} rad)")
                    print(f"Kalman gain * innovation:                 {K @ innovation}")

                y_mean = posterior.x # for next measurement, use posterior as prior
                Pyy = posterior.P

    if posterior is None: # no measurements, return prediction as posterior
        posterior = State(control.t, x=y_mean, P=Pyy)

    return posterior

def generate_sigma_points(prior: "State", alpha, kappa, beta):
    ''' generate sigma points around prior mean '''
    n = prior.x.shape[0]
    lambda_ = alpha**2 * (n + kappa) - n
    sigma_points = np.zeros((2 * n + 1, n))
    sigma_points[0] = prior.x
    sqrt_matrix = np.linalg.cholesky((n + lambda_) * prior.P)
    for i in range(n):
        sigma_points[i + 1]     = prior.x + sqrt_matrix[:, i]
        sigma_points[i + 1 + n] = prior.x - sqrt_matrix[:, i]
    return sigma_points

def compute_weights(n, alpha, kappa, beta):
    ''' compute weights for mean and covariance '''
    weights_mean = np.zeros(2 * n + 1)
    weights_cov = np.zeros(2 * n + 1)
    lambda_ = alpha**2 * (n + kappa) - n
    weights_mean[0] = lambda_ / (n + lambda_)
    weights_cov[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
    for i in range(1, 2 * n + 1):
        weights_mean[i] = 1 / (2 * (n + lambda_))
        weights_cov[i] = 1 / (2 * (n + lambda_))

    return weights_mean, weights_cov

def compute_mean_and_covariance(Y, Q, weights_mean, weights_cov):
    ''' compute mean and covariance of sigma points Y with additive noise Q '''
    mean = np.sum(weights_mean[:, None] * Y, axis=0)
    mean[-1] = math.atan2(np.sum(weights_mean * np.sin(Y[:, -1])), np.sum(weights_mean * np.cos(Y[:, -1]))) # theta, bearing

    cov = Q.copy()
    for i in range(Y.shape[0]):
        dy = Y[i] - mean
        dy[-1] = normalize_angle(dy[-1])
        cov += weights_cov[i] * np.outer(dy, dy)
    return mean, cov

def compute_cross_covariance(Y, y_mean, Z, z_mean, weights_cov):
    ''' compute cross covariance between state sigma points Y and measurement sigma points Z '''
    cross_cov = np.zeros((Y.shape[1], Z.shape[1]))
    for i in range(Y.shape[0]):
        dy = Y[i] - y_mean
        dz = Z[i] - z_mean
        dy[-1] = normalize_angle(dy[-1])
        dz[-1] = normalize_angle(dz[-1])
        cross_cov += weights_cov[i] * np.outer(dy, dz)
    return cross_cov

### PLOTTING ###

def ds_Plot(ds_States, ds_Landmark_GroundTruth, labels, colors):
    ''' plot multiple datasets, ds_States is a list of states '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
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
        xerr = [2*math.sqrt(state.P[0,0]) for state in ds_State] # 2*stddev in x direction
        yerr = [2*math.sqrt(state.P[1,1]) for state in ds_State] # 2*stddev in y direction
        ax.errorbar(ds_x, ds_y, xerr=xerr, yerr=yerr, alpha=0.1, color=color, elinewidth=1)

    ax.axis('equal')
    ax.grid(True)
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_xlim(min(ds_x)-0.05, max(ds_x)+0.05)
    ax.set_ylim(min(ds_y)-0.05, max(ds_y)+0.05)
        

def plot_landmarks(fig, ax, ds_Landmarks):
    ''' plot landmarks'''
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
        ax.quiver(state.x[0], state.x[1], math.cos(state.x[2]), math.sin(state.x[2]), width=0.005, scale=20, color='green', label='Robot orientation')
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.legend()
        ax.axis('equal')
        ax.grid(True)


def plot_state_errors(ds_GroundTruth, list_States, labels, colors): 
    ''' plot state, state error and correction (K*innovation) for a given list of states '''
    fig, ax = plt.subplots(3, 3, figsize=(20,10), sharex=True)
    labels_state = ['X position (m)', 'Y position (m)', 'Orientation (rad)']
    labels_error = ['X position error (m)', 'Y position error (m)', 'Orientation error (rad)']
    labels_correction = ['Correction x (m)', 'Correction y (m)', 'Correction theta (rad)']

    if len(list_States) > 1:
        alpha = 0.6
    else:
        alpha = 1.0

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

            errors = np.array([[states[i].x[j] - ds_GroundTruth[i].x[j] for j in range(3)] for i in range(T)])
            if i == 2: # normalize angle error
                errors[:, i] = np.array([normalize_angle(errors[j, i]) for j in range(T)])
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
    ''' plot measurement innovations for UKF'''
    fig, ax = plt.subplots(2, 2, figsize=(20,10), sharex=True)
    labels_measurement = ['Range (m)', 'Bearing (rad)']
    labels_error = ['Range innovation (m)', 'Bearing innovation (rad)']
    labels[0] = "Expected"

    if len(list_States) > 1:
        alpha = 0.6
    else:
        alpha = 1.0

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
            
            ax[i, 1].plot(t, [m[i]-p[i] for m, p in zip(measurements, predicted_measurements)], label=labels[s+1], alpha=alpha, color=colors[s+1])
        ax[i, 0].set_ylabel(labels_measurement[i])
        ax[i, 0].grid(True)
        ax[i, 1].set_ylabel(labels_error[i])
        ax[i, 1].grid(True)

    ax[1, 0].set_xlabel('Time (s)')
    ax[1, 1].set_xlabel('Time (s)')
    ax[0, 0].set_title("Landmark neasurements")
    ax[0, 1].set_title("Innovations")
    ax[0, 0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.1), ncol=len(labels))
    # ax[0, 1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.1))

### DATA STRUCTURES ###

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

if __name__ == "__main__":
    main()
