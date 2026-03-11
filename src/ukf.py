import math

import numpy as np

from models import State, Control, Measurement, Landmark
from motion_model import motion_model, normalize_angle
from measurement_model import measurement_model

DEBUG = False  # set to True for verbose per-step output


def ukf(prior: State, control: Control, measurements: list, ds_Landmark_GroundTruth: list,
        Q, R, weights_mean, weights_cov, alpha, kappa, beta) -> State:
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
                landmark = [landmark_ for landmark_ in ds_Landmark_GroundTruth if landmark_.id == measurement.id][0] # landmark corresponding to measurement
                Z = [measurement_model(sp, landmark) for sp in Y] # Z are the estimated measurement sigma points (list of Measurement objects)
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
                    from measurement_model import get_xy_measurement
                    x_dbg, y_dbg = get_xy_measurement(State(x=y_mean), measurement)
                    print(f"landmark estimated position       (x, y): ({x_dbg:.3f} m, {y_dbg:.3f} m)")
                    print(f"predicted measurement   (range, bearing): ({z_mean[0]:.3f} m, {z_mean[1]:.3f} rad)")
                    print(f"actual measurement      (range, bearing): ({measurement.z[0]:.3f} m, {measurement.z[1]:.3f} rad)")
                    print(f"innovation:                               ({innovation[0]:.3f} m, {innovation[1]:.3f} rad)")
                    print(f"Kalman gain * innovation:                 {K @ innovation}")

                y_mean = posterior.x # for next measurement, use posterior as prior
                Pyy = posterior.P

    if posterior is None: # no measurements, return prediction as posterior
        posterior = State(control.t, x=y_mean, P=Pyy)

    return posterior


def generate_sigma_points(prior: State, alpha, kappa, beta) -> np.ndarray:
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
