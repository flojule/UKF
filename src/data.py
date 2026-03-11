import numpy as np
from pathlib import Path

from models import State, Control, Landmark, Measurement

_DATA_DIR = Path(__file__).parent.parent / 'data'


def import_data(i, dt=None, export=False, robot_id=False):
    ds_Control_raw = import_dat(_DATA_DIR / f'ds{i}/ds{i}_Control.dat')
    ds_GroundTruth_raw = import_dat(_DATA_DIR / f'ds{i}/ds{i}_Groundtruth.dat')
    ds_Landmark_GroundTruth_raw = import_dat(_DATA_DIR / f'ds{i}/ds{i}_Landmark_Groundtruth.dat')
    ds_Measurement_raw = import_dat(_DATA_DIR / f'ds{i}/ds{i}_Measurement.dat')
    ds_Barcodes = import_dat(_DATA_DIR / f'ds{i}/ds{i}_Barcodes.dat')

    if dt is not None:
        ds_Control_raw, ds_Measurement_raw, ds_GroundTruth_raw = resample_data(dt, ds_Control_raw, ds_Measurement_raw, ds_GroundTruth_raw)
        ds_Control = [Control(control[0], control[1], control[2], dt) for control in ds_Control_raw]
    else:
        ds_Control = []
        for j, control in enumerate(ds_Control_raw): # convert into list of Control objects
            dt = ds_Control_raw[j+1][0] - control[0] if j < len(ds_Control_raw) - 1 else control[0] - ds_Control_raw[j-1][0] # compute dt, exception for last element
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
        (_DATA_DIR / f'ds{i}').mkdir(parents=True, exist_ok=True)
        fn_control = _DATA_DIR / f'ds{i}/ds{i}_Control.dat'
        fn_measurement = _DATA_DIR / f'ds{i}/ds{i}_Measurement.dat'
        fn_groundtruth = _DATA_DIR / f'ds{i}/ds{i}_Groundtruth.dat'
        export_dat(fn_control, ds_Control_raw)
        export_dat(fn_measurement, ds_Measurement_raw)
        export_dat(fn_groundtruth, ds_GroundTruth_raw)

        fn_barecodes = _DATA_DIR / f'ds{i}/ds{i}_Barcodes.dat' # adding to have complete ds_RS folder
        fn_landmark = _DATA_DIR / f'ds{i}/ds{i}_Landmark_Groundtruth.dat'
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
