# Unscented Kalman Filter — Mobile Robot Localization

UKF-based localization for a wheeled robot navigating an environment with known landmarks. Uses wheel odometry for prediction and range/bearing measurements for correction. Dataset: [UTIAS MRCLAM](http://asrl.utias.utoronto.ca/datasets/mrclam/).

See the [full report](report/hw0-florianjule.pdf) for derivations and analysis.

## Approach

The robot state `(x, y, theta)` is tracked using a two-step UKF loop:

**Predict** — propagate the state through the unicycle motion model using sigma points:

- `x += (v/w) * (sin(theta + w*dt) - sin(theta))`
- `y += (v/w) * (cos(theta) - cos(theta + w*dt))`
- `theta += w*dt`

(straight-line special case when `w = 0`)

**Correct** — update state using range and bearing to visible landmarks:

- `range = sqrt((lx - x)^2 + (ly - y)^2)`
- `bearing = atan2(ly - y, lx - x) - theta`

## Results

Tested on MRCLAM ds0. Dead reckoning diverges after ~150 s due to accumulated orientation error. The UKF tracks the ground truth throughout.

| Metric | UKF | Dead reckoning |
|---|---|---|
| Avg. position error | **0.107 m** | diverges |
| Avg. bearing error | **0.049 rad** | diverges |

Noise parameters: `P0 = diag(1e-6, 1e-6, 1e-6)`, `Q0 = diag(1e-6, 1e-6, 3.6e-5)`, `R0 = diag(1e-2, 1e-2)`, `alpha = 0.1`.

## Usage

```bash
pip install numpy matplotlib
python src/run.py
```

Key options at the top of `main()` in [src/run.py](src/run.py):

| Flag | Default | Effect |
|---|---|---|
| `i` | `0` | Dataset (`0` → `data/ds0/`, `1` → `data/ds1/`) |
| `use_resampled` | `False` | Use pre-resampled 50 Hz data from `data/ds0_RS/` |
| `export` | `False` | Generate and save resampled dataset |
| `DEBUG` | `False` | Truncate to 2000 steps for fast iteration |

## Structure

```
src/
├── run.py               # Entry point — main loop (Parts A & B)
├── models.py            # Data classes: State, Control, Landmark, Measurement
├── motion_model.py      # Unicycle kinematics, dead reckoning
├── measurement_model.py # Range/bearing model
├── ukf.py               # UKF, sigma points, weights
├── data.py              # Dataset loading, resampling, export
└── plot.py              # All plotting functions

data/
├── ds0/                 # Raw MRCLAM single-robot dataset
├── ds0_RS/              # ds0 resampled at 50 Hz
└── ds1/                 # Second robot run

report/                  # LaTeX source and compiled PDF
matlab/                  # Original UTIAS MATLAB dataset utilities
submissions/             # Course submission archives
```
