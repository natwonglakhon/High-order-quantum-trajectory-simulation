# High-Order Quantum Trajectory Simulation

A Python codebase for simulating and benchmarking quantum trajectory methods using higher-order stochastic integration schemes. The project reproduces numerical results from research on high-order quantum trajectory reconstruction, comparing five distinct simulation maps via trace-distance error analysis.

---

## Overview

Quantum trajectory simulation involves reconstructing a quantum state conditioned on a continuous measurement record. This project implements and compares five maps of varying order:

| Method | Label | Description |
|---|---|---|
| Itô map | `I` | First-order stochastic integration |
| Rouchon–Ralph map | `RR` | Second-order correction map |
| WWC map | `W` | Higher-order Wiener expansion |
| Robinet map | `RB` | Third-order stochastic map |
| Nearly exact map | `Φ` | High-accuracy reference map |

Error is quantified by the **trace distance** between a simulated conditioned state and the true quantum trajectory.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Dependencies: `numpy==1.26.4`, `scipy==1.11.4`, `matplotlib==3.8.2`

---

## Usage

The simulation runs in two sequential steps.

### Step 1 — Generate True Trajectories

```bash
python True_Data_Generation.py
```

This script evolves quantum states using the Rouchon–Ralph map at a fine time step (`dt = 0.0001`) and produces:

- `psi_true.npy` — true quantum state trajectories (shape: `[r, n_coarse, dim, 1]`)
- `ycg.txt` — equally-weighted coarse-grained measurement records *I*<sub>t</sub>
- `zcg.txt` — time-weighted coarse-grained measurement records *Z*<sub>t</sub>

The coarse-graining window is `Dt = 0.01`, aggregating `N = 100` fine-step records per coarse interval.

### Step 2 — Simulate and Analyse Conditioned Trajectories

```bash
python Trajectory_simulation.py
```

This script:

1. Loads the coarse-grained records from Step 1
2. Evolves all five maps at the coarse time step (`Dt = 0.01`) over `r = 5000` realisations
3. Computes trace distances against the true states
4. Prints ensemble-averaged errors to the console
5. Saves all distance arrays and histogram data to `.txt` files

### Run Both Steps Together

```bash
python run_all.py
```

---

## Simulation Examples

Five measurement scenarios are provided. Select the appropriate operator and initial state in both scripts.

| Example | System | Measurement operator | Initial state |
|---|---|---|---|
| 1 | Spin-1/2 | $\hat{c} = \sqrt{\gamma/2}\,\hat{\sigma}_z$ | $\|{+x}\rangle$ |
| 2 | Spin-1/2 | $\hat{c} = \sqrt{\gamma}\,\hat{\sigma}_-$ | $\|{+x}\rangle$ |
| 3 | Spin-1 | $\hat{c} = \sqrt{\gamma}\,\hat{S}_-$ | $\|{+x}\rangle$ |
| 4 | Spin-3/2 | $\hat{c} = \sqrt{\gamma}\,\hat{L}_-$ | $\|{+x}\rangle$ |
| 5 | Spin-1 | $\hat{c} = \sqrt{\gamma/2}\,\hat{S}_z$ | $\|{+x}\rangle$ |

The measurement operator `K` and initial state `psiIn` are defined near the top of each script and can be switched by commenting/uncommenting the relevant lines.

---

## Output Files

### Trace distances (absolute)
`Di.txt`, `Dr.txt`, `Drb.txt`, `Dw.txt`, `Dphi.txt`

### Trace distances (squared)
`Di2.txt`, `Dr2.txt`, `Drb2.txt`, `Dw2.txt`, `Dphi2.txt`

### Time-averaged RMS errors (for histograms)
`Tav_I.txt`, `Tav_R.txt`, `Tav_RB.txt`, `Tav_W.txt`, `Tav_Phi.txt`

Each file contains a matrix of shape `[r, n_coarse]` (distance files) or a vector of length `r` (time-averaged files).

---

## Error Metrics

Two summary statistics are printed for each method:

- **D̄** — ensemble and time-averaged absolute trace distance
- **σ̄²** — ensemble mean of the time-averaged squared trace distance

The nearly exact map (Φ) consistently achieves the lowest errors across all examples, confirming its higher-order accuracy.

---

## Modifying the Measurement Process

To adapt the simulation to a different physical setup:

1. **Change the measurement operator** — modify `K` and `Kd` in both `True_Data_Generation.py` and `Trajectory_simulation.py`
2. **Change the system dimension** — use the appropriate spin matrices (`sx`, `sy`, `sz` for spin-1/2; their spin-1 or spin-3/2 analogues for higher-dimensional systems)
3. **Change the initial state** — set `psiIn` to the column vector for your desired initial pure state

Ensure the matrix definitions, initial state, and measurement operator are consistent between the two scripts.

---

## Notes

- The scripts must be run **in order**: `True_Data_Generation.py` before `Trajectory_simulation.py`.
- Computational cost scales with system dimension (state vector size) and the number of realisations `r`.
- Random seeds are not fixed by default; set `np.random.seed(...)` at the top of `True_Data_Generation.py` for reproducibility.
- The fine step `dt = 0.0001` and coarse step `Dt = 0.01` can be adjusted, but the ratio `N = Dt/dt` must remain an integer.