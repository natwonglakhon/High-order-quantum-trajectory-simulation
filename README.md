# High-order Quantum Trajectory Simulation

This repository contains Python scripts for simulating quantum trajectories
using higher-order methods from the literature, and for analysing the
resulting conditioned states via trace-distance measures and histograms.

The code is intended to reproduce the numerical simulations used in our work
on high-order quantum trajectory reconstruction.

---

## Usage

The simulation consists of two main steps.

### Step 1: Generate true trajectories and measurement records

Run:
python True_Data_Generation.py

This script generates:
 - True quantum trajectories
 - Coarse-grained measurement records $I_t$ and $\phi_t$
 
The generated data are saved and used as input for the second step.

### Step 2: Simulate conditioned quantum trajectories
Run:
python Trajectory_simulation.py

This script:
 - Reconstructs quantum trajectories conditioned on the previously generated
 - coarse-grained records
 - Implements existing (lower- and higher-order) trajectory simulation methods
 - Performs trace-distance analysis and generates histograms for comparison

Example of simulation procedures:
<p align="center">
  <img src="Simulation_Procedure.pdf" width="500">
</p>

### Modifying the measurement process
To obtain results for different measurement schemes, users may modify:
 - The measurement operator $\hat c$
 - The corresponding initial quantum state

These are explicitly defined in the scripts and can be adjusted directly.

### Details for simulation examples. 
 - Example 1. Use operator $\hat c = \sqrt{\gamma/2}\hat \sigma_z$ with the initial state $|+x\rangle$ for spin 1/2 system.
 - Example 2. Use operator $\hat c = \sqrt{\gamma}\hat \sigma_-$ with the initial state $|+x\rangle$ for spin 1/2 system.
 - Example 3. Use operator $\hat c = \sqrt{\gamma}\hat \sigma_-$ with the initial state $|+x\rangle$ for spin 1 system.
 - Example 4. Use operator $\hat c = \sqrt{\gamma}\hat \sigma_-$ with the initial state $|+x\rangle$ for spin 3/2 system.
 - Example 5. Use operator $\hat c = \sqrt{\gamma/2}\hat \sigma_z$ with the initial state $|+x\rangle$ for spin 1 system.

### Numerical results (error analysis)
![Simulation_results.pdf](https://github.com/natwonglakhon/High-order-quantum-trajectory-simulation/blob/805dbf712fb9a189e6cb7f455078c2d5925391a0/Simulation_results.pdf?raw=true)

### Notes 
 - The scripts are designed to be run sequentially.
 - Numerical parameters and random seeds can be adjusted within the scripts.
 - The computational cost depends on the system dimension and time resolution.
