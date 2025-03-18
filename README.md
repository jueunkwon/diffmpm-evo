# DiffmpmEvo: Evolving Composite Soft Robots with Differentiable MPM

*Jueun Kwon*

## Overview

DiffmpmEvo is a simulation framework that leverages a differentiable Material Point Method (MPM) to model soft robots. The system evolves the actuation parameters of composite soft robots using an evolutionary algorithm. It records videos (both before and after optimization), saves simulation frames, and logs candidate parameters—all to help analyze and improve robot locomotion.

In this project, the objective is to maximize the horizontal displacement of a soft robot by optimizing its actuator parameters. The loss function is defined as the negative average x‑coordinate of the robot's solid particles; hence, minimizing the loss results in improved forward motion.

## Key Features

- **Differentiable MPM Simulation:**  
  Implements a 2D MPM simulation using Taichi with fixed parameters (grid resolution, time step, material properties, etc.).

- **Procedural Creature Generation:**  
  Generates composite soft robot structures consisting of a fixed base (a rectangle) with various attachments (rectangles or circles). The geometry remains constant while the evolutionary algorithm optimizes the control parameters.

- **Open-Loop Actuation Control:**  
  Generates actuation signals as a weighted sum of sinusoidal functions with a hyperbolic tangent applied, where each actuator is defined by its weights and bias: actuation(t, i) = tanh( Σ (weights[i,j] * sin(20*t + 2π·j/4)) + bias[i] )

- **Evolutionary Optimization:**  
    - **Candidate Representation:** Each candidate is defined solely by its actuator parameters (weights and biases).
    - **Population & Generation:** A population of 20 candidates is evolved over 50 generations (evaluating a total of 1,000 candidate designs).
    - **Selection & Mutation:** The best 20% of candidates are preserved, and offspring are generated by applying Gaussian mutations to these elite candidates.

- **Data Logging & Visualization:**  
    - Records initial creature details and candidate parameters.
    - Saves snapshots and compiles videos for both the initial (before optimization) and final (after optimization) behaviors.
    - Plots the loss evolution curve over generations.


## How to Run

**Step 1:** Install [Taichi](https://github.com/taichi-dev/taichi) with `pip`:

```bash
python3 -m pip install taichi
```

**Step 2:** Run the example scripts in the `examples` folder of the difftaichi repository to ensure that the difftaichi environment is installed correctly.

**Step 3:** Run the `diffmpm_evo.py` script:

```bash
python3 diffmpm_evo.py --gens 50 --pop 20
```

This command runs the evolutionary optimization for 50 generations with a population size of 20. Outputs (videos, snapshots, parameter logs, and a loss curve plot) are automatically saved in a folder named after the trial number (set via TRIAL_NUM).

## Example Trial

Several trial results are available in the `example_trials` folder. 

Below is a demonstration from Trial 12:

[Video](https://github.com/user-attachments/assets/cbf57f8f-5a08-43ad-843e-cf5cdeb5f5ee)