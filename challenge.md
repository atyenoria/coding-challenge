# Flow Matching Coding Challenge

## Overview

In this challenge, you will explore and explain a **Flow Matching** generative model implementation. Flow matching transforms Gaussian noise into structured data by following a learned velocity field.

**Time Limit:** 6 hours total  
**Tools Allowed:** AI assistants (ChatGPT, Claude, etc.)

---

## What is Flow Matching?

Flow matching solves this ODE from t=0 to t=1:

```
dx/dt = v(x, t)
```

- **t = 0**: Start from random Gaussian noise
- **t = 1**: Arrive at target distribution (clusters)
- **v(x, t)**: Velocity field that guides the transformation

---

## Setup Instructions

1. Copy `flow_matching.py` to Google Colab
2. Run the file to generate output and plots
3. Experiment with parameters
4. Record your video explanations

---

## Running the Code

```python
# In Google Colab, run:
!python flow_matching.py
```

This produces:
- Console output showing model behavior
- 4 PNG plots saved to the directory

---

## Code Structure

```python
# Model
FlowMatchingModel
  ├── sample_source(n)      # Get Gaussian noise
  ├── sample_target(n)      # Get target samples  
  └── get_velocity(x, t)    # Get velocity at position x, time t

# ODE Solvers (from simple to accurate)
euler_step(x, t, dt, velocity_fn)   # 1st order, 1 eval/step
heun_step(x, t, dt, velocity_fn)    # 2nd order, 2 eval/step
rk4_step(x, t, dt, velocity_fn)     # 4th order, 4 eval/step

# Generation
generate_samples(model, step_fn, n_samples, num_steps)
```

---

## Key Parameters to Experiment With

| Parameter | Description | Try These Values |
|-----------|-------------|------------------|
| `num_steps` | Integration steps | 5, 20, 50, 100 |
| `num_targets` | Number of clusters | 2, 4, 6, 8 |
| `n_samples` | Points to generate | 50, 200, 500 |
| `solver` | ODE method | euler, heun, rk4 |
| `seed` | Random seed | 42, 123, 999 |

---

## Video Submission Requirements

Record **4 separate videos** using [Loom](https://www.loom.com) with:
- ✅ Webcam visible
- ✅ Screen sharing enabled
- ✅ English only
- ✅ Public link (viewable without login)

---

### Video 1: Flow Matching Concept (2 minutes)

**Show and explain:**
1. Run the code and show the console output
2. Explain what source distribution is (Gaussian noise)
3. Explain what target distribution is (4 clusters)
4. Show `plot_3_trajectories.png` and explain how points flow from noise to clusters

**Key points to cover:**
- What does the velocity field do?
- Why do trajectories curve toward cluster centers?

---

### Video 2: ODE Solvers Comparison (2 minutes)

**Show and explain:**
1. Show `plot_1_solver_comparison.png`
2. Explain the difference between Euler, Heun, and RK4
3. Run the code with different `num_steps` values (e.g., 10 vs 50)
4. Explain why more accurate solvers give better results

**Key points to cover:**
- Which solver is fastest? Most accurate?
- What is the trade-off?

---

### Video 3: Parameter Experimentation (2 minutes)

**Modify the code and show results:**

```python
# Try different step counts
for num_steps in [5, 10, 30, 100]:
    samples = generate_samples(model, euler_step, n_samples=200, num_steps=num_steps)
    # Show how quality changes
```

**Show and explain:**
1. How does `num_steps` affect sample quality?
2. Show `plot_2_steps_comparison.png`
3. What happens with very few steps (5)? Many steps (100)?

---

### Video 4: Target Configuration (2 minutes)

**Modify the code and show results:**

```python
# Try different number of targets
config = FlowConfig(dim=2, num_targets=8, seed=42)  # Change from 4 to 8
model = FlowMatchingModel(config)
```

**Show and explain:**
1. Show `plot_4_target_configs.png`
2. Change `num_targets` to 2, 6, or 8 and regenerate
3. Explain how the flow adapts to different target distributions

---

## Expected Output

When you run `python flow_matching.py`, you should see:

```
============================================================
FLOW MATCHING DEMONSTRATION
============================================================

[1] MODEL SETUP
----------------------------------------
Dimension: 2
Number of target clusters: 4
Cluster centers:
[[ 3.  0.]
 [ 0.  3.]
 [-3.  0.]
 [ 0. -3.]]

[2] DISTRIBUTIONS
----------------------------------------
Source samples (Gaussian noise):
[[ 0.30  -1.04]
 [ 0.75   0.94]
 ...

[3] VELOCITY FIELD
----------------------------------------
Testing velocity at different positions and times:
  Time t = 0.0:
    x = [0. 0.] → v = [+0.000, +0.000]
    x = [2. 0.] → v = [+0.520, +0.000]
  ...

[4] COMPARING ODE SOLVERS
----------------------------------------
  Euler (steps=30):
    Mean: [+0.002, +0.004]
    Std:  [0.554, 0.504]
  ...

Generated 4 plots:
  1. plot_1_solver_comparison.png
  2. plot_2_steps_comparison.png
  3. plot_3_trajectories.png
  4. plot_4_target_configs.png
```

---

## Submission Checklist

| Item | Requirement |
|------|-------------|
| Video 1 | Flow matching concept explanation (2 min) |
| Video 2 | ODE solvers comparison (2 min) |
| Video 3 | Parameter experimentation with num_steps (2 min) |
| Video 4 | Target configuration changes (2 min) |

**Each video must have:**
- [ ] Webcam visible throughout
- [ ] Screen sharing with code/plots visible
- [ ] Clear English explanation
- [ ] Public Loom link

---

## Evaluation Criteria

| Criteria | Weight |
|----------|--------|
| Understanding of flow matching concept | 30% |
| Ability to explain ODE solver differences | 25% |
| Quality of parameter experimentation | 25% |
| Communication clarity | 20% |

---

## Rules

1. Maximum 6 hours total working time
2. AI assistants allowed
3. Each video must be unique (no reusing)
4. English only
5. Videos must be publicly viewable
6. Do not ask questions about the problem via LinkedIn DM

---

Good luck!