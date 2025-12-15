"""
Flow Matching - Complete Implementation with Examples
=====================================================
A simplified flow matching model for educational purposes.

Flow matching learns to transform noise into data by following a velocity field:
    dx/dt = v(x, t)  for t ∈ [0, 1]
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class FlowConfig:
    dim: int = 2
    num_targets: int = 4
    seed: Optional[int] = None


class FlowMatchingModel:
    """Mock flow matching model with Gaussian mixture target."""
    
    def __init__(self, config: FlowConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        
        # Target: Gaussian clusters arranged in a circle
        angles = np.linspace(0, 2 * np.pi, config.num_targets, endpoint=False)
        self.centers = 3.0 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    
    def sample_source(self, n: int) -> np.ndarray:
        """Sample from source distribution (standard Gaussian)."""
        return self.rng.standard_normal((n, self.config.dim))
    
    def sample_target(self, n: int) -> np.ndarray:
        """Sample from target distribution (Gaussian mixture)."""
        indices = self.rng.integers(0, len(self.centers), n)
        samples = np.array([
            self.rng.multivariate_normal(self.centers[i], np.eye(2) * 0.3)
            for i in indices
        ])
        return samples
    
    def get_velocity(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Compute velocity field v(x, t).
        
        Args:
            x: Positions, shape (batch, dim) or (dim,)
            t: Time in [0, 1]
        Returns:
            Velocities, same shape as x
        """
        squeeze = x.ndim == 1
        x = np.atleast_2d(x)
        batch_size = x.shape[0]
        
        velocities = np.zeros_like(x)
        for i in range(batch_size):
            dists = np.linalg.norm(self.centers - x[i], axis=1)
            weights = np.exp(-dists / (1 + t))
            weights /= weights.sum()
            target = np.average(self.centers, axis=0, weights=weights)
            velocities[i] = (target - x[i]) / (1 - t + 0.01)
        
        return velocities.squeeze() if squeeze else velocities


# =============================================================================
# ODE SOLVERS
# =============================================================================

def euler_step(x: np.ndarray, t: float, dt: float, 
               velocity_fn: Callable) -> np.ndarray:
    """
    Euler method: x_{n+1} = x_n + dt * v(x_n, t_n)
    - Simplest method
    - 1 function evaluation per step
    - First-order accuracy
    """
    v = velocity_fn(x, t)
    return x + dt * v


def heun_step(x: np.ndarray, t: float, dt: float,
              velocity_fn: Callable) -> np.ndarray:
    """
    Heun's method (improved Euler):
        k1 = v(x_n, t_n)
        k2 = v(x_n + dt*k1, t_n + dt)
        x_{n+1} = x_n + (dt/2) * (k1 + k2)
    - 2 function evaluations per step
    - Second-order accuracy
    """
    k1 = velocity_fn(x, t)
    k2 = velocity_fn(x + dt * k1, t + dt)
    return x + (dt / 2) * (k1 + k2)


def rk4_step(x: np.ndarray, t: float, dt: float,
             velocity_fn: Callable) -> np.ndarray:
    """
    Runge-Kutta 4th order:
        k1 = v(x, t)
        k2 = v(x + dt/2*k1, t + dt/2)
        k3 = v(x + dt/2*k2, t + dt/2)
        k4 = v(x + dt*k3, t + dt)
        x_{n+1} = x_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    - 4 function evaluations per step
    - Fourth-order accuracy (most accurate)
    """
    k1 = velocity_fn(x, t)
    k2 = velocity_fn(x + (dt/2) * k1, t + dt/2)
    k3 = velocity_fn(x + (dt/2) * k2, t + dt/2)
    k4 = velocity_fn(x + dt * k3, t + dt)
    return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


# =============================================================================
# SAMPLE GENERATION
# =============================================================================

def generate_samples(model: FlowMatchingModel, 
                     step_fn: Callable,
                     n_samples: int = 100,
                     num_steps: int = 50,
                     return_trajectories: bool = False):
    """
    Generate samples by integrating the flow from t=0 to t=1.
    
    Args:
        model: The flow matching model
        step_fn: One of euler_step, heun_step, rk4_step
        n_samples: Number of samples to generate
        num_steps: Number of integration steps
        return_trajectories: If True, return full paths
        
    Returns:
        samples: Final generated samples (n_samples, dim)
        trajectories: Optional, shape (num_steps+1, n_samples, dim)
    """
    dt = 1.0 / num_steps
    x = model.sample_source(n_samples)
    
    trajectories = [x.copy()] if return_trajectories else None
    
    for i in range(num_steps):
        t = i * dt
        x = step_fn(x, t, dt, model.get_velocity)
        if return_trajectories:
            trajectories.append(x.copy())
    
    if return_trajectories:
        return x, np.array(trajectories)
    return x


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comparison(model, samples_dict, title="Flow Matching Results"):
    """Plot source, generated samples, and target side by side."""
    n_plots = len(samples_dict) + 2  # +2 for source and target
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    
    # Source distribution
    source = model.sample_source(200)
    axes[0].scatter(source[:, 0], source[:, 1], alpha=0.5, s=15, c='blue')
    axes[0].set_title("Source\n(Gaussian Noise)")
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(-5, 5)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Generated samples
    for idx, (name, samples) in enumerate(samples_dict.items()):
        ax = axes[idx + 1]
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=15, c='green')
        ax.set_title(f"Generated\n({name})")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    # Target distribution
    target = model.sample_target(200)
    axes[-1].scatter(target[:, 0], target[:, 1], alpha=0.5, s=15, c='red')
    # Mark cluster centers
    axes[-1].scatter(model.centers[:, 0], model.centers[:, 1], 
                     c='darkred', s=100, marker='x', linewidths=2)
    axes[-1].set_title("Target\n(4 Gaussians)")
    axes[-1].set_xlim(-5, 5)
    axes[-1].set_ylim(-5, 5)
    axes[-1].set_aspect('equal')
    axes[-1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_trajectories(model, trajectories, title="Flow Trajectories"):
    """Visualize how samples move from source to target."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot target clusters in background
    target = model.sample_target(300)
    ax.scatter(target[:, 0], target[:, 1], alpha=0.2, s=10, c='red', label='Target')
    
    # Plot trajectories (limit to 30 for clarity)
    n_traj = min(30, trajectories.shape[1])
    colors = plt.cm.viridis(np.linspace(0, 1, n_traj))
    
    for i in range(n_traj):
        traj = trajectories[:, i, :]
        ax.plot(traj[:, 0], traj[:, 1], c=colors[i], alpha=0.6, linewidth=0.8)
        ax.scatter(traj[0, 0], traj[0, 1], c='blue', s=30, marker='o', zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], c='green', s=30, marker='s', zorder=5)
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend(['Target', 'Trajectory', 'Start (t=0)', 'End (t=1)'])
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_steps_comparison(model, step_fn, steps_list, title="Effect of Number of Steps"):
    """Compare results with different number of integration steps."""
    n_plots = len(steps_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    
    for idx, num_steps in enumerate(steps_list):
        # Reset seed for fair comparison
        model.rng = np.random.default_rng(42)
        samples = generate_samples(model, step_fn, n_samples=150, num_steps=num_steps)
        
        axes[idx].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=15, c='green')
        axes[idx].scatter(model.centers[:, 0], model.centers[:, 1], 
                         c='red', s=100, marker='x', linewidths=2)
        axes[idx].set_title(f"Steps = {num_steps}")
        axes[idx].set_xlim(-5, 5)
        axes[idx].set_ylim(-5, 5)
        axes[idx].set_aspect('equal')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FLOW MATCHING DEMONSTRATION")
    print("=" * 60)
    
    # Create model with fixed seed for reproducibility
    config = FlowConfig(dim=2, num_targets=4, seed=42)
    model = FlowMatchingModel(config)
    
    # -------------------------------------------------------------------------
    # 1. Basic Model Information
    # -------------------------------------------------------------------------
    print("\n[1] MODEL SETUP")
    print("-" * 40)
    print(f"Dimension: {config.dim}")
    print(f"Number of target clusters: {config.num_targets}")
    print(f"Cluster centers:\n{model.centers}")
    
    # -------------------------------------------------------------------------
    # 2. Source and Target Distributions
    # -------------------------------------------------------------------------
    print("\n[2] DISTRIBUTIONS")
    print("-" * 40)
    
    source_samples = model.sample_source(5)
    print("Source samples (Gaussian noise):")
    print(source_samples)
    
    target_samples = model.sample_target(5)
    print("\nTarget samples (4 Gaussian clusters):")
    print(target_samples)
    
    # -------------------------------------------------------------------------
    # 3. Velocity Field
    # -------------------------------------------------------------------------
    print("\n[3] VELOCITY FIELD")
    print("-" * 40)
    
    test_points = np.array([[0.0, 0.0], [2.0, 0.0], [-1.0, 1.0]])
    print("Testing velocity at different positions and times:")
    
    for t in [0.0, 0.5, 0.9]:
        print(f"\n  Time t = {t}:")
        for point in test_points:
            v = model.get_velocity(point, t)
            print(f"    x = {point} → v = [{v[0]:+.3f}, {v[1]:+.3f}]")
    
    # -------------------------------------------------------------------------
    # 4. Compare ODE Solvers
    # -------------------------------------------------------------------------
    print("\n[4] COMPARING ODE SOLVERS")
    print("-" * 40)
    
    solvers = {
        "Euler": euler_step,
        "Heun": heun_step,
        "RK4": rk4_step
    }
    
    num_steps = 30
    n_samples = 200
    
    results = {}
    for name, solver in solvers.items():
        model.rng = np.random.default_rng(42)  # Reset for fair comparison
        samples = generate_samples(model, solver, n_samples=n_samples, num_steps=num_steps)
        results[name] = samples
        
        print(f"\n  {name} (steps={num_steps}):")
        print(f"    Mean: [{samples[:, 0].mean():+.3f}, {samples[:, 1].mean():+.3f}]")
        print(f"    Std:  [{samples[:, 0].std():.3f}, {samples[:, 1].std():.3f}]")
    
    # Plot comparison
    fig1 = plot_comparison(model, results, "Comparing ODE Solvers (30 steps)")
    fig1.savefig("plot_1_solver_comparison.png", dpi=150, bbox_inches='tight')
    print("\n  → Saved: plot_1_solver_comparison.png")
    
    # -------------------------------------------------------------------------
    # 5. Effect of Number of Steps
    # -------------------------------------------------------------------------
    print("\n[5] EFFECT OF NUMBER OF STEPS")
    print("-" * 40)
    
    steps_to_test = [5, 15, 30, 100]
    
    for num_steps in steps_to_test:
        model.rng = np.random.default_rng(42)
        samples = generate_samples(model, euler_step, n_samples=150, num_steps=num_steps)
        
        # Compute average distance to nearest cluster center
        distances = []
        for s in samples:
            dist = np.min(np.linalg.norm(model.centers - s, axis=1))
            distances.append(dist)
        avg_dist = np.mean(distances)
        
        print(f"  Steps = {num_steps:3d}: Avg distance to cluster = {avg_dist:.3f}")
    
    fig2 = plot_steps_comparison(model, euler_step, steps_to_test, 
                                  "Euler Method: Effect of Step Count")
    fig2.savefig("plot_2_steps_comparison.png", dpi=150, bbox_inches='tight')
    print("\n  → Saved: plot_2_steps_comparison.png")
    
    # -------------------------------------------------------------------------
    # 6. Visualize Trajectories
    # -------------------------------------------------------------------------
    print("\n[6] FLOW TRAJECTORIES")
    print("-" * 40)
    
    model.rng = np.random.default_rng(123)
    samples, trajectories = generate_samples(
        model, heun_step, n_samples=50, num_steps=40, return_trajectories=True
    )
    
    print(f"  Trajectory shape: {trajectories.shape}")
    print(f"  (time_steps, n_samples, dimensions)")
    
    fig3 = plot_trajectories(model, trajectories, "Heun Method: Sample Trajectories")
    fig3.savefig("plot_3_trajectories.png", dpi=150, bbox_inches='tight')
    print("\n  → Saved: plot_3_trajectories.png")
    
    # -------------------------------------------------------------------------
    # 7. Different Target Configurations
    # -------------------------------------------------------------------------
    print("\n[7] DIFFERENT TARGET CONFIGURATIONS")
    print("-" * 40)
    
    fig4, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for idx, num_targets in enumerate([2, 4, 8]):
        config = FlowConfig(dim=2, num_targets=num_targets, seed=42)
        model = FlowMatchingModel(config)
        
        samples = generate_samples(model, heun_step, n_samples=200, num_steps=50)
        
        axes[idx].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=15, c='green')
        axes[idx].scatter(model.centers[:, 0], model.centers[:, 1], 
                         c='red', s=100, marker='x', linewidths=2)
        axes[idx].set_title(f"Targets = {num_targets}")
        axes[idx].set_xlim(-5, 5)
        axes[idx].set_ylim(-5, 5)
        axes[idx].set_aspect('equal')
        axes[idx].grid(True, alpha=0.3)
        
        print(f"  {num_targets} targets: samples clustered around {num_targets} centers")
    
    plt.suptitle("Effect of Number of Target Clusters", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig4.savefig("plot_4_target_configs.png", dpi=150, bbox_inches='tight')
    print("\n  → Saved: plot_4_target_configs.png")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Generated 4 plots:
  1. plot_1_solver_comparison.png  - Euler vs Heun vs RK4
  2. plot_2_steps_comparison.png   - Effect of step count
  3. plot_3_trajectories.png       - Flow paths visualization
  4. plot_4_target_configs.png     - Different target counts

Key Parameters to Experiment With:
  - num_steps: More steps = better accuracy (try 5, 20, 50, 100)
  - num_targets: Number of clusters (try 2, 4, 6, 8)
  - n_samples: Number of generated points (try 50, 200, 500)
  - solver: euler_step, heun_step, rk4_step
  - seed: Change for different random results
""")
    print("=" * 60)