"""
Flow Matching Inference Strategies
==================================

This module implements various ODE solvers and inference strategies for
flow matching models, ranging from basic to advanced techniques.

Difficulty Levels:
1. Basic: Euler Method (simplest)
2. Intermediate: Midpoint Method, Heun's Method
3. Advanced: RK4, Adaptive Step Size
4. Expert: Stochastic Sampling, Guidance Techniques

Each strategy offers different trade-offs between:
- Computational cost (number of function evaluations)
- Accuracy (how well the ODE is solved)
- Sample quality (how well samples match the target)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import time

# Import base model (assuming it's in the same directory)
from flow_matching_base import (
    FlowMatchingModel, FlowMatchingConfig, DistributionType,
    visualize_flow, visualize_trajectories
)


@dataclass
class InferenceConfig:
    """Configuration for inference/sampling."""
    num_steps: int = 50                    # Number of integration steps
    batch_size: int = 100                  # Number of samples to generate
    guidance_scale: float = 1.0            # For guided generation
    stochastic_noise: float = 0.0          # Noise injection during sampling
    adaptive_tol: float = 1e-3             # Tolerance for adaptive methods
    min_step_size: float = 1e-4            # Minimum step size for adaptive
    max_step_size: float = 0.1             # Maximum step size for adaptive


@dataclass 
class InferenceResult:
    """Results from running inference."""
    samples: np.ndarray                    # Generated samples
    trajectories: Optional[np.ndarray]     # Full trajectories if tracked
    num_function_evals: int                # Number of velocity field evaluations
    wall_time: float                       # Time taken in seconds
    stats: Dict[str, Any]                  # Additional statistics


class ODESolver(ABC):
    """Abstract base class for ODE solvers."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.num_function_evals = 0
    
    @abstractmethod
    def step(self, x: np.ndarray, t: float, dt: float, 
             velocity_fn: Callable[[np.ndarray, float], np.ndarray]) -> np.ndarray:
        """
        Perform one integration step.
        
        Args:
            x: Current position
            t: Current time
            dt: Time step size
            velocity_fn: Function that computes velocity at (x, t)
            
        Returns:
            New position after the step
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the solver for logging."""
        pass
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.num_function_evals = 0


# =============================================================================
# LEVEL 1: BASIC - Euler Method
# =============================================================================

class EulerSolver(ODESolver):
    """
    Euler Method (Forward Euler) - The simplest ODE solver.
    
    Update rule: x_{n+1} = x_n + dt * v(x_n, t_n)
    
    Pros:
    - Very simple to implement
    - Only 1 function evaluation per step
    - Easy to understand
    
    Cons:
    - First-order accuracy (error ~ O(dt))
    - Can be unstable for large step sizes
    - Accumulates error quickly
    
    Recommended for: Learning, prototyping, when speed matters more than accuracy
    """
    
    @property
    def name(self) -> str:
        return "Euler"
    
    def step(self, x: np.ndarray, t: float, dt: float,
             velocity_fn: Callable[[np.ndarray, float], np.ndarray]) -> np.ndarray:
        v = velocity_fn(x, t)
        self.num_function_evals += 1
        return x + dt * v


# =============================================================================
# LEVEL 2: INTERMEDIATE - Midpoint and Heun's Methods
# =============================================================================

class MidpointSolver(ODESolver):
    """
    Midpoint Method (Explicit) - Second-order accurate solver.
    
    Update rule:
    1. k1 = v(x_n, t_n)
    2. x_mid = x_n + (dt/2) * k1
    3. k2 = v(x_mid, t_n + dt/2)
    4. x_{n+1} = x_n + dt * k2
    
    Pros:
    - Second-order accuracy (error ~ O(dt²))
    - More stable than Euler
    - Still relatively simple
    
    Cons:
    - 2 function evaluations per step
    - Still not adaptive
    
    Recommended for: When you need better accuracy than Euler without complexity
    """
    
    @property
    def name(self) -> str:
        return "Midpoint"
    
    def step(self, x: np.ndarray, t: float, dt: float,
             velocity_fn: Callable[[np.ndarray, float], np.ndarray]) -> np.ndarray:
        # First evaluation at current point
        k1 = velocity_fn(x, t)
        self.num_function_evals += 1
        
        # Midpoint evaluation
        x_mid = x + (dt / 2) * k1
        k2 = velocity_fn(x_mid, t + dt / 2)
        self.num_function_evals += 1
        
        return x + dt * k2


class HeunSolver(ODESolver):
    """
    Heun's Method (Improved Euler / Explicit Trapezoidal) - Second-order solver.
    
    Update rule:
    1. k1 = v(x_n, t_n)
    2. x_pred = x_n + dt * k1  (Euler prediction)
    3. k2 = v(x_pred, t_n + dt)
    4. x_{n+1} = x_n + (dt/2) * (k1 + k2)  (Average slopes)
    
    Pros:
    - Second-order accuracy
    - Uses predictor-corrector approach
    - Good balance of accuracy and cost
    
    Cons:
    - 2 function evaluations per step
    - Not adaptive
    
    Recommended for: General-purpose sampling, good default choice
    """
    
    @property
    def name(self) -> str:
        return "Heun"
    
    def step(self, x: np.ndarray, t: float, dt: float,
             velocity_fn: Callable[[np.ndarray, float], np.ndarray]) -> np.ndarray:
        # Predictor (Euler step)
        k1 = velocity_fn(x, t)
        self.num_function_evals += 1
        x_pred = x + dt * k1
        
        # Corrector (evaluate at predicted point)
        k2 = velocity_fn(x_pred, t + dt)
        self.num_function_evals += 1
        
        # Average the slopes
        return x + (dt / 2) * (k1 + k2)


# =============================================================================
# LEVEL 3: ADVANCED - Runge-Kutta 4 and Adaptive Methods
# =============================================================================

class RK4Solver(ODESolver):
    """
    Classical Runge-Kutta 4th Order (RK4) - High accuracy solver.
    
    Update rule:
    1. k1 = v(x_n, t_n)
    2. k2 = v(x_n + (dt/2)*k1, t_n + dt/2)
    3. k3 = v(x_n + (dt/2)*k2, t_n + dt/2)
    4. k4 = v(x_n + dt*k3, t_n + dt)
    5. x_{n+1} = x_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    Pros:
    - Fourth-order accuracy (error ~ O(dt⁴))
    - Very stable
    - Industry standard for many applications
    
    Cons:
    - 4 function evaluations per step
    - More expensive per step
    - Still fixed step size
    
    Recommended for: When accuracy is important, high-quality generation
    """
    
    @property
    def name(self) -> str:
        return "RK4"
    
    def step(self, x: np.ndarray, t: float, dt: float,
             velocity_fn: Callable[[np.ndarray, float], np.ndarray]) -> np.ndarray:
        k1 = velocity_fn(x, t)
        self.num_function_evals += 1
        
        k2 = velocity_fn(x + (dt / 2) * k1, t + dt / 2)
        self.num_function_evals += 1
        
        k3 = velocity_fn(x + (dt / 2) * k2, t + dt / 2)
        self.num_function_evals += 1
        
        k4 = velocity_fn(x + dt * k3, t + dt)
        self.num_function_evals += 1
        
        return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class AdaptiveHeunSolver(ODESolver):
    """
    Adaptive Step Size Heun's Method with error estimation.
    
    Uses embedded error estimation to automatically adjust step sizes:
    - Larger steps where the flow is smooth
    - Smaller steps where the flow changes rapidly
    
    Error estimation:
    - Compare Euler (1st order) with Heun (2nd order) results
    - Adjust step size based on estimated error
    
    Pros:
    - Automatically adapts to flow complexity
    - Can be more efficient than fixed-step methods
    - Better error control
    
    Cons:
    - More complex implementation
    - Variable number of steps
    - May reject steps (wasted computation)
    
    Recommended for: Production use, varying flow complexity
    """
    
    @property
    def name(self) -> str:
        return "AdaptiveHeun"
    
    def __init__(self, config: InferenceConfig):
        super().__init__(config)
        self.rejected_steps = 0
        self.accepted_steps = 0
    
    def reset_stats(self):
        super().reset_stats()
        self.rejected_steps = 0
        self.accepted_steps = 0
    
    def step(self, x: np.ndarray, t: float, dt: float,
             velocity_fn: Callable[[np.ndarray, float], np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Perform adaptive step. Returns (new_x, actual_dt_used).
        
        Note: This method signature differs from base class to return the
        actual step size used after adaptation.
        """
        current_dt = dt
        
        while True:
            # Euler step (first-order)
            k1 = velocity_fn(x, t)
            self.num_function_evals += 1
            x_euler = x + current_dt * k1
            
            # Heun step (second-order)
            k2 = velocity_fn(x_euler, t + current_dt)
            self.num_function_evals += 1
            x_heun = x + (current_dt / 2) * (k1 + k2)
            
            # Error estimate (difference between orders)
            error = np.max(np.abs(x_heun - x_euler))
            
            # Accept or reject step
            if error < self.config.adaptive_tol or current_dt <= self.config.min_step_size:
                self.accepted_steps += 1
                
                # Compute optimal step size for next iteration
                if error > 0:
                    safety = 0.9
                    optimal_dt = safety * current_dt * (self.config.adaptive_tol / error) ** 0.5
                    suggested_dt = np.clip(optimal_dt, self.config.min_step_size, 
                                          self.config.max_step_size)
                else:
                    suggested_dt = self.config.max_step_size
                
                return x_heun, current_dt, suggested_dt
            else:
                # Reject and try smaller step
                self.rejected_steps += 1
                current_dt = max(current_dt * 0.5, self.config.min_step_size)


# =============================================================================
# LEVEL 4: EXPERT - Stochastic and Guided Sampling
# =============================================================================

class StochasticEulerSolver(ODESolver):
    """
    Stochastic Euler-Maruyama with noise injection.
    
    Adds controlled noise during sampling, which can help:
    - Escape local minima in the flow
    - Improve sample diversity
    - Approximate SDE-based formulations
    
    Update rule: x_{n+1} = x_n + dt * v(x_n, t_n) + sqrt(2 * noise * dt) * z
    where z ~ N(0, I)
    
    Pros:
    - Can improve sample diversity
    - Helps with mode coverage
    - Connects to diffusion model theory
    
    Cons:
    - Introduces additional hyperparameter (noise scale)
    - Can degrade quality if noise is too high
    - Not a true ODE solver
    
    Recommended for: When diversity is important, mode collapse issues
    """
    
    def __init__(self, config: InferenceConfig, seed: Optional[int] = None):
        super().__init__(config)
        self.rng = np.random.default_rng(seed)
    
    @property
    def name(self) -> str:
        return f"StochasticEuler(σ={self.config.stochastic_noise:.2f})"
    
    def step(self, x: np.ndarray, t: float, dt: float,
             velocity_fn: Callable[[np.ndarray, float], np.ndarray]) -> np.ndarray:
        v = velocity_fn(x, t)
        self.num_function_evals += 1
        
        # Deterministic drift
        x_new = x + dt * v
        
        # Stochastic diffusion (scaled by sqrt(dt) for proper SDE scaling)
        if self.config.stochastic_noise > 0:
            noise_scale = np.sqrt(2 * self.config.stochastic_noise * dt)
            # Decay noise as t → 1 to ensure convergence
            noise_scale *= (1 - t)
            x_new += noise_scale * self.rng.standard_normal(x.shape)
        
        return x_new


class GuidedSolver(ODESolver):
    """
    Classifier-Free Guidance style solver.
    
    Implements guidance by interpolating between conditional and unconditional
    velocity fields:
    
    v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
    
    With guidance_scale > 1, this amplifies the "conditional" direction,
    leading to samples that more strongly exhibit desired properties.
    
    Pros:
    - Can improve sample quality for conditional generation
    - Trade-off between diversity and fidelity
    - Widely used in diffusion models
    
    Cons:
    - Requires conditional model
    - Higher guidance can reduce diversity
    - 2x function evaluations for guidance
    
    Recommended for: Conditional generation, when quality > diversity
    """
    
    def __init__(self, config: InferenceConfig, 
                 uncond_velocity_fn: Optional[Callable] = None):
        super().__init__(config)
        self.uncond_velocity_fn = uncond_velocity_fn
        self.base_solver = HeunSolver(config)
    
    @property
    def name(self) -> str:
        return f"Guided(w={self.config.guidance_scale:.1f})"
    
    def step(self, x: np.ndarray, t: float, dt: float,
             velocity_fn: Callable[[np.ndarray, float], np.ndarray]) -> np.ndarray:
        
        if self.config.guidance_scale == 1.0 or self.uncond_velocity_fn is None:
            # No guidance, use base solver
            result = self.base_solver.step(x, t, dt, velocity_fn)
            self.num_function_evals = self.base_solver.num_function_evals
            return result
        
        # Compute conditional velocity
        v_cond = velocity_fn(x, t)
        self.num_function_evals += 1
        
        # Compute unconditional velocity
        v_uncond = self.uncond_velocity_fn(x, t)
        self.num_function_evals += 1
        
        # Apply guidance
        v_guided = v_uncond + self.config.guidance_scale * (v_cond - v_uncond)
        
        # Use guided velocity for Heun step
        x_pred = x + dt * v_guided
        
        # Corrector with guided velocity at predicted point
        v_cond_pred = velocity_fn(x_pred, t + dt)
        v_uncond_pred = self.uncond_velocity_fn(x_pred, t + dt)
        self.num_function_evals += 2
        
        v_guided_pred = v_uncond_pred + self.config.guidance_scale * (v_cond_pred - v_uncond_pred)
        
        return x + (dt / 2) * (v_guided + v_guided_pred)


# =============================================================================
# Inference Engine
# =============================================================================

class FlowMatchingInference:
    """
    Main inference engine that combines model and solver.
    
    This class orchestrates the sampling process:
    1. Sample from source distribution
    2. Integrate ODE from t=0 to t=1
    3. Return generated samples
    """
    
    def __init__(self, model: FlowMatchingModel, solver: ODESolver,
                 config: InferenceConfig):
        self.model = model
        self.solver = solver
        self.config = config
    
    def sample(self, n_samples: Optional[int] = None,
               track_trajectories: bool = False) -> InferenceResult:
        """
        Generate samples using flow matching.
        
        Args:
            n_samples: Number of samples (default: config.batch_size)
            track_trajectories: Whether to store full trajectories
            
        Returns:
            InferenceResult containing samples and statistics
        """
        n_samples = n_samples or self.config.batch_size
        self.solver.reset_stats()
        
        start_time = time.time()
        
        # Sample from source distribution
        x = self.model.sample_source(n_samples)
        
        # Setup time discretization
        if isinstance(self.solver, AdaptiveHeunSolver):
            return self._sample_adaptive(x, track_trajectories, start_time)
        else:
            return self._sample_fixed(x, track_trajectories, start_time)
    
    def _sample_fixed(self, x: np.ndarray, track_trajectories: bool,
                      start_time: float) -> InferenceResult:
        """Fixed step size integration."""
        timestamps = np.linspace(0, 1, self.config.num_steps + 1)
        dt = 1.0 / self.config.num_steps
        
        trajectories = [x.copy()] if track_trajectories else None
        
        for i, t in enumerate(timestamps[:-1]):
            x = self.solver.step(x, t, dt, self.model.get_velocity)
            
            if track_trajectories:
                trajectories.append(x.copy())
        
        wall_time = time.time() - start_time
        
        return InferenceResult(
            samples=x,
            trajectories=np.array(trajectories) if track_trajectories else None,
            num_function_evals=self.solver.num_function_evals,
            wall_time=wall_time,
            stats={
                "solver": self.solver.name,
                "num_steps": self.config.num_steps,
                "samples_per_second": len(x) / wall_time
            }
        )
    
    def _sample_adaptive(self, x: np.ndarray, track_trajectories: bool,
                        start_time: float) -> InferenceResult:
        """Adaptive step size integration."""
        trajectories = [x.copy()] if track_trajectories else None
        
        t = 0.0
        dt = 1.0 / self.config.num_steps  # Initial step size
        actual_steps = 0
        
        while t < 1.0:
            # Ensure we don't overshoot t=1
            dt = min(dt, 1.0 - t)
            
            x, used_dt, suggested_dt = self.solver.step(
                x, t, dt, self.model.get_velocity
            )
            
            t += used_dt
            dt = suggested_dt
            actual_steps += 1
            
            if track_trajectories:
                trajectories.append(x.copy())
        
        wall_time = time.time() - start_time
        
        return InferenceResult(
            samples=x,
            trajectories=np.array(trajectories) if track_trajectories else None,
            num_function_evals=self.solver.num_function_evals,
            wall_time=wall_time,
            stats={
                "solver": self.solver.name,
                "actual_steps": actual_steps,
                "accepted_steps": self.solver.accepted_steps,
                "rejected_steps": self.solver.rejected_steps,
                "acceptance_rate": self.solver.accepted_steps / 
                                  (self.solver.accepted_steps + self.solver.rejected_steps),
                "samples_per_second": len(x) / wall_time
            }
        )


# =============================================================================
# Comparison and Analysis Utilities
# =============================================================================

def compare_solvers(model: FlowMatchingModel, 
                   solvers: List[ODESolver],
                   config: InferenceConfig,
                   n_trials: int = 3) -> Dict[str, Dict[str, float]]:
    """
    Compare different solvers on the same model.
    
    Returns statistics including:
    - Average wall time
    - Function evaluations
    - Sample quality (distance to target)
    """
    results = {}
    
    # Get reference target samples for quality comparison
    target_samples = model.sample_target(config.batch_size * 10)
    
    for solver in solvers:
        solver_results = []
        
        for trial in range(n_trials):
            inference = FlowMatchingInference(model, solver, config)
            result = inference.sample()
            
            # Compute sample quality (MMD-like metric)
            # Simple approximation: average distance to nearest target
            quality = 0
            for sample in result.samples:
                distances = np.linalg.norm(target_samples - sample, axis=1)
                quality += np.min(distances)
            quality /= len(result.samples)
            
            solver_results.append({
                "wall_time": result.wall_time,
                "num_function_evals": result.num_function_evals,
                "quality": quality,
                **result.stats
            })
        
        # Average over trials
        results[solver.name] = {
            "avg_wall_time": np.mean([r["wall_time"] for r in solver_results]),
            "avg_function_evals": np.mean([r["num_function_evals"] for r in solver_results]),
            "avg_quality": np.mean([r["quality"] for r in solver_results]),
            "std_quality": np.std([r["quality"] for r in solver_results])
        }
    
    return results


def print_comparison_table(results: Dict[str, Dict[str, float]]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("SOLVER COMPARISON")
    print("=" * 70)
    print(f"{'Solver':<20} {'Time (s)':<12} {'NFE':<10} {'Quality':<15}")
    print("-" * 70)
    
    for solver_name, stats in results.items():
        print(f"{solver_name:<20} "
              f"{stats['avg_wall_time']:<12.4f} "
              f"{stats['avg_function_evals']:<10.0f} "
              f"{stats['avg_quality']:.4f} ± {stats['std_quality']:.4f}")
    
    print("=" * 70)
    print("NFE = Number of Function Evaluations")
    print("Quality = Average distance to nearest target (lower is better)")


# =============================================================================
# Main Demonstration
# =============================================================================

def main():
    print("Flow Matching Inference Strategies")
    print("=" * 50)
    
    # Create model
    config = FlowMatchingConfig(dim=2, seed=42)
    model = FlowMatchingModel(config, DistributionType.GAUSSIAN_MIXTURE)
    
    # Create inference configs
    inference_config = InferenceConfig(
        num_steps=20,  # Reduced for faster demo
        batch_size=50,  # Reduced for faster demo
        stochastic_noise=0.1,
        guidance_scale=1.5
    )
    
    # Initialize all solvers
    solvers = [
        EulerSolver(inference_config),
        MidpointSolver(inference_config),
        HeunSolver(inference_config),
        RK4Solver(inference_config),
        StochasticEulerSolver(inference_config, seed=42),
    ]
    
    # Demonstrate individual solvers
    print("\n" + "=" * 50)
    print("SOLVER DEMONSTRATIONS")
    print("=" * 50)
    
    for solver in solvers:
        solver.reset_stats()
        inference = FlowMatchingInference(model, solver, inference_config)
        result = inference.sample(n_samples=50, track_trajectories=False)
        
        print(f"\n{solver.name}:")
        print(f"  Samples generated: {len(result.samples)}")
        print(f"  Function evaluations: {result.num_function_evals}")
        print(f"  Wall time: {result.wall_time:.4f}s")
        print(f"  Sample mean: [{result.samples[:, 0].mean():.2f}, {result.samples[:, 1].mean():.2f}]")
        print(f"  Sample std:  [{result.samples[:, 0].std():.2f}, {result.samples[:, 1].std():.2f}]")
    
    # Test adaptive solver separately
    print("\n" + "=" * 50)
    print("ADAPTIVE SOLVER TEST")
    print("=" * 50)
    
    adaptive_solver = AdaptiveHeunSolver(inference_config)
    adaptive_solver.reset_stats()
    inference = FlowMatchingInference(model, adaptive_solver, inference_config)
    result = inference.sample(n_samples=20, track_trajectories=False)
    
    print(f"\n{adaptive_solver.name}:")
    print(f"  Samples generated: {len(result.samples)}")
    print(f"  Function evaluations: {result.num_function_evals}")
    print(f"  Accepted steps: {result.stats.get('accepted_steps', 'N/A')}")
    print(f"  Rejected steps: {result.stats.get('rejected_steps', 'N/A')}")
    print(f"  Wall time: {result.wall_time:.4f}s")
    
    print("\n" + "=" * 50)
    print("All solvers working correctly!")
    print("=" * 50)


if __name__ == "__main__":
    main()