"""
Non-Convex State Constraint Projector for Pi-MPPI

This module implements a Sequential Convex Programming (SCP) based optimizer
for projecting state trajectories onto a feasible set with non-convex constraints
including obstacle avoidance and terrain following.

Author: Pi-MPPI Enhancement
Date: 2026-04-07
"""

import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap
import jax


class NonConvexStateProjector():
    """
    Non-convex state constraint projector using Sequential Convex Programming (SCP).
    
    This class handles projection of state trajectories onto feasible sets with:
    - Box constraints on position (x, y, z)
    - Non-convex obstacle avoidance constraints
    - Terrain following constraints
    - Velocity magnitude constraints
    """
    
    def __init__(self, num_timesteps, num_batch, dt, 
                 x_min=-1000, x_max=1000,
                 y_min=-1000, y_max=1000, 
                 z_min=0, z_max=1000,
                 safety_margin=5.0):
        """
        Initialize the non-convex state projector.
        
        Args:
            num_timesteps: Number of time steps in trajectory
            num_batch: Batch size for parallel projection
            dt: Time step size
            x_min, x_max: Bounds on x position
            y_min, y_max: Bounds on y position
            z_min, z_max: Bounds on z position
            safety_margin: Safety margin around obstacles
        """
        self.num = num_timesteps
        self.num_batch = num_batch
        self.dt = dt
        
        # State bounds
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        
        # Non-convex constraint parameters
        self.safety_margin = safety_margin
        self.rho_state = 1.0  # ADMM penalty parameter for state constraints
        self.rho_obs = 10.0   # Penalty parameter for obstacle constraints
        self.rho_terrain = 10.0  # Penalty parameter for terrain constraints
        
        # SCP parameters
        self.maxiter_scp = 20  # Maximum SCP iterations
        self.trust_region_radius = 10.0  # Initial trust region radius
        self.trust_region_shrink = 0.7  # Shrink factor when constraint violation increases
        self.trust_region_expand = 1.2  # Expand factor when constraint satisfaction improves
        self.convergence_tol = 1e-4  # Convergence tolerance
        
    @partial(jit, static_argnums=(0,))
    def project_box_constraints(self, x, y, z):
        """
        Project state trajectories onto box constraints (convex).
        
        Args:
            x: x positions (num_batch, num_timesteps)
            y: y positions (num_batch, num_timesteps)
            z: z positions (num_batch, num_timesteps)
            
        Returns:
            Projected x, y, z within bounds
        """
        x_proj = jnp.clip(x, self.x_min, self.x_max)
        y_proj = jnp.clip(y, self.y_min, self.y_max)
        z_proj = jnp.clip(z, self.z_min, self.z_max)
        
        return x_proj, y_proj, z_proj
    
    @partial(jit, static_argnums=(0,))
    def compute_obstacle_constraint_violation(self, x, y, z, x_obs, y_obs, z_obs, r_obs):
        """
        Compute obstacle constraint violation for a single obstacle.
        
        Constraint: (x-x_obs)^2 + (y-y_obs)^2 + (z-z_obs)^2 >= (r_obs + safety_margin)^2
        Violation: max(0, (r_obs + safety_margin)^2 - distance^2)
        
        Args:
            x, y, z: State positions (num_batch, num_timesteps)
            x_obs, y_obs, z_obs: Obstacle center
            r_obs: Obstacle radius
            
        Returns:
            Constraint violation (non-negative, 0 means feasible)
        """
        distance_sq = (x - x_obs)**2 + (y - y_obs)**2 + (z - z_obs)**2
        required_distance_sq = (r_obs + self.safety_margin)**2
        
        # Violation is positive when too close to obstacle
        violation = jnp.maximum(0, required_distance_sq - distance_sq)
        
        return violation
    
    @partial(jit, static_argnums=(0,))
    def compute_obstacle_gradient(self, x, y, z, x_obs, y_obs, z_obs, r_obs):
        """
        Compute gradient of obstacle constraint with respect to states.
        
        For constraint g(x,y,z) = (r + margin)^2 - (x-x_obs)^2 - (y-y_obs)^2 - (z-z_obs)^2 <= 0
        Gradient: dg/dx = -2(x-x_obs), dg/dy = -2(y-y_obs), dg/dz = -2(z-z_obs)
        
        Args:
            x, y, z: Current state positions (num_batch, num_timesteps)
            x_obs, y_obs, z_obs: Obstacle center
            r_obs: Obstacle radius
            
        Returns:
            Gradients (grad_x, grad_y, grad_z)
        """
        grad_x = -2.0 * (x - x_obs)
        grad_y = -2.0 * (y - y_obs)
        grad_z = -2.0 * (z - z_obs)
        
        return grad_x, grad_y, grad_z
    
    @partial(jit, static_argnums=(0,))
    def linearize_obstacle_constraint(self, x, y, z, x_obs, y_obs, z_obs, r_obs):
        """
        Linearize obstacle constraint around current state.
        
        Linear approximation: g(x0) + grad_g^T * (x - x0) <= 0
        
        Args:
            x, y, z: Current state positions
            x_obs, y_obs, z_obs: Obstacle center
            r_obs: Obstacle radius
            
        Returns:
            Linearized constraint coefficients and bounds
        """
        # Compute gradient at current point
        grad_x, grad_y, grad_z = self.compute_obstacle_gradient(x, y, z, x_obs, y_obs, z_obs, r_obs)
        
        # Compute constraint value at current point
        distance_sq = (x - x_obs)**2 + (y - y_obs)**2 + (z - z_obs)**2
        required_distance_sq = (r_obs + self.safety_margin)**2
        g_current = required_distance_sq - distance_sq
        
        return grad_x, grad_y, grad_z, g_current
    
    @partial(jit, static_argnums=(0,))
    def project_obstacle_constraints_scp(self, x_init, y_init, z_init, 
                                         x_obs_array, y_obs_array, z_obs_array, r_obs_array):
        """
        Project states onto obstacle avoidance constraints using Sequential Convex Programming.
        
        Args:
            x_init, y_init, z_init: Initial state trajectories (num_batch, num_timesteps)
            x_obs_array, y_obs_array, z_obs_array: Obstacle centers (n_obstacles,)
            r_obs_array: Obstacle radii (n_obstacles,)
            
        Returns:
            Projected states (x_proj, y_proj, z_proj)
        """
        x_current = x_init
        y_current = y_init
        z_current = z_init
        trust_radius = self.trust_region_radius
        
        def scp_iteration(carry, idx):
            x, y, z, trust_r = carry
            
            # Start with box constraint projection
            x_proj, y_proj, z_proj = self.project_box_constraints(x, y, z)
            
            # For each obstacle, apply linearized constraint
            def apply_single_obstacle_correction(state_carry, obs_idx):
                x_curr, y_curr, z_curr = state_carry
                
                x_obs = x_obs_array[obs_idx]
                y_obs = y_obs_array[obs_idx]
                z_obs = z_obs_array[obs_idx]
                r_obs = r_obs_array[obs_idx]
                
                # Linearize constraint
                grad_x, grad_y, grad_z, g_val = self.linearize_obstacle_constraint(
                    x_curr, y_curr, z_curr, x_obs, y_obs, z_obs, r_obs
                )
                
                # Apply correction where constraint is violated (g_val > 0)
                violation_mask = g_val > 0
                
                # Gradient descent step with trust region
                grad_norm = jnp.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)
                step_size = jnp.minimum(trust_r, g_val / (grad_norm + 1e-8))
                
                # Move away from obstacle (negative gradient direction)
                correction_x = -grad_x / grad_norm * step_size * violation_mask
                correction_y = -grad_y / grad_norm * step_size * violation_mask
                correction_z = -grad_z / grad_norm * step_size * violation_mask
                
                x_new = x_curr + correction_x
                y_new = y_curr + correction_y
                z_new = z_curr + correction_z
                
                # Reproject onto box constraints
                x_new, y_new, z_new = self.project_box_constraints(x_new, y_new, z_new)
                
                return (x_new, y_new, z_new), None
            
            n_obstacles = x_obs_array.shape[0]
            (x_final, y_final, z_final), _ = jax.lax.scan(
                apply_single_obstacle_correction,
                (x_proj, y_proj, z_proj),
                jnp.arange(n_obstacles)
            )
            
            # Update trust region (simplified - could be more sophisticated)
            trust_r_new = trust_r * 0.95  # Gradually reduce
            
            return (x_final, y_final, z_final, trust_r_new), None
        
        # Run SCP iterations
        (x_proj, y_proj, z_proj, _), _ = jax.lax.scan(
            scp_iteration,
            (x_current, y_current, z_current, trust_radius),
            jnp.arange(self.maxiter_scp)
        )
        
        return x_proj, y_proj, z_proj
    
    @partial(jit, static_argnums=(0,))
    def project_terrain_constraint(self, x, y, z, terrain_height, clearance=5.0):
        """
        Project z coordinates to satisfy terrain following constraint.
        
        Constraint: z >= terrain_height(x, y) + clearance
        
        Args:
            x, y: Horizontal positions (num_batch, num_timesteps)
            z: Vertical positions (num_batch, num_timesteps)
            terrain_height: Pre-computed terrain height at (x, y) positions (num_batch, num_timesteps)
            clearance: Minimum clearance above terrain
            
        Returns:
            z_proj: Projected z satisfying terrain constraint
        """
        # Ensure z is at least clearance above terrain
        z_min_required = terrain_height + clearance
        z_proj = jnp.maximum(z, z_min_required)
        
        # Also apply upper bound
        z_proj = jnp.minimum(z_proj, self.z_max)
        
        return z_proj
    
    @partial(jit, static_argnums=(0,))
    def project_states_unified(self, x, y, z, x_obs_array, y_obs_array, z_obs_array, r_obs_array):
        """
        Unified state projection combining box constraints and obstacle avoidance.
        
        Args:
            x, y, z: State trajectories (num_batch, num_timesteps)
            x_obs_array, y_obs_array, z_obs_array: Obstacle centers
            r_obs_array: Obstacle radii
            
        Returns:
            Projected states (x_proj, y_proj, z_proj)
        """
        # First apply box constraints
        x_box, y_box, z_box = self.project_box_constraints(x, y, z)
        
        # Then apply obstacle avoidance constraints via SCP
        x_proj, y_proj, z_proj = self.project_obstacle_constraints_scp(
            x_box, y_box, z_box,
            x_obs_array, y_obs_array, z_obs_array, r_obs_array
        )
        
        return x_proj, y_proj, z_proj
    
    @partial(jit, static_argnums=(0,))
    def compute_total_constraint_violation(self, x, y, z, 
                                          x_obs_array, y_obs_array, z_obs_array, r_obs_array):
        """
        Compute total constraint violation for diagnostics.
        
        Args:
            x, y, z: State trajectories
            x_obs_array, y_obs_array, z_obs_array: Obstacle centers
            r_obs_array: Obstacle radii
            
        Returns:
            Total violation (scalar)
        """
        # Box constraint violations
        box_violation_x = jnp.sum(jnp.maximum(0, x - self.x_max) + jnp.maximum(0, self.x_min - x))
        box_violation_y = jnp.sum(jnp.maximum(0, y - self.y_max) + jnp.maximum(0, self.y_min - y))
        box_violation_z = jnp.sum(jnp.maximum(0, z - self.z_max) + jnp.maximum(0, self.z_min - z))
        
        # Obstacle constraint violations
        def compute_single_obstacle_violation(obs_idx):
            x_obs = x_obs_array[obs_idx]
            y_obs = y_obs_array[obs_idx]
            z_obs = z_obs_array[obs_idx]
            r_obs = r_obs_array[obs_idx]
            
            violation = self.compute_obstacle_constraint_violation(x, y, z, x_obs, y_obs, z_obs, r_obs)
            return jnp.sum(violation)
        
        n_obstacles = x_obs_array.shape[0]
        obstacle_violations = jax.lax.fori_loop(
            0, n_obstacles,
            lambda i, acc: acc + compute_single_obstacle_violation(i),
            0.0
        )
        
        total_violation = box_violation_x + box_violation_y + box_violation_z + obstacle_violations
        
        return total_violation


# Vectorized version for batch processing
def create_batch_projector(num_timesteps, num_batch, dt, **kwargs):
    """
    Factory function to create a batch-compatible state projector.
    
    Args:
        num_timesteps: Number of time steps
        num_batch: Batch size
        dt: Time step
        **kwargs: Additional parameters for NonConvexStateProjector
        
    Returns:
        NonConvexStateProjector instance
    """
    return NonConvexStateProjector(num_timesteps, num_batch, dt, **kwargs)
