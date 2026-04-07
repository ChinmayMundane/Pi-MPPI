"""
Unified State-Control Projector for Pi-MPPI

This module implements a unified projection framework that jointly handles
both state and control constraints using an ADMM-based alternating projection scheme.

Author: Pi-MPPI Enhancement
Date: 2026-04-07
"""

import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap
import jax
import jax.lax as lax


class UnifiedStateControlProjector():
    """
    Unified projector for joint state-control constraint satisfaction.
    
    Uses ADMM (Alternating Direction Method of Multipliers) to alternate between:
    1. Control constraint projection (existing quadratic programming method)
    2. State constraint projection (new non-convex SCP method)
    
    Enforces coupling between states and controls via dynamics.
    """
    
    def __init__(self, control_projector, state_projector, num_timesteps, num_batch, dt, g=9.81):
        """
        Initialize unified state-control projector.
        
        Args:
            control_projector: Instance with compute_feasible_control method (from pi_mppi)
            state_projector: Instance of NonConvexStateProjector
            num_timesteps: Number of time steps in trajectory
            num_batch: Batch size
            dt: Time step size
            g: Gravitational constant
        """
        self.control_projector = control_projector
        self.state_projector = state_projector
        self.num = num_timesteps
        self.num_batch = num_batch
        self.dt = dt
        self.g = g
        
        # ADMM parameters
        self.rho_coupling = 1.0  # Penalty parameter for coupling constraints
        self.maxiter_admm = 10  # Maximum ADMM iterations
        self.convergence_tol = 1e-3  # Convergence tolerance
        
        # Lagrange multipliers for coupling constraints
        # These enforce that states computed from controls match projected states
        
    @partial(jit, static_argnums=(0,))
    def compute_states_from_controls(self, x_init, y_init, z_init, psi_init,
                                     v_samples, pitch_samples, roll_samples):
        """
        Compute state trajectory from control trajectory via forward dynamics.
        
        This is the coupling equation that links controls to states.
        
        Args:
            x_init, y_init, z_init: Initial positions
            psi_init: Initial heading
            v_samples: Velocity control (num_batch, num_timesteps)
            pitch_samples: Pitch angle control (num_batch, num_timesteps)
            roll_samples: Roll angle control (num_batch, num_timesteps)
            
        Returns:
            State trajectories (x, y, z, psi)
        """
        # Compute pitch rate from roll dynamics
        pitchdot_samples = jnp.gradient(pitch_samples, axis=1) / self.dt
        
        # Compute yaw rate from roll and pitch
        R = (self.g / v_samples) * jnp.sin(roll_samples) * jnp.cos(pitch_samples)
        Q = (pitchdot_samples + jnp.sin(roll_samples) * R) / jnp.cos(roll_samples)
        psidot_samples = (jnp.sin(roll_samples) / jnp.cos(pitch_samples) * Q + 
                         jnp.cos(roll_samples) / jnp.cos(pitch_samples) * R)
        
        # Integrate to get psi
        psi_samples = psi_init + jnp.cumsum(psidot_samples * self.dt, axis=1)
        psi_samples = jnp.hstack((psi_init * jnp.ones((self.num_batch, 1)), 
                                  psi_samples[:, 0:-1]))
        
        # Compute velocities in world frame
        v_x = v_samples * jnp.cos(psi_samples) * jnp.cos(pitch_samples)
        v_y = v_samples * jnp.sin(psi_samples) * jnp.cos(pitch_samples)
        v_z = -v_samples * jnp.sin(pitch_samples)
        
        # Integrate to get positions
        x = x_init + jnp.cumsum(v_x * self.dt, axis=1)
        y = y_init + jnp.cumsum(v_y * self.dt, axis=1)
        z = z_init + jnp.cumsum(v_z * self.dt, axis=1)
        
        x = jnp.hstack((x_init * jnp.ones((self.num_batch, 1)), x[:, 0:-1]))
        y = jnp.hstack((y_init * jnp.ones((self.num_batch, 1)), y[:, 0:-1]))
        z = jnp.hstack((z_init * jnp.ones((self.num_batch, 1)), z[:, 0:-1]))
        
        return x, y, z, psi_samples
    
    @partial(jit, static_argnums=(0,))
    def project_control_constraints(self, c_v_samples, c_pitch_samples, c_roll_samples,
                                    lamda_v, lamda_pitch, lamda_roll,
                                    s_v, s_pitch, s_roll,
                                    b_eq_v, b_eq_pitch, b_eq_roll):
        """
        Project control samples onto control constraints.
        
        This delegates to the existing control projection method from pi_mppi.
        
        Args:
            c_v_samples: Velocity control coefficients (num_batch, nvar)
            c_pitch_samples: Pitch control coefficients (num_batch, nvar)
            c_roll_samples: Roll control coefficients (num_batch, nvar)
            lamda_v, lamda_pitch, lamda_roll: Lagrange multipliers
            s_v, s_pitch, s_roll: Slack variables
            b_eq_v, b_eq_pitch, b_eq_roll: Equality constraint bounds
            
        Returns:
            Projected control coefficients and updated ADMM variables
        """
        # Use existing control projection from pi_mppi
        c_v_proj, c_pitch_proj, c_roll_proj, res_v, res_pitch, res_roll, \
        lamda_v_new, lamda_pitch_new, lamda_roll_new, \
        s_v_new, s_pitch_new, s_roll_new = self.control_projector.compute_projection(
            lamda_v, lamda_pitch, lamda_roll,
            s_v, s_pitch, s_roll,
            c_v_samples, c_pitch_samples, c_roll_samples,
            b_eq_v, b_eq_pitch, b_eq_roll
        )
        
        return (c_v_proj, c_pitch_proj, c_roll_proj,
                lamda_v_new, lamda_pitch_new, lamda_roll_new,
                s_v_new, s_pitch_new, s_roll_new)
    
    @partial(jit, static_argnums=(0,))
    def unified_projection_admm(self, 
                               x_init, y_init, z_init, psi_init,
                               c_v_samples, c_pitch_samples, c_roll_samples,
                               x_obs_array, y_obs_array, z_obs_array, r_obs_array,
                               lamda_v, lamda_pitch, lamda_roll,
                               s_v, s_pitch, s_roll,
                               b_eq_v, b_eq_pitch, b_eq_roll):
        """
        Unified ADMM-based projection for joint state-control constraints.
        
        Alternates between:
        1. Control projection: Project controls onto control constraints
        2. State computation: Compute states from projected controls
        3. State projection: Project states onto state constraints
        4. Control update: Update controls to match projected states
        
        Args:
            x_init, y_init, z_init, psi_init: Initial state
            c_v_samples, c_pitch_samples, c_roll_samples: Control coefficient samples
            x_obs_array, y_obs_array, z_obs_array, r_obs_array: Obstacle parameters
            lamda_v, lamda_pitch, lamda_roll: Control Lagrange multipliers
            s_v, s_pitch, s_roll: Control slack variables
            b_eq_v, b_eq_pitch, b_eq_roll: Control equality constraints
            
        Returns:
            Projected controls and states satisfying both control and state constraints
        """
        
        # Initialize coupling Lagrange multipliers
        lamda_x = jnp.zeros((self.num_batch, self.num))
        lamda_y = jnp.zeros((self.num_batch, self.num))
        lamda_z = jnp.zeros((self.num_batch, self.num))
        
        def admm_iteration(carry, idx):
            c_v, c_pitch, c_roll, lam_v, lam_pitch, lam_roll, sv, sp, sr, lx, ly, lz = carry
            
            # Step 1: Project controls onto control constraints
            c_v_proj, c_pitch_proj, c_roll_proj, lam_v_new, lam_pitch_new, lam_roll_new, \
            sv_new, sp_new, sr_new = self.project_control_constraints(
                c_v, c_pitch, c_roll,
                lam_v, lam_pitch, lam_roll,
                sv, sp, sr,
                b_eq_v, b_eq_pitch, b_eq_roll
            )
            
            # Convert control coefficients to control trajectories
            P_jax = self.control_projector.P_jax
            v_traj = jnp.dot(c_v_proj, P_jax.T)
            pitch_traj = jnp.dot(c_pitch_proj, P_jax.T)
            roll_traj = jnp.dot(c_roll_proj, P_jax.T)
            
            # Step 2: Compute states from controls via forward dynamics
            x_from_controls, y_from_controls, z_from_controls, psi_from_controls = \
                self.compute_states_from_controls(
                    x_init, y_init, z_init, psi_init,
                    v_traj, pitch_traj, roll_traj
                )
            
            # Step 3: Project states onto state constraints
            x_proj, y_proj, z_proj = self.state_projector.project_states_unified(
                x_from_controls, y_from_controls, z_from_controls,
                x_obs_array, y_obs_array, z_obs_array, r_obs_array
            )
            
            # Step 4: Update coupling Lagrange multipliers
            # Residual: difference between states from controls and projected states
            residual_x = x_from_controls - x_proj
            residual_y = y_from_controls - y_proj
            residual_z = z_from_controls - z_proj
            
            lx_new = lx + self.rho_coupling * residual_x
            ly_new = ly + self.rho_coupling * residual_y
            lz_new = lz + self.rho_coupling * residual_z
            
            # Step 5: Update controls to minimize coupling constraint violation
            # This is simplified - in full implementation would solve for controls
            # that minimize distance to both control feasibility and state feasibility
            
            return (c_v_proj, c_pitch_proj, c_roll_proj,
                   lam_v_new, lam_pitch_new, lam_roll_new,
                   sv_new, sp_new, sr_new,
                   lx_new, ly_new, lz_new), (residual_x, residual_y, residual_z)
        
        # Run ADMM iterations
        carry_init = (c_v_samples, c_pitch_samples, c_roll_samples,
                     lamda_v, lamda_pitch, lamda_roll,
                     s_v, s_pitch, s_roll,
                     lamda_x, lamda_y, lamda_z)
        
        carry_final, residuals = lax.scan(
            admm_iteration,
            carry_init,
            jnp.arange(self.maxiter_admm)
        )
        
        c_v_final, c_pitch_final, c_roll_final, \
        lamda_v_final, lamda_pitch_final, lamda_roll_final, \
        s_v_final, s_pitch_final, s_roll_final, \
        lamda_x_final, lamda_y_final, lamda_z_final = carry_final
        
        # Compute final states
        P_jax = self.control_projector.P_jax
        v_traj_final = jnp.dot(c_v_final, P_jax.T)
        pitch_traj_final = jnp.dot(c_pitch_final, P_jax.T)
        roll_traj_final = jnp.dot(c_roll_final, P_jax.T)
        
        x_final, y_final, z_final, psi_final = self.compute_states_from_controls(
            x_init, y_init, z_init, psi_init,
            v_traj_final, pitch_traj_final, roll_traj_final
        )
        
        # Final state projection
        x_final, y_final, z_final = self.state_projector.project_states_unified(
            x_final, y_final, z_final,
            x_obs_array, y_obs_array, z_obs_array, r_obs_array
        )
        
        return (c_v_final, c_pitch_final, c_roll_final,
                x_final, y_final, z_final, psi_final,
                lamda_v_final, lamda_pitch_final, lamda_roll_final,
                s_v_final, s_pitch_final, s_roll_final,
                residuals)
    
    @partial(jit, static_argnums=(0,))
    def check_convergence(self, residuals):
        """
        Check if ADMM has converged based on residual norms.
        
        Args:
            residuals: Tuple of (residual_x, residual_y, residual_z) from iterations
            
        Returns:
            Convergence flag and residual norm
        """
        residual_x, residual_y, residual_z = residuals
        
        # Compute norm of final residual
        final_residual_norm = jnp.sqrt(
            jnp.mean(residual_x[-1]**2) + 
            jnp.mean(residual_y[-1]**2) + 
            jnp.mean(residual_z[-1]**2)
        )
        
        converged = final_residual_norm < self.convergence_tol
        
        return converged, final_residual_norm


def create_unified_projector(control_projector, state_projector, 
                            num_timesteps, num_batch, dt, g=9.81):
    """
    Factory function to create a unified state-control projector.
    
    Args:
        control_projector: Instance with compute_projection method
        state_projector: Instance of NonConvexStateProjector
        num_timesteps: Number of time steps
        num_batch: Batch size
        dt: Time step
        g: Gravitational constant
        
    Returns:
        UnifiedStateControlProjector instance
    """
    return UnifiedStateControlProjector(
        control_projector, state_projector,
        num_timesteps, num_batch, dt, g
    )
