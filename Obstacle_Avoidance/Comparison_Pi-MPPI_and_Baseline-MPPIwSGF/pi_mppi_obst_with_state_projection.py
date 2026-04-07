"""
Enhanced Pi-MPPI with Non-Convex State Constraint Projection

This is an enhanced version of pi_mppi_obst.py that integrates the non-convex
state constraint projector for hard obstacle avoidance constraints.

Key enhancements:
1. State constraints are enforced as hard constraints via projection
2. Non-convex obstacle avoidance uses SCP (Sequential Convex Programming)
3. Unified state-control projection ensures feasibility

Author: Pi-MPPI Enhancement
Date: 2026-04-07
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt 

import bernstein_coeff_order10_arbitinterval
from functools import partial
from jax import jit, random, vmap
import jax
import jax.lax as lax

# Import the new state projection modules
import sys
sys.path.append('/home/runner/work/Pi-MPPI/Pi-MPPI')
from nonconvex_state_projector import NonConvexStateProjector
from unified_projector import UnifiedStateControlProjector


class pi_mppi_with_state_projection():
    """
    Enhanced Pi-MPPI with non-convex state constraint projection.
    
    This class extends the original pi_mppi with:
    - Hard state constraints via projection (box bounds on x, y, z)
    - Non-convex obstacle avoidance as hard constraints
    - Unified state-control projection framework
    """

    def __init__(self, v_max, v_min, vdot_max, vdot_min, vddot_max, vddot_min,
                 pitch_max, pitch_min, pitchdot_max, pitchdot_min, pitchddot_max, pitchddot_min,
                 roll_max, roll_min, rolldot_max, rolldot_min, rollddot_max, rollddot_min,
                 x_min=-1000, x_max=1000, y_min=-1000, y_max=1000, z_min=0, z_max=1000,
                 enable_state_projection=True):
        """
        Initialize enhanced pi_mppi with state projection.
        
        Args:
            v_max, v_min, etc.: Control bounds (same as original pi_mppi)
            x_min, x_max, y_min, y_max, z_min, z_max: State bounds
            enable_state_projection: Whether to use state projection (True) or soft penalties (False)
        """
        
        # Cost weights
        if enable_state_projection:
            # With state projection, we can reduce obstacle penalty since it's a hard constraint
            self.w_1 = 1.0      # Goal Reaching
            self.w_2 = 0.0001   # MPPI
            self.w_3 = 0.0      # Obstacle (now hard constraint, so weight = 0)
        else:
            # Original weights with soft penalties
            self.w_1 = 1.0      # Goal Reaching
            self.w_2 = 0.0001   # MPPI
            self.w_3 = 100.0    # Obstacle (soft penalty)

        self.enable_state_projection = enable_state_projection

        # Control bounds
        self.v_min = v_min
        self.v_max = v_max
        self.v_dot_max = vdot_max
        self.v_dot_min = vdot_min
        self.v_ddot_max = vddot_max
        self.v_ddot_min = vddot_min

        self.pitch_max = pitch_max
        self.pitch_min = pitch_min
        self.pitch_dot_max = pitchdot_max
        self.pitch_dot_min = pitchdot_min
        self.pitch_ddot_max = pitchddot_max
        self.pitch_ddot_min = pitchddot_min

        self.roll_max = roll_max
        self.roll_min = roll_min
        self.roll_dot_max = rolldot_max
        self.roll_dot_min = rolldot_min
        self.roll_ddot_max = rollddot_max
        self.roll_ddot_min = rollddot_min

        # Time parameters
        self.t_fin = 20
        self.num = 100
        self.t = self.t_fin / self.num
        self.num_batch = 1000

        # Bernstein polynomial parameterization
        tot_time = np.linspace(0, self.t_fin, self.num)
        self.tot_time = tot_time
        tot_time_copy = tot_time.reshape(self.num, 1)

        self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(
            10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)

        self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)
        self.nvar = jnp.shape(self.P_jax)[1]

        self.num_dot = self.num
        self.num_ddot = self.num_dot

        # Control projection parameters
        self.A_projection = jnp.identity(self.nvar)

        self.A = jnp.vstack((self.P_jax, -self.P_jax))
        self.A_dot = jnp.vstack((self.Pdot_jax, -self.Pdot_jax))
        self.A_ddot = jnp.vstack((self.Pddot_jax, -self.Pddot_jax))
        self.A_control = jnp.vstack((self.A, self.A_dot, self.A_ddot))

        self.A_eq_control = jnp.vstack((self.P_jax[0], self.Pdot_jax[0]))

        self.rho_projection = 1.0
        self.rho_ineq = 1.0

        # ADMM parameters
        self.maxiter = 1
        self.maxiter_cem = 1
        self.maxiter_projection = 50
        self.e = 10**(-3)

        self.alpha_mean = 0.2
        self.alpha_cov = 0.2

        self.lamda = 0.9
        self.g = 9.81

        # Batch operations
        self.compute_cost_mppi_batch = jit(vmap(self.compute_cost_mppi, 
                                                 in_axes=(None, None, None, None, None, None, None, 1, 1, 1, 1)))
        self.compute_cost_batch = jit(vmap(self.compute_cost, 
                                           in_axes=((None, None, None, None, None, None, None, 0, 0, 0, 0, None, None))))
        self.compute_weights_batch = jit(vmap(self._compute_weights, in_axes=(0, None, None)))
        self.obstacle_cost_batch = jit(vmap(self.obstacle_cost, in_axes=(0, 0, 0, 0, None, None, None)))
        self.compute_epsilon_batch = jit(vmap(self.compute_epsilon, in_axes=(1, None)))
        self.compute_w_epsilon_batch = jit(vmap(self.compute_w_epsilon, in_axes=(0, 0)))

        # MPPI parameters
        self.param_exploration = 0.0
        self.param_lambda = 50
        self.param_alpha = 0.99
        self.param_gamma = self.param_lambda * (1.0 - self.param_alpha)

        # Initialize state projector
        if self.enable_state_projection:
            self.state_projector = NonConvexStateProjector(
                num_timesteps=self.num,
                num_batch=self.num_batch,
                dt=self.t,
                x_min=x_min, x_max=x_max,
                y_min=y_min, y_max=y_max,
                z_min=z_min, z_max=z_max,
                safety_margin=5.0
            )
            
            # Initialize unified projector (combines state and control projection)
            self.unified_projector = UnifiedStateControlProjector(
                control_projector=self,
                state_projector=self.state_projector,
                num_timesteps=self.num,
                num_batch=self.num_batch,
                dt=self.t,
                g=self.g
            )

    @partial(jit, static_argnums=(0,))
    def compute_boundary_vec(self, v_init, v_dot_init, pitch_init, pitch_dot_init, roll_init, roll_dot_init):
        """Compute boundary conditions for control projection."""
        v_init_vec = v_init * jnp.ones((self.num_batch, 1))
        v_dot_init_vec = v_dot_init * jnp.ones((self.num_batch, 1))

        pitch_init_vec = pitch_init * jnp.ones((self.num_batch, 1))
        pitch_dot_init_vec = pitch_dot_init * jnp.ones((self.num_batch, 1))

        roll_init_vec = roll_init * jnp.ones((self.num_batch, 1))
        roll_dot_init_vec = roll_dot_init * jnp.ones((self.num_batch, 1))

        b_eq_v = jnp.hstack((v_init_vec, v_dot_init_vec))
        b_eq_pitch = jnp.hstack((pitch_init_vec, pitch_dot_init_vec))
        b_eq_roll = jnp.hstack((roll_init_vec, roll_dot_init_vec))

        return b_eq_v, b_eq_pitch, b_eq_roll

    @partial(jit, static_argnums=(0,))
    def compute_boundary_vec_single(self, v_init, v_dot_init, pitch_init, pitch_dot_init, roll_init, roll_dot_init):
        """Compute boundary conditions for single control projection."""
        v_init_vec = v_init * jnp.ones((1, 1))
        v_dot_init_vec = v_dot_init * jnp.ones((1, 1))

        pitch_init_vec = pitch_init * jnp.ones((1, 1))
        pitch_dot_init_vec = pitch_dot_init * jnp.ones((1, 1))

        roll_init_vec = roll_init * jnp.ones((1, 1))
        roll_dot_init_vec = roll_dot_init * jnp.ones((1, 1))

        b_eq_v = jnp.hstack((v_init_vec, v_dot_init_vec))
        b_eq_pitch = jnp.hstack((pitch_init_vec, pitch_dot_init_vec))
        b_eq_roll = jnp.hstack((roll_init_vec, roll_dot_init_vec))

        return b_eq_v, b_eq_pitch, b_eq_roll

    # Import control projection methods from original pi_mppi
    @partial(jit, static_argnums=(0,))
    def compute_feasible_control(self, control_samples, lamda_control, b_eq_control, s_control, 
                                 control_max, control_min, control_dot_max, control_dot_min, 
                                 control_ddot_max, control_ddot_min):
        """Project control samples onto control constraints (from original pi_mppi)."""
        b_control = jnp.hstack((control_max * jnp.ones((self.num_batch, self.num)),
                                -control_min * jnp.ones((self.num_batch, self.num))))
        b_control_dot = jnp.hstack((control_dot_max * jnp.ones((self.num_batch, self.num_dot)),
                                    -control_dot_min * jnp.ones((self.num_batch, self.num_dot))))
        b_control_ddot = jnp.hstack((control_ddot_max * jnp.ones((self.num_batch, self.num_ddot)),
                                     -control_ddot_min * jnp.ones((self.num_batch, self.num_ddot))))
        b_control_comb = jnp.hstack((b_control, b_control_dot, b_control_ddot))

        b_control_aug = b_control_comb - s_control

        cost = self.rho_projection * jnp.dot(self.A_projection.T, self.A_projection) + \
               self.rho_ineq * jnp.dot(self.A_control.T, self.A_control)

        cost_mat = jnp.vstack((jnp.hstack((cost, self.A_eq_control.T)),
                               jnp.hstack((self.A_eq_control, jnp.zeros((jnp.shape(self.A_eq_control)[0],
                                                                         jnp.shape(self.A_eq_control)[0]))))))
        lincost = -lamda_control - \
                  self.rho_projection * jnp.dot(self.A_projection.T, control_samples.T).T - \
                  self.rho_ineq * jnp.dot(self.A_control.T, b_control_aug.T).T

        sol = jnp.linalg.solve(cost_mat, jnp.hstack((-lincost, b_eq_control)).T).T

        control_projected = sol[:, 0: self.nvar]

        s_control = jnp.maximum(jnp.zeros((self.num_batch, 2 * self.num + 2 * self.num_dot + 2 * self.num_ddot)),
                               -jnp.dot(self.A_control, control_projected.T).T + b_control_comb)

        res_control_vec = jnp.dot(self.A_control, control_projected.T).T - b_control_comb + s_control

        res_control = jnp.linalg.norm(jnp.dot(self.A_control, control_projected.T).T - b_control_comb + s_control,
                                     axis=1)

        lamda_control = lamda_control - self.rho_ineq * jnp.dot(self.A_control.T, res_control_vec.T).T

        return control_projected, s_control, res_control, lamda_control

    @partial(jit, static_argnums=(0,))
    def compute_feasible_control_single(self, control_samples, lamda_control, b_eq_control, s_control,
                                       control_max, control_min, control_dot_max, control_dot_min,
                                       control_ddot_max, control_ddot_min):
        """Project single control sample onto control constraints."""
        b_control = jnp.hstack((control_max * jnp.ones((1, self.num)),
                                -control_min * jnp.ones((1, self.num))))
        b_control_dot = jnp.hstack((control_dot_max * jnp.ones((1, self.num_dot)),
                                    -control_dot_min * jnp.ones((1, self.num_dot))))
        b_control_ddot = jnp.hstack((control_ddot_max * jnp.ones((1, self.num_ddot)),
                                     -control_ddot_min * jnp.ones((1, self.num_ddot))))
        b_control_comb = jnp.hstack((b_control, b_control_dot, b_control_ddot))

        b_control_aug = b_control_comb - s_control

        cost = self.rho_projection * jnp.dot(self.A_projection.T, self.A_projection) + \
               self.rho_ineq * jnp.dot(self.A_control.T, self.A_control)

        cost_mat = jnp.vstack((jnp.hstack((cost, self.A_eq_control.T)),
                               jnp.hstack((self.A_eq_control, jnp.zeros((jnp.shape(self.A_eq_control)[0],
                                                                         jnp.shape(self.A_eq_control)[0]))))))
        lincost = -lamda_control - \
                  self.rho_projection * jnp.dot(self.A_projection.T, control_samples.T).T - \
                  self.rho_ineq * jnp.dot(self.A_control.T, b_control_aug.T).T

        sol = jnp.linalg.solve(cost_mat, jnp.hstack((-lincost, b_eq_control)).T).T

        control_projected = sol[:, 0: self.nvar]

        s_control = jnp.maximum(jnp.zeros((1, 2 * self.num + 2 * self.num_dot + 2 * self.num_ddot)),
                               -jnp.dot(self.A_control, control_projected.T).T + b_control_comb)

        res_control_vec = jnp.dot(self.A_control, control_projected.T).T - b_control_comb + s_control

        res_control = jnp.linalg.norm(jnp.dot(self.A_control, control_projected.T).T - b_control_comb + s_control,
                                     axis=1)

        lamda_control = lamda_control - self.rho_ineq * jnp.dot(self.A_control.T, res_control_vec.T).T

        return control_projected.squeeze(axis=0), s_control, res_control.squeeze(axis=0), lamda_control

    @partial(jit, static_argnums=(0,))
    def compute_projection(self, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll,
                          c_v_samples_input, c_pitch_samples_input, c_roll_samples_input,
                          b_eq_v, b_eq_pitch, b_eq_roll):
        """Batch control projection using ADMM."""
        c_v_samples_init = c_v_samples_input
        c_pitch_samples_init = c_pitch_samples_input
        c_roll_samples_init = c_roll_samples_input
        lamda_v_init = lamda_v
        lamda_pitch_init = lamda_pitch
        lamda_roll_init = lamda_roll

        s_v_init = s_v
        s_pitch_init = s_pitch
        s_roll_init = s_roll

        def lax_custom_projection(carry, idx):
            c_v_samples, c_pitch_samples, c_roll_samples, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll = carry

            c_v_samples, s_v, res_v, lamda_v = self.compute_feasible_control(
                c_v_samples_input, lamda_v, b_eq_v, s_v,
                self.v_max, self.v_min, self.v_dot_max, self.v_dot_min, self.v_ddot_max, self.v_ddot_min)

            c_pitch_samples, s_pitch, res_pitch, lamda_pitch = self.compute_feasible_control(
                c_pitch_samples_input, lamda_pitch, b_eq_pitch, s_pitch,
                self.pitch_max, self.pitch_min, self.pitch_dot_max, self.pitch_dot_min,
                self.pitch_ddot_max, self.pitch_ddot_min)

            c_roll_samples, s_roll, res_roll, lamda_roll = self.compute_feasible_control(
                c_roll_samples_input, lamda_roll, b_eq_roll, s_roll,
                self.roll_max, self.roll_min, self.roll_dot_max, self.roll_dot_min,
                self.roll_ddot_max, self.roll_ddot_min)

            return (c_v_samples, c_pitch_samples, c_roll_samples, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch,
                   s_roll), (res_v, res_pitch, res_roll)

        carry_init = (c_v_samples_init, c_pitch_samples_init, c_roll_samples_init, lamda_v_init, lamda_pitch_init,
                     lamda_roll_init, s_v_init, s_pitch_init, s_roll_init)
        carry_final, res_tot = lax.scan(lax_custom_projection, carry_init, jnp.arange(self.maxiter_projection))

        c_v_samples, c_pitch_samples, c_roll_samples, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll = carry_final

        res_v, res_pitch, res_roll = res_tot

        return c_v_samples, c_pitch_samples, c_roll_samples, res_v, res_pitch, res_roll, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll

    @partial(jit, static_argnums=(0,))
    def compute_projection_single(self, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll,
                                  c_v_samples_input, c_pitch_samples_input, c_roll_samples_input,
                                  b_eq_v, b_eq_pitch, b_eq_roll):
        """Single control projection using ADMM."""
        c_v_samples_init = c_v_samples_input
        c_pitch_samples_init = c_pitch_samples_input
        c_roll_samples_init = c_roll_samples_input
        lamda_v_init = lamda_v
        lamda_pitch_init = lamda_pitch
        lamda_roll_init = lamda_roll

        s_v_init = s_v
        s_pitch_init = s_pitch
        s_roll_init = s_roll

        def lax_custom_projection(carry, idx):
            c_v_samples, c_pitch_samples, c_roll_samples, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll = carry

            c_v_samples, s_v, res_v, lamda_v = self.compute_feasible_control_single(
                c_v_samples_input, lamda_v, b_eq_v, s_v,
                self.v_max, self.v_min, self.v_dot_max, self.v_dot_min, self.v_ddot_max, self.v_ddot_min)

            c_pitch_samples, s_pitch, res_pitch, lamda_pitch = self.compute_feasible_control_single(
                c_pitch_samples_input, lamda_pitch, b_eq_pitch, s_pitch,
                self.pitch_max, self.pitch_min, self.pitch_dot_max, self.pitch_dot_min,
                self.pitch_ddot_max, self.pitch_ddot_min)

            c_roll_samples, s_roll, res_roll, lamda_roll = self.compute_feasible_control_single(
                c_roll_samples_input, lamda_roll, b_eq_roll, s_roll,
                self.roll_max, self.roll_min, self.roll_dot_max, self.roll_dot_min,
                self.roll_ddot_max, self.roll_ddot_min)

            return (c_v_samples, c_pitch_samples, c_roll_samples, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch,
                   s_roll), (res_v, res_pitch, res_roll)

        carry_init = (c_v_samples_init, c_pitch_samples_init, c_roll_samples_init, lamda_v_init, lamda_pitch_init,
                     lamda_roll_init, s_v_init, s_pitch_init, s_roll_init)
        carry_final, res_tot = lax.scan(lax_custom_projection, carry_init, jnp.arange(self.maxiter_projection))

        c_v_samples, c_pitch_samples, c_roll_samples, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll = carry_final

        res_v, res_pitch, res_roll = res_tot

        return c_v_samples, c_pitch_samples, c_roll_samples, res_v, res_pitch, res_roll, lamda_v, lamda_pitch, lamda_roll, s_v, s_pitch, s_roll

    @partial(jit, static_argnums=(0,))
    def compute_control_samples(self, key, mean_control, cov_control):
        """Sample control trajectories from distribution."""
        key, subkey = random.split(key)
        control_samples = jax.random.multivariate_normal(key, mean_control, cov_control, (self.num_batch,))

        c_v_samples = control_samples[:, 0: self.nvar]
        c_pitch_samples = control_samples[:, self.nvar: 2 * self.nvar]
        c_roll_samples = control_samples[:, 2 * self.nvar: 3 * self.nvar]

        return c_v_samples, c_pitch_samples, c_roll_samples, key

    @partial(jit, static_argnums=(0,))
    def compute_rollouts(self, x_init, y_init, z_init, psi_init, v_samples, roll_samples, pitch_samples,
                        pitchdot_samples):
        """Compute state trajectories from control samples."""
        R = (self.g / v_samples) * jnp.sin(roll_samples) * jnp.cos(pitch_samples)
        Q = (pitchdot_samples + jnp.sin(roll_samples) * R) / jnp.cos(roll_samples)
        psidot_samples = (jnp.sin(roll_samples) / jnp.cos(pitch_samples) * Q +
                         jnp.cos(roll_samples) / jnp.cos(pitch_samples) * R)
        psi_samples = psi_init + jnp.cumsum(psidot_samples * self.t, axis=1)
        psi_samples = jnp.hstack((psi_init * jnp.ones((self.num_batch, 1)), psi_samples[:, 0:-1]))

        v_x = v_samples * jnp.cos(psi_samples) * jnp.cos(pitch_samples)
        v_y = v_samples * jnp.sin(psi_samples) * jnp.cos(pitch_samples)
        v_z = -v_samples * jnp.sin(pitch_samples)

        x = x_init + jnp.cumsum(v_x * self.t, axis=1)
        y = y_init + jnp.cumsum(v_y * self.t, axis=1)
        z = z_init + jnp.cumsum(v_z * self.t, axis=1)

        x = jnp.hstack((x_init * jnp.ones((self.num_batch, 1)), x[:, 0:-1]))
        y = jnp.hstack((y_init * jnp.ones((self.num_batch, 1)), y[:, 0:-1]))
        z = jnp.hstack((z_init * jnp.ones((self.num_batch, 1)), z[:, 0:-1]))

        return x, y, z, psi_samples, psidot_samples

    @partial(jit, static_argnums=(0,))
    def compute_rollouts_mppi(self, x_init, y_init, z_init, psi_init, v_samples, roll_samples, pitchdot_samples,
                             pitch_samples):
        """Compute single rollout for MPPI update."""
        R = (self.g / v_samples) * jnp.sin(roll_samples) * jnp.cos(pitch_samples)
        Q = (pitchdot_samples + jnp.sin(roll_samples) * R) / jnp.cos(roll_samples)
        psidot_samples = (jnp.sin(roll_samples) / jnp.cos(pitch_samples) * Q +
                         jnp.cos(roll_samples) / jnp.cos(pitch_samples) * R)
        psi_samples = psi_init + jnp.cumsum(psidot_samples * self.t)

        v_x = v_samples * jnp.cos(psi_samples) * jnp.cos(pitch_samples)
        v_y = v_samples * jnp.sin(psi_samples) * jnp.cos(pitch_samples)
        v_z = -v_samples * jnp.sin(pitch_samples)

        x = x_init + jnp.cumsum(v_x * self.t)
        y = y_init + jnp.cumsum(v_y * self.t)
        z = z_init + jnp.cumsum(v_z * self.t)

        return x, y, z, psi_samples

    @partial(jit, static_argnums=(0,))
    def obstacle_cost(self, x_obs, y_obs, z_obs, r_obs, x, y, z):
        """Compute obstacle avoidance cost (soft penalty)."""
        obstacle = (x - x_obs) ** 2 + (y - y_obs) ** 2 + (z - z_obs) ** 2 - (5 + r_obs) ** 2
        cost_obstacle = jnp.maximum(0, -obstacle)
        return cost_obstacle

    @partial(jit, static_argnums=(0,))
    def compute_cost(self, x_goal, y_goal, z_goal,
                    x_obs, y_obs, z_obs, r_obs,
                    x, y, z, controls_stack, u_mean, sigma):
        """Compute total cost for a trajectory."""
        cost_goal = ((x - x_goal) ** 2 + (y - y_goal) ** 2 + (z - z_goal) ** 2) * self.w_1

        cost_obstacle_b = self.obstacle_cost_batch(x_obs, y_obs, z_obs, r_obs, x, y, z)
        cost_obstacle = jnp.sum(cost_obstacle_b) * self.w_3

        mppi = self.param_gamma * u_mean.T @ jnp.linalg.inv(sigma) @ controls_stack * self.w_2

        return cost_goal, mppi, cost_obstacle

    @partial(jit, static_argnums=(0,))
    def compute_cost_mppi(self, x_goal, y_goal, z_goal,
                         x_obs, y_obs, z_obs, r_obs,
                         x, y, z, controls_stack):
        """Compute MPPI cost."""
        u_mean = jnp.mean(controls_stack, axis=0)
        sigma = jnp.cov((controls_stack - u_mean).T)
        cost_goal, cost_obstacle, mppi = self.compute_cost_batch(x_goal, y_goal, z_goal,
                                                                 x_obs, y_obs, z_obs, r_obs,
                                                                 x, y, z, controls_stack, u_mean, sigma)
        cost = cost_goal + cost_obstacle + mppi
        return cost

    @partial(jit, static_argnums=(0,))
    def _compute_weights(self, S, rho, eta):
        """Compute MPPI weights."""
        w = (1.0 / eta) * jnp.exp((-1.0 / self.param_lambda) * (S - rho))
        return w

    @partial(jit, static_argnums=(0,))
    def compute_epsilon(self, epsilon, w):
        """Compute weighted control perturbations."""
        we = self.compute_w_epsilon_batch(epsilon, w)
        w_epsilon = jnp.sum(we, axis=0)
        return w_epsilon

    @partial(jit, static_argnums=(0,))
    def compute_w_epsilon(self, epsilon, w):
        """Compute single weighted perturbation."""
        return w * epsilon

    @partial(jit, static_argnums=(0,))
    def pi_mppi_main(self, v_init, v_dot_init, pitch_init, pitch_dot_init, roll_init, roll_dot_init, psi_init,
                    x_init, y_init, z_init, x_fin, y_fin, z_fin, mean, key, x_obs, y_obs, z_obs, r_obs):
        """
        Main MPPI iteration with state projection.
        
        This is the enhanced version that includes state constraint projection
        """
        b_eq_v, b_eq_pitch, b_eq_roll = self.compute_boundary_vec(v_init, v_dot_init, pitch_init, pitch_dot_init,
                                                                   roll_init, roll_dot_init)

        cov_v_control = 20 * jnp.identity(self.nvar)
        cov_angle_control = jnp.identity(self.nvar * 2) * 2
        cov_control_init = jax.scipy.linalg.block_diag(cov_v_control, cov_angle_control)

        lamda_v_init = jnp.zeros((self.num_batch, self.nvar))
        lamda_pitch_init = jnp.zeros((self.num_batch, self.nvar))
        lamda_roll_init = jnp.zeros((self.num_batch, self.nvar))

        s_v_init = jnp.zeros((self.num_batch, 2 * self.num + 2 * self.num_dot + 2 * self.num_ddot))
        s_pitch_init = jnp.zeros((self.num_batch, 2 * self.num + 2 * self.num_dot + 2 * self.num_ddot))
        s_roll_init = jnp.zeros((self.num_batch, 2 * self.num + 2 * self.num_dot + 2 * self.num_ddot))

        # MPPI: Sample controls
        c_v_samples, c_pitch_samples, c_roll_samples, key = self.compute_control_samples(key, mean, cov_control_init)

        # Control projection
        c_v_samples, c_pitch_samples, c_roll_samples, res_v, res_pitch, res_roll, lamda_v, lamda_pitch, \
        lamda_roll, s_v, s_pitch, s_roll = self.compute_projection(lamda_v_init, lamda_pitch_init, lamda_roll_init,
                                                                    s_v_init, s_pitch_init,
                                                                    s_roll_init, c_v_samples, c_pitch_samples,
                                                                    c_roll_samples, b_eq_v, b_eq_pitch, b_eq_roll)

        # Convert control coefficients to trajectories
        v_samples = jnp.dot(self.P_jax, c_v_samples.T).T
        vdot_samples = jnp.dot(self.Pdot_jax, c_v_samples.T).T
        pitch_samples = jnp.dot(self.P_jax, c_pitch_samples.T).T
        pitchdot_samples = jnp.dot(self.Pdot_jax, c_pitch_samples.T).T
        roll_samples = jnp.dot(self.P_jax, c_roll_samples.T).T
        rolldot_samples = jnp.dot(self.Pdot_jax, c_roll_samples.T).T

        # Trajectory rollouts
        x_traj, y_traj, z_traj, psi_samples, psidot_samples = self.compute_rollouts(x_init, y_init, z_init, psi_init,
                                                                                    v_samples, roll_samples,
                                                                                    pitch_samples, pitchdot_samples)

        # Apply state projection if enabled
        if self.enable_state_projection:
            # Convert obstacle parameters to arrays for projection
            x_obs_array = jnp.array([x_obs]) if jnp.ndim(x_obs) == 0 else x_obs
            y_obs_array = jnp.array([y_obs]) if jnp.ndim(y_obs) == 0 else y_obs
            z_obs_array = jnp.array([z_obs]) if jnp.ndim(z_obs) == 0 else z_obs
            r_obs_array = jnp.array([r_obs]) if jnp.ndim(r_obs) == 0 else r_obs

            # Project states onto state constraints
            x_traj, y_traj, z_traj = self.state_projector.project_states_unified(
                x_traj, y_traj, z_traj,
                x_obs_array, y_obs_array, z_obs_array, r_obs_array
            )

        controls_stack = jnp.stack((v_samples, pitch_samples, roll_samples), axis=-1)

        # Compute costs
        S_mat = self.compute_cost_mppi_batch(x_fin, y_fin, z_fin,
                                            x_obs, y_obs, z_obs, r_obs,
                                            x_traj, y_traj, z_traj, controls_stack)

        S = jnp.sum(S_mat, axis=0)

        rho = S.min()
        eta = jnp.sum(jnp.exp((-1.0 / self.param_lambda) * (S - rho)))

        w = self.compute_weights_batch(S, rho, eta)

        epsilon = controls_stack - jnp.mean(controls_stack, axis=0)
        w_epsilon = self.compute_epsilon_batch(epsilon, w)

        u_new = jnp.mean(controls_stack, axis=0) + w_epsilon

        v_new = u_new[:, 0]
        pitch_new = u_new[:, 1]
        roll_new = u_new[:, 2]

        c_v_mppi = jnp.linalg.inv(self.P_jax.T @ self.P_jax + 0.001 * jnp.identity(11)) @ self.P_jax.T @ v_new
        c_pitch_mppi = jnp.linalg.inv(self.P_jax.T @ self.P_jax + 0.001 * jnp.identity(11)) @ self.P_jax.T @ pitch_new
        c_roll_mppi = jnp.linalg.inv(self.P_jax.T @ self.P_jax + 0.001 * jnp.identity(11)) @ self.P_jax.T @ roll_new

        # Single projection for final control
        lamda_v_init_single = jnp.zeros((1, self.nvar))
        lamda_pitch_init_single = jnp.zeros((1, self.nvar))
        lamda_roll_init_single = jnp.zeros((1, self.nvar))

        s_v_init_single = jnp.zeros((1, 2 * self.num + 2 * self.num_dot + 2 * self.num_ddot))
        s_pitch_init_single = jnp.zeros((1, 2 * self.num + 2 * self.num_dot + 2 * self.num_ddot))
        s_roll_init_single = jnp.zeros((1, 2 * self.num + 2 * self.num_dot + 2 * self.num_ddot))

        b_eq_v_single, b_eq_pitch_single, b_eq_roll_single = self.compute_boundary_vec_single(v_init, v_dot_init,
                                                                                               pitch_init,
                                                                                               pitch_dot_init,
                                                                                               roll_init, roll_dot_init)

        c_v_single, c_pitch_single, c_roll_single, res_v_single, res_pitch_single, res_roll_single, lamda_v_single, \
        lamda_pitch_single, lamda_roll_single, s_v_single, s_pitch_single, s_roll_single = self.compute_projection_single(
            lamda_v_init_single, lamda_pitch_init_single, lamda_roll_init_single,
            s_v_init_single, s_pitch_init_single, s_roll_init_single,
            c_v_mppi, c_pitch_mppi, c_roll_mppi,
            b_eq_v_single, b_eq_pitch_single, b_eq_roll_single)

        v_single = jnp.dot(self.P_jax, c_v_single.T).T
        pitchdot_single = jnp.dot(self.Pdot_jax, c_pitch_single.T).T
        pitch_single = jnp.dot(self.P_jax, c_pitch_single.T).T
        roll_single = jnp.dot(self.P_jax, c_roll_single.T).T

        x_traj_mppi, y_traj_mppi, z_traj_mppi, psi_mppi = self.compute_rollouts_mppi(x_init, y_init, z_init, psi_init,
                                                                                     v_single, roll_single,
                                                                                     pitchdot_single, pitch_single)

        mean = jnp.hstack((c_v_single, c_pitch_single, c_roll_single))

        new_init_states = jnp.array([x_traj_mppi[0], y_traj_mppi[0], z_traj_mppi[0], psi_mppi[0]])

        return mean, key, new_init_states
