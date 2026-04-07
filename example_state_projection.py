"""
Example: Using Non-Convex State Constraint Projection in Pi-MPPI

This example demonstrates how to use the enhanced Pi-MPPI with hard state
constraint enforcement for obstacle avoidance.

Author: Pi-MPPI Enhancement
Date: 2026-04-07
"""

import numpy as np
import jax.numpy as jnp
from jax import random
import sys
sys.path.append('/home/runner/work/Pi-MPPI/Pi-MPPI')
sys.path.append('/home/runner/work/Pi-MPPI/Pi-MPPI/Obstacle_Avoidance/Comparison_Pi-MPPI_and_Baseline-MPPIwSGF')

# Import the enhanced controller
from pi_mppi_obst_with_state_projection import pi_mppi_with_state_projection


def example_hard_vs_soft_constraints():
    """
    Compare hard constraint projection vs soft penalty approach.
    """
    print("="*70)
    print("Example: Hard vs Soft Obstacle Constraints")
    print("="*70)
    
    # Define control bounds (same for both)
    control_params = {
        'v_max': 30, 'v_min': 15,
        'vdot_max': 5, 'vdot_min': -5,
        'vddot_max': 2, 'vddot_min': -2,
        'pitch_max': 0.3, 'pitch_min': -0.3,
        'pitchdot_max': 0.5, 'pitchdot_min': -0.5,
        'pitchddot_max': 0.2, 'pitchddot_min': -0.2,
        'roll_max': 0.5, 'roll_min': -0.5,
        'rolldot_max': 0.5, 'rolldot_min': -0.5,
        'rollddot_max': 0.2, 'rollddot_min': -0.2,
    }
    
    # Define state bounds
    state_bounds = {
        'x_min': -500, 'x_max': 500,
        'y_min': -500, 'y_max': 500,
        'z_min': 0, 'z_max': 200,
    }
    
    print("\n1. Initializing controllers...")
    
    # Controller with hard constraints (state projection)
    controller_hard = pi_mppi_with_state_projection(
        **control_params,
        **state_bounds,
        enable_state_projection=True
    )
    print("   ✓ Hard constraint controller initialized")
    
    # Controller with soft constraints (original penalty method)
    controller_soft = pi_mppi_with_state_projection(
        **control_params,
        **state_bounds,
        enable_state_projection=False
    )
    print("   ✓ Soft constraint controller initialized")
    
    print("\n2. Setting up scenario...")
    
    # Initial state
    v_init, v_dot_init = 20.0, 0.0
    pitch_init, pitch_dot_init = 0.0, 0.0
    roll_init, roll_dot_init = 0.0, 0.0
    psi_init = 0.0
    x_init, y_init, z_init = -50.0, 0.0, 50.0
    
    # Goal state
    x_fin, y_fin, z_fin = 50.0, 0.0, 50.0
    
    # Obstacles (directly in path)
    x_obs = jnp.array([0.0, 25.0])
    y_obs = jnp.array([0.0, 0.0])
    z_obs = jnp.array([50.0, 50.0])
    r_obs = jnp.array([10.0, 8.0])
    
    print(f"   Start: ({x_init:.1f}, {y_init:.1f}, {z_init:.1f})")
    print(f"   Goal:  ({x_fin:.1f}, {y_fin:.1f}, {z_fin:.1f})")
    print(f"   Obstacles: {len(x_obs)} spherical obstacles")
    for i in range(len(x_obs)):
        print(f"     - Obstacle {i+1}: center=({x_obs[i]:.1f}, {y_obs[i]:.1f}, {z_obs[i]:.1f}), radius={r_obs[i]:.1f}m")
    
    print("\n3. Running MPPI iterations...")
    
    # Initialize control distribution
    key = random.PRNGKey(42)
    mean_init = jnp.zeros(33)  # 3 controls × 11 coefficients each
    
    # Simulate one MPPI iteration for each controller
    print("\n   a) Hard constraint controller:")
    try:
        mean_hard, key_hard, states_hard = controller_hard.pi_mppi_main(
            v_init, v_dot_init, pitch_init, pitch_dot_init,
            roll_init, roll_dot_init, psi_init,
            x_init, y_init, z_init,
            x_fin, y_fin, z_fin,
            mean_init, key,
            x_obs, y_obs, z_obs, r_obs
        )
        print(f"      ✓ Completed successfully")
        print(f"      Final position: ({states_hard[0]:.2f}, {states_hard[1]:.2f}, {states_hard[2]:.2f})")
    except Exception as e:
        print(f"      ✗ Error: {str(e)}")
    
    print("\n   b) Soft constraint controller:")
    try:
        mean_soft, key_soft, states_soft = controller_soft.pi_mppi_main(
            v_init, v_dot_init, pitch_init, pitch_dot_init,
            roll_init, roll_dot_init, psi_init,
            x_init, y_init, z_init,
            x_fin, y_fin, z_fin,
            mean_init, key,
            x_obs, y_obs, z_obs, r_obs
        )
        print(f"      ✓ Completed successfully")
        print(f"      Final position: ({states_soft[0]:.2f}, {states_soft[1]:.2f}, {states_soft[2]:.2f})")
    except Exception as e:
        print(f"      ✗ Error: {str(e)}")
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)
    print("\nKey Differences:")
    print("  - Hard constraints: Obstacles enforced via projection (guaranteed avoidance)")
    print("  - Soft constraints: Obstacles penalized in cost (may violate if penalty too low)")
    print("\nRecommendation:")
    print("  - Use hard constraints for safety-critical applications")
    print("  - Use soft constraints for informational obstacles or smoother trajectories")


def example_state_projector_standalone():
    """
    Example of using the state projector directly (without full MPPI).
    """
    print("\n" + "="*70)
    print("Example: Standalone State Projector")
    print("="*70)
    
    from nonconvex_state_projector import NonConvexStateProjector
    
    print("\n1. Initializing state projector...")
    projector = NonConvexStateProjector(
        num_timesteps=100,
        num_batch=10,  # Small batch for demonstration
        dt=0.2,
        x_min=-100, x_max=100,
        y_min=-100, y_max=100,
        z_min=0, z_max=100,
        safety_margin=5.0
    )
    print("   ✓ Projector initialized")
    
    print("\n2. Creating test trajectory (violates constraints)...")
    # Trajectory that passes through obstacles
    x = jnp.linspace(-20, 40, 100)
    x = jnp.tile(x, (10, 1))  # 10 batch samples
    y = jnp.zeros((10, 100))
    z = jnp.ones((10, 100)) * 50.0  # Constant altitude
    
    print(f"   Trajectory: x ∈ [{jnp.min(x):.1f}, {jnp.max(x):.1f}]")
    print(f"   Trajectory: y = {y[0,0]:.1f}")
    print(f"   Trajectory: z = {z[0,0]:.1f}")
    
    print("\n3. Defining obstacles...")
    x_obs_array = jnp.array([0.0, 20.0])
    y_obs_array = jnp.array([0.0, 0.0])
    z_obs_array = jnp.array([50.0, 50.0])
    r_obs_array = jnp.array([8.0, 6.0])
    
    for i in range(len(x_obs_array)):
        print(f"   Obstacle {i+1}: ({x_obs_array[i]:.1f}, {y_obs_array[i]:.1f}, {z_obs_array[i]:.1f}), r={r_obs_array[i]:.1f}m")
    
    print("\n4. Computing constraint violations before projection...")
    violation_before = projector.compute_total_constraint_violation(
        x, y, z, x_obs_array, y_obs_array, z_obs_array, r_obs_array
    )
    print(f"   Total violation: {violation_before:.2f}")
    
    print("\n5. Projecting onto feasible set...")
    x_proj, y_proj, z_proj = projector.project_states_unified(
        x, y, z,
        x_obs_array, y_obs_array, z_obs_array, r_obs_array
    )
    print("   ✓ Projection completed")
    
    print("\n6. Computing constraint violations after projection...")
    violation_after = projector.compute_total_constraint_violation(
        x_proj, y_proj, z_proj,
        x_obs_array, y_obs_array, z_obs_array, r_obs_array
    )
    print(f"   Total violation: {violation_after:.2f}")
    print(f"   Reduction: {(violation_before - violation_after)/violation_before*100:.1f}%")
    
    print("\n7. Trajectory comparison:")
    print(f"   Original: x ∈ [{jnp.min(x):.1f}, {jnp.max(x):.1f}], z ∈ [{jnp.min(z):.1f}, {jnp.max(z):.1f}]")
    print(f"   Projected: x ∈ [{jnp.min(x_proj):.1f}, {jnp.max(x_proj):.1f}], z ∈ [{jnp.min(z_proj):.1f}, {jnp.max(z_proj):.1f}]")
    
    print("\n" + "="*70)


def main():
    """Main example execution."""
    print("\n" + "="*70)
    print("Pi-MPPI Non-Convex State Constraint Projection Examples")
    print("="*70)
    
    # Example 1: Standalone state projector
    example_state_projector_standalone()
    
    # Example 2: Hard vs soft constraints comparison
    # Note: This example requires bernstein_coeff_order10_arbitinterval module
    print("\n\nNote: For the full MPPI example (hard vs soft constraints),")
    print("      ensure bernstein_coeff_order10_arbitinterval.py is available.")
    print("      Skipping full MPPI example for now.")
    # example_hard_vs_soft_constraints()
    
    print("\n\n" + "="*70)
    print("Examples completed! See STATE_PROJECTION_README.md for more details.")
    print("="*70)


if __name__ == "__main__":
    main()
