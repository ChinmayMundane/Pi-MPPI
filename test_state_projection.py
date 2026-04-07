"""
Unit Tests for Non-Convex State Constraint Projection

This module contains tests to validate the state projection modules:
- NonConvexStateProjector
- UnifiedStateControlProjector

Author: Pi-MPPI Enhancement
Date: 2026-04-07
"""

import numpy as np
import jax.numpy as jnp
import jax
from jax import random

# Import the new modules
import sys
sys.path.append('/home/runner/work/Pi-MPPI/Pi-MPPI')
from nonconvex_state_projector import NonConvexStateProjector, create_batch_projector
from unified_projector import UnifiedStateControlProjector, create_unified_projector


class TestNonConvexStateProjector:
    """Test suite for NonConvexStateProjector class."""
    
    def __init__(self):
        """Initialize test parameters."""
        self.num_timesteps = 100
        self.num_batch = 10
        self.dt = 0.2
        
        # Create projector instance
        self.projector = NonConvexStateProjector(
            num_timesteps=self.num_timesteps,
            num_batch=self.num_batch,
            dt=self.dt,
            x_min=-100, x_max=100,
            y_min=-100, y_max=100,
            z_min=0, z_max=100,
            safety_margin=5.0
        )
        
        print("TestNonConvexStateProjector initialized successfully")
    
    def test_box_constraints(self):
        """Test box constraint projection."""
        print("\n=== Testing Box Constraint Projection ===")
        
        # Create test trajectories that violate bounds
        key = random.PRNGKey(42)
        key, subkey = random.split(key)
        
        x = random.uniform(subkey, (self.num_batch, self.num_timesteps), 
                          minval=-150, maxval=150)
        key, subkey = random.split(key)
        y = random.uniform(subkey, (self.num_batch, self.num_timesteps),
                          minval=-150, maxval=150)
        key, subkey = random.split(key)
        z = random.uniform(subkey, (self.num_batch, self.num_timesteps),
                          minval=-50, maxval=150)
        
        # Project onto box constraints
        x_proj, y_proj, z_proj = self.projector.project_box_constraints(x, y, z)
        
        # Verify projections satisfy bounds
        assert jnp.all(x_proj >= self.projector.x_min), "X projection violates lower bound"
        assert jnp.all(x_proj <= self.projector.x_max), "X projection violates upper bound"
        assert jnp.all(y_proj >= self.projector.y_min), "Y projection violates lower bound"
        assert jnp.all(y_proj <= self.projector.y_max), "Y projection violates upper bound"
        assert jnp.all(z_proj >= self.projector.z_min), "Z projection violates lower bound"
        assert jnp.all(z_proj <= self.projector.z_max), "Z projection violates upper bound"
        
        print("✓ Box constraint projection test passed")
        print(f"  Original x range: [{jnp.min(x):.2f}, {jnp.max(x):.2f}]")
        print(f"  Projected x range: [{jnp.min(x_proj):.2f}, {jnp.max(x_proj):.2f}]")
        print(f"  Original z range: [{jnp.min(z):.2f}, {jnp.max(z):.2f}]")
        print(f"  Projected z range: [{jnp.min(z_proj):.2f}, {jnp.max(z_proj):.2f}]")
        
        return True
    
    def test_obstacle_constraint_violation(self):
        """Test obstacle constraint violation computation."""
        print("\n=== Testing Obstacle Constraint Violation ===")
        
        # Create states at various distances from obstacle
        x_obs, y_obs, z_obs = 0.0, 0.0, 50.0
        r_obs = 10.0
        
        # State very close to obstacle (should violate)
        x_close = jnp.ones((self.num_batch, self.num_timesteps)) * 5.0
        y_close = jnp.ones((self.num_batch, self.num_timesteps)) * 0.0
        z_close = jnp.ones((self.num_batch, self.num_timesteps)) * 50.0
        
        violation_close = self.projector.compute_obstacle_constraint_violation(
            x_close, y_close, z_close, x_obs, y_obs, z_obs, r_obs
        )
        
        # State far from obstacle (should not violate)
        x_far = jnp.ones((self.num_batch, self.num_timesteps)) * 50.0
        y_far = jnp.ones((self.num_batch, self.num_timesteps)) * 0.0
        z_far = jnp.ones((self.num_batch, self.num_timesteps)) * 50.0
        
        violation_far = self.projector.compute_obstacle_constraint_violation(
            x_far, y_far, z_far, x_obs, y_obs, z_obs, r_obs
        )
        
        assert jnp.sum(violation_close) > 0, "Close state should violate obstacle constraint"
        assert jnp.sum(violation_far) == 0, "Far state should not violate obstacle constraint"
        
        print("✓ Obstacle constraint violation test passed")
        print(f"  Violation for close state: {jnp.mean(violation_close):.4f}")
        print(f"  Violation for far state: {jnp.mean(violation_far):.4f}")
        
        return True
    
    def test_obstacle_projection_scp(self):
        """Test obstacle avoidance projection using SCP."""
        print("\n=== Testing Obstacle Avoidance Projection (SCP) ===")
        
        # Create trajectory that passes through obstacle
        x_obs_array = jnp.array([0.0, 30.0])
        y_obs_array = jnp.array([0.0, 0.0])
        z_obs_array = jnp.array([50.0, 50.0])
        r_obs_array = jnp.array([10.0, 8.0])
        
        # Create initial trajectory that violates constraints
        key = random.PRNGKey(123)
        x_init = jnp.linspace(-10, 40, self.num_timesteps)
        x_init = jnp.tile(x_init, (self.num_batch, 1))
        y_init = jnp.zeros((self.num_batch, self.num_timesteps))
        z_init = jnp.ones((self.num_batch, self.num_timesteps)) * 50.0
        
        # Project onto obstacle constraints
        x_proj, y_proj, z_proj = self.projector.project_obstacle_constraints_scp(
            x_init, y_init, z_init,
            x_obs_array, y_obs_array, z_obs_array, r_obs_array
        )
        
        # Compute violations before and after
        violation_before = self.projector.compute_total_constraint_violation(
            x_init, y_init, z_init,
            x_obs_array, y_obs_array, z_obs_array, r_obs_array
        )
        
        violation_after = self.projector.compute_total_constraint_violation(
            x_proj, y_proj, z_proj,
            x_obs_array, y_obs_array, z_obs_array, r_obs_array
        )
        
        print("✓ Obstacle projection (SCP) test completed")
        print(f"  Total violation before projection: {violation_before:.4f}")
        print(f"  Total violation after projection: {violation_after:.4f}")
        print(f"  Violation reduction: {(violation_before - violation_after)/violation_before*100:.1f}%")
        
        return True
    
    def test_terrain_constraint(self):
        """Test terrain following constraint projection."""
        print("\n=== Testing Terrain Following Constraint ===")
        
        # Create trajectory below terrain
        x = jnp.linspace(0, 50, self.num_timesteps)
        x = jnp.tile(x, (self.num_batch, 1))
        y = jnp.zeros((self.num_batch, self.num_timesteps))
        z = jnp.ones((self.num_batch, self.num_timesteps)) * 5.0  # Below terrain
        
        # Compute simple sinusoidal terrain heights
        terrain_heights = 10.0 + 5.0 * jnp.sin(x / 10.0) * jnp.cos(y / 10.0)
        
        # Project onto terrain constraint
        clearance = 5.0
        z_proj = self.projector.project_terrain_constraint(x, y, z, terrain_heights, clearance)
        
        # Verify z is above terrain + clearance
        min_required_z = terrain_heights + clearance
        
        assert jnp.all(z_proj >= min_required_z - 1e-5), "Projected z violates terrain constraint"
        
        print("✓ Terrain following constraint test passed")
        print(f"  Original z range: [{jnp.min(z):.2f}, {jnp.max(z):.2f}]")
        print(f"  Projected z range: [{jnp.min(z_proj):.2f}, {jnp.max(z_proj):.2f}]")
        print(f"  Terrain + clearance range: [{jnp.min(min_required_z):.2f}, {jnp.max(min_required_z):.2f}]")
        
        return True
    
    def test_unified_state_projection(self):
        """Test unified state projection combining multiple constraints."""
        print("\n=== Testing Unified State Projection ===")
        
        # Create obstacles
        x_obs_array = jnp.array([20.0])
        y_obs_array = jnp.array([0.0])
        z_obs_array = jnp.array([50.0])
        r_obs_array = jnp.array([12.0])
        
        # Create trajectory with violations
        key = random.PRNGKey(456)
        x = jnp.linspace(0, 40, self.num_timesteps)
        x = jnp.tile(x, (self.num_batch, 1))
        y = jnp.zeros((self.num_batch, self.num_timesteps))
        z = jnp.ones((self.num_batch, self.num_timesteps)) * 50.0
        
        # Apply unified projection
        x_proj, y_proj, z_proj = self.projector.project_states_unified(
            x, y, z,
            x_obs_array, y_obs_array, z_obs_array, r_obs_array
        )
        
        # Verify box constraints
        assert jnp.all((x_proj >= self.projector.x_min) & (x_proj <= self.projector.x_max))
        assert jnp.all((y_proj >= self.projector.y_min) & (y_proj <= self.projector.y_max))
        assert jnp.all((z_proj >= self.projector.z_min) & (z_proj <= self.projector.z_max))
        
        # Compute constraint violations
        violation_before = self.projector.compute_total_constraint_violation(
            x, y, z, x_obs_array, y_obs_array, z_obs_array, r_obs_array
        )
        violation_after = self.projector.compute_total_constraint_violation(
            x_proj, y_proj, z_proj, x_obs_array, y_obs_array, z_obs_array, r_obs_array
        )
        
        print("✓ Unified state projection test passed")
        print(f"  Violation before: {violation_before:.4f}")
        print(f"  Violation after: {violation_after:.4f}")
        
        return True
    
    def run_all_tests(self):
        """Run all test cases."""
        print("\n" + "="*60)
        print("Running NonConvexStateProjector Test Suite")
        print("="*60)
        
        tests = [
            self.test_box_constraints,
            self.test_obstacle_constraint_violation,
            self.test_obstacle_projection_scp,
            self.test_terrain_constraint,
            self.test_unified_state_projection,
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                test()
                passed += 1
            except Exception as e:
                print(f"✗ Test {test.__name__} failed: {str(e)}")
                failed += 1
        
        print("\n" + "="*60)
        print(f"Test Results: {passed} passed, {failed} failed")
        print("="*60)
        
        return failed == 0


def main():
    """Main test execution function."""
    print("Starting State Projection Tests...")
    print(f"JAX version: {jax.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    # Run tests
    tester = TestNonConvexStateProjector()
    success = tester.run_all_tests()
    
    if success:
        print("\n✓ All tests passed successfully!")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
