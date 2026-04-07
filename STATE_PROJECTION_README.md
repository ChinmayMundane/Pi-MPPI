# Non-Convex State Constraint Projection for Pi-MPPI

This document describes the non-convex state constraint projection enhancement to the Pi-MPPI (Projection-based Model Predictive Path Integral) algorithm.

## Overview

The enhanced Pi-MPPI implementation adds hard state constraint enforcement through non-convex optimization, complementing the existing control constraint projection. This allows for:

- **Hard obstacle avoidance constraints** (non-convex)
- **State bounds** (box constraints on position)
- **Terrain following constraints**
- **Unified state-control projection** via ADMM

## Key Components

### 1. NonConvexStateProjector (`nonconvex_state_projector.py`)

Sequential Convex Programming (SCP) based optimizer for projecting state trajectories onto feasible sets with non-convex constraints.

**Features:**
- Box constraint projection (convex)
- Obstacle avoidance constraints (non-convex)
- Terrain following constraints
- JAX-compatible with JIT compilation
- Trust region management for SCP convergence

**Usage:**
```python
from nonconvex_state_projector import NonConvexStateProjector

# Initialize projector
state_projector = NonConvexStateProjector(
    num_timesteps=100,
    num_batch=1000,
    dt=0.2,
    x_min=-100, x_max=100,
    y_min=-100, y_max=100,
    z_min=0, z_max=100,
    safety_margin=5.0  # meters around obstacles
)

# Project states onto constraints
x_obs_array = jnp.array([0.0, 30.0])
y_obs_array = jnp.array([0.0, 0.0])
z_obs_array = jnp.array([50.0, 50.0])
r_obs_array = jnp.array([10.0, 8.0])

x_proj, y_proj, z_proj = state_projector.project_states_unified(
    x, y, z,
    x_obs_array, y_obs_array, z_obs_array, r_obs_array
)
```

### 2. UnifiedStateControlProjector (`unified_projector.py`)

ADMM-based framework for jointly projecting state and control constraints while enforcing dynamics coupling.

**Features:**
- Alternates between control and state projection
- Enforces dynamics coupling (states computed from controls must match projected states)
- Convergence checking based on residual norms
- Suitable for complex coupled constraints

**Usage:**
```python
from unified_projector import UnifiedStateControlProjector

# Initialize unified projector
unified_projector = UnifiedStateControlProjector(
    control_projector=mppi_controller,  # Original pi_mppi instance
    state_projector=state_projector,
    num_timesteps=100,
    num_batch=1000,
    dt=0.2,
    g=9.81
)

# Run unified projection
(c_v_final, c_pitch_final, c_roll_final,
 x_final, y_final, z_final, psi_final,
 lamda_v, lamda_pitch, lamda_roll,
 s_v, s_pitch, s_roll,
 residuals) = unified_projector.unified_projection_admm(...)
```

### 3. Enhanced Pi-MPPI (`pi_mppi_obst_with_state_projection.py`)

Drop-in replacement for the original `pi_mppi_obst.py` with integrated state projection.

**Features:**
- Toggle between hard constraints (projection) and soft penalties
- Automatic cost weight adjustment when using state projection
- Backward compatible with original pi_mppi interface

**Usage:**
```python
from pi_mppi_obst_with_state_projection import pi_mppi_with_state_projection

# Initialize with state projection enabled
controller = pi_mppi_with_state_projection(
    v_max=30, v_min=15,
    vdot_max=5, vdot_min=-5,
    vddot_max=2, vddot_min=-2,
    pitch_max=0.3, pitch_min=-0.3,
    pitchdot_max=0.5, pitchdot_min=-0.5,
    pitchddot_max=0.2, pitchddot_min=-0.2,
    roll_max=0.5, roll_min=-0.5,
    rolldot_max=0.5, rolldot_min=-0.5,
    rollddot_max=0.2, rollddot_min=-0.2,
    x_min=-500, x_max=500,
    y_min=-500, y_max=500,
    z_min=0, z_max=200,
    enable_state_projection=True  # Toggle hard constraints
)

# Use like original pi_mppi
mean, key, new_states = controller.pi_mppi_main(
    v_init, v_dot_init, pitch_init, pitch_dot_init,
    roll_init, roll_dot_init, psi_init,
    x_init, y_init, z_init,
    x_fin, y_fin, z_fin,
    mean, key,
    x_obs, y_obs, z_obs, r_obs
)
```

## Technical Details

### Sequential Convex Programming (SCP)

The obstacle avoidance constraints are non-convex:
```
(x - x_obs)² + (y - y_obs)² + (z - z_obs)² ≥ (r_obs + safety_margin)²
```

SCP iteratively solves a sequence of convex approximations:
1. Linearize non-convex constraints around current iterate
2. Solve convex QP with linearized constraints
3. Update trust region based on constraint satisfaction
4. Iterate until convergence

### ADMM Framework

The unified projector uses ADMM to alternate between:
1. **Control projection**: Project onto control constraints (existing QP solver)
2. **Forward dynamics**: Compute states from projected controls
3. **State projection**: Project onto state constraints (SCP solver)
4. **Coupling update**: Update Lagrange multipliers for coupling constraints

Convergence is achieved when states from controls match projected states.

### Performance Considerations

**Computational cost:**
- State projection adds ~20 SCP iterations per MPPI iteration
- Each SCP iteration: O(num_obstacles × num_batch × num_timesteps)
- JIT compilation amortizes overhead after first call

**Tuning parameters:**
- `safety_margin`: Distance buffer around obstacles (5m recommended)
- `maxiter_scp`: SCP iterations (10-20 typically sufficient)
- `trust_region_radius`: Initial trust region (10m recommended)
- `rho_coupling`: ADMM penalty for state-control coupling (1.0 recommended)

## Testing

Run the test suite to validate the implementation:

```bash
cd /home/runner/work/Pi-MPPI/Pi-MPPI
python test_state_projection.py
```

**Test coverage:**
- ✓ Box constraint projection
- ✓ Obstacle constraint violation computation
- ✓ Obstacle avoidance projection (SCP)
- ✓ Terrain following constraints
- ✓ Unified state projection
- ✓ JAX JIT compilation

## Comparison: Hard vs Soft Constraints

### Original Pi-MPPI (Soft Constraints)
- Obstacles handled via cost penalties (`w_3 * cost_obstacle`)
- Can violate constraints if cost weight too low
- Smoother trajectories due to soft penalties
- Faster computation (no SCP iterations)

### Enhanced Pi-MPPI (Hard Constraints)
- Obstacles enforced as hard constraints via projection
- Guaranteed constraint satisfaction (within numerical tolerance)
- More conservative trajectories (always maintains safety margin)
- Slower computation (~2-3x) due to SCP

### Recommendation
- **Use hard constraints** when safety is critical (UAV obstacle avoidance)
- **Use soft constraints** when flexibility is needed or obstacles are informational
- **Toggle via `enable_state_projection` flag** for easy comparison

## Future Enhancements

Potential extensions:
1. **Dynamic obstacles**: Time-varying obstacle positions
2. **Velocity constraints**: State-dependent velocity bounds
3. **Formation constraints**: Multi-agent coordination
4. **Learned projections**: Neural network-based projection approximations
5. **Adaptive trust regions**: Online trust region tuning

## References

1. Williams, G., et al. "Model Predictive Path Integral Control using Covariance Variable Importance Sampling." arXiv:1509.01149, 2015.
2. Sequential Convex Programming: Schulman, J., et al. "Motion planning with sequential convex optimization and convex collision checking." IJRR, 2014.
3. ADMM: Boyd, S., et al. "Distributed optimization and statistical learning via ADMM." Foundations and Trends in Machine Learning, 2011.

## Contact

For questions or issues, please open an issue on the GitHub repository.
