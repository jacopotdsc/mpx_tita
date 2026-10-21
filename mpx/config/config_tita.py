import jax.numpy as jnp
import jax
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, '..')) + '/data/tita/tita.xml'

joints_name = [
    'joint_left_leg_1', 'joint_left_leg_2', 'joint_left_leg_3', 'joint_left_leg_4',
    'joint_right_leg_1', 'joint_right_leg_2', 'joint_right_leg_3', 'joint_right_leg_4',
]

contact_frame = ['left_leg_4_collision', 'right_leg_4_collision']
body_name = ['left_leg_4', 'right_leg_4']

simulation_frequency = 500
whole_body_frequency = 100
mpc_frequency = 40
dt_mpc = 0.025
N = 12
mpc_horizon_s = N * dt_mpc
mpc_iterations = 1
wbc_lookahead_dt = 0.0

dt_sim = 1.0 / simulation_frequency
dt_wbc = 1.0 / whole_body_frequency
mpc_period_steps = simulation_frequency / mpc_frequency
wbc_period_steps = round(simulation_frequency / whole_body_frequency)
mpc_shift_nodes = round(1.0 / (mpc_frequency * dt_mpc))

T_TRAJECTORY = 60
grav = 9.81

robot_height = 0.44  # Robot base height in meters
h_min = 0.28
h_max = 0.4
com_z_to_track = 0.4
clearence_speed = 0.4

mu = 0.9
use_terrain_estimator = False

p0 = jnp.array([0, 0, robot_height])
p0_c = jnp.array([0, 0, 0])
quat0 = jnp.array([1, 0, 0, 0])

q0 = jnp.array([0.0, 0.5, -1.0, 0.0, 0.0, 0.5, -1.0, 0.0])

n_joints = len(joints_name)
n_contact = len(contact_frame)
nx = 13
nu = 9
mass = 27.68978  # Sum of inertial masses in the canonical tita_running.urdf
d = 0.567  # leg distance (m)
# Centroidal inertia at q0 from the aligned model, expressed in the base frame.
inertia = jnp.array([
    [1.090008860808705, -2.6318929583168255e-05, -0.029887671948266012],
    [-2.6318929583168255e-05, 0.4984221717169791, -2.0791225890770537e-05],
    [-0.029887671948266012, -2.0791225890770537e-05, 0.7010938745906233],
])

grf_ref = jnp.array(n_contact*[0.0, 0.0, (grav*mass)/2.0])
u_ref = jnp.concatenate([jnp.array([0.0, 0.0, 0.0]), jnp.array(grf_ref)])

Kp = jnp.diag(jnp.tile(jnp.array([500,500,500]),n_contact))
Kd = jnp.diag(jnp.tile(jnp.array([20,20,20]),n_contact))

# Whole-body controller
base_body_name = 'base_link'
wheel_radius = 0.0925

Kp_motion = 5e1
Kd_motion = 3e1
Kp_wheel = 5e1
Kd_wheel = 3e1
Kp_reg = 1e2
Kd_reg = 2e1

w_posture = 0.1
posture_joint_ids = (0, 4)

w_qddot = 1e-12
w_force = 1e-4  # WBC QP force regularization; stabilizes convergence while braking
w_com = 1e0
w_lwheel = 1e0
w_rwheel = 1e0
w_base = 1e0

# MPC cost weights
w_pcomxy = 0e0   # CoM position in the xy plane
w_pcomz = 2e4    # CoM height
w_vcomxy = 3e2   # CoM velocity in the xy plane
w_vcomz = 1e1    # CoM vertical velocity
w_c = 0e0        # Ground-projected CoM position
w_vcz = 0e0      # Ground-projected CoM vertical velocity
w_theta = 0e0    # Heading

w_v = 1e1        # Ground-projected CoM forward velocity
# NOTE: was 5. With w_vcomxy=300 on WORLD-frame CoM velocity and a reference asking
# for 5 m/s^2 braking (the pendulum delivers ~1.5), weight 5 made yawing the cheapest
# way to cut CoM-velocity error: alpha opposed omega_ref. Since omega_ref ramps from
# MEASURED omega, the excursion fed itself (vx 2->0, wz 0->0.6 measured 3.3 rad/s).
# 50 removes it on the flat test set; braking also needs the WBC joint-limit relaxation
# in mpc_utils.py or the knee-limit QP becomes infeasible.
w_omega = 5e1    # Angular velocity

# Control weights for the wheel-related inputs
w_a = 1e-1       # Linear acceleration
w_ac_z = 1e-1    # Vertical acceleration
w_alpha = 1e-3   # Angular acceleration

# Control weights for the ground reaction forces
w_fcxy = 1e-7    # Horizontal forces
w_fcz = 1e-4     # Vertical force
w_eq = 1e6       # Equality-constraint penalty
w_stab = w_eq    # Terminal constraint: CoM over the wheel-contact midpoint

# Bounds on the wheel-related MPC inputs, |u| <= bound: an inequality constraint in the
# MPC, penalised with w_eq like the other constraints. 100 = inactive: in W they break
# braking (a <= 3 m/s^2 from 1 m/s: the robot sits down, Gazebo 2026-10-21), so only
# W_takeover sets them.
a_max = 100.0      # m/s^2, linear acceleration
ac_z_max = 100.0   # m/s^2, vertical acceleration
alpha_max = 100.0  # rad/s^2, angular acceleration


def _weights(**override):
    """MPC weight matrix from the values above, with some of them overridden by name."""
    v = {k: override.get(k, globals()[k]) for k in (
        'w_pcomxy', 'w_pcomz', 'w_vcomxy', 'w_vcomz', 'w_c', 'w_vcz', 'w_theta', 'w_v',
        'w_omega', 'w_a', 'w_ac_z', 'w_alpha', 'w_fcxy', 'w_fcz', 'w_eq', 'w_stab',
        'a_max', 'ac_z_max', 'alpha_max')}  # the last three are input bounds, not weights
    return jnp.diag(jnp.array(list(v.values())))


W = _weights()

# MPC weights while standing up from transform_up (C++ node: from the takeover until the
# COM holds at h_mpc), then W. At the end of transform_up the shins rest on the ground and
# the CoM is ~2.7 cm behind the wheels: with W the MPC sees a falling pendulum and jerks
# the wheels (a up to 12 m/s^2, hips swinging at 7 rad/s). A soft terminal stability and
# weighted wheel accelerations move the wheels under the CoM while it rises
# (Gazebo 2026-10-21: w_a 5 falls, 2-3 stands; bounds remove the ac_z chatter).
W_takeover = _weights(w_stab=1e4, w_a=2e0, w_ac_z=1e0, w_alpha=1e-1,
                      a_max=3.0, ac_z_max=1.0, alpha_max=1.0)
