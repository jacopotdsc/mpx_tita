import jax
import math
import jax.numpy as jnp
from functools import partial
from flax import struct
import mpx.utils.mpc_utils as mpc_utils
import mpx.utils.models as mpc_dyn_model
import mpx.utils.objectives as mpc_objectives
import mujoco
from mujoco import mjx
import mpx.jax_ocp_solvers.optimizers as optimizers


@struct.dataclass
class ControlSol:
    a     : jax.Array
    ac_z  : jax.Array
    alpha : jax.Array
    grf   : jax.Array


@struct.dataclass
class MPCState:
    """Tutto lo stato mutabile dell'MPC in un pytree JAX (compatibile con jax.vmap)."""
    sol        : ControlSol
    X0_shifted : jax.Array
    U0_shifted : jax.Array
    D0_shifted : jax.Array
    ref_rate   : jax.Array   # [v, acc_v, omega, acc_omega] the next reference starts from


class BatchedMPCControllerWrapper:
    def __init__(self, config, n_env, model=None):
        jax.config.update("jax_compilation_cache_dir", "./jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

        self.n_env = n_env
        self.config = config

        self.mpc_frequency = config.mpc_frequency
        shift = 1.0 / (config.mpc_frequency * config.dt_mpc)
        if not math.isfinite(shift) or abs(shift - round(shift)) > 1e-6 or not 1 <= round(shift) <= config.N:
            raise ValueError('MPC period must span a whole number of prediction nodes within the horizon')
        self.shift = int(round(shift))
        self.mpc_iterations = int(getattr(config, "mpc_iterations", 1))
        self.wbc_lookahead_dt = float(getattr(config, "wbc_lookahead_dt", 0.0))

        if model is None:
            print("[Warning] Model not provided. Loading model from XML path:", config.model_path)
            model = mujoco.MjModel.from_xml_path(config.model_path)
        mjx_model = mjx.put_model(model)
        self._mjx_model = mjx_model

        contact_id   = [mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in config.contact_frame]
        body_id      = [mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_BODY, n) for n in config.body_name]
        base_body_id = mjx.name2id(mjx_model, mujoco.mjtObj.mjOBJ_BODY, config.base_body_name)

        pcom_init  = config.p0.copy()
        dpcom_init = jnp.zeros(3)
        c_init     = pcom_init.copy().at[2].set(0.0)
        vcz_init   = jnp.zeros(1)
        theta      = jnp.arctan2(c_init[1], c_init[0])
        v          = jnp.zeros(1)
        omega      = jnp.zeros(1)
        self.initial_state = jnp.concatenate([pcom_init, dpcom_init, c_init, vcz_init, jnp.array([theta]), v, omega])

        self._nj = mjx_model.nv - 6

        self._index_variables_qp = {
            'com_pos': 0,     'com_vel': 3,     'com_acc': 6,
            'lwheel_pos': 9,  'lwheel_vel': 12, 'lwheel_acc': 15,
            'rwheel_pos': 18, 'rwheel_vel': 21, 'rwheel_acc': 24,
            'base_rot': 27,   'base_omega': 36, 'base_alpha': 39,
            'joints': 42,
        }
        self._desired_size = self._index_variables_qp['joints'] + 3 * self._nj

        posture_joint_ids = getattr(config, "posture_joint_ids", None)
        if posture_joint_ids is None:
            self._posture_mask = None
        else:
            self._posture_mask = jnp.zeros(self._nj).at[jnp.array(list(posture_joint_ids))].set(1.0)

        cost     = partial(mpc_objectives.wheeled_dfcip_obj, config.d, config.N)
        hess     = partial(mpc_objectives.wheeled_dfcip_hessian_gn, config.d, config.N)
        dynamics = partial(mpc_dyn_model.wheeled_dfcip_dynamics, mjx_model, config.mass, config.grav, config.dt_mpc)
        fddp     = partial(optimizers.fddp_mpc, cost, dynamics, hess, False)
        ref_gen  = partial(mpc_utils.reference_generator_dfcip_online, config.N, config.dt_mpc, config.mass, config.grav, config.h_min, config.h_max,
                           config.a_default, config.tau_v,
                           config.alpha_default, config.tau_omega, config.az_default)

        def work(reference, parameter, W, x0, X_init, U_init):
            X_it, U_it, D_it = X_init, U_init, None
            for _ in range(self.mpc_iterations):
                X_it, U_it, D_it = fddp(reference, parameter, W, x0, X_it, U_it)
            return X_it, U_it, D_it

        dt_wbc = config.dt_wbc
        n_contacts = 1

        def whole_body_control(ww, qpos, qvel, desired, com_err_integral, force_ref, base_err_integral):
            # ww: WBC gains and weights of a phase (config.W_wbc "weights"), a runtime input.
            return mpc_utils.whole_body_interface_wheeled_legged_qp(
                mjx_model, config.mass, config.grav, config.d,
                contact_id, body_id, base_body_id,
                config.wheel_radius, dt_wbc, n_contacts,
                ww['Kp_motion'], ww['Kd_motion'], ww['Kp_wheel'], ww['Kd_wheel'], ww['Kp_reg'], ww['Kd_reg'],
                ww['w_posture'],
                ww['w_qddot'], ww['w_force'], ww['w_com'], ww['w_lwheel'], ww['w_rwheel'], ww['w_base'],
                config.mu,
                qpos, qvel, desired,
                posture_mask=self._posture_mask,
                ref_layout=self._index_variables_qp,
                Ki_motion=ww['Ki_com'],
                com_err_integral=com_err_integral,
                force_ref=force_ref,
                Ki_base=ww['Ki_base'],
                base_err_integral=base_err_integral,
                w_track=ww['w_track'], Kp_track=ww['Kp_track'],
                track_ref=config.d,
            )

        self._solve                = jax.jit(jax.vmap(work))
        self._ref_gen              = jax.jit(jax.vmap(ref_gen))
        self._whole_body_interface = jax.jit(jax.vmap(whole_body_control, in_axes=(None, 0, 0, 0, 0, 0, 0)))
        self._build_desired_jit    = jax.jit(self._build_desired_impl)
        self._process_state_jit    = jax.jit(self._process_state_impl)

        U0 = jnp.tile(config.u_ref, (config.N, 1))
        X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
        D0 = jnp.zeros((config.N + 1, config.nx))
        self._U0_init = jnp.tile(U0, (n_env, 1, 1))
        self._X0_init = jnp.tile(X0, (n_env, 1, 1))
        self._D0_init = jnp.tile(D0, (n_env, 1, 1))

    def init_state(self, x0: jax.Array | None = None) -> MPCState:
        n, cfg = self.n_env, self.config
        a     = jnp.full((n,), cfg.u_ref[0])
        ac_z  = jnp.full((n,), cfg.u_ref[1])
        alpha = jnp.full((n,), cfg.u_ref[2])
        grf   = jnp.tile(cfg.u_ref[3:], (n, 1))

        if x0 is None:
            X_init = self._X0_init
            ref_rate = jnp.zeros((n, 4))
        else:
            x0 = jnp.asarray(x0)
            if x0.ndim == 1:
                x0 = x0[None, :]
            X_init = jnp.broadcast_to(x0[:, None, :], (self.n_env, cfg.N + 1, cfg.nx))
            # The reference starts at the measured velocities, with zero acceleration.
            zero = jnp.zeros_like(x0[:, 11])
            ref_rate = jnp.stack([x0[:, 11], zero, x0[:, 12], zero], axis=1)

        return MPCState(
            sol=ControlSol(a=a, ac_z=ac_z, alpha=alpha, grf=grf),
            X0_shifted=X_init,
            U0_shifted=self._U0_init,
            D0_shifted=self._D0_init,
            ref_rate=ref_rate,
        )

    def process_state(self, pcom, vcom, centers, Rs, radii, feet_vel, theta_prev):
        return self._process_state_jit(pcom, vcom, centers, Rs, radii, feet_vel, theta_prev)

    @partial(jax.jit, static_argnums=(0,))
    def shift_warm_start(self, trajectory):
        """Advance the solver guess by whole nodes and hold the terminal value."""
        s = self.shift
        return jnp.concatenate([trajectory[:, s:, :],
                                jnp.tile(trajectory[:, -1:, :], (1, s, 1))], axis=1)

    def run(self, state: MPCState, x0, cmd, phase="W") -> MPCState:
        cfg = self.config

        x_ref, u_ref, rates = self._ref_gen(x0, cmd, state.ref_rate)
        reference = jnp.concatenate([x_ref, u_ref], axis=-1)

        X, U, D = self.solve(reference, None, x0, state.X0_shifted, state.U0_shifted, phase)

        new_a     = U[:, 0, 0]
        new_ac_z  = U[:, 0, 1]
        new_alpha = U[:, 0, 2]
        new_grf   = U[:, 0, 3:]

        D = jnp.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)

        new_X0 = self.shift_warm_start(X)
        new_U0 = self.shift_warm_start(U)
        new_D0 = self.shift_warm_start(D)

        new_state = MPCState(
            sol=ControlSol(a=new_a, ac_z=new_ac_z, alpha=new_alpha, grf=new_grf),
            X0_shifted=new_X0,
            U0_shifted=new_U0,
            D0_shifted=new_D0,
            ref_rate=rates[:, self.shift],  # one MPC period ahead
        )
        return new_state, reference

    def solve(self, reference, parameter, x0, X_init, U_init, phase="W", W=None):
        """MPC solve with the weight matrix W, else the weights of the phase. Inside a jitted
        caller pass W as one of its inputs (else XLA compiles it as a constant)."""
        if W is None:
            W = self.weight_matrix(self.config.W_mpc[phase]["weights"])
        return self._solve(reference, parameter, jnp.tile(W, (self.n_env, 1, 1)), x0, X_init, U_init)

    def weight_matrix(self, weights):
        """MPC weight matrix W from the named weights of a phase (config W_mpc "weights").

        The diagonal follows the key order of the "W" phase (config _W_mpc), which is the index
        order the cost and the Hessian (objectives.wheeled_dfcip_*) read W by.
        """
        keys = list(self.config.W_mpc["W"]["weights"])
        if list(weights) != keys:
            raise ValueError(f'MPC weights must have the keys of W_mpc["W"] in the same order: {keys}')
        return jnp.diag(jnp.array([float(weights[k]) for k in keys], dtype=jnp.float64))

    def whole_body_run(self, state: MPCState, x0, qpos, qvel, pl_world, pr_world, dpl_world, dpr_world, com_height_target=None, com_err_integral=None, base_err_integral=None, phase="W", ww=None):
        """WBC with the gains ww (config.W_wbc "weights"), else those of the phase. Inside a jitted caller
        pass ww as one of its inputs (else XLA compiles it as a constant)."""
        desired = self._build_desired_jit(x0, qpos, state, pl_world, pr_world, dpl_world, dpr_world)
        if com_height_target is not None:
            # ROS/Gazebo runs asynchronously and has different contact dynamics.
            # Keep actual vertical task feedback instead of cancelling its PD
            # error by setting the desired position/velocity to the measurement.
            # MPC still supplies feedforward acceleration. Standalone callers
            # retain their original behavior when no height target is supplied.
            desired = desired.at[:, self._index_variables_qp['com_pos'] + 2].set(com_height_target)
            desired = desired.at[:, self._index_variables_qp['com_vel'] + 2].set(0.0)
        if com_err_integral is None:
            com_err_integral = jnp.zeros((desired.shape[0], 3))
        force_ref = state.sol.grf if getattr(self.config, 'wbc_force_reg_mpc', True) else jnp.zeros_like(state.sol.grf)
        if base_err_integral is None:
            base_err_integral = jnp.zeros((desired.shape[0], 3))
        if ww is None:
            ww = self.config.W_wbc[phase]["weights"]
        tau_cmd, qddot, fl, fr = self._whole_body_interface(ww, qpos, qvel, desired, com_err_integral, force_ref, base_err_integral)
        return state, tau_cmd, qddot, fl, fr, desired

    def reset(self) -> MPCState:
        print("MPC Controller Reset")
        return self.init_state()

    def _process_state_impl(self, pcom, vcom, centers, Rs, radii, feet_vel, theta_prev):

        l_rcp = mpc_utils.get_rCP(Rs[0], radii[0])
        r_rcp = mpc_utils.get_rCP(Rs[1], radii[1])

        pl_world = centers[0] + l_rcp
        pr_world = centers[1] + r_rcp
        dpl_world, dpr_world = feet_vel[0], feet_vel[1]

        tita_state = jnp.concatenate([pcom, vcom, pl_world, pr_world, dpl_world, dpr_world])

        c_world = (pl_world + pr_world) / 2.0
        vc_world = (dpl_world + dpr_world) / 2.0

        # Heading and its rate must use the same wheel-center baseline as
        # feet_vel. The actual spacing changes with leg posture; config.d is
        # a nominal prediction-model parameter, not a measured track width.
        diff = centers[0] - centers[1]
        theta_wrapped = jnp.arctan2(-diff[0], diff[1])
        a = (theta_wrapped - theta_prev + jnp.pi) % (2 * jnp.pi)
        a = jnp.where(a < 0, a + 2 * jnp.pi, a) - jnp.pi
        theta = theta_prev + a

        ct, st = jnp.cos(theta), jnp.sin(theta)
        R = jnp.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
        dpl_b = R.T @ dpl_world
        dpr_b = R.T @ dpr_world
        track_width = jnp.maximum(jnp.linalg.norm(diff[:2]), 1e-6)
        w = (dpr_b[0] - dpl_b[0]) / track_width
        v = (dpr_b[0] + dpl_b[0]) / 2.0

        x0 = jnp.concatenate([
            pcom,
            vcom,
            c_world,
            jnp.array([vc_world[2]]),
            jnp.array([theta]),
            jnp.array([v]),
            jnp.array([w]),
        ])
        return tita_state, x0, theta

    def _build_desired_impl(self, x0, qpos, state, pl_world, pr_world, dpl_world, dpr_world):
        cfg = self.config
        B = qpos.shape[0]
        desired = jnp.zeros((B, self._desired_size))
        dt = self.wbc_lookahead_dt

        x_mpc = x0[None, :]
        u_mpc = jnp.concatenate([
            state.sol.a[:, None],
            state.sol.ac_z[:, None],
            state.sol.alpha[:, None],
            state.sol.grf,
        ], axis=1)

        a     = u_mpc[:, 0]
        ac_z  = u_mpc[:, 1]
        alpha = u_mpc[:, 2]
        fcl   = u_mpc[:, 3:6]
        fcr   = u_mpc[:, 6:9]

        pcom_curr  = x_mpc[:, 0:3]
        vcom_curr  = x_mpc[:, 3:6]
        pl_curr    = pl_world
        pr_curr    = pr_world
        dpl_curr   = dpl_world
        dpr_curr   = dpr_world
        theta_curr = x_mpc[:, 10]
        v_curr     = x_mpc[:, 11]
        w_curr     = x_mpc[:, 12]

        cos_t = jnp.cos(theta_curr)
        sin_t = jnp.sin(theta_curr)

        g_vec = jnp.array([0.0, 0.0, -cfg.grav])
        vector_off = jnp.array([0.0, cfg.d / 2.0, 0.0])

        dR_curr = jnp.stack([
            jnp.stack([-sin_t, -cos_t, jnp.zeros((B,))], axis=1),
            jnp.stack([ cos_t, -sin_t, jnp.zeros((B,))], axis=1),
            jnp.zeros((B, 3)),
        ], axis=1)
        ddR_curr = jnp.stack([
            jnp.stack([-cos_t,  sin_t, jnp.zeros((B,))], axis=1),
            jnp.stack([-sin_t, -cos_t, jnp.zeros((B,))], axis=1),
            jnp.zeros((B, 3)),
        ], axis=1)

        ddc = jnp.stack([
            a * cos_t - v_curr * sin_t * w_curr,
            a * sin_t + v_curr * cos_t * w_curr,
            ac_z,
        ], axis=1)

        acc_com_ = (fcl + fcr) / cfg.mass + g_vec[None, :]
        vel_com_ = vcom_curr + dt * acc_com_
        pos_com_ = pcom_curr + dt * vcom_curr

        acc_pl_ = ddc + (
            jnp.einsum('bij,j->bi', ddR_curr, vector_off) * w_curr[:, None] * w_curr[:, None]
            + jnp.einsum('bij,j->bi', dR_curr, vector_off) * alpha[:, None]
        )
        vel_pl_ = dpl_curr + dt * acc_pl_
        pos_pl_ = pl_curr + dt * dpl_curr

        acc_pr_ = ddc - (
            jnp.einsum('bij,j->bi', ddR_curr, vector_off) * w_curr[:, None] * w_curr[:, None]
            + jnp.einsum('bij,j->bi', dR_curr, vector_off) * alpha[:, None]
        )
        vel_pr_ = dpr_curr + dt * acc_pr_
        pos_pr_ = pr_curr + dt * dpr_curr

        alpha_ = alpha
        omega_ = w_curr + dt * alpha_
        theta_ = theta_curr + dt * w_curr

        com_pos_ref = pos_com_
        com_vel_ref = vel_com_
        com_acc_ref = acc_com_

        lwheel_pos_ref = pos_pl_.at[:, 2].set(pos_pl_[:, 2] + cfg.wheel_radius)
        lwheel_vel_ref = vel_pl_
        lwheel_acc_ref = acc_pl_
        rwheel_pos_ref = pos_pr_.at[:, 2].set(pos_pr_[:, 2] + cfg.wheel_radius)
        rwheel_vel_ref = vel_pr_
        rwheel_acc_ref = acc_pr_

        cos_theta_ = jnp.cos(theta_)
        sin_theta_ = jnp.sin(theta_)
        R_theta = jnp.stack([
            jnp.stack([cos_theta_, -sin_theta_, jnp.zeros((B,))], axis=1),
            jnp.stack([sin_theta_,  cos_theta_, jnp.zeros((B,))], axis=1),
            jnp.stack([jnp.zeros((B,)), jnp.zeros((B,)), jnp.ones((B,))], axis=1),
        ], axis=1)
        base_rot_ref   = R_theta.reshape(B, 9)
        base_omega_ref = jnp.zeros((B, 3)).at[:, 2].set(omega_)
        base_alpha_ref = jnp.zeros((B, 3)).at[:, 2].set(alpha_)

        qjnt_ref     = jnp.tile(cfg.q0, (B, 1))
        qjntdot_ref  = jnp.zeros((B, self._nj))
        qjntddot_ref = jnp.zeros((B, self._nj))

        nj = self._nj
        idx = self._index_variables_qp
        desired = desired.at[:, idx['com_pos']:idx['com_pos'] + 3].set(com_pos_ref)
        desired = desired.at[:, idx['com_vel']:idx['com_vel'] + 3].set(com_vel_ref)
        desired = desired.at[:, idx['com_acc']:idx['com_acc'] + 3].set(com_acc_ref)
        desired = desired.at[:, idx['lwheel_pos']:idx['lwheel_pos'] + 3].set(lwheel_pos_ref)
        desired = desired.at[:, idx['rwheel_pos']:idx['rwheel_pos'] + 3].set(rwheel_pos_ref)
        desired = desired.at[:, idx['lwheel_vel']:idx['lwheel_vel'] + 3].set(lwheel_vel_ref)
        desired = desired.at[:, idx['rwheel_vel']:idx['rwheel_vel'] + 3].set(rwheel_vel_ref)
        desired = desired.at[:, idx['lwheel_acc']:idx['lwheel_acc'] + 3].set(lwheel_acc_ref)
        desired = desired.at[:, idx['rwheel_acc']:idx['rwheel_acc'] + 3].set(rwheel_acc_ref)
        desired = desired.at[:, idx['base_rot']:idx['base_rot'] + 9].set(base_rot_ref)
        desired = desired.at[:, idx['base_omega']:idx['base_omega'] + 3].set(base_omega_ref)
        desired = desired.at[:, idx['base_alpha']:idx['base_alpha'] + 3].set(base_alpha_ref)
        desired = desired.at[:, idx['joints']:idx['joints'] + nj].set(qjnt_ref)
        desired = desired.at[:, idx['joints'] + nj:idx['joints'] + 2 * nj].set(qjntdot_ref)
        desired = desired.at[:, idx['joints'] + 2 * nj:idx['joints'] + 3 * nj].set(qjntddot_ref)

        return desired
