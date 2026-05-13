"""
JAX/MJX (GPU) rollout backend.

Drop-in replacement for SimMjRollout. Swap via Hydra:
    task.sim._target_: sbto.sim.sim_mjx_rollout.SimMjxRollout

Requirements:
    pip install --upgrade "jax[cuda12]"   # GPU
    # or:
    pip install --upgrade "jax[cpu]"      # CPU-JAX (for debugging)
    mujoco >= 3.3.0  (ships mujoco.mjx)
"""
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Tuple, Optional
import copy

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)

from sbto.sim.sim_base import SimRolloutBase, Array
from sbto.sim.scene_mj import MjScene
from sbto.sim.action_scaling import Scaling


@dataclass
class ConfigMjxRollout:
    T: int
    step_knots: int = 25
    keyframe_x0: str = ""
    interp_kind: str = "linear"
    scaling_kind: str = ""
    device: str = "gpu"


class SimMjxRollout(SimRolloutBase):
    """
    GPU-batched rollout backend using MuJoCo XLA (mujoco.mjx).

    The host side keeps the regular mj_model/mj_data for sensor/index lookups
    used by the rest of the pipeline. The device side keeps a parallel mjx
    model and runs N rollouts in one shot via vmap+lax.scan.
    """

    def __init__(
        self,
        mj_scene: MjScene,
        cfg: ConfigMjxRollout,
        scaling: Optional[Scaling] = None,
    ):
        self.cfg = cfg
        self.mj_scene = mj_scene

        super().__init__(
            mj_scene.Nq,
            mj_scene.Nv,
            mj_scene.Nu,
            cfg.T,
            cfg.step_knots,
            cfg.interp_kind,
            scaling,
        )

        try:
            self.device = jax.devices(cfg.device)[0]
        except RuntimeError:
            print(f"[SimMjxRollout] {cfg.device!r} not available, falling back to CPU.")
            self.device = jax.devices("cpu")[0]
        print(f"[SimMjxRollout] running on {self.device}")

        self.mjx_model = mjx.put_model(mj_scene.mj_model, device=self.device)
        self.dt = mj_scene.mj_model.opt.timestep

        if cfg.keyframe_x0:
            self.set_initial_state_from_keyframe(cfg.keyframe_x0)

        self.set_act_limits(
            mj_scene.q_min,
            mj_scene.q_max,
            self.x_0[mj_scene.act_qposadr],
        )

        self._rollout_fn = None
        self._last_T = -1
        self._last_N = -1

        self.nstep_allocated = self.T
        self.t0 = 0.0

    @property
    def duration(self):
        return self.dt * self.T

    # ------------------------------------------------------------------
    # Setup helpers (mirror SimMjRollout)
    # ------------------------------------------------------------------

    def set_act_limits(self, q_min, q_max, q_nom=None):
        if q_nom is None and not np.all(self.x_0 == 0.0):
            q_nom = self.x_0[self.mj_scene.act_qposadr]
        super().set_act_limits(q_min, q_max, q_nom)

    def set_initial_state_from_keyframe(self, keyframe_name: str, with_obj: bool = False) -> None:
        keyframe = self.mj_scene.mj_model.keyframe(keyframe_name)

        if not with_obj:
            x_p_0 = self.mj_scene.mj_data.qpos
            x_v_0 = self.mj_scene.mj_data.qvel
            qpos_adr = self.mj_scene.act_qposadr
            qvel_adr = self.mj_scene.act_dofadr

            if self.mj_scene.is_floating_base:
                qpos_base = np.arange(qpos_adr[0])
                qvel_base = np.arange(qvel_adr[0])
                qpos_adr = np.concatenate((qpos_base, qpos_adr))
                qvel_adr = np.concatenate((qvel_base, qvel_adr))

            x_p_0[qpos_adr] = np.array(keyframe.qpos)[qpos_adr]
            x_v_0[qvel_adr] = np.array(keyframe.qvel)[qvel_adr]
        else:
            x_p_0 = np.array(keyframe.qpos)
            x_v_0 = np.array(keyframe.qvel)

        self.mj_scene.update_data(x_p_0, x_v_0)
        self.set_initial_state(np.concatenate((x_p_0, x_v_0)))

    # ------------------------------------------------------------------
    # Core: JIT-compiled batched rollout
    # ------------------------------------------------------------------

    def _build_rollout_fn(self, T: int):
        """Build the JIT'd function that runs N rollouts of length T."""
        mjx_model = self.mjx_model
        Nq = self.Nq

        def single_rollout(x0, u_traj):
            data = mjx.make_data(mjx_model)
            data = data.replace(qpos=x0[:Nq], qvel=x0[Nq:])
            data = mjx.forward(mjx_model, data)

            def step(d, u):
                d = d.replace(ctrl=u)
                d = mjx.step(mjx_model, d)
                x = jp.concatenate([d.qpos, d.qvel])
                return d, (x, d.sensordata)

            _, (x_traj, obs_traj) = jax.lax.scan(step, data, u_traj)
            return x_traj, obs_traj

        batched = jax.vmap(single_rollout, in_axes=(None, 0))
        return jax.jit(batched, device=self.device)

    def _rollout_dynamics(
        self,
        u_traj: Array,
        with_x0: bool = False,
    ) -> Tuple[Array, Array, Array, Array]:
        """
        u_traj: [N, T, Nu] numpy.
        Returns (t, x, u, obs) as numpy arrays, matching SimMjRollout shapes.
        """
        N, T, _ = u_traj.shape

        if self._rollout_fn is None or T != self._last_T:
            self._rollout_fn = self._build_rollout_fn(T)
            self._last_T = T
            self._last_N = N

        u_jax = jp.asarray(u_traj)
        x0_jax = jp.asarray(self.x_0)

        x_traj, obs_traj = self._rollout_fn(x0_jax, u_jax)
        x_traj.block_until_ready()

        x_np = np.asarray(x_traj, dtype=np.float64)
        obs_np = np.asarray(obs_traj, dtype=np.float64)

        t = (np.arange(1, T + 1, dtype=np.float64) * self.dt)[None, :, None]
        t = np.broadcast_to(t, (N, T, 1)).copy()

        if with_x0:
            x0_b = np.broadcast_to(self.x_0[None, None, :], (N, 1, self.Nx)).copy()
            x_np = np.concatenate([x0_b, x_np], axis=1)
            t = np.concatenate([np.zeros((N, 1, 1)), t], axis=1)
            obs0 = np.zeros((N, 1, obs_np.shape[-1]))
            obs_np = np.concatenate([obs0, obs_np], axis=1)

        self.nstep_allocated = T
        return t, x_np, u_traj, obs_np

    # ------------------------------------------------------------------
    # Multiple shooting (optional; CPU parity)
    # ------------------------------------------------------------------

    def rollout_multiple_shooting(
        self,
        u_knots: Array,
        x_shooting: Array,
        with_x0: bool = False,
    ) -> Tuple[Array, Array, Array, Array]:
        """
        Sequential per-interval rollout, each interval batched on GPU.
        Mirrors SimMjRollout.rollout_multiple_shooting.
        """
        u_knots = u_knots.reshape(-1, self.Nknots, self.Nu)
        if self.scaling:
            u_knots = self.scaling(u_knots)
        u_traj = self.interpolate(u_knots)

        t_full, x_full, u_full, obs_full = [], [], [], []

        for t_start, t_end, x_start in zip(
            self.t_knots[:-1], self.t_knots[1:], x_shooting
        ):
            self.set_initial_state(x_start)
            self._last_T = -1
            include_x0 = with_x0 and len(t_full) == 0
            t, x, u, obs = self._rollout_dynamics(
                u_traj[:, t_start:t_end, :], include_x0
            )
            t_full.append(t + t_start * self.dt)
            x_full.append(x)
            u_full.append(u)
            obs_full.append(obs)

        self.set_initial_state(x_shooting[0])
        self._last_T = -1
        return (
            np.concatenate(t_full, axis=1),
            np.concatenate(x_full, axis=1),
            np.concatenate(u_full, axis=1),
            np.concatenate(obs_full, axis=1),
        )
