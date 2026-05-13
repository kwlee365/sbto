"""
MJX-Warp (NVIDIA GPU) rollout backend.

Drop-in alternative to SimMjxRollout. Swap via Hydra:
    task/g1/sim@task.sim=default_mjwarp

Requirements:
    pip install warp-lang mujoco-warp
    mujoco >= 3.8.0
"""
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Tuple, Optional

import warp as wp
import mujoco_warp as mjw

from sbto.sim.sim_base import SimRolloutBase, Array
from sbto.sim.scene_mj import MjScene
from sbto.sim.action_scaling import Scaling


@dataclass
class ConfigMjwarpRollout:
    T: int
    step_knots: int = 25
    keyframe_x0: str = ""
    interp_kind: str = "linear"
    scaling_kind: str = ""
    device: str = "cuda:0"
    # Per-world solver buffer limits passed to mjw.make_data.
    # If None, mjwarp picks a default that may be too small for contact-heavy
    # scenes — increase if you see "nefc overflow" / "nacon overflow" warnings.
    njmax: Optional[int] = 128
    nconmax: Optional[int] = None
    naconmax: Optional[int] = None


@wp.kernel
def _write_ctrl_at_t(
    u_traj: wp.array3d(dtype=wp.float32),  # (T, N, Nu)
    t: int,
    ctrl: wp.array2d(dtype=wp.float32),    # (N, Nu)
):
    w, u = wp.tid()
    ctrl[w, u] = u_traj[t, w, u]


@wp.kernel
def _record_2d_into_3d(
    src: wp.array2d(dtype=wp.float32),  # (N, D)
    t: int,
    dst: wp.array3d(dtype=wp.float32),  # (T, N, D)
):
    w, d = wp.tid()
    dst[t, w, d] = src[w, d]


class SimMjwarpRollout(SimRolloutBase):
    """
    NVIDIA GPU rollout via MJX-Warp.

    Uses mujoco_warp's batched simulation: a single mjw.step(m, d) advances all
    N worlds in parallel. Per-timestep we copy ctrl into d.ctrl from a device
    buffer and record qpos/qvel/sensordata into device history buffers. Final
    sync + download once after the full rollout.
    """

    def __init__(
        self,
        mj_scene: MjScene,
        cfg: ConfigMjwarpRollout,
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

        # Initialize warp device
        try:
            wp.set_device(cfg.device)
            self.device = cfg.device
        except Exception:
            wp.set_device("cuda:0")
            self.device = "cuda:0"
        print(f"[SimMjwarpRollout] running on {self.device}")

        # Upload model once
        self.mjm = mj_scene.mj_model
        self.m_warp = mjw.put_model(self.mjm)
        self.dt = self.mjm.opt.timestep

        if cfg.keyframe_x0:
            self.set_initial_state_from_keyframe(cfg.keyframe_x0)

        self.set_act_limits(
            mj_scene.q_min,
            mj_scene.q_max,
            self.x_0[mj_scene.act_qposadr],
        )

        # Lazily allocated per-N device buffers
        self._N = -1
        self._T_buf = -1
        self._d = None
        self._u_dev = None       # (T, N, Nu) device
        self._qpos_hist = None   # (T, N, Nq) device
        self._qvel_hist = None   # (T, N, Nv) device
        self._obs_hist = None    # (T, N, Nobs) device

        self.nstep_allocated = self.T
        self.t0 = 0.0

    @property
    def duration(self):
        return self.dt * self.T

    # --- Setup helpers (mirror SimMjxRollout) ---

    def set_act_limits(self, q_min, q_max, q_nom=None):
        if q_nom is None and not np.all(self.x_0 == 0.0):
            q_nom = self.x_0[self.mj_scene.act_qposadr]
        super().set_act_limits(q_min, q_max, q_nom)

    def set_initial_state_from_keyframe(
        self, keyframe_name: str, with_obj: bool = False
    ) -> None:
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

    # --- Buffer allocation ---

    def _ensure_buffers(self, N: int, T: int):
        Nobs = self.mj_scene.Nobs
        realloc_N = N != self._N
        realloc_T = T != self._T_buf

        if realloc_N:
            make_data_kwargs = {"nworld": N}
            if self.cfg.njmax is not None:
                make_data_kwargs["njmax"] = self.cfg.njmax
            if self.cfg.nconmax is not None:
                make_data_kwargs["nconmax"] = self.cfg.nconmax
            if self.cfg.naconmax is not None:
                make_data_kwargs["naconmax"] = self.cfg.naconmax
            self._d = mjw.make_data(self.mjm, **make_data_kwargs)
        if realloc_N or realloc_T:
            self._u_dev = wp.zeros((T, N, self.Nu), dtype=wp.float32)
            self._qpos_hist = wp.zeros((T, N, self.Nq), dtype=wp.float32)
            self._qvel_hist = wp.zeros((T, N, self.Nv), dtype=wp.float32)
            self._obs_hist = wp.zeros((T, N, Nobs), dtype=wp.float32)
        self._N = N
        self._T_buf = T

    # --- Core: batched rollout ---

    def _rollout_dynamics(
        self,
        u_traj: Array,
        with_x0: bool = False,
    ) -> Tuple[Array, Array, Array, Array]:
        """
        u_traj: [N, T, Nu] numpy.
        Returns (t, x, u, obs) as numpy arrays in (N, T, *) layout, matching
        SimMjxRollout / SimMjRollout.
        """
        N, T, _ = u_traj.shape
        Nobs = self.mj_scene.Nobs
        self._ensure_buffers(N, T)

        # Upload u_traj as (T, N, Nu) device buffer
        u_TNU = np.transpose(u_traj, (1, 0, 2)).astype(np.float32)
        self._u_dev.assign(u_TNU)

        # Initialize qpos/qvel for all N worlds (broadcast x_0)
        qpos_init = np.tile(self.x_0[: self.Nq].astype(np.float32), (N, 1))
        qvel_init = np.tile(self.x_0[self.Nq :].astype(np.float32), (N, 1))
        self._d.qpos.assign(qpos_init)
        self._d.qvel.assign(qvel_init)
        mjw.forward(self.m_warp, self._d)

        # Rollout loop: enqueue T steps; all stays on device
        for t in range(T):
            wp.launch(
                _write_ctrl_at_t,
                dim=(N, self.Nu),
                inputs=[self._u_dev, t, self._d.ctrl],
            )
            mjw.step(self.m_warp, self._d)
            wp.launch(
                _record_2d_into_3d,
                dim=(N, self.Nq),
                inputs=[self._d.qpos, t, self._qpos_hist],
            )
            wp.launch(
                _record_2d_into_3d,
                dim=(N, self.Nv),
                inputs=[self._d.qvel, t, self._qvel_hist],
            )
            if Nobs > 0:
                wp.launch(
                    _record_2d_into_3d,
                    dim=(N, Nobs),
                    inputs=[self._d.sensordata, t, self._obs_hist],
                )

        # Sync and download
        wp.synchronize()
        qpos_TNQ = self._qpos_hist.numpy()  # (T, N, Nq)
        qvel_TNV = self._qvel_hist.numpy()  # (T, N, Nv)
        obs_TNO = (
            self._obs_hist.numpy() if Nobs > 0 else np.zeros((T, N, 0), dtype=np.float32)
        )

        # Reshape to (N, T, *)
        qpos = np.transpose(qpos_TNQ, (1, 0, 2))
        qvel = np.transpose(qvel_TNV, (1, 0, 2))
        obs = np.transpose(obs_TNO, (1, 0, 2))

        # Time vector (broadcast over N): t at each step is (k+1)*dt + t0
        t_steps = (np.arange(1, T + 1, dtype=np.float64) * self.dt + self.t0).astype(
            np.float64
        )
        t_arr = np.broadcast_to(t_steps[None, :, None], (N, T, 1)).copy()

        # State: concat qpos + qvel along last dim → (N, T, Nx)
        x = np.concatenate([qpos, qvel], axis=-1).astype(np.float64)

        # Prepend x_0 if requested (match SimMjxRollout's with_x0 layout)
        if with_x0:
            t_arr = np.concatenate(
                [
                    np.broadcast_to(
                        np.array([[self.t0]], dtype=np.float64)[None], (N, 1, 1)
                    ).copy(),
                    t_arr,
                ],
                axis=1,
            )
            x0 = np.broadcast_to(self.x_0[None, None, :], (N, 1, self.Nx)).copy()
            x = np.concatenate([x0.astype(np.float64), x], axis=1)

        return t_arr, x, u_traj.astype(np.float64), obs.astype(np.float64)
