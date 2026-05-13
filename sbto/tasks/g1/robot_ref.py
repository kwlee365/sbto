import numpy as np
from dataclasses import dataclass
from typing import Optional

import sbto.tasks.g1.constants as G1
from sbto.sim.sim_mj_rollout import SimMjRollout
from sbto.tasks.task_mj_ref import TaskMjRef, MjScene, ConfigRefMotion
from sbto.tasks.cost import quadratic_cost_nb, quaternion_dist_logmap_nb, hamming_dist_nb

@dataclass
class ConfigG1RobotRef():

    # --- State costs ---
    joint_pos_weight: float = 0.1
    joint_vel_weight: float = 0.05
    base_pos_weight: float = 5.
    base_quat_weight: float = 1.

    # --- Torso pose/vel ---
    torso_pos_weight: float = 30.
    torso_quat_weight: float = 1.
    torso_quat_weight_terminal: float = 10.
    torso_linvel_weight: float = 1.
    torso_angvel_weight: float = 1.

    # --- Hand pose cost ---
    hand_position: float = 5.
    hand_orientation: float = 0.1

    # --- Foot pose ---
    foot_position: float = 10.
    foot_orientation: float = 0.1

    # --- Feet Contact cost ---
    contact_feet_weight: float = 0.25
    contact_force_feet_weight: float = 1.0e-6

    # --- Self collision ---
    self_collision: float = 1.

    # --- Arm vs floor collision ---
    arm_floor_collision: float = 5.0

class G1RobotRef(TaskMjRef):

    def __init__(
        self,
        sim: SimMjRollout,
        cfg: ConfigG1RobotRef,
        cfg_ref: ConfigRefMotion,
        mj_scene_ref: Optional[MjScene] = None,
        ):
        super().__init__(sim, cfg_ref, mj_scene_ref)

        sensor_names = [
            G1.Sensors.TORSO_POS,
            G1.Sensors.TORSO_QUAT,
            G1.Sensors.TORSO_LINVEL,
            G1.Sensors.TORSO_ANGVEL,
            *G1.Sensors.FEET_CONTACTS,
            *G1.Sensors.FEET_POS,
            *G1.Sensors.FEET_QUAT,
            *G1.Sensors.HAND_POS,
            *G1.Sensors.HAND_QUAT,
        ]
        self.ref.compute_sensor_data(sensor_names)
        sim.set_initial_state(self.ref.x0)
        q_min = sim.mj_scene.q_min
        q_max = sim.mj_scene.q_max
        sim.set_act_limits(q_min, q_max)

        ### SET CONTACT PLAN
        # --- Contact plan feet ---
        N_feet_cnt = len(G1.Sensors.FEET_CONTACTS)
        contact_plan_feet = np.zeros((self.T, N_feet_cnt), dtype=np.int32)
        for i, foot_cnt in enumerate(G1.Sensors.FEET_CONTACTS):
            contact_plan_feet[:, i] = self.ref.sensor_data[foot_cnt][:self.T, 0]

        contact_plan_feet[contact_plan_feet > 1] = 1

        # Setup contact plan for plot and remove unused sensors
        unused_sensors = []
        contact_plan = []
        cnt_sns = []

        if cfg.hand_orientation == 0.:
            unused_sensors.extend(G1.Sensors.HAND_QUAT)
            unused_sensors.extend(G1.Sensors.HAND_QUAT_OBJ_FRAME)

        if cfg.foot_orientation == 0.:
            unused_sensors.extend(G1.Sensors.FEET_QUAT)

        if cfg.contact_feet_weight > 0.:
            cnt_sns.extend(G1.Sensors.FEET_CONTACTS)
            contact_plan.append(contact_plan_feet)
        elif cfg.contact_force_feet_weight == 0.:
            unused_sensors.extend(G1.Sensors.FEET_CONTACTS)


        self.mj_scene.edit.delete_sensors(unused_sensors)

        if len(contact_plan) > 0:
            self.contact_plan = np.int32(np.concatenate(contact_plan, axis=-1))

        if cnt_sns:
            dim_sns = 4
            cnt_sns_sub_id = list(range(0, len(cnt_sns) * dim_sns, dim_sns))
            self.set_contact_sensor_id(cnt_sns, cnt_sns_sub_id)

        # --- G1 costs ---
        self.add_state_cost_from_ref(
            "joint_ref",
            quadratic_cost_nb,
            sim.mj_scene.act_qposadr,
            weights=cfg.joint_pos_weight,
            weights_terminal=cfg.joint_pos_weight,
        )
        self.add_state_cost_from_ref(
            "base_position",
            quadratic_cost_nb,
            [0, 1, 2],
            weights=cfg.base_pos_weight,
            weights_terminal=cfg.base_pos_weight,
        )
        self.add_state_cost_from_ref(
            "base_quat",
            quaternion_dist_logmap_nb,
            [3, 4, 5, 6],
            weights=cfg.base_quat_weight,
            weights_terminal=cfg.base_quat_weight,
        )
        self.add_state_cost_from_ref(
            "joint_vel",
            quadratic_cost_nb,
            sim.mj_scene.act_vel_adr,
            weights=cfg.joint_vel_weight,
            weights_terminal=cfg.joint_vel_weight,
        )
        self.add_sensor_cost_from_ref(
            G1.Sensors.TORSO_POS,
            quadratic_cost_nb,
            weights=cfg.torso_pos_weight,
            weights_terminal=cfg.torso_pos_weight,
        )
        self.add_sensor_cost_from_ref(
            G1.Sensors.TORSO_QUAT,
            quaternion_dist_logmap_nb,
            weights=cfg.torso_quat_weight,
            weights_terminal=cfg.torso_quat_weight,
        )
        self.add_sensor_cost_from_ref(
            G1.Sensors.TORSO_LINVEL,
            quadratic_cost_nb,
            weights=cfg.torso_linvel_weight,
        )
        self.add_sensor_cost_from_ref(
            G1.Sensors.TORSO_ANGVEL,
            quadratic_cost_nb,
            weights=cfg.torso_angvel_weight,
        )
        # Hand position (world and obj frame)
        self.add_sensor_cost_from_ref(
            G1.Sensors.HAND_POS,
            quadratic_cost_nb,
            weights=cfg.hand_position,
        )
        # Hand orientation (world and obj frame)
        self.add_sensor_cost_from_ref(
            G1.Sensors.HAND_QUAT,
            quaternion_dist_logmap_nb,
            weights=cfg.hand_orientation,
        )
        # Foot position
        self.add_sensor_cost_from_ref(
            G1.Sensors.FEET_POS,
            quadratic_cost_nb,
            weights=cfg.foot_position,
        )
        # Foot orientation
        self.add_sensor_cost_from_ref(
            G1.Sensors.FEET_QUAT,
            quaternion_dist_logmap_nb,
            weights=cfg.foot_orientation,
        )
        # Feet contact
        self.add_sensor_cost(
            G1.Sensors.FEET_CONTACTS,
            hamming_dist_nb,
            sub_idx_sensor=G1.Sensors.id_cnt_status_feet,
            ref_values=contact_plan_feet,
            weights=cfg.contact_feet_weight,
        )
        # Feet-env contact force
        self.add_sensor_cost(
            G1.Sensors.FEET_CONTACTS,
            quadratic_cost_nb,
            sub_idx_sensor=G1.Sensors.id_cnt_force_feet,
            weights=cfg.contact_force_feet_weight,
        )
        # Self collision robot-robot
        self.add_sensor_cost(
            G1.Sensors.SELF_COLLISION,
            hamming_dist_nb,
            ref_values=np.zeros((self.T-1, len(G1.Sensors.SELF_COLLISION)), dtype=np.int32),
            weights=cfg.self_collision,
        )
        # Arm vs floor collision
        self.add_sensor_cost(
            G1.Sensors.ARM_FLOOR_COLLISION,
            hamming_dist_nb,
            ref_values=np.zeros((self.T-1, len(G1.Sensors.ARM_FLOOR_COLLISION)), dtype=np.int32),
            weights=cfg.arm_floor_collision,
        )