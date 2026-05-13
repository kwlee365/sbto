import numpy as np
from dataclasses import dataclass
from typing import Optional

import sbto.tasks.g1.constants as G1
from sbto.sim.sim_mj_rollout import SimMjRollout
from sbto.tasks.task_mj_ref import TaskMjRef, MjScene, ConfigRefMotion
from sbto.tasks.cost import quadratic_cost_nb, quaternion_dist_logmap_nb, hamming_dist_nb

@dataclass
class ConfigG1RobotObjRef():
    
    t_hand_cnt_before_lift: float = 0.15
    t_hand_cnt_after_place: float = 0.15

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

    # --- Obj pose cost ---
    obj_pos_weight: float = 20.
    obj_pos_weight_terminal: float = 20.
    obj_quat_weight: float = 5.
    obj_quat_weight_terminal: float = 5.
    obj_v_weight: float = 1.
    obj_w_weight: float = 1.

    # --- Obj/hands Contact cost ---
    contact_obj_weight: float = 1.
    contact_hands_weight: float = 0.25
    collision_obj_robot: float = 0.25
    self_collision: float = 1.

class G1RobotObjRef(TaskMjRef):

    def __init__(
        self,
        sim: SimMjRollout,
        cfg: ConfigG1RobotObjRef,
        cfg_ref: ConfigRefMotion,
        mj_scene_ref: Optional[MjScene] = None,
        ):
        super().__init__(sim, cfg_ref, mj_scene_ref)
        dt = sim.mj_scene.dt

        sensor_names = [
            G1.Sensors.TORSO_POS,
            G1.Sensors.TORSO_QUAT,
            G1.Sensors.TORSO_LINVEL,
            G1.Sensors.TORSO_ANGVEL,
            *G1.Sensors.FEET_CONTACTS,
            *G1.Sensors.HAND_CONTACTS,
            *G1.Sensors.OBJ_FLOOR_CONTACT,
            *G1.Sensors.FEET_POS,
            *G1.Sensors.FEET_QUAT,
            *G1.Sensors.HAND_POS,
            *G1.Sensors.HAND_QUAT,
            *G1.Sensors.HAND_POS_OBJ_FRAME,
            *G1.Sensors.HAND_QUAT_OBJ_FRAME,
        ]
        self.ref.compute_sensor_data(sensor_names)
        sim.set_initial_state(self.ref.x0)
        q_min = sim.mj_scene.q_min
        q_max = sim.mj_scene.q_max
        sim.set_act_limits(q_min, q_max)

       ### SET CONTACT PLAN
        # --- Contact plan feet ---
        N_feet_cnt = len(G1.Sensors.FEET_CONTACTS)
        N_hands_cnt = len(G1.Sensors.HAND_CONTACTS)
        N_obj_cnt = len(G1.Sensors.OBJ_FLOOR_CONTACT)
        N_cnt = N_feet_cnt + N_hands_cnt + N_obj_cnt


        contact_plan_feet = np.zeros((self.T, N_feet_cnt), dtype=np.int32)
        for i, foot_cnt in enumerate(G1.Sensors.FEET_CONTACTS):
            contact_plan_feet[:, i] = self.ref.sensor_data[foot_cnt][:self.T, 0]

        # --- Contact plan obj from ref ---
        contact_plan_obj_env = self.ref.sensor_data[G1.Sensors.OBJ_FLOOR_CONTACT[0]][:self.T, :1]

        # --- Contact plan hands ---
        contact_hand_ref = np.zeros((self.T, len(G1.Sensors.HAND_CONTACTS)), dtype=np.int32)
        for i, hand_cnt in enumerate(G1.Sensors.HAND_CONTACTS):
            contact_hand_ref[:, i] = self.ref.sensor_data[hand_cnt][:self.T, 0]
        
        # If the object is lifed:
        # Hands make contact slightly before and release slightly after
        nodes_lifted = np.where(contact_plan_obj_env == 0)[0]
        time_lifted = np.sum(nodes_lifted) * dt
        hand_cnt = np.where(contact_hand_ref > 0)[0]
        MIN_TIME_LIFTED = 0.4
        contact_hands_weight = cfg.contact_hands_weight

        # MIN_TIME_LIFTED to prevent misdetection
        contact_plan_hands = np.zeros((self.T, N_hands_cnt), dtype=np.int32)
        if len(nodes_lifted) > 0 and time_lifted > MIN_TIME_LIFTED:
            node_grasp_hands = max(nodes_lifted[0] - int(cfg.t_hand_cnt_before_lift / dt), 0)
            node_release_hands = min(nodes_lifted[-1] + int(cfg.t_hand_cnt_after_place / dt), self.T-1)
            contact_plan_hands[node_grasp_hands:node_release_hands, :] = 1
        # If obj is not lifted, no hands contact
        else:
            contact_hands_weight = 0.

        contact_plan_hands[contact_plan_hands > 1] = 1
        contact_plan_obj_env[contact_plan_obj_env > 1] = 1
        contact_plan_feet[contact_plan_feet > 1] = 1

        # Setup contact plan for plot and remove unused sensors
        unused_sensors = []
        contact_plan = []
        cnt_sns = []

        cnt_sns.extend(G1.Sensors.OBJ_FLOOR_CONTACT)
        contact_plan.append(contact_plan_obj_env)

        if cfg.hand_orientation == 0.:
            unused_sensors.extend(G1.Sensors.HAND_QUAT)
            unused_sensors.extend(G1.Sensors.HAND_QUAT_OBJ_FRAME)

        if cfg.foot_orientation == 0.:
            unused_sensors.extend(G1.Sensors.FEET_QUAT)

        if cfg.contact_feet_weight > 0.:
            cnt_sns.extend(G1.Sensors.FEET_CONTACTS)
            contact_plan.append(contact_plan_feet)

        if contact_hands_weight == 0:
            unused_sensors.extend(G1.Sensors.HAND_CONTACTS)
        else:
            cnt_sns.extend(G1.Sensors.HAND_CONTACTS)
            contact_plan.append(contact_plan_hands)

        self.mj_scene.edit.delete_sensors(unused_sensors)

        self.contact_plan = np.int32(np.concatenate(contact_plan, axis=-1))

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
            quadratic_cost_nb,
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
        # --- Obj cost ---
        self.add_state_cost_from_ref(
            "obj_position",
            quadratic_cost_nb,
            sim.mj_scene.obj_pos_adr,
            weights=cfg.obj_pos_weight,
            weights_terminal=cfg.obj_pos_weight_terminal,
        )
        self.add_state_cost_from_ref(
            "obj_quat",
            quaternion_dist_logmap_nb,
            sim.mj_scene.obj_quat_adr,
            weights=cfg.obj_quat_weight,
            weights_terminal=cfg.obj_quat_weight_terminal,
        )
        self.add_state_cost_from_ref(
            "obj_vel",
            quadratic_cost_nb,
            sim.mj_scene.obj_v_adr,
            weights=cfg.obj_v_weight,
            weights_terminal=cfg.obj_v_weight,
        )
        self.add_state_cost_from_ref(
            "obj_w",
            quadratic_cost_nb,
            sim.mj_scene.obj_w_adr,
            weights=cfg.obj_w_weight,
            weights_terminal=cfg.obj_w_weight,
        )
        # Hand position (world and obj frame)
        self.add_sensor_cost_from_ref(
            G1.Sensors.HAND_POS + G1.Sensors.HAND_POS_OBJ_FRAME,
            quadratic_cost_nb,
            weights=cfg.hand_position,
        )
        # Hand orientation (world and obj frame)
        self.add_sensor_cost_from_ref(
            G1.Sensors.HAND_QUAT + G1.Sensors.HAND_QUAT_OBJ_FRAME,
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
        # Obj-env contact
        self.add_sensor_cost(
            G1.Sensors.OBJ_STATIC_CONTACT,
            hamming_dist_nb,
            sub_idx_sensor=[0],
            ref_values=contact_plan_obj_env,
            weights=cfg.contact_obj_weight,
        )
        # Hands-obj contact
        contact_hands_weight_arr = np.zeros_like(contact_plan_hands)
        contact_hands_weight_arr[contact_plan_hands > 0] = contact_hands_weight
        self.add_sensor_cost(
            G1.Sensors.HAND_CONTACTS,
            hamming_dist_nb,
            sub_idx_sensor=G1.Sensors.id_cnt_status_hands,
            ref_values=contact_plan_hands,
            weights=contact_hands_weight_arr,
        )
        # Collision obj - robot
        self.add_sensor_cost(
            G1.Sensors.OBJ_ROBOT_COLLISION,
            hamming_dist_nb,
            ref_values=np.zeros((self.T-1, len(G1.Sensors.OBJ_ROBOT_COLLISION)), dtype=np.int32),
            weights=cfg.collision_obj_robot,
        )
        # Self collision robot-robot
        self.add_sensor_cost(
            G1.Sensors.SELF_COLLISION,
            hamming_dist_nb,
            ref_values=np.zeros((self.T-1, len(G1.Sensors.SELF_COLLISION)), dtype=np.int32),
            weights=cfg.self_collision,
        )