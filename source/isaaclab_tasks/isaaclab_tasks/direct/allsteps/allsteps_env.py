# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate_inverse, subtract_frame_transforms, scale_transform, euler_xyz_from_quat
from isaaclab.markers import VisualizationMarkers

from isaaclab.envs import DirectRLEnv
from .allsteps_env_cfg import AllstepsEnvCfg

DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi

class AllstepsEnv(DirectRLEnv):
    cfg: AllstepsEnvCfg

    def __init__(self, cfg: AllstepsEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Terrain info
        self.dist_range = torch.tensor([0.65, 1.25], dtype=torch.float32, device=self.device)
        self.pitch_range = torch.tensor([-30, 30], dtype=torch.float32, device=self.device)
        self.yaw_range = torch.tensor([-20, 20], dtype=torch.float32, device=self.device)
        self.tilt_range = torch.tensor([-15, 15], dtype=torch.float32, device=self.device)
        self.curriculum = 0
        self.max_curriculum = 9
        self.num_steps = 20
        self.terrain_info = self.generate_foot_steps()

        # action offset and scale for PD controller
        dof_lower_limits = self.robot.data.joint_limits[0, :, 0]
        dof_upper_limits = self.robot.data.joint_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits

        # joint gears for torque controller
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.device)
        self.force_scale = self.cfg.force_scale
        
        self.foot_names = self.cfg.foot_names
        key_body_names = self.foot_names # right foot, left foot
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.torso_index = self.robot.data.body_names.index(self.cfg.torso_name)

        # to-target potentials (will be updated in reset_idx)
        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.old_potentials = self.potentials.clone()

        # Marker index
        self.marker_idx = torch.zeros((self.num_envs * 3), dtype=torch.int64, device=self.device)
        self.marker.visualize(translations=self.handhold_pos.view(-1, 3), marker_indices=self.marker_idx)
    
        
    def generate_foot_steps(self):
        self.curriculum = min(self.curriculum, self.max_curriculum) # check overflow
        ratio = self.curriculum / self.max_curriculum # progress is 0 now

        # {self.max_curriculum + 1} levels in total
        dist_upper = np.linspace(*self.dist_range, self.max_curriculum + 1) # divide the upper bounds to 10 levels
        dist_range = np.array([self.dist_range[0], dist_upper[self.curriculum]])
        yaw_range = self.yaw_range * DEG2RAD * 0.7
        pitch_range = self.pitch_range * ratio * DEG2RAD + np.pi / 2
        tilt_range = self.tilt_range * ratio * DEG2RAD

        N = self.num_steps
        dr = np.random.uniform(*dist_range, size=N)
        dphi = np.random.uniform(*yaw_range, size=N)
        dtheta = np.random.uniform(*pitch_range, size=N)
        x_tilt = np.random.uniform(*tilt_range, size=N)
        y_tilt = np.random.uniform(*tilt_range, size=N)

        # make first step below feet
        dr[0] = 0.0
        dphi[0] = 0.0
        dtheta[0] = np.pi / 2

        dr[1:3] = self.init_step_separation
        dphi[1:3] = 0.0
        dtheta[1:3] = np.pi / 2

        x_tilt[0:3] = 0
        y_tilt[0:3] = 0

        dphi = np.cumsum(dphi)

        dx = dr * np.sin(dtheta) * np.cos(dphi)
        dy = dr * np.sin(dtheta) * np.sin(dphi)
        dz = dr * np.cos(dtheta)

        # Fix overlapping steps
        dx_max = np.maximum(np.abs(dx[2:]), self.step_radius * 2.5)
        dx[2:] = np.sign(dx[2:]) * np.minimum(dx_max, self.dist_range[1])

        x = np.cumsum(dx)
        y = np.cumsum(dy)
        z = np.cumsum(dz)

        return torch.from_numpy(np.stack((x, y, z, dphi, x_tilt, y_tilt), axis=1)).to(self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # create marker
        # self.marker = VisualizationMarkers(self.cfg.handhold_markers)
        

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        
    def _apply_action(self):
        # Torque controller
        forces = self.force_scale * self.joint_gears * self.actions
        self.robot.set_joint_effort_target(forces)

    def _compute_useful_values(self):
        self.left_wrist_pos_w = self.robot.data.body_pos_w[:, self.key_body_indexes[LimbNames.LeftHand.value]]
        self.right_wrist_pos_w = self.robot.data.body_pos_w[:, self.key_body_indexes[LimbNames.RightHand.value]]
        self.left_toe_pos_w = self.robot.data.body_pos_w[:, self.key_body_indexes[LimbNames.LeftFoot.value]]
        self.right_toe_pos_w = self.robot.data.body_pos_w[:, self.key_body_indexes[LimbNames.RightFoot.value]]
        self.chest_pos_w = self.robot.data.body_pos_w[:, self.torso_index]
        
        self.lower_foot_z_w = torch.min(self.left_toe_pos_w[:, 2], self.right_toe_pos_w[:, 2])

        self.chest_to_feet_height = self.chest_pos_w[:, 2] - self.lower_foot_z_w

        self.roll, self.pitch, self.yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)

        self.current_target_pos_w = self.handhold_pos[torch.arange(self.num_envs), self.target_sequence[torch.arange(self.num_envs), self.current_target_idx]]
        self.next_target_pos_w = self.handhold_pos[torch.arange(self.num_envs), self.target_sequence[torch.arange(self.num_envs), clamp_index(self.current_target_idx + 1, 0, self.cfg.num_targets - 1)]]
        self.next_2nd_target_pos_w = self.handhold_pos[torch.arange(self.num_envs), self.target_sequence[torch.arange(self.num_envs), clamp_index(self.current_target_idx + 2, 0, self.cfg.num_targets - 1)]]

        self.current_target_pos_b = subtract_frame_transforms(
            t01=self.robot.data.root_pos_w,
            q01=self.robot.data.root_quat_w,
            t02=self.current_target_pos_w
        )[0]

        self.next_target_pos_b = subtract_frame_transforms(
            t01=self.robot.data.root_pos_w,
            q01=self.robot.data.root_quat_w,
            t02=self.next_target_pos_w
        )[0]

        self.next_2nd_target_pos_b = subtract_frame_transforms(
            t01=self.robot.data.root_pos_w,
            q01=self.robot.data.root_quat_w,
            t02=self.next_2nd_target_pos_w
        )[0]

        self.left_wrist_pos_b = subtract_frame_transforms(
            t01=self.robot.data.root_pos_w,
            q01=self.robot.data.root_quat_w,
            t02=self.left_wrist_pos_w
        )[0]

        self.right_wrist_pos_b = subtract_frame_transforms(
            t01=self.robot.data.root_pos_w,
            q01=self.robot.data.root_quat_w,
            t02=self.right_wrist_pos_w
        )[0]

        self.left_toe_pos_b = subtract_frame_transforms(
            t01=self.robot.data.root_pos_w,
            q01=self.robot.data.root_quat_w,
            t02=self.left_toe_pos_w
        )[0]

        self.right_toe_pos_b = subtract_frame_transforms(
            t01=self.robot.data.root_pos_w,
            q01=self.robot.data.root_quat_w,
            t02=self.right_toe_pos_w
        )[0]
        
        self.joint_pos_scaled = scale_transform(
            self.robot.data.joint_pos,
            self.robot.data.joint_limits[:, :, 0],
            self.robot.data.joint_limits[:, :, 1],
        )

        self.root_vec_b = quat_rotate_inverse(self.robot.data.root_quat_w, self.robot.data.root_lin_vel_w)

        self.root_ang_vec_b = quat_rotate_inverse(self.robot.data.root_quat_w, self.robot.data.root_ang_vel_w)

        # update potentials
        limb_to_target = self.robot.data.body_pos_w[:, self.key_body_indexes[LimbNames.LeftHand.value]] - self.handhold_pos[torch.arange(self.num_envs), self.target_sequence[torch.arange(self.num_envs), self.current_target_idx]]
        self.old_potentials = self.potentials.clone()
        self.potentials = - torch.linalg.vector_norm(limb_to_target, dim=-1) / self.step_dt

    def _get_observations(self) -> dict:
        # build task observation
        obs = torch.cat(
            (
                self.chest_to_feet_height.unsqueeze(-1), # 1
                self.robot.data.root_pos_w[:, 2].unsqueeze(-1), # 1
                self.roll.unsqueeze(-1), # 1
                self.pitch.unsqueeze(-1), # 1
                self.yaw.unsqueeze(-1), # 1
                self.root_vec_b, # 3
                self.root_ang_vec_b * self.cfg.angular_velocity_scale, # 3
                self.joint_pos_scaled, # 28
                self.robot.data.joint_vel * self.cfg.dof_vel_scale, # 28
                # self.actions, # 29
                # self.left_toe_pos_b, # 3
                # self.right_toe_pos_b, # 3
                self.left_wrist_pos_b, # 3
                # self.right_wrist_pos_b, # 3
                self.current_target_pos_b, # 3
                # self.next_target_pos_b, # 3
                # self.next_2nd_target_pos_b, # 3
            ), 
            dim=-1
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        
        # Reward for no falling
        alive_reward = torch.ones_like(self.chest_to_feet_height) * self.cfg.alive_reward_scale

        # A small reward for reaching target
        progress = self.potentials - self.old_potentials
        
        
        # total reward
        total_reward = (
            alive_reward
            + progress
        )
        
        # adjust reward for falling
        total_reward = torch.where(self.reset_terminated, self.cfg.death_cost * torch.ones_like(total_reward), total_reward)
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_useful_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        root_spd = torch.norm(self.robot.data.root_com_lin_vel_w, dim=-1) # Barrier condition that prevents the character from flying out 
        died = root_spd >= 5 # TODO: Roughly chosen. but perpectly prevent. 
        distance = torch.norm((self.current_target_pos_w - self.robot.data.root_pos_w), dim=-1) # Barrier condition that prevents the character from flying out 
        died = torch.where(distance >= 5, torch.ones_like(died), died) # TODO: I think using spd condition is enough.
        died = torch.where(self.robot.data.root_pos_w[:, 2] < self.cfg.termination_height, torch.ones_like(died), died) # Standing root height (hip) is ~0.95m
        # died = self.robot.data.root_pos_w[:, 2] < self.cfg.termination_height
        
        fell = self.chest_to_feet_height < self.cfg.termination_height_chest_to_feet
        return died | fell, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        self.old_potentials[env_ids] = 0
        self.potentials[env_ids] = 0

        # Reset to a fixed pose for now
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        self._compute_useful_values()