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
from isaaclab.sensors import ContactSensor

from isaaclab.envs import DirectRLEnv
from .allsteps_env_cfg import AllstepsEnvCfg

DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi

RIGHT_FOOT = 0
LEFT_FOOT = 1

EPSILON = 1e-4

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
        self.init_step_separation = 0.75
        self.step_radius = self.cfg.step_radius
        self.target_dim = 3
        
        # pre allocate foot steps
        self.steps_pos = torch.zeros((self.num_envs, self.num_steps, 3), dtype=torch.float32, device=self.device)
        self.steps_dphi = torch.zeros((self.num_envs, self.num_steps), dtype=torch.float32, device=self.device)
        self.targets_w = torch.zeros((self.num_envs, self.look_ahead + self.look_behind, self.target_dim), dtype=torch.float32, device=self.device)
        self.targets_b = torch.zeros((self.num_envs, self.look_ahead + self.look_behind, self.target_dim), dtype=torch.float32, device=self.device)
        self._generate_foot_steps_torch()
        
        self.steps_dphi = self.steps_dphi.repeat(self.num_envs, 1)
        self.look_ahead = 2
        self.look_behind = 1

        # pre allocate buffers
        self.swing_leg = torch.ones(self.num_envs, dtype=torch.int64, device=self.device) # Right leg to start with
        self.curr_target_index = torch.ones(self.num_envs, dtype=torch.int64, device=self.device) # 1
        self.prev_target_index = self.curr_target_index - 1
        self.next_target_index = torch.clamp(self.curr_target_index + 1, 0, self.num_steps - 1)
        self.target_reach_count = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.target_reach_count_threshold = 120 # 2 seconds -> 60hz control step
        self.foot_contact = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device) # (N, B)

        # action offset and scale for PD controller
        dof_lower_limits = self.robot.data.joint_limits[0, :, 0]
        dof_upper_limits = self.robot.data.joint_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits

        # joint gears for torque controller
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.device)
        self.force_scale = self.cfg.force_scale
        
        # Foot names and indices
        self.foot_names = self.cfg.foot_names
        self.foot_indices = [self.robot.data.body_names.index(name) for name in self.foot_names]
        self.torso_index = self.robot.data.body_names.index(self.cfg.torso_name)

        # to-target potentials (will be updated in reset_idx)
        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.old_potentials = self.potentials.clone()

        # Marker index
        self.marker_idx = torch.zeros((self.num_envs * self.num_steps), dtype=torch.int64, device=self.device)
        self.marker.visualize(translations=self.steps_pos.view(-1, 3), marker_indices=self.marker_idx)
    
    def _generate_foot_steps_torch(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        temp_steps_pos, temp_steps_dphi = self.generate_foot_steps_np()
        temp_steps_pos = temp_steps_pos + self.scene.env_origins.unsqueeze(1) # (num_envs, num_steps, 3)
        temp_steps_dphi = temp_steps_dphi.repeat(self.num_envs, 1)
        
        if len(env_ids) > 0:
            self.steps_pos[env_ids] = temp_steps_pos[env_ids]
            self.steps_dphi[env_ids] = temp_steps_dphi[env_ids]
        else:
            self.steps_pos = temp_steps_pos
            self.steps_dphi = temp_steps_dphi
        
    def generate_foot_steps_np(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.curriculum = min(self.curriculum, self.max_curriculum)
        ratio = self.curriculum / self.max_curriculum

        dist_upper = np.linspace(*self.dist_range, self.max_curriculum + 1)
        dist_range = np.array([self.dist_range[0], dist_upper[self.curriculum]])
        yaw_range = self.yaw_range * 0.7 * DEG2RAD
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

        z[:] = 0 # set vertical to be 0 for now

        return torch.from_numpy(np.stack((x, y, z), axis=1)).to(self.device), torch.from_numpy(dphi).to(self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.sensor = ContactSensor(self.cfg.foot_contacts)
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
        self.scene.sensors["foot_contacts"] = self.sensor
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # create marker
        self.marker = VisualizationMarkers(self.cfg.step_markers)
        
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        
    def _apply_action(self):
        # Torque controller
        forces = self.force_scale * self.joint_gears * self.actions
        self.robot.set_joint_effort_target(forces)

    def _compute_useful_values(self):
        self.right_foot_pos_w = self.robot.data.body_pos_w[:, self.foot_indices[RIGHT_FOOT]]
        self.left_foot_pos_w = self.robot.data.body_pos_w[:, self.foot_indices[LEFT_FOOT]]
        self.torso_pos_w = self.robot.data.body_pos_w[:, self.torso_index]
        
        self.lower_foot_z_w = torch.min(self.left_foot_pos_w[:, 2], self.right_foot_pos_w[:, 2])

        self.torso_to_feet_height = self.torso_pos_w[:, 2] - self.lower_foot_z_w

        self.roll, self.pitch, self.yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        
        self.joint_pos_scaled = scale_transform(
            self.robot.data.joint_pos,
            self.robot.data.joint_limits[:, :, 0],
            self.robot.data.joint_limits[:, :, 1],
        )

        self.root_vec_b = quat_rotate_inverse(self.robot.data.root_quat_w, self.robot.data.root_lin_vel_w)

        self.root_ang_vec_b = quat_rotate_inverse(self.robot.data.root_quat_w, self.robot.data.root_ang_vel_w)
        
        # compute foot states
        self._calculate_foot_state() # here we still update state for every env
        
        # calculate targets
        self._calculate_targets()
        self.targets_b = subtract_frame_transforms(
            t01=self.robot.data.root_pos_w,
            q01=self.robot.data.root_quat_w,
            t02=self.targets_w.view(-1, 3)
        )[0].view(self.num_envs, -1, 3)

        # update potentials
        self._calculate_body_potentials()

    def _get_observations(self) -> dict:
        # build task observation
        obs = torch.cat(
            (
                self.torso_to_feet_height.unsqueeze(-1), # 1
                self.roll.unsqueeze(-1), # 1
                self.pitch.unsqueeze(-1), # 1
                self.root_vec_b, # 3
                self.joint_pos_scaled, # 28
                torch.clamp(self.robot.data.joint_vel * self.cfg.dof_vel_scale, -5, 5), # 28
                self.foot_contact, # 2
                self.targets_b.view(self.num_envs, -1) # 3 * 3 = 9
            ), 
            dim=-1
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        
        # Reward for no falling
        alive_reward = torch.ones_like(self.torso_to_feet_height) * self.cfg.alive_reward_scale

        # A small reward for reaching target
        progress = self.potentials - self.old_potentials
        
        # Some regularization terms 
        roll_violation = (self.roll > 0.4) | (self.roll < -0.2)
        pitch_violation = (self.pitch > 0.4) | (self.pitch < -0.4)
        roll_cost = torch.where(roll_violation, self.roll.abs(), torch.zeros_like(self.roll))
        pitch_cost = torch.where(pitch_violation, self.pitch.abs(), torch.zeros_like(self.pitch))
        
        speed = torch.linalg.vector_norm(self.robot.data.root_lin_vel_w, dim=-1)
        speed_cost = torch.where(speed > 1.6, speed - 1.6, torch.zeros_like(speed))
        
        action_cost = self.cfg.actions_cost_scale * torch.linalg.vector_norm(self.actions, dim=-1)
        energy_cost = self.cfg.energy_cost_scale * torch.sum(torch.abs(self.cfg.dof_vel_scale * self.robot.data.joint_vel * self.actions), dim=-1)

        joint_at_limit_cost = torch.count_nonzero(torch.abs(self.joint_pos_scaled) > 0.98, dim=-1).float() * self.cfg.joint_at_limit_cost_scale
        
        # Step reward to encourage the foot to step on the center of the target
        step_reward_condition = self.target_reached and self.target_reach_count == 1 and self.curr_target_index < self.num_steps - 1
        dist = self.foot_to_target_dist_xy[torch.arange(self.num_envs), self.swing_leg]
        step_reward = torch.where(step_reward_condition, 50 * torch.exp( -dist / 0.25), torch.zeros_like(step_reward_condition))
        
        target_bonus_condition = self.curr_target_index == self.num_steps - 1 and self.body_dist_to_target_xy < 0.15
        target_bonus = torch.where(target_bonus_condition, torch.ones_like(target_bonus_condition), torch.zeros_like(target_bonus_condition))
        
        # total reward
        total_reward = (
            alive_reward
            + progress
            - roll_cost
            - pitch_cost
            - speed_cost
            - energy_cost
            - action_cost
            - joint_at_limit_cost
            + step_reward
            + target_bonus
        )
        
        # adjust reward for falling
        total_reward = torch.where(self.reset_terminated, self.cfg.death_cost * torch.ones_like(total_reward), total_reward)
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_useful_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self.robot.data.root_pos_w[:, 2] < self.cfg.termination_height
        fell = self.torso_to_feet_height < self.cfg.termination_height_torso_to_feet

        return died | fell, time_out
    
    def _calculate_body_potentials(self):
        walk_target_delta = self.targets_w[:, -1] - self.robot.data.root_pos_w
        self.body_dist_to_target_xy = torch.linalg.vector_norm(walk_target_delta[:, 0:2], dim=-1)
        
        self.old_potentials = self.potentials.clone()
        self.potentials = -self.body_dist_to_target_xy / self.step_dt
        
    def _calculate_foot_state(self):
        '''Calculate the foot state and update the target index.
        '''
        contact_forces = torch.linalg.vector_norm(self.sensor.data.net_forces_w, dim=-1) # (N, B)
        binary_contact = contact_forces > EPSILON # (N, B)
        self.foot_contact[:] = binary_contact.float()
        
        # XY distance
        target_pos_xy = self.steps_pos[torch.arange(self.num_envs), self.curr_target_index, :2] # (N, 2)
        foot_pos_xy = self.robot.data.body_pos_w[:, self.foot_indices, :2] # (N, 2, 2)
        self.foot_to_target_dist_xy = torch.linalg.vector_norm(foot_pos_xy - target_pos_xy[:, None, :], dim=-1)
        
        self.target_reached = binary_contact[torch.arange(self.num_envs), self.swing_leg] > 0
        
        self.target_reach_count[self.target_reached] += 1
        
        can_progress = self.target_reach_count >= 2
        self.swing_leg[can_progress] = self.swing_leg[can_progress] ^ 1 # flip the leg
            
        self.curr_target_index[can_progress] = torch.clamp(
            self.curr_target_index[can_progress] + 1,
            0,
            self.num_steps - 1,
        )
        
        self.prev_target_index[can_progress] = torch.clamp(
            self.curr_target_index[can_progress] - 1,
            0,
            self.num_steps - 1,
        )
        
        self.next_target_index[can_progress] = torch.clamp(
            self.curr_target_index[can_progress] + 1,
            0,
            self.num_steps - 1,
        )
        self.target_reach_count[can_progress] = 0
        
    def _calculate_targets(self) -> torch.Tensor:
        N, _, _ = self.targets_w.shape
        
        prev_target_index = torch.clamp(self.prev_target_index - 1, 0, self.num_steps - 1)
        current_target_index = torch.clamp(self.curr_target_index - 1, 0, self.num_steps - 1)
        next_target_index = torch.clamp(self.next_target_index - 1, 0, self.num_steps - 1)
        
        prev_target_pos = self.steps_pos[torch.arange(N), prev_target_index]
        current_target_pos = self.steps_pos[torch.arange(N), current_target_index]
        next_target_pos = self.steps_pos[torch.arange(N), next_target_index]
        
        result = torch.stack([prev_target_pos, current_target_pos, next_target_pos], dim=1)
        self.targets_w[:] = result
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        # put potentials to zero first, later we will update them in _compute_useful_values
        self.old_potentials[env_ids] = 0.0
        self.potentials[env_ids] = 0.0
        
        self.target_reach_count[env_ids] = 0
        self.curr_target_index[env_ids] = self.look_behind # 1
        self.prev_target_index[env_ids] = torch.clamp(self.curr_target_index[env_ids] - 1, 0, self.num_steps - 1)
        self.next_target_index[env_ids] = torch.clamp(self.curr_target_index[env_ids] + 1, 0, self.num_steps - 1)
        
        # if we progress over half of the steps, we need to generate new foot steps
        over_half_ids = (self.curr_target_index > self.num_steps // 2).nonzero(as_tuple=False).flatten()
        replace_ids = env_ids[torch.isin(env_ids, over_half_ids)]
        if len(replace_ids) > 0:
            self._generate_foot_steps_torch(replace_ids)
        
        # Reset to a fixed pose for now TODO: add randomization
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        self._compute_useful_values() # here we still update state for every env