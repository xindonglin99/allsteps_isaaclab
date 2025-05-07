# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate, quat_rotate_inverse, subtract_frame_transforms, scale_transform, unscale_transform, euler_xyz_from_quat, sample_uniform
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor

from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

from isaaclab.envs import DirectRLEnv
from .allsteps_env_cfg import AllstepsEnvCfg

import copy

RIGHT_FOOT = 0
LEFT_FOOT = 1

EPSILON = 1e-4

class AllstepsEnv(DirectRLEnv):
    cfg: AllstepsEnvCfg

    def __init__(self, cfg: AllstepsEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Terrain info
        self.dist_range = torch.tensor([0.75, 0.9], dtype=torch.float32, device=self.device)
        self.pitch_range = torch.tensor([-30, 30], dtype=torch.float32, device=self.device)
        self.yaw_range = torch.tensor([-20, 20], dtype=torch.float32, device=self.device)
        self.tilt_range = torch.tensor([-15, 15], dtype=torch.float32, device=self.device)
        self.max_curriculum = torch.tensor(9, dtype=torch.int64, device=self.device)
        self.termination_curriculum = torch.linspace(0.75, 0.45, self.max_curriculum + 1).to(self.device)
        self.applied_gain_curriculum = torch.linspace(1.2, 1.2, self.max_curriculum + 1).to(self.device)
        self.curriculum = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.num_steps = self.cfg.num_steps
        self.init_step_separation = 0.75
        self.step_radius = self.cfg.step_radius
        self.target_dim = 3
        self.curriculum_progess_theshold = 12
        self.foot_sep = 0.16
        self.mirrored = False
        self.stop_frames = 2 # 0.5 seconds -> 60hz control step
        
        
        self.look_ahead = 2
        self.look_behind = 1

        # Marker index
        self.marker_idx = torch.zeros((self.num_envs * self.num_steps), dtype=torch.int64, device=self.device)
        
        # pre allocate foot steps
        self.steps_pos = torch.zeros((self.num_envs, self.num_steps, 3), dtype=torch.float32, device=self.device)
        self.steps_dphi = torch.zeros((self.num_envs, self.num_steps), dtype=torch.float32, device=self.device)
        self.targets_w = torch.zeros((self.num_envs, self.look_ahead + self.look_behind, self.target_dim), dtype=torch.float32, device=self.device)
        self.targets_b = torch.zeros((self.num_envs, self.look_ahead + self.look_behind, self.target_dim), dtype=torch.float32, device=self.device)
        self.pre_defined_swing_leg = torch.ones((self.num_envs, self.num_steps), dtype=torch.int64, device=self.device) # Given from the steps generator
        self._generate_foot_steps()

        # pre allocate buffers
        self.swing_leg = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device) 
        self.curr_target_index = torch.ones(self.num_envs, dtype=torch.int64, device=self.device) # WARNING: if change this, need to change the reset() as well
        self.prev_target_index = torch.clamp(self.curr_target_index - 1, 0, self.num_steps - 1)
        self.next_target_index = torch.clamp(self.curr_target_index + 1, 0, self.num_steps - 1)
        self.target_reach_count = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.foot_contact = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device) # (N, B)

        # joint gears for torque controller
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.device)
        self.force_scale = self.cfg.force_scale
        
        # Foot names and indices
        self.foot_names = self.cfg.foot_names
        self.foot_indices = [self.robot.data.body_names.index(name) for name in self.foot_names]
        self.torso_index = self.robot.data.body_names.index(self.cfg.torso_name)
        self.hip_y_index = torch.tensor([self.robot.data.joint_names.index(name) for name in self.cfg.hip_y_names], dtype=torch.int64, device=self.device)
        self.right_body_indices = torch.tensor([self.robot.data.joint_names.index(name) for name in self.cfg.right_body_names], dtype=torch.int64, device=self.device)
        self.left_body_indices = torch.tensor([self.robot.data.joint_names.index(name) for name in self.cfg.left_body_names], dtype=torch.int64, device=self.device)
        self.negation_body_indices = torch.tensor([self.robot.data.joint_names.index(name) for name in self.cfg.negation_body_names], dtype=torch.int64, device=self.device)

        # to-target potentials (will be updated in reset_idx)
        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.old_potentials = self.potentials.clone()

        # DEBUGGING
        self.old_obs = None
        self.old_action = None
        self.old_robot_data = None
        self.curriculum_counter = 0

        
    
    def _generate_foot_steps(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        temp_steps_pos, temp_steps_dphi, temp_swing_leg = self._generate_foot_steps_allsteps()
        temp_steps_pos = temp_steps_pos + self.scene.env_origins.unsqueeze(1) # (num_envs, num_steps, 3)
        temp_steps_dphi = temp_steps_dphi.repeat(self.num_envs, 1)
        temp_swing_leg = temp_swing_leg.repeat(self.num_envs, 1)
        
        self.steps_pos[env_ids] = temp_steps_pos[env_ids]
        self.steps_dphi[env_ids] = temp_steps_dphi[env_ids]
        self.pre_defined_swing_leg[env_ids] = temp_swing_leg[env_ids]
        
        full_steps_pose = torch.cat((self.steps_pos, torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device).repeat(self.num_envs, self.num_steps, 1)), dim=-1) # (num_envs, num_steps, 7)
        self.steps.write_object_pose_to_sim(full_steps_pose[env_ids], env_ids)
        pos_on_step = self.steps_pos.clone()
        pos_on_step[:, :, 2] += 0.225 / 2
        self.marker.visualize(translations=pos_on_step.view(-1, 3), marker_indices=self.marker_idx)
        
    def _generate_foot_steps_allsteps(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.curriculum = torch.minimum(self.curriculum, self.max_curriculum) # (num_envs,)
        ratio = self.curriculum / self.max_curriculum # (num_envs,)

        dist_upper = torch.linspace(*self.dist_range, self.max_curriculum + 1).to(self.device) # (max_curriculum + 1, )
        dist_range = torch.stack([self.dist_range[0].repeat(self.num_envs), dist_upper[self.curriculum]], dim=-1) # (num_envs, 2)
        yaw_range = torch.deg2rad(self.yaw_range.unsqueeze(0) * ratio.unsqueeze(1)) # (num_envs, 2)
        pitch_range = torch.deg2rad(self.pitch_range.unsqueeze(0) * ratio.unsqueeze(1)) + torch.pi / 2 # (num_envs, 2)
        tilt_range = torch.deg2rad(self.tilt_range.unsqueeze(0) * ratio.unsqueeze(1)) # (num_envs, 2)

        N = self.num_steps

        dr = torch.lerp(dist_range[:, 0].unsqueeze(1), dist_range[:, 1].unsqueeze(1), torch.rand((self.num_envs, N), device=self.device))  # shape: (4096, N)
        dphi = torch.lerp(yaw_range[:, 0].unsqueeze(1), yaw_range[:, 1].unsqueeze(1), torch.rand((self.num_envs, N), device=self.device))  # shape: (4096, N)
        dtheta = torch.lerp(pitch_range[:, 0].unsqueeze(1), pitch_range[:, 1].unsqueeze(1), torch.rand((self.num_envs, N), device=self.device))  # shape: (4096, N)
        x_tilt = torch.lerp(tilt_range[:, 0].unsqueeze(1), tilt_range[:, 1].unsqueeze(1), torch.rand((self.num_envs, N), device=self.device))  # shape: (4096, N)
        y_tilt = torch.lerp(tilt_range[:, 0].unsqueeze(1), tilt_range[:, 1].unsqueeze(1), torch.rand((self.num_envs, N), device=self.device))  # shape: (4096, N)

        # make first step below feet
        dr[:, 0] = 0.0
        dphi[:, 0] = 0.0
        dtheta[:, 0] = torch.pi / 2

        dr[:, 1:3] = self.init_step_separation
        dphi[:, 1:3] = 0.0
        dtheta[:, 1:3] = torch.pi / 2

        x_tilt[:, :3] = 0
        y_tilt[:, :3] = 0

        dphi = torch.cumsum(dphi, dim=1)

        dx = dr * torch.sin(dtheta) * torch.cos(dphi)
        dy = dr * torch.sin(dtheta) * torch.sin(dphi)
        dz = dr * torch.cos(dtheta)

        # Fix overlapping steps
        # dx_max = torch.maximum(torch.abs(dx[:, 2:]), torch.full_like(dx[:, 2:], torch.tensor(self.step_radius * 2.5)))
        # dx[:, 2:] = torch.sign(dx[:, 2:]) * torch.minimum(dx_max, self.dist_range[1])

        x = torch.cumsum(dx, dim=1)
        y = torch.cumsum(dy, dim=1)
        z = torch.cumsum(dz, dim=1)

        # z[:] = 0  # set vertical to be 0 for now

        swing_legs = torch.ones(N, dtype=torch.int64)
        swing_legs[1:N:2] = 0 # [1,0,1,0,...] --> [Left, Right, Left, Right, ...]

        return torch.stack((x, y, z), axis=2).to(self.device), dphi.to(self.device), swing_legs.to(self.device)

    def _generate_foot_steps_allsteps_not_used(self) -> tuple[torch.Tensor, torch.Tensor]:
        N = self.num_steps

        dr = torch.ones(N) * self.init_step_separation
        dphi = torch.zeros(N)  # Placeholder for yaw range if needed later
        dtheta = torch.ones(N) * torch.pi / 2

        # make first step below feet
        dr[0] = 0.0
        dphi[0] = 0.0
        dphi[1] = 0.0
        dphi[2] = 0.0

        # Make the first 3 steps easier
        dr[1] = 0.3
        dr[2] = 0.5

        swing_legs = torch.ones(N, dtype=torch.int64)
        swing_legs[:N:2] = 0

        dphi = torch.cumsum(dphi, dim=0)

        dy = dr * torch.sin(dtheta) * torch.sin(dphi)
        dx = dr * torch.sin(dtheta) * torch.cos(dphi)
        dz = dr * torch.cos(dtheta)

        x = torch.cumsum(dx, dim=0)
        y = torch.cumsum(dy, dim=0)
        z = torch.cumsum(dz, dim=0)

        # Calculate shifts
        left_shifts = torch.stack([torch.cos(dphi + torch.pi / 2), torch.sin(dphi + torch.pi / 2)])
        right_shifts = torch.stack([torch.cos(dphi - torch.pi / 2), torch.sin(dphi - torch.pi / 2)])

        # Select left or right shifts based on swing legs
        left_mask = swing_legs == 1
        right_mask = ~left_mask

        shift_x = torch.where(left_mask, left_shifts[0], right_shifts[0])
        shift_y = torch.where(left_mask, left_shifts[1], right_shifts[1])

        x += shift_x * self.foot_sep
        y += shift_y * self.foot_sep

        return torch.stack((x, y, z), dim=1).to(self.device), dphi.to(self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.sensor = ContactSensor(self.cfg.foot_contacts)
        # add ground plane
        # spawn_ground_plane(
        #     prim_path="/World/ground",
        #     cfg=GroundPlaneCfg(
        #         physics_material=sim_utils.RigidBodyMaterialCfg(
        #             static_friction=1.0,
        #             dynamic_friction=1.0,
        #             restitution=0.0,
        #         ),
        #     ),
        # )
        self.steps = RigidObjectCollection(self.cfg.steps)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["foot_contacts"] = self.sensor
        self.scene.rigid_object_collections["steps"] = self.steps
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # create marker
        self.marker = VisualizationMarkers(self.cfg.step_markers)

        # Adjust camera
        self.sim.set_camera_view(eye=self.cfg.camera_pos, target=(0.0, 0.0, 0.0))
        
    def _pre_physics_step(self, actions: torch.Tensor):
        ##### DEBUGGING #####
        self.old_action = self.actions.clone()
        self.old_robot_root_pos = self.robot.data.root_pos_w.clone()
        self.old_robot_root_quat = self.robot.data.root_quat_w.clone()
        self.old_robot_joint_pos = self.robot.data.joint_pos.clone()
        self.old_robot_joint_vel = self.robot.data.joint_vel.clone()
        self.old_robot_root_vel_w = self.robot.data.root_lin_vel_w.clone()
        ####### DEBUGGING #####

        self.actions = actions.clone()
        self.actions = torch.clamp(self.actions, -1.0, 1.0)
        
    def _apply_action(self):
        # Torque controller
        # forces = self.force_scale * self.joint_gears * self.actions
        forces = self.applied_gain_curriculum[self.curriculum].unsqueeze(-1) * self.joint_gears.unsqueeze(0) * self.actions
        self.robot.set_joint_effort_target(forces)

    def _compute_useful_values(self):
        self.right_foot_pos_w = self.robot.data.body_pos_w[:, self.foot_indices[RIGHT_FOOT]]
        self.left_foot_pos_w = self.robot.data.body_pos_w[:, self.foot_indices[LEFT_FOOT]]
        self.torso_pos_w = self.robot.data.body_pos_w[:, self.torso_index]
        
        self.lower_foot_z_w = torch.minimum(self.left_foot_pos_w[:, 2], self.right_foot_pos_w[:, 2])

        self.torso_to_feet_height = self.torso_pos_w[:, 2] - self.lower_foot_z_w

        self.roll, self.pitch, self.yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        
        self.joint_pos_scaled = scale_transform(
            self.robot.data.joint_pos,
            self.robot.data.joint_pos_limits[:, :, 0],
            self.robot.data.joint_pos_limits[:, :, 1],
        )

        self.root_vec_b = quat_rotate_inverse(self.robot.data.root_quat_w, self.robot.data.root_lin_vel_w)

        self.root_ang_vec_b = quat_rotate_inverse(self.robot.data.root_quat_w, self.robot.data.root_ang_vel_w)
        
        # compute foot states
        self._calculate_foot_state() # here we still update state for every env
        
        # calculate targets
        self._calculate_targets()
        self.targets_b[:, 0] = subtract_frame_transforms(
            t01=self.robot.data.root_pos_w,
            q01=self.robot.data.root_quat_w,
            t02=self.targets_w[:, 0]
        )[0]
        self.targets_b[:, 1] = subtract_frame_transforms(
            t01=self.robot.data.root_pos_w,
            q01=self.robot.data.root_quat_w,
            t02=self.targets_w[:, 1]
        )[0]
        self.targets_b[:, 2] = subtract_frame_transforms(
            t01=self.robot.data.root_pos_w,
            q01=self.robot.data.root_quat_w,
            t02=self.targets_w[:, 2]
        )[0]

        # update potentials
        self._calculate_body_potentials()


        # Adjust camera
        cam_pos_w = torch.tensor(self.cfg.camera_pos, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1) + self.robot.data.root_pos_w
        self.sim.set_camera_view(eye=tuple(cam_pos_w[0].tolist()), target=tuple(self.robot.data.root_pos_w[0].tolist()))

    def _get_observations(self) -> dict:
        # if self.num_envs == 1:
        #     print(self.curr_target_index)
        # build task observation
        obs = torch.cat(
            (
                self.torso_to_feet_height.unsqueeze(-1), # 1
                self.roll.unsqueeze(-1), # 1
                self.pitch.unsqueeze(-1), # 1
                self.root_vec_b, # 3
                self.joint_pos_scaled, # 21
                torch.clamp(self.robot.data.joint_vel * self.cfg.dof_vel_scale, -5, 5), # 21
                self.foot_contact, # 2
                self.targets_b.reshape(self.num_envs, -1) # 3 * 3 = 9
                # self.steps_pos[:, -1, 0:2]
            ), 
            dim=-1
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        
        # Reward for no falling
        alive_reward = torch.ones_like(self.torso_to_feet_height) * self.cfg.alive_reward_scale

        # A small reward for body reaching target
        progress = self.potentials - self.old_potentials
        
        # Some regularization terms 
        roll_violation = (self.roll > 0.4) | (self.roll < -0.4)
        pitch_violation = (self.pitch > 0.4) | (self.pitch < -0.2)
        roll_cost = torch.where(roll_violation, self.roll.abs(), torch.zeros_like(self.roll))
        pitch_cost = torch.where(pitch_violation, self.pitch.abs(), torch.zeros_like(self.pitch))
        
        speed = torch.linalg.vector_norm(self.robot.data.root_lin_vel_w, dim=-1)
        speed_cost = torch.where(speed > 1.6, speed - 1.6, torch.zeros_like(speed))
        
        action_cost = self.cfg.actions_cost_scale * torch.linalg.vector_norm(self.actions, dim=-1)
        energy_cost = self.cfg.energy_cost_scale * torch.sum(torch.abs(self.robot.data.joint_vel * self.actions), dim=-1)

        joint_at_limit_cost = torch.count_nonzero(torch.abs(self.joint_pos_scaled) > 0.99, dim=-1).float() * self.cfg.joint_at_limit_cost_scale
        
        # Step reward to encourage the foot to step on the center of the target
        step_reward_condition = (self.target_reached) & (self.target_reach_count == 1) & (self.curr_target_index < self.num_steps - 1) 
        dist = self.foot_to_target_dist_xy[torch.arange(self.num_envs), self.swing_leg]
        step_reward = torch.where(step_reward_condition, 50 * torch.exp( -dist / 0.25), torch.zeros_like(step_reward_condition))
        
        target_bonus_condition = (self.curr_target_index == self.num_steps - 1) & (self.body_dist_to_target_xy < 0.15)
        target_bonus = torch.where(target_bonus_condition, 10 * torch.ones_like(target_bonus_condition), torch.zeros_like(target_bonus_condition))
        
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
        # fell = self.torso_to_feet_height < self.cfg.termination_height_torso_to_feet
        fell = self.torso_to_feet_height < self.termination_curriculum[self.curriculum]
        so_fast = torch.linalg.vector_norm(self.robot.data.root_lin_vel_w, dim=-1) > 5.0 # To avoid weird NaN issues
        died = self.robot.data.root_pos_w[:, 2] < self.cfg.termination_height_absolute

        return fell | so_fast | died, time_out
    
    def _calculate_body_potentials(self):
        walk_target_delta = self.targets_w[:, -1] - self.robot.data.root_pos_w
        # walk_target_delta = self.steps_pos[:, -1] - self.robot.data.root_pos_w
        self.body_dist_to_target_xy = torch.linalg.vector_norm(walk_target_delta[:, 0:2], dim=-1)

        foot_delta = self.targets_w[:, 1] - self.robot.data.body_pos_w[:, self.foot_indices][torch.arange(self.num_envs), self.swing_leg]
        self.foot_dist_to_target = torch.linalg.vector_norm(foot_delta, dim=-1)
        
        self.old_potentials = self.potentials.clone()
        self.potentials = -(self.body_dist_to_target_xy) / self.step_dt
        
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
        
        self.target_reached = (binary_contact[torch.arange(self.num_envs), self.swing_leg] > 0) & ((self.foot_to_target_dist_xy < self.step_radius)[torch.arange(self.num_envs), self.swing_leg])
        
        self.target_reach_count[self.target_reached] += 1
        
        can_progress = self.target_reach_count >= self.stop_frames
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
        
        prev_target_pos = self.steps_pos[torch.arange(N), self.prev_target_index] # (N, 3)
        current_target_pos = self.steps_pos[torch.arange(N), self.curr_target_index]
        next_target_pos = self.steps_pos[torch.arange(N), self.next_target_index]
        
        result = torch.stack([prev_target_pos, current_target_pos, next_target_pos], dim=1) # (N, 3, 3)
        self.targets_w[:] = result
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        # if we progress over half of the steps, we need to generate new foot steps
        if self.curr_target_index.float().mean() > self.curriculum_progess_theshold:
            self.curriculum = torch.clamp(self.curriculum + 1, 0, self.max_curriculum)
            if self.curriculum_counter % 500 == 0:
                print("--------------------------------------------")
                print(f"The current curriculum : {self.curriculum[0].item()}")
                print("--------------------------------------------")
                self.curriculum_counter = 0
            
            self.curriculum_counter += 1

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        # put potentials to zero first, later we will update them in _compute_useful_values
        self.old_potentials[env_ids] = 0.0
        self.potentials[env_ids] = 0.0
        
        self.target_reach_count[env_ids] = 0
        self.swing_leg[env_ids] = 0
        self.curr_target_index[env_ids] = 1 # start from the second target
        self.prev_target_index[env_ids] = torch.clamp(self.curr_target_index[env_ids] - 1, 0, self.num_steps - 1)
        self.next_target_index[env_ids] = torch.clamp(self.curr_target_index[env_ids] + 1, 0, self.num_steps - 1)
        
        # if we progress over half of the steps, we need to generate new foot steps
        over_half_ids = (self.curr_target_index > self.num_steps // 2).nonzero(as_tuple=False).flatten()
        replace_ids = env_ids[torch.isin(env_ids, over_half_ids)]
        if len(replace_ids) > 0:
            self._generate_foot_steps(replace_ids)

        # self.extras["mean_curriculum"] = self.curriculum.float().mean()
        
        # IMPORTANT: running start of the pose
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, [12, 17]] = -torch.pi / 8 # Right leg, hip_y, knee
        joint_pos[:, 15] = torch.pi / 10 # Left leg back, hip_y
        joint_pos[:, [2, 5]] = torch.pi / 3 # Left shoulder x, right shoulder x
        joint_pos[:, 4] = -torch.pi / 6 # Right shoulder z
        joint_pos[:, 7] = torch.pi / 6 # Left shoulder z
        joint_pos[:, [9, 10]] = torch.pi / 3 # Left elbow, right elbow

        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # 50% chance mirror the starting pose
        mirror_masks = torch.rand(env_ids.shape, device=env_ids.device) > 0.5
        subset_mirror_ids = torch.nonzero(mirror_masks, as_tuple=True)[0]
        original_mirror_ids = env_ids[mirror_masks]
    
        mirrored_joint_pos = joint_pos.clone()
        mirrored_joint_pos[subset_mirror_ids[:, None], self.right_body_indices] = joint_pos[subset_mirror_ids[:, None], self.left_body_indices]
        mirrored_joint_pos[subset_mirror_ids[:, None], self.left_body_indices] = joint_pos[subset_mirror_ids[:, None], self.right_body_indices]
        mirrored_joint_pos[subset_mirror_ids[:, None], self.negation_body_indices] *= -1
        joint_pos[subset_mirror_ids] = mirrored_joint_pos[subset_mirror_ids] 

        mirrored_joint_vel = joint_vel.clone()
        mirrored_joint_vel[subset_mirror_ids[:, None], self.right_body_indices] = joint_vel[subset_mirror_ids[:, None], self.left_body_indices]
        mirrored_joint_vel[subset_mirror_ids[:, None], self.left_body_indices] = joint_vel[subset_mirror_ids[:, None], self.right_body_indices]
        mirrored_joint_vel[subset_mirror_ids[:, None], self.negation_body_indices] *= -1
        joint_vel[subset_mirror_ids] = mirrored_joint_vel[subset_mirror_ids]

        mirrored_root_state = default_root_state.clone()
        mirrored_root_state[subset_mirror_ids, 4:7] *= -1 # flipped the quat axis (w, x, y, z) -> flip x,y,z
        default_root_state[subset_mirror_ids] = mirrored_root_state[subset_mirror_ids]

        self.swing_leg[original_mirror_ids] = self.swing_leg[original_mirror_ids] ^ 1 # flip the leg

        ############## NOISE ##############
        # Add noise to the joint position
        joint_pos[:] += sample_uniform(
            self.cfg.initial_joint_angle_range[0],
            self.cfg.initial_joint_angle_range[1],
            joint_pos.shape,
            joint_pos.device,
        )
        normalized_joint_pos = scale_transform(
            joint_pos,
            self.robot.data.joint_pos_limits[env_ids, :, 0],
            self.robot.data.joint_pos_limits[env_ids, :, 1],
        )
        
        normalized_joint_pos = torch.clamp(normalized_joint_pos, self.cfg.initial_joint_angle_clip_range[0], self.cfg.initial_joint_angle_clip_range[1])

        clipped_joint_pos = unscale_transform(
            normalized_joint_pos,
            self.robot.data.joint_pos_limits[env_ids, :, 0],
            self.robot.data.joint_pos_limits[env_ids, :, 1],
        )
        ##########################################

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(clipped_joint_pos, joint_vel, None, env_ids)
        
        self._compute_useful_values() # here we still update state for every env 


def get_symmetric_states_rsl_rl(
        obs: torch.Tensor | None, actions: torch.Tensor | None, env: RslRlVecEnvWrapper | RlGamesVecEnvWrapper, is_critic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    
    right_joint_indices = env.unwrapped.right_body_indices
    left_joint_indices = env.unwrapped.left_body_indices
    negation_joint_indices = env.unwrapped.negation_body_indices

    K = 2 if env.unwrapped.observation_space.shape[1] == 56 else 3
    steps_negation_indices = torch.tensor([K * i + 1 for i in range(3)], dtype=torch.int64, device=env.device) # y
    root_negation_indices = torch.tensor([1, 4], dtype=torch.int64, device=env.device) # roll, vy

    right_obs_indices = torch.cat((right_joint_indices + 6, right_joint_indices + 6 + env.unwrapped.action_space.shape[1], torch.tensor([6 + env.unwrapped.action_space.shape[1] * 2], dtype=torch.int64, device=env.device))) # joint pos, joint vel, foot contact
    left_obs_indices = torch.cat((left_joint_indices + 6, left_joint_indices + 6 + env.unwrapped.action_space.shape[1], torch.tensor([6 + env.unwrapped.action_space.shape[1] * 2 + 1], dtype=torch.int64, device=env.device))) # joint pos, joint vel, foot contact
    negation_obs_indices = torch.cat((root_negation_indices, 6 + negation_joint_indices, 6 + env.unwrapped.action_space.shape[1] + negation_joint_indices, 6 + env.unwrapped.action_space.shape[1] * 2 + 2 + steps_negation_indices)) # roll, vy, joint pos, joint vel, steps y

    if obs is None:
        mirrored_obs = None
        return_obs = None
    else:
        mirrored_obs = obs.clone()
        mirrored_obs[:, right_obs_indices] = obs[:, left_obs_indices]
        mirrored_obs[:, left_obs_indices] = obs[:, right_obs_indices]
        mirrored_obs[:, negation_obs_indices] = -obs[:, negation_obs_indices]

        return_obs = torch.vstack((obs, mirrored_obs))

    if actions is None:
        mirrored_actions = None
        return_actions = None
    else:
        mirrored_actions = actions.clone()
        mirrored_actions[:, right_joint_indices] = actions[:, left_joint_indices]
        mirrored_actions[:, left_joint_indices] = actions[:, right_joint_indices]
        mirrored_actions[:, negation_joint_indices] = -actions[:, negation_joint_indices]

        return_actions = torch.vstack((actions, mirrored_actions))

    return return_obs, return_actions


def get_symmetric_states_rl_games(
        obs: torch.Tensor | None, actions: torch.Tensor | None, env: RslRlVecEnvWrapper | RlGamesVecEnvWrapper, is_critic: bool, mus: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    right_joint_indices = env.unwrapped.right_body_indices
    left_joint_indices = env.unwrapped.left_body_indices
    negation_joint_indices = env.unwrapped.negation_body_indices

    K = 2 if env.unwrapped.observation_space.shape[1] == 56 else 3
    steps_negation_indices = torch.tensor([K * i + 1 for i in range(3)], dtype=torch.int64, device=env.device) # y
    root_negation_indices = torch.tensor([1, 4], dtype=torch.int64, device=env.device) # roll, vy

    right_obs_indices = torch.cat((right_joint_indices + 6, right_joint_indices + 6 + env.unwrapped.action_space.shape[1], torch.tensor([6 + env.unwrapped.action_space.shape[1] * 2], dtype=torch.int64, device=env.device))) # joint pos, joint vel, foot contact
    left_obs_indices = torch.cat((left_joint_indices + 6, left_joint_indices + 6 + env.unwrapped.action_space.shape[1], torch.tensor([6 + env.unwrapped.action_space.shape[1] * 2 + 1], dtype=torch.int64, device=env.device))) # joint pos, joint vel, foot contact
    negation_obs_indices = torch.cat((root_negation_indices, 6 + negation_joint_indices, 6 + env.unwrapped.action_space.shape[1] + negation_joint_indices, 6 + env.unwrapped.action_space.shape[1] * 2 + 2 + steps_negation_indices)) # roll, vy, joint pos, joint vel, steps y

    if obs is None:
        mirrored_obs = None
        return_obs = None
    else:
        mirrored_obs = obs.clone()
        mirrored_obs[:, right_obs_indices] = obs[:, left_obs_indices]
        mirrored_obs[:, left_obs_indices] = obs[:, right_obs_indices]
        mirrored_obs[:, negation_obs_indices] = -obs[:, negation_obs_indices]

        return_obs = torch.vstack((obs, mirrored_obs))

    if actions is None:
        mirrored_actions = None
        return_actions = None
    else:
        mirrored_actions = actions.clone()
        mirrored_actions[:, right_joint_indices] = actions[:, left_joint_indices]
        mirrored_actions[:, left_joint_indices] = actions[:, right_joint_indices]
        mirrored_actions[:, negation_joint_indices] = -actions[:, negation_joint_indices]

        return_actions = torch.vstack((actions, mirrored_actions))

    if mus is None:
        mirrored_mus = None
        return_mus = None
    else:
        mirrored_mus = mus.clone()
        mirrored_mus[:, right_joint_indices] = mus[:, left_joint_indices]
        mirrored_mus[:, left_joint_indices] = mus[:, right_joint_indices]
        mirrored_mus[:, negation_joint_indices] = -mus[:, negation_joint_indices]

        return_mus = torch.vstack((mus, mirrored_mus))

    return return_obs, return_actions, return_mus

