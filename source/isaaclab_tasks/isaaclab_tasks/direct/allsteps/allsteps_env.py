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

class AllstepsEnv(DirectRLEnv):
    cfg: AllstepsEnvCfg

    def __init__(self, cfg: AllstepsEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # action offset and scale for PD controller
        dof_lower_limits = self.robot.data.joint_limits[0, :, 0]
        dof_upper_limits = self.robot.data.joint_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits

        # joint gears for torque controller
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.device)
        self.joint_gears *= 0.5
        self.force_scale = self.cfg.force_scale

        # Setup grab conditions
        # self.grab_conditions = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.random_target = self.cfg.random_target
        if not self.random_target and self.cfg.num_targets > len(self.cfg.handhold_pos):
            raise ValueError("Number of targets should be less than the number of handholds")
        self.grab_conditions = torch.zeros((self.num_envs), dtype=torch.bool, device=self.device) # Set using left hand to grab
        self.handhold_pos = torch.tensor(self.cfg.handhold_pos, device=self.device) + self.scene.env_origins.unsqueeze(1) # (num_envs, 3, 3)
        self.current_target_idx = torch.zeros((self.num_envs), dtype=torch.int64, device=self.device)
        self.hold_counter = torch.zeros((self.num_envs), dtype=torch.int64, device=self.device)
        self.hold_counter_old = self.hold_counter.clone()
        self.release_counter = torch.zeros((self.num_envs), dtype=torch.int64, device=self.device)
        self.release_counter_old = self.release_counter.clone()
        self.target_sequence = torch.zeros((self.num_envs, self.cfg.num_targets), dtype=torch.int64, device=self.device)
        self.limb_names = self.cfg.limb_names
        key_body_names = self.limb_names # left toe, right toe, left wrist, right wrist
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.chest_index = self.robot.data.body_names.index(self.cfg.chest_name)
        
        
        # Dynamic control API
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self.d6_force_limit = 5000
        self.d6_joint_stiffness = 700
        self.d6_joint_damping = 70
        self.d6_props = {}
        self.d6_joints = {}
        self.d6_info = {}
        self.d6_env_ids = torch.zeros((self.num_envs), dtype=torch.bool, device=self.device)

        # to-target potentials (will be updated in reset_idx)
        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.old_potentials = self.potentials.clone()
        
        self._pre_create_d6_joints()
        if self.random_target:
            self._generate_random_target_idx()
        else:
            self._generate_fix_target_idx()

        # Marker index
        self.marker_idx = torch.zeros((self.num_envs * 3), dtype=torch.int64, device=self.device)
        self.marker.visualize(translations=self.handhold_pos.view(-1, 3), marker_indices=self.marker_idx)
    
        
    def _generate_random_target_idx(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        self.target_sequence[env_ids] = torch.randint(high=self.handhold_pos.shape[1], size=(self.num_envs, self.cfg.num_targets), device=self.device)[env_ids]


    def _generate_fix_target_idx(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        self.target_sequence[env_ids] = torch.arange(len(self.cfg.handhold_pos), device=self.device).repeat((self.num_envs, 1))[env_ids]

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
        ## add anchor
        self.anchor = RigidObject(self.cfg.anchor)
        ## add handholds
        # self.handholds = RigidObjectCollection(self.cfg.handholds)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # create marker
        self.marker = VisualizationMarkers(self.cfg.handhold_markers)
        
    def _pre_create_d6_joints(self):
        rel_pose = self.robot.data.body_pos_w[torch.arange(self.num_envs), self.key_body_indexes[LimbNames.LeftHand.value]] - self.anchor.data.root_pos_w # changed here to also use GPU data
        for i in range(self.num_envs):
            base_path = f"/World/envs/env_{i}/" + self.cfg.base_path_robot
            limb_paths = [ base_path + '/' + body_name for body_name in self.limb_names ]
            hold_path = "/World/anchor"
            for limb_path in limb_paths:
                tmp_d6_properties = _dynamic_control.D6JointProperties()
                tmp_limb_handle = self.dc.get_rigid_body(limb_path)
                ground_handle = self.dc.get_rigid_body(hold_path)
                tmp_d6_properties.axes = _dynamic_control.AXIS_NONE
                tmp_d6_properties.body0 = ground_handle
                tmp_d6_properties.body1 = tmp_limb_handle

                # rel_pose = self.dc.get_relative_body_poses(ground_handle, [tmp_limb_handle])
                tmp_d6_properties.pose0.p = (rel_pose[i, 0], rel_pose[i, 1], rel_pose[i, 2])
                tmp_d6_properties.force_limit = self.d6_force_limit
                tmp_d6_properties.stiffness = self.d6_joint_stiffness
                tmp_d6_properties.damping = self.d6_joint_damping
                d6_joint_name = limb_path + "_hold"
                tmp_d6_properties.name = d6_joint_name
                self.d6_props[d6_joint_name] = tmp_d6_properties
                self.d6_joints[d6_joint_name] = self.dc.create_d6_joint(tmp_d6_properties)

    def _grab(self, hold_path: str, relative_pos: torch.Tensor):
        # Put on constraints
        self.d6_props[hold_path].axes = _dynamic_control.AXIS_ALL_TRANSLATION
        self.d6_props[hold_path].pose0.p = (relative_pos[0], relative_pos[1], relative_pos[2])
        # You can see the difference of values when run on GPU.
        #print(relative_pos, self.dc.get_relative_body_poses(self.d6_props[hold_path].body0, [self.d6_props[hold_path].body1])[0].p) 
        # self.dc.set_d6_joint_properties(self.d6_joints[hold_path], self.d6_props[hold_path])

        self.d6_info[hold_path] = True

    def _release(self, hold_path: str):
        # Disable constraints
        self.d6_props[hold_path].axes = _dynamic_control.AXIS_NONE

        # self.dc.set_d6_joint_properties(self.d6_joints[hold_path], self.d6_props[hold_path])
        
        # print(f"Releasing {hold_path}")
        self.d6_info[hold_path] = False

    def _reset_release(self, env_ids: torch.Tensor) -> None:
        # already_grabbed = self.hold_counter > 0
        # mask = already_grabbed & self.reset_buf 
        # # TODO : In some conditions although the characters are not grabbing(self.hold_counter == 0), they didn't release grab.
        # # We have to find the cause of this happening.
        # root_spd = torch.norm(self.robot.data.root_com_lin_vel_w, dim=-1)
        # mask = torch.where(root_spd >= 3, torch.ones_like(mask), mask) # TODO: temporal solution. But this is not fundamental solution.
        # masked_ids = mask.nonzero(as_tuple=False).squeeze(-1)
        release_path = self._generate_hold_path(env_ids)
        self.d6_env_ids[env_ids] = False
        for path in release_path:
            self._release(path)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

        # Last action is grab if > 0, release if < 0. Now only for left hand.
        self.actions_grab = self.actions[:, -1]
        self.actions_joints = self.actions[:, :-1]

        self._grab_or_release()

        already_grabbed_idx = self.d6_env_ids.nonzero(as_tuple=False).flatten() # this gives the env idx
        marker_offset = self.target_sequence[torch.arange(self.num_envs), self.current_target_idx] # (num_envs)
        self.marker_idx.fill_(0)
        if len(already_grabbed_idx) > 0:
            self.marker_idx.view(self.num_envs, 3)[already_grabbed_idx, marker_offset[already_grabbed_idx]] = 1
        self.marker.visualize(translations=self.handhold_pos.view(-1, 3), marker_indices=self.marker_idx)
        
    def _generate_hold_path(self, idx, limb_names = ["LeftWrist"]):
        return [f"/World/envs/env_{i}/" + self.cfg.base_path_robot + f"/{limb_names[0]}_hold" for i in idx]
        
    def _grab_or_release(self):
        # Check grab conditions
        grabbed = self.actions_grab > 0 # Grab action
        limb_to_target = self.robot.data.body_pos_w[:, self.key_body_indexes[LimbNames.LeftHand.value]] - self.handhold_pos[torch.arange(self.num_envs), self.target_sequence[torch.arange(self.num_envs), self.current_target_idx]]
        reached = torch.linalg.vector_norm(limb_to_target, dim=-1) < self.cfg.grab_threshold # Reach target

        ####### Debugging #######
        # print(self.robot.data.root_pos_w[:, 2])
        # print(grabbed)
        # print(reached)
        #########################

        self.grab_conditions: torch.Tensor = grabbed & reached

        not_yet_grabbed = self.hold_counter == 0
        already_grabbed = self.hold_counter > 0

        # Get index of the idx where we need to call set_d6_joint_properties, this way we can reduce the number of calls
        need_d6_change_grab_idx = (not_yet_grabbed & self.grab_conditions).nonzero(as_tuple=False).flatten() # first time grab
        need_d6_change_release_idx = (already_grabbed & (~self.grab_conditions)).nonzero(as_tuple=False).flatten() # grabbed that need to release

        # Generate hold path (name of the d6 joint) and set the properties
        if len(need_d6_change_grab_idx) > 0:
            grab_path = self._generate_hold_path(need_d6_change_grab_idx)
            self.d6_env_ids[need_d6_change_grab_idx] = True
            rel_pos = self.robot.data.body_pos_w[need_d6_change_grab_idx, self.key_body_indexes[LimbNames.LeftHand.value]] - self.anchor.data.root_pos_w
            for i, path in enumerate(grab_path):
                self._grab(path, rel_pos[i])

        if len(need_d6_change_release_idx) > 0:
            release_path = self._generate_hold_path(need_d6_change_release_idx)
            self.d6_env_ids[need_d6_change_release_idx] = False

            for path in release_path:
                self._release(path)
        
        # Save old hold counter 
        self.hold_counter_old = self.hold_counter.clone()

        # Update hold counter
        self.hold_counter[self.grab_conditions] += 1
        self.hold_counter[~self.grab_conditions] = 0

        # Move to next target
        can_progress = self.hold_counter > self.cfg.hold_counter_stop_frames
        self.current_target_idx[can_progress] = clamp_index(self.current_target_idx[can_progress] + 1, 0, self.cfg.num_targets - 1)
        self.hold_counter[can_progress] = 0

    def _apply_action(self):
        if self.cfg.pd_control:
            # PD controller
            target = self.action_offset + self.action_scale * self.actions_joints
            self.robot.set_joint_position_target(target)
        else:
        # Torque controller
            forces = self.force_scale * self.joint_gears * self.actions_joints
            self.robot.set_joint_effort_target(forces)

    def _compute_useful_values(self):
        # hard code position of interests now, could be turned into a loop later. A bit ugly now.
        ####### Debugging #######
        # print(self.robot.data.root_pos_w[:, 2])
        # print(self.current_target_idx)
        # print(self.grab_conditions)
        #########################
        
        self.left_wrist_pos_w = self.robot.data.body_pos_w[:, self.key_body_indexes[LimbNames.LeftHand.value]]
        self.right_wrist_pos_w = self.robot.data.body_pos_w[:, self.key_body_indexes[LimbNames.RightHand.value]]
        self.left_toe_pos_w = self.robot.data.body_pos_w[:, self.key_body_indexes[LimbNames.LeftFoot.value]]
        self.right_toe_pos_w = self.robot.data.body_pos_w[:, self.key_body_indexes[LimbNames.RightFoot.value]]
        self.chest_pos_w = self.robot.data.body_pos_w[:, self.chest_index]
        
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
        
        # Reward for holding it
        hold_reward = torch.where(self.hold_counter > self.hold_counter_old, torch.ones_like(self.hold_counter), torch.zeros_like(self.hold_counter))
        
        # total reward
        total_reward = (
            alive_reward
            + progress
            + hold_reward
        )
        
        # adjust reward for falling
        total_reward = torch.where(self.reset_terminated, self.cfg.death_cost * torch.ones_like(total_reward), total_reward)
        
        if self.num_envs < 1:
            for i in range(self.num_envs):
                print(f"Env: {i} ---------------------------------")
                print(f"Reward: {total_reward[i].item()}")
                print(f"Alive Reward: {alive_reward[i].item()}")
                print(f"Progress Reward: {progress[i].item()}")
                print(f"Hold Reward: {hold_reward[i].item()}")
                print(f"Hold Counter: {self.hold_counter[i].item()}")
                print(f"Old Hold Counter: {self.hold_counter_old[i].item()}")
                print(f"Current Index: {self.current_target_idx[i].item()}")
                print(f"Current Target: {self.target_sequence[i, self.current_target_idx[i].item()]}")
                print("--------------------------------------------")
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

        self._reset_release(env_ids)
        self.hold_counter[env_ids] = 0  
        self.hold_counter_old[env_ids] = 0
        
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

        self.current_target_idx[env_ids] = 0
        if self.random_target:
            self._generate_random_target_idx(env_ids)
        else:
            self._generate_fix_target_idx(env_ids)
        
        self._compute_useful_values()

@torch.jit.script
def clamp_index(index: torch.Tensor, min_index: int, max_index: int) -> torch.Tensor:
    min_index_tensor = torch.ones_like(index) * min_index
    max_index_tensor = torch.ones_like(index) * max_index

    return torch.where( index > max_index_tensor, max_index_tensor, torch.where(index < min_index_tensor, min_index_tensor, index))
