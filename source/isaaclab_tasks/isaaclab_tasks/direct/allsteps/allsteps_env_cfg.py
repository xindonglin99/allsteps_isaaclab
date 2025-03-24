from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg

from isaaclab_assets import WALKER_CFG, HUMANOID_28_CFG

@configclass
class AllstepsEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 4
    action_scale = 1.0
    action_space = 21
    observation_space = 117 - 3 - 29 - 3 * 4
    state_space = 0
    pd_control = False

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = WALKER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    joint_gears: list = [
        60, # abdomen_z
        80, # abdomen_y 
        60, # abdomen_x
        80, # right_hip_x
        60, # right_hip_z
        100, # right_hip_y
        90, # right_knee
        60, # right_ankle
        80, # left_hip_x
        60, # left_hip_z
        100, # left_hip_y
        90, # left_knee
        60, # left_ankle
        60, # right_shoulder_x
        60, # right_shoulder_z
        50, # right_shoulder_y
        60, # right_elbow
        60, # left_shoulder_x
        60, # left_shoulder_z
        50, # left_shoulder_y
        60, # left_elbow
    ] # 21
    
    force_scale = 1.0
    
    torso_name: str = "torso"
    foot_names: list = ["right_foot", "left_foot"]
    heading_weight: float = 0.5
    up_weight: float = 0.1

    grab_threshold: float = 0.05

    hold_counter_stop_frames: int = 60

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 0.7
    termination_height_chest_to_feet: float = 0.5

    angular_velocity_scale: float = 0.25

    