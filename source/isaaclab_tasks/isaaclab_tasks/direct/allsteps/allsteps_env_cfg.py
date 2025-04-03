from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import ContactSensorCfg

from isaaclab_assets import WALKER_CFG, HUMANOID_28_CFG, HUMANOID_CFG

@configclass
class AllstepsEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 4
    action_scale = 1.0
    action_space = 21
    observation_space = 59
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=decimation)
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
    robot: ArticulationCfg = HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    step_radius: float = 0.25

    marker_radius: float = 0.05

    # foot step markers
    step_markers: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/stepMarkers",
        markers={
            "marker1": sim_utils.SphereCfg(
                radius=marker_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "marker2": sim_utils.SphereCfg(
                radius=marker_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        }
    )

    # foot contact sensors
    foot_contacts: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_foot", update_period=0.0, history_length=4, debug_vis=True
    )
    
    # joint_gears: list = [
    #     60, # abdomen_z
    #     80, # abdomen_y 
    #     60, # abdomen_x
    #     80, # right_hip_x
    #     60, # right_hip_z
    #     100, # right_hip_y
    #     90, # right_knee
    #     60, # right_ankle
    #     80, # left_hip_x
    #     60, # left_hip_z
    #     100, # left_hip_y
    #     90, # left_knee
    #     60, # left_ankle
    #     60, # right_shoulder_x
    #     60, # right_shoulder_z
    #     50, # right_shoulder_y
    #     60, # right_elbow
    #     60, # left_shoulder_x
    #     60, # left_shoulder_z
    #     50, # left_shoulder_y
    #     60, # left_elbow
    # ] # 21 WALER3d joint gears
    
    joint_gears: list = [
        67.5000,  # lower_waist
        67.5000,  # lower_waist
        67.5000,  # right_upper_arm
        67.5000,  # right_upper_arm
        67.5000,  # left_upper_arm
        67.5000,  # left_upper_arm
        67.5000,  # pelvis
        45.0000,  # right_lower_arm
        45.0000,  # left_lower_arm
        45.0000,  # right_thigh: x
        135.0000,  # right_thigh: y
        45.0000,  # right_thigh: z
        45.0000,  # left_thigh: x
        135.0000,  # left_thigh: y
        45.0000,  # left_thigh: z
        90.0000,  # right_knee
        90.0000,  # left_knee
        22.5,  # right_foot
        22.5,  # right_foot
        22.5,  # left_foot
        22.5,  # left_foot
    ] # HUMANOID 21 joint gears
    
    # joint_gears: list = [
    #     60.0000,  # 'abdomen_x'
    #     80.0000,  # 'abdomen_y'
    #     60.0000,  # 'abdomen_z'
    #     20.0000,  # 'neck_x'
    #     20.0000,  # 'neck_y'
    #     20.0000,  # 'neck_z'
    #     60.0000,  # 'right_shoulder_x'
    #     60.0000,  # 'right_shoulder_y'
    #     60.0000,  # 'right_shoulder_z'
    #     60.0000,  # 'left_shoulder_x'
    #     60.0000,  # 'left_shoulder_y'
    #     60.0000,  # 'left_shoulder_z'
    #     80.0000,  # 'right_hip_x'
    #     100.0000, # 'right_hip_y'
    #     60.0000,  # 'right_hip_z'
    #     80.0000,  # 'left_hip_x'
    #     100.0000, # 'left_hip_y'
    #     60.0000,  # 'left_hip_z'
    #     60.0000,  # 'right_elbow'
    #     60.0000,  # 'left_elbow'
    #     90.0000,  # 'right_knee'
    #     90.0000,  # 'left_knee'
    #     20.0000,  # 'right_ankle_x'
    #     60.0000,  # 'right_ankle_y'
    #     20.0000,  # 'right_ankle_z'
    #     20.0000,  # 'left_ankle_x'
    #     60.0000,  # 'left_ankle_y'
    #     20.0000,  # 'left_ankle_z'
    # ] # HUMANOID 28 joint gears
    
    force_scale = 1.0
    
    torso_name: str = "torso"
    foot_names: list = ["right_foot", "left_foot"]

    energy_cost_scale: float = 0.214
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1
    joint_at_limit_cost_scale: float = 0.1

    death_cost: float = -1.0
    termination_height_torso_to_feet: float = 0.70

    angular_velocity_scale: float = 0.25

    