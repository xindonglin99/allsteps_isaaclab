from __future__ import annotations
from typing import Tuple

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

def create_step_cfg(num_steps: int, size: Tuple[float, float, float]) -> RigidObjectCollectionCfg:
    """Create a step configuration."""
    collection = {}
    spawn_cfg = sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1), metallic=0.2),
        # physics_material=sim_utils.RigidBodyMaterialCfg(
        #     friction_combine_mode="average",
        #     restitution_combine_mode="average",
        #     static_friction=1.0,
        #     dynamic_friction=1.0,
        #     restitution=0.0,
        # ),
    )
    initial_state = RigidObjectCfg.InitialStateCfg()

    for i in range(num_steps):
        key = "step_" + str(i)
        current_prim_path = "/World/envs/env_.*/" + key 
        collection[key] = RigidObjectCfg(
            prim_path=current_prim_path,
            spawn=spawn_cfg,
            init_state=initial_state,
            collision_group=0,
            debug_vis=False,
    )

        
    return RigidObjectCollectionCfg(rigid_objects=collection)

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
    robot: ArticulationCfg = WALKER_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Steps
    num_steps: int = 20
    steps: RigidObjectCollectionCfg = create_step_cfg(num_steps=num_steps, size=(0.5, 0.8, 0.225))

    camera_pos: Tuple[float, float, float] = (1.5, -4.0, 1.5)

    step_radius: float = 0.35

    marker_radius: float = 0.35

    # foot step markers
    step_markers: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/stepMarkers",
        markers={
            "marker1": sim_utils.CylinderCfg(
                radius=0.1,
                height=0.00001,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "marker2": sim_utils.CylinderCfg(
                radius=0.1,
                height=0.00001,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        }
    )

    # foot contact sensors
    foot_contacts: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/walker3d/.*_foot", update_period=0.0, history_length=4, debug_vis=True
    )
    
    # Joint correspondence is different than in original file
    joint_gears: list = [
        60, # abdomen_z
        80, # abdomen_y
        60, # right_shoulder_x
        50, # right_shoulder_y
        60, # right_shoulder_z
        60, # left_shoulder_x
        50, # left_shoulder_y
        60, # left_shoulder_z
        60, # abdomen_x
        60, # right_elbow
        60, # left_elbow
        80, # right_hip_x
        100, # right_hip_y
        60, # right_hip_z
        80, # left_hip_x
        100, # left_hip_y
        60, # left_hip_z
        90, # right_knee
        90, # left_knee
        60, # right_ankle
        60, # left_ankle
    ] # 21 WALER3d joint gears
    
    # joint_gears: list = [
    #     67.5000,  # lower_waist
    #     67.5000,  # lower_waist
    #     67.5000,  # right_upper_arm
    #     67.5000,  # right_upper_arm
    #     67.5000,  # left_upper_arm
    #     67.5000,  # left_upper_arm
    #     67.5000,  # pelvis
    #     45.0000,  # right_lower_arm
    #     45.0000,  # left_lower_arm
    #     45.0000,  # right_thigh: x
    #     135.0000,  # right_thigh: y
    #     45.0000,  # right_thigh: z
    #     45.0000,  # left_thigh: x
    #     135.0000,  # left_thigh: y
    #     45.0000,  # left_thigh: z
    #     90.0000,  # right_knee
    #     90.0000,  # left_knee
    #     22.5,  # right_foot
    #     22.5,  # right_foot
    #     22.5,  # left_foot
    #     22.5,  # left_foot
    # ] # HUMANOID 21 joint gears
    
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
    
    force_scale = 1.5
    
    torso_name: str = "torso"
    foot_names: list = ["right_foot", "left_foot"]

    energy_cost_scale: float = 0.214
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1
    joint_at_limit_cost_scale: float = 0.1

    death_cost: float = -1.0
    termination_height_absolute: float = 0.2

    angular_velocity_scale: float = 0.25

    initial_joint_angle_range: list = [-0.1, 0.1] # rad
    initial_joint_angle_clip_range: list = [-0.95, 0.95] # rad


