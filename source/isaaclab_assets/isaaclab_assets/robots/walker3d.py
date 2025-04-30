"""Configuration for the Allsteps Humanoid robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import os

## Point to the data folder
ASSET_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')

##
# Configuration
##

WALKER_CFG = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_ROOT_DIR}/usd/walker3d.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),

        # Visual material seems to cause problem in cloning the robot
        # visual_material=sim_utils.PreviewSurfaceCfg(
        #     diffuse_color=(240.0 / 256.0, 0.0 / 256.0, 0.0 / 256.0),
        #     emissive_color=(209.0 / 256.0, 42.0 / 256.0, 148.0 / 256.0),
        #     metallic=0.2,
        #     roughness=0.1,
        # ),  
        copy_from_source=False,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.1, 0.0, 1.5),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for the Allsteps Humanoid robot."""
