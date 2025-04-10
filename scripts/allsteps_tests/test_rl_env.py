# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to run the RL environment for the cartpole balancing task."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.device = "cuda:0"

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from isaaclab_tasks.direct.allsteps.allsteps_env import AllstepsEnv

from isaaclab_tasks.direct.allsteps.allsteps_env_cfg import AllstepsEnvCfg


def main():
    """Main function."""
    # create environment configuration
    env_cfg = AllstepsEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup RL environment
    env = AllstepsEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 1800 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            # joint_efforts = 2 * torch.rand(env.action_space.shape) - 1
            joint_efforts_zero = torch.zeros(env.action_space.shape)
            # paths = env._generate_hold_path([0, 1])
            # if count % 1800 == 0:
            #     for path in paths:
            #         env._grab(path)
            # if count % 1800 == 900:
            #     for path in paths:
            #         env._release(path)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts_zero)
            # print current orientation of pole
            # print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
