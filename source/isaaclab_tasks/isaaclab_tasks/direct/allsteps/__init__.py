"""
ALLSTEPS Humanoid environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Allsteps-v0",
    entry_point=f"{__name__}.allsteps_env:AllstepsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allsteps_env_cfg:AllstepsEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:rsl_rl_ppo_cfg:AllstepsPPORunnerCfg",
    },
)
