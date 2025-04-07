## ALLSTEPS IsaacLab
**This is an implementation of the paper [ALLSTEPS](https://www.cs.ubc.ca/~van/papers/2020-allsteps/index.html) in IsaacLab framework.**
The code can be run after correctly installing all the dependencies on [IsaacLab website](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html), using the following command: 

Training:
```
cd allsteps_isaaclab
python scripts/reinforcement_learning/rl_games/train.py --task Allsteps-v0 --headless
```

Testing:
```
cd allsteps_isaaclab
python scripts/reinforcement_learning/rl_games/play.py --task Allsteps-v0 --num_envs 1
```
