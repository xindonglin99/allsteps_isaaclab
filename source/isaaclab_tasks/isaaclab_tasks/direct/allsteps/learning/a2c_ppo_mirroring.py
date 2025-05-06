from rl_games.algos_torch.a2c_continuous import A2CAgent
from isaaclab_tasks.direct.allsteps.allsteps_env import get_symmetric_states

class A2CAgentSymmetry(A2CAgent):
    """
    A2CAgent with symmetry loss for data augmentation.
    """
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.symmetry = params['config'].get('symmetry', False)

    def play_steps(self):
        normal_batch =  super().play_steps()

        if self.symmetry:
            normal_batch['played_frames'] = 2 * normal_batch['played_frames']
            normal_batch['returns'] = normal_batch['returns'].repeat(2, 1)
            normal_batch['dones'] = normal_batch['dones'].repeat(2, 1)
            normal_batch['values'] = normal_batch['values'].repeat(2, 1)
            normal_batch['sigmas'] = normal_batch['sigmas'].repeat(2, 1)
            normal_batch['mus'] = normal_batch['mus'].repeat(2, 1)
            normal_batch['neglogpacs'] = normal_batch['neglogpacs'].repeat(2, 1)

            new_obs, new_actions = get_symmetric_states(
                normal_batch['obses'], normal_batch['actions'], self.vec_env.env, False
            )

            normal_batch['obses'] = new_obs
            normal_batch['actions'] = new_actions

        return normal_batch


    