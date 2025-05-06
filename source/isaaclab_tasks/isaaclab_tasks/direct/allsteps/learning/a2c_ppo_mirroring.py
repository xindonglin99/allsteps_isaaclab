from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common import datasets
from isaaclab_tasks.direct.allsteps.allsteps_env import get_symmetric_states_rl_games

class A2CAgentSymmetry(A2CAgent):
    """
    A2CAgent with symmetry loss for data augmentation.
    """
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.symmetry = params['config'].get('symmetry', False)

        if self.symmetry:
            self.batch_size *= 2
            self.batch_size_envs *= 2
            self.num_minibatches = self.batch_size // self.minibatch_size
            self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)


    def play_steps(self):
        normal_batch =  super().play_steps()

        if self.symmetry:
            # normal_batch['played_frames'] = 2 * normal_batch['played_frames']
            normal_batch['returns'] = normal_batch['returns'].repeat(2, 1)
            normal_batch['dones'] = normal_batch['dones'].repeat(2)
            normal_batch['values'] = normal_batch['values'].repeat(2, 1)
            normal_batch['sigmas'] = normal_batch['sigmas'].repeat(2, 1)
            # normal_batch['mus'] = normal_batch['mus'].repeat(2, 1)
            normal_batch['neglogpacs'] = normal_batch['neglogpacs'].repeat(2)

            new_obs, new_actions, new_mus = get_symmetric_states_rl_games(
                normal_batch['obses'], normal_batch['actions'], self.vec_env.env, False, normal_batch['mus']
            )

            normal_batch['obses'] = new_obs
            normal_batch['actions'] = new_actions
            normal_batch['mus'] = new_mus

        return normal_batch


    