# from rsl_rl.algorithms.ppo import PPO


# class CurriculumPPO(PPO):
#     """
#     CurriculumPPO is a subclass of the PPO algorithm that implements a curriculum learning approach.
#     It allows for the training of reinforcement learning agents using a curriculum of tasks.
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.curriculum = kwargs.get("curriculum", None)

#     def train(self):
#         if self.curriculum:
#             self.curriculum.start_curriculum()
#         super().train()
#         if self.curriculum:
#             self.curriculum.end_curriculum()