# import gym
# from version2.core import Environment, State
#
#
# class GymState(State):
#
#     def __init__(self, observation):
#         super().__init__()
#         self.observation = observation
#
#     def __str__(self) -> str:
#         return str(self.observation)
#
#     def copy(self):
#         c = GymState(self.observation)
#         c.terminal = self.terminal
#         return c
#
#
# class GymEnvironment(Environment):
#
#     def __init__(self, name, render=False):
#         self.env = gym.make(name)
#         self.state = None
#         self.reset()
#         self.render = render
#
#     def sample_action(self):
#         return self.env.action_space.sample()
#
#     def step(self, action, update=True) -> tuple:
#         if self.state.is_terminal():
#             raise Exception('Cannot perform action on terminal state!')
#         s = self.state if update else self.state.copy()
#         if self.render:
#             self.env.render()
#         # print(action)
#         s.observation, reward, s.terminal, info = self.env.step(action)
#         # print(s.observation, reward)
#
#         return s.copy() if update else s, reward
#
#     def reset(self):
#         self.env.reset()
#         self.state = GymState(None)
#         return self.state.copy()
#
#
# if __name__ == '__main__':
#     import numpy as np
#     from version2.linear_func_approx import SarsaLambda
#
#     # e = Easy21()
#     e = GymEnvironment('CartPole-v0', render=True)
#     # e = GymEnvironment('Blackjack-v0', render=False)
#
#     def feature_vector(o):
#         fs = np.zeros(4)
#         for i in range(len(fs)):
#             fs[i] = int(o[i])
#         return fs.T
#
#     SarsaLambda(e, features=feature_vector, weights=np.zeros(4))
#
#     # e.step(e.action_space().sample())