import gym
from version2.core import State, DiscreteActionEnvironment


class CartPoleState(State):

    def __init__(self, observation):
        super().__init__()
        self.observation = observation

    def __str__(self) -> str:
        return str(self.observation)

    def copy(self):
        c = CartPoleState(self.observation)
        c.terminal = self.terminal
        return c


class CartPole(DiscreteActionEnvironment):

    def __init__(self, render=False):
        self.env = gym.make('CartPole-v1')
        self.state = self.reset()
        self.render = render

    def sample_action(self):
        return self.env.action_space.sample()

    def step(self, action, update=True) -> tuple:
        if self.state.is_terminal():
            raise Exception('Cannot perform action on terminal state!')
        s = self.state if update else self.state.copy()
        if self.render:
            self.env.render()
        s.observation, reward, s.terminal, info = self.env.step(action)

        return s.copy() if update else s, reward

    def reset(self):
        self.state = CartPoleState(self.env.reset())
        return self.state.copy()

    def action_space(self, state) -> set:
        return {0, 1}


if __name__ == '__main__':
    import numpy as np
    from version2.linear_func_approx import SarsaLambda

    e = CartPole(render=True)
    # e = GymEnvironment('Blackjack-v0', render=False)

    # def feature_vector(o, a):
    #     fs = np.zeros(8)
    #     for i in range(len(fs)):
    #         for _a in {0, 1}:
    #             fs[i] = int(o[i])
    #     return fs.T

    feature_vector = lambda x, y: np.zeros(4)

    SarsaLambda(e, features=feature_vector, weights=np.zeros(4)).policy_eval()

    # e.step(0)

    # print(e.env.action_space.sample())

    # e.step()
