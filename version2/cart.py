import gym
from version2.core import Environment, State


class CartPoleState(State):

    def __init__(self, observation):
        super().__init__()
        self.observation = observation

    def copy(self):
        c = CartPoleState(self.observation)
        c.terminal = self.terminal
        return c


class CartPole(Environment):

    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.state = None
        self.reset()

    def sample_action(self):
        return self.env.action_space.sample()

    def step(self, action, update=True, render=True) -> tuple:
        if self.state.is_terminal():
            raise Exception('Cannot perform action on terminal state!')
        s = self.state if update else self.state.copy()
        if render:
            self.env.render()
        print(action)
        s.observation, reward, s.terminal, info = self.env.step(action)
        print(s.observation, reward)

        return s.copy() if update else s, reward

    def reset(self):
        self.env.reset()
        self.state = CartPoleState(None)
        return self.state.copy()


if __name__ == '__main__':
    from version2.sarsa import SarsaLambda

    e = CartPole()

    SarsaLambda(e).policy_eval()

    # e.step(e.action_space().sample())