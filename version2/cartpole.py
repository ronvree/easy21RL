import gym
from version2.core import State, DiscreteActionEnvironment

"""
    Environment wrapper for OpenAI Gym's CartPole
"""


class CartPoleState(State):
    """
        A CartPole environment state
    """

    def __init__(self, observation):
        """
        Create a new CartPole state
        :param observation: Environment observation to be stored in this state
        """
        super().__init__()
        self.observation = observation

    def __str__(self) -> str:
        """
        :return: A string representation of this state
        """
        return str(self.observation)

    def copy(self):
        """
        :return: A copy of this state
        """
        c = CartPoleState(self.observation)
        c.terminal = self.terminal
        return c


class CartPole(DiscreteActionEnvironment):
    """
        CartPole environment class
    """

    def __init__(self, render=False):
        self.env = gym.make('CartPole-v1')
        self.state = self.reset()
        self.render = render

    def sample_action(self):
        """
        :return: A random sample from the action space
        """
        return self.env.action_space.sample()

    def step(self, action, update=True) -> tuple:
        """
        Perform an action on the current environment state
        :param action: The action to be performed
        :param update: A boolean indicating whether the change in the environment should be stored
        :return: A two-tuple of (observation, reward)
        """
        if self.state.is_terminal():
            raise Exception('Cannot perform action on terminal state!')
        s = self.state if update else self.state.copy()
        if self.render:
            self.env.render()
        s.observation, reward, s.terminal, info = self.env.step(action)

        return s.copy() if update else s, reward

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial observation
        """
        self.state = CartPoleState(self.env.reset())
        return self.state.copy()

    def action_space(self, state) -> set:
        """
        Get the actions that can be performed on the specified state
        :param state: The state on which an action should be performed
        :return: A set of actions
        """
        return {0, 1}  # Actions independent of state


if __name__ == '__main__':
    import numpy as np
    from version2.linear_func_approx import SarsaLambda

    e = CartPole(render=True)

    feature_vector = lambda x, y: np.zeros(4)

    SarsaLambda(e, features=feature_vector, weights=np.zeros(4)).policy_eval()
