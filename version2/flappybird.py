import random

import numpy as np
import pygame
from ple import PLE
# from ple.games import FlappyBird
import ple.games

from version2.core import State, DiscreteActionEnvironment

pygame.init()


class FlappyBirdState(State):
    """
        A FlappyBird environment state
    """

    def __init__(self, observation):
        """
        Create a new FlappyBird state
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
        c = FlappyBirdState(self.observation)
        c.terminal = self.terminal
        return c

    def set_observation(self, observation):
        self.observation = observation


class FlappyBird(DiscreteActionEnvironment):
    """
        FlappyBird environment class
    """

    def __init__(self, size: tuple = (48, 48)):
        self.width, self.height = size
        self.game = ple.games.FlappyBird(width=self.width, height=self.height)
        self.game.screen = pygame.display.set_mode(self.game.getScreenDims(), 0, 32)
        self.game.clock = pygame.time.Clock()
        self.game.rng = np.random.RandomState(24)

        self.game.rewards['loss'] = -1
        self.game.rewards['win'] = 1

        self.ple = PLE(self.game)
        # self.game.init()
        self.ple.init()

        self.i = 0

        self.state = self.reset()

    def sample_action(self):
        """
        :return: A random sample from the action space
        """
        return bool(random.getrandbits(1))

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

        if action:
            reward = self.ple.act(pygame.K_w)
        else:
            reward = self.ple.act(self.ple.NOOP)

        s.set_observation(self.game.getGameState())
        s.terminal = self.ple.game_over()

        # if self.i % 10 == 0:
        pygame.display.update()

        return s.copy() if update else s, reward

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial observation
        """
        self.ple.reset_game()
        self.state = FlappyBirdState(self.game.getGameState())

        self.i += 1

        return self.state.copy()

    def action_space(self, state) -> set:
        """
        Get the actions that can be performed on the specified state
        :param state: The state on which an action should be performed
        :return: A set of actions
        """
        return {False, True}  # Actions independent of state


if __name__ == '__main__':
    import numpy as np
    import time

    width, height = size = 256, 256
    e = FlappyBird(size)

    _s = e.reset()
    while not _s.is_terminal():
        _s, r = e.step(np.random.choice([True, False], p=[0.1, 0.9]))
        print(r)
        if _s.is_terminal():
            _s = e.reset()
        time.sleep(0.1)
