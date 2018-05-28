import random

import numpy as np
import pygame
from ple import PLE
import ple.games

from version2.core import State, DiscreteActionEnvironment

pygame.init()


class WaterWorldState(State):
    """
        A WaterWorld environment state
    """

    def __init__(self, observation):
        """
        Create a new WaterWorld state
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
        c = WaterWorldState(self.observation)
        c.terminal = self.terminal
        return c

    def set_observation(self, observation):
        self.observation = observation


class WaterWorld(DiscreteActionEnvironment):
    """
        WaterWorld environment class
    """

    def __init__(self, size: tuple = (48, 48), num_creeps=5):
        self.width, self.height = size
        self.game = ple.games.WaterWorld(width=self.width, height=self.height, num_creeps=num_creeps)
        self.game.screen = pygame.display.set_mode(self.game.getScreenDims(), 0, 32)
        self.game.clock = pygame.time.Clock()
        self.game.rng = np.random.RandomState(24)

        self.ple = PLE(self.game)
        self.ple.init()

        self.i = 0

        self.actions = {'up', 'down', 'left', 'right'}

        self.actions = {
            "up": pygame.K_w,
            "left": pygame.K_a,
            "right": pygame.K_d,
            "down": pygame.K_s
        }

        self.state = self.reset()

    def sample_action(self):
        """
        :return: A random sample from the action space
        """
        return random.choice(self.actions.keys())

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

        reward = self.ple.act(self.actions[action])

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
        self.state = WaterWorldState(self.game.getGameState())

        self.i += 1

        return self.state.copy()

    def action_space(self, state) -> set:
        """
        Get the actions that can be performed on the specified state
        :param state: The state on which an action should be performed
        :return: A set of actions
        """
        return set(self.actions.keys())  # Actions independent of state


if __name__ == '__main__':
    import numpy as np
    import time

    width, height = size = 256, 256
    e = WaterWorld(size)

    _s = e.reset()
    while not _s.is_terminal():
        _s, r = e.step(np.random.choice([True, False], p=[0.1, 0.9]))
        print(r)
        if _s.is_terminal():
            _s = e.reset()
        time.sleep(0.1)
