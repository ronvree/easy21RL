import random

import numpy as np
import pygame
from ple import PLE
from ple.games import Pixelcopter

from version2.core import State, DiscreteActionEnvironment

pygame.init()


class PixelCopterState(State):
    """
        A PixelCopter environment state
    """

    def __init__(self, observation):
        """
        Create a new PixelCopter state
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
        c = PixelCopterState(self.observation)
        c.terminal = self.terminal
        return c

    def set_observation(self, observation):
        self.observation = observation


class PixelCopter(DiscreteActionEnvironment):
    """
        PixelCopter environment class
    """

    def __init__(self, size: tuple = (48, 48)):
        self.width, self.height = size
        self.game = Pixelcopter(width=self.width, height=self.height)
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
        self.state = PixelCopterState(self.game.getGameState())

        self.i += 1

        return self.state.copy()

    def action_space(self, state) -> set:
        """
        Get the actions that can be performed on the specified state
        :param state: The state on which an action should be performed
        :return: A set of actions
        """
        return {False, True}  # Actions independent of state


class VisualPixelCopter(DiscreteActionEnvironment):
    """
        PixelCopter environment class giving screen captures as observation
    """

    def __init__(self, size: tuple = (48, 48)):
        self.width, self.height = size
        self.game = Pixelcopter(width=self.width, height=self.height)
        self.game.screen = pygame.display.set_mode(self.game.getScreenDims(), 0, 32)
        self.game.clock = pygame.time.Clock()
        self.game.rng = np.random.RandomState(24)

        self.game.rewards['loss'] = -1
        self.game.rewards['win'] = 1

        self.ple = PLE(self.game)
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

        s.set_observation(self.ple.getScreenGrayscale())
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
        self.state = PixelCopterState(self.ple.getScreenGrayscale())

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
    from version2.linear_func_approx import SarsaLambda

    width, height = size = 256, 256
    e = PixelCopter(size)

    _s = e.reset()
    while not _s.is_terminal():
        _s, r = e.step(np.random.choice([True, False], p=[0.1, 0.9]))
        print(r)
        if _s.is_terminal():
            _s = e.reset()
        time.sleep(0.1)

    # h_seg = [(0 * height, (1 / 4) * height),
    #          ((1 / 4) * height, (1 / 2) * height),
    #          ((1 / 2) * height, (3 / 4) * height),
    #          ((3 / 4) * height, 1 * height)
    #          ]
    #
    # d_seg = [(0, height / 10), (height / 10, height / 4), (height / 4, height / 2)]
    #
    # w_seg = [(0 * width, (1 / 4) * width),
    #          ((1 / 4) * width, (1 / 2) * width),
    #          ((1 / 2) * width, (3 / 4) * width),
    #          ((3 / 4) * width, 1 * width)
    #          ]
    #
    # def reduce_state(s, a):
    #     # o = np.zeros(2 * 4 * 4 * 4 * 4 * 4 * 4)
    #     o = np.zeros(2 * 4 * 4 * 4 * 2 * 2 * 2)
    #     i = 0
    #     for _a in e.action_space(s):
    #         for h_l, h_r in h_seg:
    #             o1 = s.observation['player_y'] / height
    #             for d1_l, d1_r in d_seg:
    #                 o2 = s.observation['player_dist_to_ceil'] / height
    #                 for d2_l, d2_r in d_seg:
    #                     o3 = s.observation['player_dist_to_floor'] / height
    #                     for b1_l, b1_r in h_seg[:2]:
    #                         o4 = s.observation['next_gate_block_top'] / height
    #                         for b2_l, b2_r in h_seg[:2]:
    #                             o5 = s.observation['next_gate_block_bottom'] / height
    #                             for w_l, w_r in w_seg[:2]:
    #                                 o6 = s.observation['next_gate_dist_to_player'] / width
    #
    #                                 o[i] = 1 if a == _a and \
    #                                             h_l < o1 <= h_r and \
    #                                             d1_l < o2 <= d1_r and \
    #                                             d2_l < o3 <= d2_r and \
    #                                             b1_l < o4 <= b1_l and \
    #                                             b2_l < o5 <= b2_r and \
    #                                             w_l < o6 <= w_r \
    #                                     else 0
    #                                 i += 1
    #     return o
    #
    # sl = SarsaLambda(e, features=reduce_state, weights=np.zeros(128 * 2 * 2 * 2)).policy_eval()
