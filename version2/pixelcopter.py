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


class PixelCopter(DiscreteActionEnvironment):
    """
        PixelCopter environment class
    """

    def __init__(self, size: tuple=(48, 48)):
        self.width, self.height = size
        self.game = Pixelcopter(width=self.width, height=self.height)
        self.game.screen = pygame.display.set_mode(self.game.getScreenDims(), 0, 32)
        self.game.clock = pygame.time.Clock()
        self.game.rng = np.random.RandomState(24)
        self.ple = PLE(self.game)
        # self.game.init()
        self.ple.init()

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

        s.observation = self.game.getGameState()
        s.terminal = self.ple.game_over()

        pygame.display.update()

        return s.copy() if update else s, reward

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial observation
        """
        print(self.ple.score())
        self.ple.reset_game()
        self.state = PixelCopterState(self.game.getGameState())
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
    from version2.linear_func_approx import SarsaLambda

    width, height = size = 256, 256
    e = PixelCopter(size)

    def reduce_state(s, a):
        o = np.zeros(2 * 4 * 4 * 4)

        h_seg = [(0 * height, (1 / 4) * height),
                 ((1 / 4) * height, (1 / 2) * height),
                 ((1 / 2) * height, (3 / 4) * height),
                 ((3 / 4) * height, 1 * height)]

        d_seg = [(0, height / 10), (height / 10, height / 4), (height / 4, height / 2)]

        i = 0
        for _a in e.action_space(s):
            for h_l, h_r in h_seg:
                o1 = s.observation['player_y'] / height
                for d1_l, d1_r in d_seg:
                    o2 = s.observation['player_dist_to_ceil']
                    for d2_l, d2_r in d_seg:
                        o3 = s.observation['player_dist_to_floor']
                        o[i] = 1 if a == _a and h_l < o1 <= h_r and d1_l < o2 <= d1_r and d2_l < o3 <= d2_r else 0
                        i += 1
        return o


    # def reduce_state(s, a):
    #     s = s.observation
    #     o = np.zeros(4)
    #     print(s)
    #     o[0] = int(a)
    #     o[1] = int(s['player_y'] / 10)
    #     o[2] = int(s['player_dist_to_ceil'] / 10)
    #     o[3] = int(s['player_dist_to_floor'] / 10)
    #     # o[4] = int(s['next_gate_block_top'] / 10)
    #     # o[5] = int(s['next_gate_block_bottom'] / 10)
    #     # o[6] = int(s['player_vel'])
    #     # o[7] = int(s['next_gate_dist_to_player'] / 10)
    #     return o

    sl = SarsaLambda(e, features=reduce_state, weights=np.zeros(128)).policy_eval()
