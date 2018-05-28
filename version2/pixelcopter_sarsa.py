from version2.pixelcopter import PixelCopter, PixelCopterState


SIZE = WIDTH, HEIGHT = 256, 256

# h_seg = [((0 / 1) * HEIGHT, (1 / 8) * HEIGHT),
#          ((1 / 8) * HEIGHT, (1 / 4) * HEIGHT),
#          ((1 / 4) * HEIGHT, (1 / 3) * HEIGHT),
#          ((1 / 3) * HEIGHT, (1 / 2) * HEIGHT),
#          ((1 / 2) * HEIGHT, (1 / 1) * HEIGHT),
#          ]
#
# d_seg = [((0 / 1) * HEIGHT, (1 / 8) * HEIGHT),
#          ((1 / 8) * HEIGHT, (1 / 4) * HEIGHT),
#          ((1 / 4) * HEIGHT, (1 / 3) * HEIGHT),
#          ((1 / 3) * HEIGHT, (1 / 2) * HEIGHT),
#          ]
#
# w_seg = [(0 * WIDTH, (1 / 4) * WIDTH),
#          ((1 / 4) * WIDTH, (1 / 2) * WIDTH),
#          ((1 / 2) * WIDTH, (3 / 4) * WIDTH),
#          ((3 / 4) * WIDTH, 1 * WIDTH)
#          ]


def create_intervals(length, n):
    return [(i * (length / n), (i + 1) * (length / n)) for i in range(n)]


h_seg = create_intervals(HEIGHT, 6)
d_seg = create_intervals(HEIGHT, 12)[:6]
w_seg = create_intervals(WIDTH, 6)


def get_interval_index(value, intervals):
    for i, (lower, upper) in enumerate(intervals):
        if lower <= value <= upper:
            return i
    return -1


class DiscretizedPixelCopterState(PixelCopterState):

    def __init__(self):
        super().__init__(None)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, DiscretizedPixelCopterState) and o.observation == self.observation

    def __hash__(self) -> int:
        h = 2 if self.terminal else 0
        h += 3 * int(self.observation['player_y'])
        h += 5 * int(self.observation['player_dist_to_ceil'])
        h += 7 * int(self.observation['player_dist_to_floor'])
        h += 11 * int(self.observation['player_vel'])
        h += 13 * int(self.observation['next_gate_dist_to_player'])
        h += 17 * int(self.observation['next_gate_block_top'])
        h += 19 * int(self.observation['next_gate_block_bottom'])
        return h

    def set_observation(self, observation):
        self.observation = self._discretize(observation)

    def copy(self):
        """
        :return: A copy of this state
        """
        c = DiscretizedPixelCopterState()
        c.observation = self.observation
        c.terminal = self.terminal
        return c

    @staticmethod
    def _discretize(observation):
        return {
            'player_y': get_interval_index(observation['player_y'], h_seg),
            'player_dist_to_ceil': get_interval_index(observation['player_dist_to_ceil'], d_seg),
            'player_dist_to_floor': get_interval_index(observation['player_dist_to_floor'], d_seg),
            # 'next_gate_dist_to_player': get_interval_index(observation['next_gate_dist_to_player'], w_seg),
            'next_gate_dist_to_player': 0,
            # 'next_gate_block_top': get_interval_index(observation['next_gate_block_top'], d_seg),
            'next_gate_block_top': 0,
            # 'next_gate_block_bottom': get_interval_index(observation['next_gate_block_bottom'], d_seg),
            'next_gate_block_bottom': 0,
            'player_vel': int(observation['player_vel'] / 3),
            # 'player_vel': 0,
        }


class DiscretizedPixelCopter(PixelCopter):

    def __init__(self):
        super().__init__(size=SIZE)

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial observation
        """
        self.ple.reset_game()
        self.state = DiscretizedPixelCopterState()
        self.state.set_observation(self.game.getGameState())

        self.i += 1
        return self.state.copy()


if __name__ == '__main__':
    from version2.sarsa import SarsaLambda

    e = DiscretizedPixelCopter()

    sl = SarsaLambda(e, lam=0.4)

    sl.policy_eval()
