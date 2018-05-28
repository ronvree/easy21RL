from version2.flappybird import FlappyBirdState, FlappyBird

SIZE = WIDTH, HEIGHT = 288, 512


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


class DiscretizedFlappyBirdState(FlappyBirdState):

    def __init__(self):
        super().__init__(None)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, DiscretizedFlappyBirdState) and o.observation == self.observation

    def __hash__(self) -> int:
        h = 2 if self.terminal else 0
        h += 3 * int(self.observation['player_y'])
        h += 5 * int(self.observation['player_vel'])
        h += 7 * int(self.observation['next_pipe_dist_to_player'])
        h += 11 * int(self.observation['next_pipe_top_y'])
        # h += 13 * int(self.observation['next_pipe_bottom_y'])
        h += 17 * int(self.observation['next_next_pipe_dist_to_player'])
        h += 19 * int(self.observation['next_next_pipe_top_y'])
        # h += 23 * int(self.observation['next_next_pipe_bottom_y'])
        return h

    def set_observation(self, observation):
        self.observation = self._discretize(observation)

    def copy(self):
        """
        :return: A copy of this state
        """
        c = DiscretizedFlappyBirdState()
        c.observation = self.observation
        c.terminal = self.terminal
        return c

    @staticmethod
    def _discretize(observation):
        return {
            'player_y': get_interval_index(observation['player_y'], h_seg),
            'player_vel': int(observation['player_vel'] / 4),
            'next_pipe_dist_to_player': get_interval_index(observation['next_pipe_dist_to_player'], w_seg),
            'next_pipe_top_y': get_interval_index(observation['next_pipe_top_y'], d_seg),
            # 'next_pipe_bottom_y': get_interval_index(observation['next_pipe_bottom_y'], d_seg),
            'next_next_pipe_dist_to_player': get_interval_index(observation['next_next_pipe_dist_to_player'], w_seg),
            'next_next_pipe_top_y': get_interval_index(observation['next_next_pipe_top_y'], d_seg),
            # 'next_next_pipe_bottom_y': get_interval_index(observation['next_next_pipe_bottom_y'], d_seg),
        }


class DiscretizedFlappyBird(FlappyBird):

    def __init__(self):
        super().__init__(size=SIZE)

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial observation
        """
        self.ple.reset_game()
        self.state = DiscretizedFlappyBirdState()
        self.state.set_observation(self.game.getGameState())

        self.i += 1
        return self.state.copy()


if __name__ == '__main__':
    from version2.sarsa import SarsaLambda

    e = DiscretizedFlappyBird()

    sl = SarsaLambda(e, lam=0.8)
    # sl.epsilon = lambda x: 0.00000000001

    sl.policy_eval()
