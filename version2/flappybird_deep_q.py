import numpy as np

from version2.deep_q import DeepQLearning
from version2.dqn import DQN
from version2.flappybird import FlappyBird

if __name__ == '__main__':
    import keras as ks

    nn = ks.models.Sequential()
    nn.add(ks.layers.Dense(32, activation='sigmoid', input_shape=(8,)))
    nn.add(ks.layers.Dense(32, activation='sigmoid'))
    nn.add(ks.layers.Dense(2, activation='linear'))

    nn.compile(optimizer=ks.optimizers.Adam(lr=0.001),
               loss='mse')

    width, height = size = (256, 256)
    _e = FlappyBird(size)
    _out_map = [False, True]

    def normalize_state(s):
        o = np.zeros(shape=(1, 8))
        o[0, 0] = s.observation['player_y'] / height
        o[0, 1] = s.observation['player_vel']
        o[0, 2] = s.observation['next_pipe_dist_to_player'] / width
        o[0, 3] = s.observation['next_pipe_top_y'] / (height / 2)
        o[0, 4] = s.observation['next_pipe_bottom_y'] / (height / 2)

        o[0, 5] = s.observation['next_next_pipe_dist_to_player'] / width
        o[0, 6] = s.observation['next_next_pipe_top_y'] / (height / 2)
        o[0, 7] = s.observation['next_next_pipe_bottom_y'] / (height / 2)
        return o


    dqn = DQN(nn, _out_map, normalize_state)

    dql = DeepQLearning(_e, dqn)

    q = dql.policy_eval()
