import numpy as np

from version2.deep_q import DeepQLearning
from version2.dqn import DQN
from version2.pixelcopter import VisualPixelCopter

if __name__ == '__main__':
    import keras as ks

    width, height = size = (32, 32)
    _e = VisualPixelCopter(size)
    _out_map = [False, True]

    nn = ks.models.Sequential()
    nn.add(ks.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid', input_shape=size + (1,)))
    nn.add(ks.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid'))
    nn.add(ks.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid'))
    nn.add(ks.layers.Flatten())
    nn.add(ks.layers.Dense(units=16, activation='sigmoid'))
    nn.add(ks.layers.Dense(units=2,  activation='linear'))

    nn.compile(optimizer=ks.optimizers.Adam(lr=0.0001),
               loss='mse')

    print(nn.summary())

    def normalize_state(s):
        return np.reshape(s.observation / 256, newshape=(1,) + size + (1,))


    dqn = DQN(nn, _out_map, normalize_state)

    dql = DeepQLearning(_e, dqn)

    q = dql.policy_eval()
