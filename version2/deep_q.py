from collections import deque

import numpy as np

from version2.core import DiscreteActionEnvironment
from version2.dqn import DQN

NUM_EPISODES = 100000  # Number of times the environment is run
MINIBATCH_SIZE = 32    # Size of batches used to train the network
GAMMA = 0.99           # Reward discount factor


class DeepQLearning:
    """
        Deep Q Learning algorithm implementation
    """

    def __init__(self, environment: DiscreteActionEnvironment, model: DQN):
        """
        Create a new Deep Q Learning procedure
        :param environment:  The environment the algorithm is subjected to
        :param model:  The neural network used in this procedure
        """
        self.env = environment
        self.model = model
        self.replay_memory = deque(maxlen=3000)

    def policy_eval(self):
        """
        Train the Q-Network
        :return: the estimated Q function
        """
        Q = self.model
        for e in range(NUM_EPISODES):
            s = self.env.reset()                                        # Initialize the environment
            while not s.is_terminal():                                  # Repeat until environment is terminal:
                a = self.sample_derived_policy(s, epsilon=0.05)         # - Epsilon-greedily pick an action
                s_p, r = self.env.step(a)                               # - Perform the action, obtain feedback
                self.add_to_replay_memory(s, a, r, s_p)                 # - Store result as sample to be trained on
                Q.fit_on_samples(self.sample_minibatch())               # - Train the model on a random batch of samples
                s = s_p                                                 # - Continue to next state

        return lambda s_, a_: Q.predict(s_, a_)                         # Return the estimated Q function

    def sample_derived_policy(self, s, epsilon: float=0):
        """
        Epsilon-greedy sample an action from a derived policy from the current Q-network.
        :param s: The state from which an action should be performed
        :param epsilon: The probability of picking a random action
        :return: The sampled action
        """
        assert 0 <= epsilon <= 1                                # Ensure epsilon is a valid probability
        if np.random.random() < epsilon:                        # With probability epsilon:
            return self.env.sample_action()                     # - Return a random action
        else:                                                   # Otherwise:
            pi = self.model.pi(s)                               # - Greedily sample an action
            return max(self.env.action_space(s), key=pi.get)    # - Return the action for which Q is maximized

    def sample_minibatch(self) -> list:
        """
        Get a random minibatch of samples from current replay memory
        :return: A list of samples
        """
        ixs = np.random.choice(range(len(self.replay_memory)), size=MINIBATCH_SIZE)
        return [self.replay_memory[i] for i in ixs]

    def add_to_replay_memory(self, s, a, r, sp):
        """
        Add one sample to the replay memory
        :param s: State
        :param a: Action performed on that state
        :param r: Reward obtained from performing the action
        :param sp: Resulting state
        """
        self.replay_memory.append((s, a, r, sp))


if __name__ == '__main__':
    from version2.cartpole import CartPole
    import keras as ks

    nn = ks.models.Sequential()
    nn.add(ks.layers.Dense(32, activation='sigmoid', input_shape=(4,)))
    nn.add(ks.layers.Dense(32, activation='sigmoid'))
    nn.add(ks.layers.Dense(2, activation='linear'))

    nn.compile(optimizer=ks.optimizers.Adam(lr=0.001),
               loss='mse')

    _e = CartPole(render=True)
    _out_map = [0, 1]

    dqn = DQN(nn, _out_map, lambda x: np.reshape(x.observation, newshape=(1, 4)))

    dql = DeepQLearning(_e, dqn)

    q = dql.policy_eval()
