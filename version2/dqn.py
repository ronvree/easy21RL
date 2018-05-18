import keras as ks
import numpy as np


class DQN:
    """
        Deep Q-Network class that wraps around a suitable Keras model
    """

    def __init__(self, model: ks.Model, out_map: list, feature_ex: callable=lambda x: x):
        """
        Create a new Deep Q-Network
        :param model: Keras model to be used in the network
        :param out_map: A list mapping the model's output to actions (by index)
        :param feature_ex: Function that is applied on a state to transform it into suitable input for the model
        """
        assert (None, len(out_map)) == model.output_shape  # Make sure all outputs can be mapped to actions
        self.model = model
        self.phi = feature_ex
        self.out_map = out_map

    def predict(self, s, a):
        """
        Predict a Q value for the state action pair
        :param s: The state parameter
        :param a: The action parameter
        :return: Q(s, a)
        """
        out = self.model.predict([self.phi(s)])[0]  # Predict Q for the single transformed state and all actions
        return out[self.out_map.index(a)]           # Return the relevant Q value

    def pi(self, s):
        """
        Predict the Q values for all actions that can be performed from a state
        :param s: The state parameter
        :return: A dictionary mapping all actions to their expected Q value
        """
        pi = dict()
        for i, v in enumerate(self.model.predict([self.phi(s)])[0]):  # Perform a forward pass in the network
            pi[self.out_map[i]] = v                                   # Map actions to their predicted Q-value
        return pi

    def fit_on_samples(self, samples, gamma: float):
        """
        Train the network on a minibatch of samples
        :param samples: A list of four-tuples (State, Action, Reward, Next State)
        :param gamma: Reward discount factor
        :return:
        """
        for s, a, r, s_p in samples:                                             # Iterate through all samples
            phi_s = self.phi(s)                                                  # Prepare state for model input
            qs = self.model.predict([phi_s])[0]                                  # Compute model output
            if s_p.is_terminal():                                                # Get model target
                qs[self.out_map.index(a)] = r
            else:
                qs[self.out_map.index(a)] = r + gamma * max(qs)
            self.model.fit(phi_s, np.reshape(qs, newshape=(1, len(self.out_map))), epochs=1)  # Train model on target


if __name__ == '__main__':
    import numpy as np

    _model = ks.models.Sequential()
    _model.add(ks.layers.Dense(64, activation='sigmoid', input_shape=(16,)))
    _model.add(ks.layers.Dense(2, activation='sigmoid'))

    _model.compile(optimizer='sgd', loss='mse')

    _out_map = [False, True]

    _feature_ex = lambda x: x / 10

    dqn = DQN(_model, _out_map, _feature_ex)

    _s = np.random.random(size=(1, 16))

    print(dqn.pi(_s))
    print(dqn.predict(_s, False))
