from collections import defaultdict

from version5.agent import Agent
from version5.core import FiniteActionEnvironment
from version5.policy import EpsilonGreedyPolicy
from version5.q_table import QTable


class MonteCarlo(Agent):
    """
        Monte Carlo Agent implementation
    """

    def __init__(self, env: FiniteActionEnvironment, gamma: float = 1.0):
        """
        Create a new MonteCarlo Agent
        :param env: The environment the agent will learn from
        :param gamma: Reward discount factor
        """
        super().__init__(env)
        self.q_table = QTable()
        self.visit_count = defaultdict(int)
        self.policy = self.q_table.derive_policy(EpsilonGreedyPolicy,
                                                 env.valid_actions_from,
                                                 epsilon=self.epsilon)
        self.gamma = gamma

    def learn(self, num_iter=100000) -> EpsilonGreedyPolicy:
        """
        Learn a policy from the environment
        :param num_iter: The number of iterations the algorithm should run
        :return: the derived policy
        """
        Q, N, pi = self.q_table, self.visit_count, self.policy
        for _ in range(num_iter):
            s = self.env.reset()
            e, r = [], 0
            while not s.is_terminal():                          # Execute an episode
                a = pi.sample(s)
                e += [[s, a]]
                s, r = self.env.step(a)
                e[-1] += [r]
            
            for i, (s, a, r) in enumerate(reversed(e)):         # Reverse rewards so G can be computed efficiently
                g = r if i == 0 else g * self.gamma + r
                N[s, a] += 1
                N[s] += 1
                Q[s, a] += (1 / N[s, a]) * (g - Q[s, a])
        return pi

    def epsilon(self, s):
        N_0, N = 100, self.visit_count
        return N_0 / (N_0 + N[s])


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from version5.environments.easy21 import Easy21

    _env = Easy21()

    procedure = MonteCarlo(_env)

    q = procedure.learn(num_iter=1000000)

    table = procedure.q_table

    print(table)

    vs = np.zeros(shape=(21, 10))

    for (state, action), value in table.items():
        vs[state.p_sum - 1, state.d_sum - 1] = max([table[state, a] for a in _env.valid_actions()])

    plt.imshow(vs)
    plt.show()
