from collections import defaultdict

from version2.core import Environment
from version2.qtable import QTable

N_0 = 100
NUM_ITER = 100000


class MonteCarlo:

    def __init__(self, env: Environment):
        self.Q, self.N = QTable(), defaultdict(int)
        self.env = env

    def policy_eval(self):
        Q, N = self.Q, self.N
        for _ in range(NUM_ITER):
            s = self.env.reset()
            episode, r = [], 0
            while not s.is_terminal():
                try:
                    a = Q.sample_epsilon_greedy(s, epsilon=N_0 / (N_0 + N[s]), action_space=self.env.sample_action)
                except KeyError:
                    a = self.env.sample_action()
                episode.append([s, a])
                s, r = self.env.step(a)

            for e in episode:
                e.append(r)

            for s, a, r in episode:
                N[s, a] += 1
                N[s] += 1
                Q[s, a] += (1 / N[s, a]) * (r - Q[s, a])

        return Q


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from version2.easy21 import Easy21

    procedure = MonteCarlo(Easy21())

    q = procedure.policy_eval()

    vs = np.zeros(shape=(21, 10))

    for (state, action), value in q.items():
        if (state, not action) in q.keys():
            vs[state.p_sum - 1, state.d_sum - 1] = max([q[state, True], q[state, False]])
        else:
            vs[state.p_sum - 1, state.d_sum - 1] = q[state, action]

    plt.imshow(vs)

    plt.show()
