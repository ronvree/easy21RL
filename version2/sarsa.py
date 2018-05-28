import random
from collections import defaultdict

from version2.core import Environment
from version2.qtable import QTable

N_0 = 100
GAMMA = 1
NUM_ITER = 1000000000


class SarsaLambda:

    def __init__(self, env: Environment, lam: float=0.2):
        assert 0 <= lam <= 1
        self.Q, self.N, self.E = QTable(), defaultdict(int), defaultdict(int)
        self.env = env
        self.lam = lam

    def policy_eval(self) -> QTable:
        N, Q, E = self.N, self.Q, self.E
        for _ in range(NUM_ITER):
            E.clear()
            s = self.env.reset()
            a = self.env.sample_action()

            N[s] += 1
            N[s, a] += 1

            while not s.is_terminal():
                s_p, r = self.env.step(a)
                N[s_p] += 1

                a_p = self.sample_derived_policy(s_p, self.epsilon(s_p))

                E[s, a] += 1
                N[s_p, a_p] += 1

                delta = r + GAMMA * Q[s_p, a_p] - Q[s, a]

                for k in E.keys():
                    Q[k] += (1 / N[k]) * delta * E[k]
                    E[k] *= GAMMA * self.lam

                s, a = s_p, a_p
            # print(len(Q))
            # print(Q)
        return Q

    def sample_derived_policy(self, s, epsilon=0):
        assert 0 <= epsilon <= 1
        if random.random() < epsilon:
            return self.env.sample_action()
        else:
            try:
                a, v = self.Q.sample_greedy(s)
                if v < 0:
                    return self.env.sample_action()
                else:
                    return a
            except KeyError:
                return self.env.sample_action()

    def epsilon(self, s):
        return N_0 / (N_0 + self.N[s])


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from version2.easy21 import Easy21

    procedure = SarsaLambda(Easy21(), lam=0.2)

    q = procedure.policy_eval()

    print(q)

    vs = np.zeros(shape=(21, 10))

    for (state, action), value in q.items():
        if (state, not action) in q.keys():
            vs[state.p_sum - 1, state.d_sum - 1] = max([q[state, True], q[state, False]])
        else:
            vs[state.p_sum - 1, state.d_sum - 1] = q[state, action]

    plt.imshow(vs)
    plt.show()
