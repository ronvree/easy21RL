from collections import defaultdict

from version2.easy21 import Easy21
from version2.core import Environment
from version2.policy import Policy

N_0 = 100
GAMMA = 1
NUM_ITER = 1000000


class SarsaLambda:

    def __init__(self, env: Environment, lam: float = 0.2):
        assert 0 <= lam <= 1
        self.Q, self.N, self.E = Policy(), defaultdict(int), defaultdict(int)
        self.env = env
        self.lam = lam

    def policy_eval(self) -> Policy:
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

                try:
                    a_p = Q.sample_epsilon_greedy(s_p, epsilon=N_0 / (N_0 + N[s_p]))
                except KeyError:
                    a_p = self.env.sample_action()

                E[s, a] += 1
                N[s_p, a_p] += 1

                delta = r + GAMMA * Q[s_p, a_p] - Q[s, a]

                for k in E.keys():
                    Q[k] += (1 / N[k]) * delta * E[k]
                    E[k] *= GAMMA * self.lam

                s, a = s_p, a_p
        return Q


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    procedure = SarsaLambda(Easy21(), lam=0.2)

    q = procedure.policy_eval()

    vs = np.zeros(shape=(21, 10))

    for (key, _), v in q.items():
        vs[key.p_sum - 1, key.d_sum - 1] = max([q[key, True], q[key, False]])

    plt.imshow(vs)
    plt.show()
