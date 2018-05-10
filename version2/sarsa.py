import random

from version2.util import ZeroDict

N_0 = 100
GAMMA = 1
NUM_ITER = 1000000


class SarsaLambda:

    def __init__(self, game: callable, lam: float = 0.2):
        assert 0 <= lam <= 1
        self.Q, self.N, self.E = ZeroDict(), ZeroDict(), ZeroDict()
        self.game = game
        self.lam = lam

    def policy_eval(self) -> dict:
        N, Q, E = self.N, self.Q, self.E
        for _ in range(NUM_ITER):
            E.clear()
            s = self.game()
            a = random.choice(s.actions())

            N[s] += 1
            N[s, a] += 1

            while not s.is_terminal():
                s_p, r = s.step(a, inplace=False)

                N[s_p] += 1
                epsilon = N_0 / (N_0 + N[s_p])
                
                actions = s_p.actions()
                if len(actions) == 0:
                    a_p = None
                else:
                    if random.random() < epsilon:
                        a_p = random.choice(actions)
                    else:
                        a_p = max(actions, key=lambda _a: Q[s_p, _a])

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

    from version2.game import Easy21

    procedure = SarsaLambda(Easy21, lam=0.2)

    q = procedure.policy_eval()

    vs = np.zeros(shape=(21, 10))

    for (key, _), v in q.items():
        vs[key.p_sum - 1, key.d_sum - 1] = max([q[key, True], q[key, False]])

    plt.imshow(vs)

    plt.show()
