from numpy import random

from version2.util import ZeroDict

N_0 = 100
NUM_ITER = 1000000


class MonteCarlo:

    def __init__(self, game: callable):
        self.Q, self.N = ZeroDict(), ZeroDict()
        self.game = game

    def policy_eval(self):
        Q, N = self.Q, self.N
        for _ in range(NUM_ITER):
            s = self.game()
            episode, r = [], 0
            while not s.is_terminal():
                epsilon = N_0 / (N_0 + N[s])

                actions = s.actions()

                if random.random() < epsilon:
                    a = random.choice(actions)
                else:
                    a = max(actions, key=lambda _a: Q[s, _a])

                episode.append([s, a])

                s, r = s.step(a, inplace=False)

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

    from version2.game import Easy21

    procedure = MonteCarlo(Easy21)

    q = procedure.policy_eval()

    vs = np.zeros(shape=(21, 10))

    for (key, _), v in q.items():
        vs[key.p_sum - 1, key.d_sum - 1] = max([q[key, True], q[key, False]])

    plt.imshow(vs)

    plt.show()
