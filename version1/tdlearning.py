from version1 import game

import numpy.random as rnd

from version1.util import ZeroDict

N_0 = 100
gamma = 1


def sarsa_lambda_policy_eval(lam=0.2):
    Q, N = ZeroDict(), ZeroDict()
    for _ in range(100000):                                  # Repeat for each episode
        E = ZeroDict()

        S = _, _, terminal = game.draw_init_state()           # Initialize S and A
        A = rnd.choice([True, False])

        N[S] += 1
        N[S, A] += 1

        while not terminal:                                     # Repeat for each step of episode
            S_p, reward = game.step(S, A)                       # Take action A, observe R, S_p

            N[S_p] += 1
            epsilon = N_0 / (N_0 + N[S_p])

            if rnd.choice([True, False], p=[epsilon, 1 - epsilon]):     # Sample A_p from Q using epsilon-greedy
                A_p = rnd.choice([True, False])
            else:
                A_p = False if Q[S, False] > Q[S, True] else True

            E[S, A] += 1
            N[S_p, A_p] += 1

            delta = reward + gamma * Q[S_p, A_p] - Q[S, A]

            for k in E.keys():
                Q[k] += (1 / N[k]) * delta * E[k]
                E[k] *= gamma * lam

            S, A = S_p, A_p
            _, _, terminal = S
    return Q


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    q = sarsa_lambda_policy_eval(lam=0.2)

    print(q)

    vs = np.zeros(shape=(10, 21))

    for d_sum in range(10):
        for p_sum in range(21):
            vs[d_sum, p_sum] = max([q.setdefault(((p_sum + 1, d_sum + 1, False), a), 0) for a in [True, False]])

    plt.imshow(vs)

    plt.show()

