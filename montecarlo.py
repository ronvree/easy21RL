import game
import numpy.random as rnd
import matplotlib.pyplot as plt

from util import ZeroDict

N_0 = 100


def monte_carlo_policy_eval():
    N, Q = ZeroDict(), ZeroDict()
    for _ in range(100000):
        # Run episode
        episode, reward = [], 0
        S = _, _, terminal = game.draw_init_state()
        while not terminal:

            epsilon = N_0 / (N_0 + N[S])

            if rnd.choice([True, False], p=[epsilon, 1 - epsilon]):
                a = rnd.choice([True, False])
            else:
                a = False if Q[S, False] > Q[S, True] else True

            episode.append([S, a])

            S, reward = game.step(S, a)
            _, _, terminal = S

        # Append reward to episode
        for e in episode:
            e.append(reward)

        # Update values
        for s, a, r in episode:
            N[s, a] += 1
            N[s] += 1
            Q[s, a] += (1 / N[s, a]) * (r - Q[s, a])

    return Q


if __name__ == '__main__':
    import numpy as np

    q = monte_carlo_policy_eval()

    vs = np.zeros(shape=(10, 21))

    for d_sum in range(10):
        for p_sum in range(21):
            vs[d_sum, p_sum] = max([q.setdefault(((p_sum + 1, d_sum + 1, False), a), 0) for a in [True, False]])

    plt.imshow(vs)

    plt.show()


