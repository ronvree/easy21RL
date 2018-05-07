import game

import numpy.random as rnd


N_0 = 100
gamma = 1


def sarsa_lambda_policy_eval(lam=0.2):
    Q, N = dict(), dict()
    for _ in range(100000):                                  # Repeat for each episode
        terminal, E = False, dict()

        S = game.draw_init_values() + (terminal,)              # Initialize S and A
        A = rnd.choice([True, False])

        N.setdefault(S, 1)
        N.setdefault((S, A), 1)
        Q.setdefault((S, A), 0)

        while not terminal:                                     # Repeat for each step of episode
            S_p, reward = game.step(S, A)                       # Take action A, observe R, S_p

            N.setdefault(S_p, 0)
            N[S_p] += 1
            epsilon = N_0 / (N_0 + N[S_p])

            if rnd.choice([True, False], p=[epsilon, 1 - epsilon]):     # Sample A_p from Q using epsilon-greedy
                A_p = rnd.choice([True, False])
            else:
                A_p = False if Q.setdefault((S, False), 0) > Q.setdefault((S, True), 0) else True

            Q.setdefault((S_p, A_p), 0)

            E.setdefault((S, A), 0)
            E[S, A] += 1

            N.setdefault((S_p, A_p), 0)
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

    vs = np.zeros(shape=(len(game.card_values), 21))

    for d_sum in range(len(game.card_values)):
        for p_sum in range(21):
            vs[d_sum, p_sum] = max([q.setdefault(((p_sum + 1, d_sum + 1, False), a), 0) for a in [True, False]])

    plt.imshow(vs)

    plt.show()




