import game

import numpy.random as rnd


N_0 = 100
gamma = 1


def sarsa_lambda_policy_eval(lam=0.2):
    Q, N = dict(), dict()
    for _ in range(10000):                                  # Repeat for each episode
        episode, terminal, reward, E = [], False, 0, dict()

        S = game.draw_init_values() + (terminal,)              # Initialize S and A
        A = rnd.choice([True, False])

        while not terminal:                                     # Repeat for each step of episode
            S_p, reward = game.step(S, A)                       # Take action A, observe R, S_p

            N.setdefault(S_p, 0)
            epsilon = N_0 / (N_0 + N[S_p] + 1)

            v_t = Q.setdefault((S, True), 0)
            v_f = Q.setdefault((S, False), 0)
            # N.setdefault(state, 0)

            if rnd.choice([True, False], p=[epsilon, 1 - epsilon]):     # Sample A_p from Q using epsilon-greedy
                A_p = rnd.choice([True, False], p=[0.5, 0.5])
            else:
                A_p = False if v_f > v_t else True

            delta = reward + gamma * Q[S_p, A_p] - Q[S, A]
            E[S, A] += 1

            for k in E.keys():
                Q[k] += (1 / N[k]) * delta * E[k]
                E[k] = gamma * lam * E[k]

            S, A = S_p, A_p





