if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt

    from version4.agents.montecarlo import MonteCarlo
    from version4.environments.easy21 import Easy21

    env = Easy21()

    procedure = MonteCarlo(env)

    q = procedure.learn(num_iter=1000000)

    table = procedure.q_table

    print(table)

    vs = np.zeros(shape=(21, 10))

    for (state, action), value in table.items():
        vs[state.p_sum - 1, state.d_sum - 1] = max([table[state, a] for a in env.valid_actions()])

    plt.imshow(vs)
    plt.show()
