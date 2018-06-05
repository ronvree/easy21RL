import random
from collections import defaultdict

from version3.agent import Agent
from version3.core import Action, Environment
from version3.policy import Policy
from version3.qtable import QTable


class MonteCarloAgent(Agent):

    def __init__(self, env: Environment):
        super().__init__(env)
        self.q_table = QTable(env.observation_space, env.action_space)
        self.visit_count = defaultdict(int)
        self.N_0 = 100
        self.gamma = 1

    def learn(self, num_iter=100000) -> Policy:
        Q, N = self.q_table, self.visit_count
        for _ in range(num_iter):
            s = self.env.reset()
            e, r, t = [], 0, False
            while not t:
                a = None  # TODO
                e += [[s, a]]
                s, r, t = self.env.step(a)
                e[-1] += [r]

            for i, (s, a, r) in enumerate(reversed(e)):
                g = r if i == 0 else g * self.gamma + r
                N[s, a] += 1
                N[s] += 1
                Q[s, a] += (1 / N[s, a]) * (g - Q[s, a])


    def sample_action(self, s, epsilon: float=0) -> Action:
        assert 0 <= epsilon <= 1
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            try:  # TODO -- move to table
                pass
            except:
                pass