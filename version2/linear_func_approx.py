import random
import numpy as np

from version2.core import DiscreteActionEnvironment

N_0 = 100
GAMMA = 1
NUM_ITER = 100000


class SarsaLambda:

    def __init__(self,
                 env: DiscreteActionEnvironment,
                 features: callable,
                 weights,
                 lam: float=0.2,
                 eta: float=0.01
                 ):

        assert 0 <= lam <= 1
        self.env = env
        self.lam = lam
        self.eta = eta
        self.x = features
        self.w = weights

    def policy_eval(self) -> callable:
        q, w, x = self.q, self.w, self.x
        for _ in range(NUM_ITER):
            s = self.env.reset()
            a = self.env.sample_action()

            while not s.is_terminal():
                s_p, r = self.env.step(a)

                a_p = self.sample_derived_policy(s_p, epsilon=0.05)

                delta = r + GAMMA * q(s_p, a_p) - q(s, a)

                for i, f in enumerate(x(s, a)):
                    w[i] += self.eta * delta * f

                s, a = s_p, a_p
        return self.q

    def sample_derived_policy(self, s, epsilon: float=0):
        assert 0 <= epsilon <= 1
        if random.random() < epsilon:
            return self.env.sample_action()
        else:
            return max(self.env.action_space(s), key=lambda a: self.q(s, a))

    def q(self, s, a):
        return np.dot(self.x(s, a), self.w)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from version2.easy21 import Easy21, Easy21State


    def easy21_feature_vector(s, a):
        i = 0
        fs = np.zeros(36)
        for d_interval in [range(1, 5), range(4, 8), range(7, 11)]:
            for p_interval in [range(1, 7), range(4, 10), range(7, 13), range(10, 16), range(13, 19), range(16, 22)]:
                for _a in [True, False]:
                    fs[i] = 1 if s.p_sum in p_interval and s.d_sum in d_interval and a == _a else 0
                    i += 1
        return fs.T


    procedure = SarsaLambda(Easy21(), features=easy21_feature_vector, weights=np.zeros(36), lam=0.2, eta=0.01)

    _q = procedure.policy_eval()

    vs = np.zeros(shape=(21, 10))

    for p_sum in range(1, 22):
        for d_sum in range(1, 11):
            _s = Easy21State(p_sum, d_sum)
            vs[p_sum - 1, d_sum - 1] = max([_q(_s, True), _q(_s, False)])

    plt.imshow(vs)
    plt.show()
