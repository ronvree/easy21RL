import collections
import random


class Policy(collections.MutableMapping):

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))

    @staticmethod
    def __state_transform__(s):
        return s

    def __getitem__(self, key: tuple) -> float:
        s, a = key
        return self.store.get(self.__state_transform__(s), dict()).get(a, 0)

    def __setitem__(self, key: tuple, value: float):
        s, a = key
        self.store.setdefault(self.__state_transform__(s), dict())[a] = value

    def __delitem__(self, key: tuple):
        s, a = key
        del self.store[self.__state_transform__(s)][a]

    def __iter__(self):
        for state, actions in self.store.items():
            for action in iter(actions):
                yield (state, action)

    def __len__(self) -> int:
        return sum([len(actions) for actions in self.store.values()])

    def sample_greedy(self, state):
        actions = self.store[self.__state_transform__(state)]
        return max(actions, key=actions.get)

    def sample_epsilon_greedy(self, state, epsilon: float):
        assert 0 <= epsilon <= 1
        actions = self.store[self.__state_transform__(state)]
        if random.random() < epsilon:
            return random.choice(actions)
        else:
            return self.sample_greedy(state)


if __name__ == '__main__':

    pi = Policy()

    pi[0, 0] = 1  # State 0, action 0
    pi[0, 1] = 2  # State 0, action 1

    pi[1, 0] = 2  # State 1, action 0

    pi[1, 0] += 1

    print(pi[0, 1])
    print(pi[0, 2])
    print(pi[1, 0])
    print([v for v in pi])
    print(pi.sample_greedy(0))
    print(pi.sample_epsilon_greedy(0, 0.5))
