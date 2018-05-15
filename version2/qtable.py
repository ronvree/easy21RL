import collections
import random


class QTable(collections.MutableMapping):

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

    def __str__(self) -> str:
        t = '| Q Table                                    Value    |\n'
        t += '+------------------------------------------+----------+\n'
        t_entry = '| {:<40} | {:8.3f} |\n'
        for s, a in self:
            t += t_entry.format(str(s) + ', ' + str(a), self[s, a])
        t += '+------------------------------------------+----------+\n'
        return t

    def sample_greedy(self, state):
        actions = self.store[self.__state_transform__(state)]
        return max(actions.items(), key=lambda x: actions[x[0]])

    # def sample_epsilon_greedy(self, state, epsilon: float, action_space: callable):
    #     assert 0 <= epsilon <= 1
    #     if random.random() < epsilon:
    #         return action_space()
    #     else:
    #         return self.sample_greedy(state, action_space)


if __name__ == '__main__':

    table = QTable()

    table[0, 0] = 1  # State 0, action 0
    table[0, 1] = 2  # State 0, action 1

    table[1, 0] = 2  # State 1, action 0

    table[1, 0] += 1

    print(table[0, 1])
    print(table[0, 2])
    print(table[1, 0])
    print([v for v in table])
    print(len(table))
    print(table.sample_greedy(0, lambda x: -1))
    print(table.sample_epsilon_greedy(0, 0.5, lambda x: -1))
    print(table)
