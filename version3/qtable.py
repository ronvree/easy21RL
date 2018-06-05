import collections
import random

from version3.space import Space, FiniteSpace


class QTable(collections.MutableMapping):
    """
        Q Table implementation
    """

    def __init__(self, state_space: Space, action_space: FiniteSpace, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))
        self.state_space = state_space
        self.action_space = action_space

    def __getitem__(self, key: tuple) -> float:
        """
        Get the Q-value corresponding to the given key
        :param key: A two-tuple of (state, action)
        :return: The Q-value corresponding to the (state, action) pair. Return 0 if the pair is not in this table
        """
        s, a = key
        if s not in self.state_space or a not in self.action_space:
            raise KeyError
        return self.store.get(s, dict()).get(a, 0)

    def __setitem__(self, key: tuple, value: float):
        """
        Set a Q-value for a given (state, action) pair
        :param key: Two-tuple of (state, action)
        :param value: Q-value corresponding to the key
        """
        s, a = key
        if s not in self.state_space or a not in self.action_space:
            raise KeyError
        self.store.setdefault(s, dict())[a] = value

    def __delitem__(self, key: tuple):
        """
        Remove an entry from this table
        :param key: The (state, action) pair of the entry that should be removed
        """
        s, a = key
        if s not in self.state_space or a not in self.action_space:
            raise KeyError
        del self.store[s][a]

    def __iter__(self):  # TODO -- anders of compleet weg halen
        """
        :return: An iterator that iterates through all (state, action) pairs stored in this table
        """
        for state, actions in self.store.items():
            for action in iter(actions):
                yield (state, action)

    def __len__(self) -> int:  # TODO anders of compleet weg halen
        """
        :return: The number of entries in this table
        """
        return sum([len(actions) for actions in self.store.values()])

    def __str__(self) -> str:
        """
        :return: A pretty string representation  TODO -- fix
        """
        t = '| Q Table                                    Value    |\n'
        t += '+------------------------------------------+----------+\n'
        t_entry = '| {:<40} | {:8.3f} |\n'
        for s, a in self:
            t += t_entry.format(str(s) + ', ' + str(a), self[s, a])
        t += '+------------------------------------------+----------+\n'
        return t

    def sample_greedy(self, state):
        """
        Greedily choose an action for the given state
        :param state: The state from which an action should be chosen
        :return: The greedily sampled action to be performed from the specified state
        """
        if state not in self.state_space:
            raise KeyError
        return max(self.action_space.elements(),   # Pick the (action, value pair) that maximizes Q(s, a)
                   key=lambda x: self[state, x])

    def sample_epsilon_greedy(self, state, epsilon):
        assert 0 <= epsilon <= 1
        if random.random() < epsilon:
            return self.action_space.sample()
        else:
            return self.sample_greedy(state)


if __name__ == '__main__':
    from version3.space import BooleanSpace, FiniteSpace

    table = QTable(BooleanSpace(), FiniteSpace([0, 1, 2]))

    table[0, 0] = 1  # State 0, action 0
    table[0, 1] = 2  # State 0, action 1

    table[1, 0] = 2  # State 1, action 0

    table[1, 0] += 1

    print(table[0, 1])
    print(table[0, 2])
    print(table[1, 0])
    print([v for v in table])
    print(len(table))
    print(table.sample_greedy(0))
    print(table)
