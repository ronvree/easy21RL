import collections


class QTable(collections.MutableMapping):
    """
        Q Table implementation
    """

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __state_transform__(self, s):
        """
        Allows state to be modified (e.g. hashed) before being stored in the table
        :param s: The state entry
        :return: A modified state entry
        """
        return s

    def __getitem__(self, key: tuple) -> float:
        """
        Get the Q-value corresponding to the given key
        :param key: A two-tuple of (state, action)
        :return: The Q-value corresponding to the (state, action) pair. Return 0 if the pair is not in this table
        """
        s, a = key
        return self.store.get(self.__state_transform__(s), dict()).get(a, 0)

    def __setitem__(self, key: tuple, value: float):
        """
        Set a Q-value for a given (state, action) pair
        :param key: Two-tuple of (state, action)
        :param value: Q-value corresponding to the key
        """
        s, a = key
        self.store.setdefault(self.__state_transform__(s), dict())[a] = value

    def __delitem__(self, key: tuple):
        """
        Remove an entry from this table
        :param key: The (state, action) pair of the entry that should be removed
        """
        s, a = key
        del self.store[self.__state_transform__(s)][a]

    def __iter__(self):
        """
        :return: An iterator that iterates through all (state, action) pairs stored in this table
        """
        for state, actions in self.store.items():
            for action in iter(actions):
                yield (state, action)

    def __len__(self) -> int:
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
        actions = self.store[self.__state_transform__(state)]     # Obtain all actions stored with the state
        return max(actions.items(), key=lambda x: actions[x[0]])  # Pick the (action, value pair) that maximizes Q(s, a)


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
    print(table.sample_greedy(0))
    print(table)
