import copy
import random


class Space:

    def __init__(self):
        self.subspaces = [self]

    def sample(self):
        raise NotImplementedError

    def contains(self, item) -> bool:
        raise NotImplementedError

    def __contains__(self, item):
        return self.contains(item)

    def __mul__(self, other):
        if isinstance(other, Space):
            s_c = copy.copy(self)
            s_c.subspaces = self.subspaces + other.subspaces
            s_c.contains = lambda x: \
                isinstance(x, tuple) and \
                len(x) == len(s_c.subspaces) and\
                all([s.contains(xi) for s, xi in zip(s_c.subspaces, x)])
            s_c.sample = lambda: tuple(s.sample() for s in s_c.subspaces)
            return s_c
        else:
            raise TypeError

    def __pow__(self, power):
        if isinstance(power, int):
            s_p = copy.copy(self)
            for _ in range(power - 1):
                s_p *= self
            return s_p
        else:
            raise TypeError


class FiniteSpace(Space):

    def __init__(self, elements):
        super().__init__()
        self._elements = elements

    def sample(self):
        return random.choice(self._elements)

    def contains(self, item) -> bool:
        return item in self._elements

    def elements(self):
        return list(self.__iter__())

    def __iter__(self):
        for i in self._elements:
            yield i

    def __len__(self):
        return len(self.elements())

    def __mul__(self, other):
        if isinstance(other, Space):
            if isinstance(other, FiniteSpace):
                s_c = copy.copy(self)
                s_c.__len__ = lambda: len(self) * len(other)
            else:
                s_c = Space()
            s_c.subspaces = self.subspaces + other.subspaces
            s_c.contains = lambda x: \
                isinstance(x, tuple) and \
                len(x) == len(s_c.subspaces) and\
                all([s.contains(xi) for s, xi in zip(s_c.subspaces, x)])
            s_c.sample = lambda: tuple(s.sample() for s in s_c.subspaces)
            if isinstance(other, FiniteSpace):
                def elem_iter():
                    for ei in self.elements():
                        for ej in other.elements():
                            if isinstance(ei, tuple):
                                if isinstance(ej, tuple):
                                    yield ei + ej
                                else:
                                    yield ei + (ej,)
                            else:
                                if isinstance(ej, tuple):
                                    yield (ei,) + ej
                                else:
                                    yield (ei, ej)
                s_c.__iter__ = elem_iter
            return s_c
        else:
            raise TypeError

    def __pow__(self, power):
        if isinstance(power, int):
            s_p = copy.copy(self)
            for _ in range(power - 1):
                s_p *= self
            return s_p
        else:
            raise TypeError


class BooleanSpace(FiniteSpace):

    def __init__(self):
        super().__init__([False, True])


if __name__ == '__main__':

    class TestSpace(Space):

        def __init__(self, items):
            super().__init__()
            self.items = items

        def sample(self):
            return random.choice(self.items)

        def contains(self, item) -> bool:
            return item in self.items

    booleans = FiniteSpace([True, False])
    bits = FiniteSpace([0, 1])

    foos = TestSpace(['blorp', 'hic', 'et'])

    test = booleans * bits

    test2 = booleans ** 3

    test3 = booleans * foos

    foos *= foos

    print(booleans.sample())
    print(bits.sample())
    print(test.sample())
    print(test2.sample())
    print([e for e in test2.elements()])
    print(len(test2))

    print(test3.sample())
    print(test3)

    print(foos.sample())

    print(test.contains((False, 0)))
    print(test.contains((2, 0)))
    print(test.contains((1,)))
