import random

from version2.core import State, DiscreteActionEnvironment


class Easy21State(State):

    def __init__(self, p_sum, d_sum):
        super().__init__()
        self.p_sum, self.d_sum = p_sum, d_sum

    def __str__(self):
        return 'P: {:<3}, D: {:<3}, T: {}'.format(self.p_sum, self.d_sum, 'y' if self.terminal else 'n')

    def __eq__(self, other):
        if not isinstance(other, Easy21State):
            return False
        else:
            return self.p_sum == other.p_sum and\
                   self.d_sum == other.d_sum and\
                   self.terminal == other.terminal

    def __hash__(self) -> int:
        h = 2 if self.terminal else 0
        h += self.p_sum * 3
        h += self.d_sum * 5
        return h

    def copy(self):
        c = Easy21State(self.p_sum, self.d_sum)
        c.terminal = self.terminal
        return c


class Easy21(DiscreteActionEnvironment):

    def __init__(self, p_red: float = 1 / 3):
        assert 0 <= p_red <= 1
        self.p_red = p_red
        self.state = self._draw_init_state()

    @staticmethod
    def _draw_card_value() -> int:
        return random.randint(1, 10)

    @staticmethod
    def _card_value(card: tuple) -> int:
        v, c = card
        return v if c else -v

    @staticmethod
    def _valid_score(score: int) -> bool:
        return 1 <= score <= 21

    def _draw_card_color(self) -> bool:
        return random.random() > self.p_red

    def _draw_card(self, force_black: bool=False) -> tuple:
        if force_black:
            return self._draw_card_value(), True
        return self._draw_card_value(), self._draw_card_color()

    def _draw_init_state(self) -> Easy21State:
        return Easy21State(
            p_sum=self._card_value(self._draw_card(force_black=True)),
            d_sum=self._card_value(self._draw_card(force_black=True)),
        )

    def _reward(self, state: Easy21State) -> int:
        if not state.is_terminal():
            return 0
        if not self._valid_score(state.p_sum):
            return -1
        if not self._valid_score(state.d_sum):
            return 1
        if state.p_sum > state.d_sum:
            return 1
        if state.p_sum < state.d_sum:
            return -1
        return 0

    '''
        GameState functions
    '''

    def sample_action(self):
        if self.state.is_terminal():
            return None
        else:
            return bool(random.getrandbits(1))

    def step(self, action: bool, update=True) -> tuple:
        if self.state.is_terminal():
            raise Exception('Cannot perform action on terminal state!')
        s = self.state if update else self.state.copy()
        if action:  # Hit
            s.p_sum += self._card_value(self._draw_card())
            s.terminal = not self._valid_score(s.p_sum)
        else:  # Stick -> play dealer
            while self._valid_score(s.d_sum) and s.d_sum < 17:
                s.d_sum += self._card_value(self._draw_card())
            s.terminal = True
        return s.copy() if update else s, self._reward(s)

    def reset(self):
        self.state = self._draw_init_state()
        return self.state.copy()

    def action_space(self, state) -> set:
        return {True, False}
