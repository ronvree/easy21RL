import copy
import random


class GameState:

    def actions(self) -> list:
        raise Exception('Not implemented!')

    def action_space(self) -> list:
        raise Exception('Not implemented!')

    def step(self, action, inplace=False) -> tuple:
        raise Exception('Not implemented!')

    def is_terminal(self) -> bool:
        raise Exception('Not implemented!')

    def reset(self):
        raise Exception('Not implemented!')


class Easy21(GameState):

    def __init__(self, p_red: float=1/3):
        assert 0 <= p_red <= 1
        self.p_sum, self.d_sum, self.terminal = 0, 0, False
        self.p_red = p_red
        self.reset()

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Easy21):
            return False
        else:
            return self.p_sum == o.p_sum and self.d_sum == o.d_sum and self.terminal == o.terminal

    def __hash__(self) -> int:
        h = 2 if self.terminal else 0
        h += self.p_sum * 3
        h += self.d_sum * 5
        return h

    def __str__(self):
        return 'P: {}, D: {}, T: {}'.format(self.p_sum, self.d_sum, self.terminal)

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

    def _draw_init_values(self) -> tuple:
        return self._card_value(self._draw_card(force_black=True)), self._card_value(self._draw_card(force_black=True))

    def _reward(self) -> int:
        if not self.terminal:
            return 0
        if not self._valid_score(self.p_sum):
            return -1
        if not self._valid_score(self.d_sum):
            return 1
        if self.p_sum > self.d_sum:
            return 1
        if self.p_sum < self.d_sum:
            return -1
        return 0

    '''
        GameState functions
    '''

    def actions(self) -> list:
        return [True, False] if not self.terminal else []

    def action_space(self) -> list:
        return [True, False]

    def step(self, action: bool, inplace=False) -> tuple:
        if self.terminal:
            raise Exception('Cannot perform action on terminal state!')
        s = self if inplace else copy.deepcopy(self)
        if action:  # Hit
            s.p_sum += self._card_value(self._draw_card())
            s.terminal = not self._valid_score(s.p_sum)
        else:  # Stick -> play dealer
            while self._valid_score(s.d_sum) and s.d_sum < 17:
                s.d_sum += self._card_value(self._draw_card())
            s.terminal = True
        return s, s._reward()

    def is_terminal(self) -> bool:
        return self.terminal

    def reset(self):
        self.p_sum, self.d_sum = self._draw_init_values()
        self.terminal = False
        return self
