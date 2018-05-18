import random

from version2.core import State, DiscreteActionEnvironment

"""
    Easy21 environment implementation
"""


class Easy21State(State):
    """
        Easy21 game state
    """

    def __init__(self, p_sum, d_sum):
        """
        Create a new Easy21 state
        :param p_sum: The initial player score
        :param d_sum: The initial dealer score
        """
        super().__init__()
        self.p_sum, self.d_sum = p_sum, d_sum

    def __str__(self):
        """
        :return: A string representation of this state
        """
        return 'P: {:<3}, D: {:<3}, T: {}'.format(self.p_sum, self.d_sum, 'y' if self.terminal else 'n')

    def __eq__(self, other):
        """
        Defines equality between two states
        :param other: Object to compare this state with
        :return: Whether the specified object is equal to this state
        """
        if not isinstance(other, Easy21State):
            return False
        else:
            return self.p_sum == other.p_sum and\
                   self.d_sum == other.d_sum and\
                   self.terminal == other.terminal

    def __hash__(self) -> int:
        """
        :return: A unique hash corresponding to this state
        """
        h = 2 if self.terminal else 0
        h += self.p_sum * 3
        h += self.d_sum * 5
        return h

    def copy(self):
        """
        :return: A copy of this state
        """
        c = Easy21State(self.p_sum, self.d_sum)
        c.terminal = self.terminal
        return c


class Easy21(DiscreteActionEnvironment):
    """
        Easy21 environment class
    """

    def __init__(self, p_red: float = 1 / 3):
        """
        Create a new Easy21 environment
        :param p_red: Probability of drawing a red card
        """
        assert 0 <= p_red <= 1
        self.p_red = p_red
        self.state = self._draw_init_state()

    @staticmethod
    def _draw_card_value() -> int:
        """
        :return: A random card value
        """
        return random.randint(1, 10)

    @staticmethod
    def _card_value(card: tuple) -> int:
        """
        Get the effect of the specified card. Red cards subtract value, black cards add value
        :param card: The card for which its value should be calculated
        :return: The value of the specified card
        """
        v, c = card
        return v if c else -v

    @staticmethod
    def _valid_score(score: int) -> bool:
        """
        Checks whether the given score is greater or equal to 1 and smaller or equal than 21
        :param score: The score to be checked
        :return: Whether the score is valid
        """
        return 1 <= score <= 21

    def _draw_card_color(self) -> bool:
        """
        :return: A boolean specifying whether the color is black (with p(True)=p_red)
        """
        return random.random() > self.p_red

    def _draw_card(self, force_black: bool=False) -> tuple:
        """
        Draw a random card from the pile
        :param force_black: If set to true, the card drawn will always be black
        :return: The drawn card
        """
        if force_black:
            return self._draw_card_value(), True
        return self._draw_card_value(), self._draw_card_color()

    def _draw_init_state(self) -> Easy21State:
        """
        :return: A random initial state for Easy21
        """
        return Easy21State(
            p_sum=self._card_value(self._draw_card(force_black=True)),
            d_sum=self._card_value(self._draw_card(force_black=True)),
        )

    def _reward(self, state: Easy21State) -> int:
        """
        Compute the reward corresponding to the specified state
        :param state: The state for which the reward should be determined
        :return:
        """
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
        """
        :return: A random action from the action space
        """
        if self.state.is_terminal():
            return None
        else:
            return bool(random.getrandbits(1))

    def step(self, action: bool, update=True) -> tuple:
        """
        Perform an action on the environment state
        :param action: The action to be performed
        :param update: A boolean indicating whether the change in environment model should be saved
        :return: A two-tuple of (observation, reward)
        """
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
        """
        Reset the Easy21 environment
        :return: An initial observation
        """
        self.state = self._draw_init_state()
        return self.state.copy()

    def action_space(self, state) -> set:
        """
        Get the actions that can be performed on the specified state
        :param state: The state on which an action should be performed
        :return: A set of actions
        """
        return {True, False}
