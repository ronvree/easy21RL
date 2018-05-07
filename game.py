import numpy as np


card_values = range(1, 11)
card_colors = ['r', 'b']

p_red = 1 / 3
p_black = 1 - p_red


def draw_card():
    return np.random.choice(card_values), np.random.choice(card_colors, p=[p_red, p_black])


def draw_black_card():
    return np.random.choice(card_values), card_colors[1]


def draw_init_values():
    return card_value(draw_black_card()), card_value(draw_black_card())


def draw_init_state():
    return draw_init_values() + (False,)


def card_value(card):
    v, c = card
    return v if c == card_colors[1] else -v


def valid_score(score):
    return 1 <= score <= 21


def reward(state):
    p_sum, d_sum, _ = state
    if not valid_score(p_sum):
        return -1
    if not valid_score(d_sum):
        return 1
    if p_sum > d_sum:
        return 1
    if p_sum < d_sum:
        return -1
    return 0


def step(s: tuple, a: bool):
    p_sum, d_sum, terminal = s
    if terminal:
        raise Exception('Cannot perform action on terminal state!')
    if a:
        p_sum += card_value(draw_card())
        return (p_sum, d_sum, not valid_score(p_sum)), reward((p_sum, d_sum, terminal))
    else:
        while valid_score(d_sum) and d_sum < 17:
            d_sum += card_value(draw_card())

        return (p_sum, d_sum, True), reward((p_sum, d_sum, True))
