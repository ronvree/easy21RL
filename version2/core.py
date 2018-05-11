

class Environment:

    def sample_action(self):
        raise Exception('Not implemented!')

    def step(self, action, update=True) -> tuple:
        raise Exception('Not implemented!')

    def reset(self):
        raise Exception('Not implemented!')


class State:

    def __init__(self):
        self.terminal = False

    def is_terminal(self) -> bool:
        return self.terminal

