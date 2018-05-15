

class Environment:

    def sample_action(self):
        raise NotImplementedError

    def step(self, action, update=True) -> tuple:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class State:

    def __init__(self):
        self.terminal = False

    def is_terminal(self) -> bool:
        return self.terminal


class DiscreteActionEnvironment(Environment):

    def sample_action(self):
        return super().sample_action()

    def step(self, action, update=True) -> tuple:
        return super().step(action, update)

    def reset(self):
        return super().reset()

    def action_space(self, state):
        raise NotImplementedError
