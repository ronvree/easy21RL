from version3.core import Environment
from version3.policy import Policy


class Agent:

    def __init__(self, env: Environment):
        self.env = env

    def learn(self) -> Policy:
        raise NotImplementedError
