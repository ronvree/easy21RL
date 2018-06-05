

class Action:
    pass


class Observation:
    pass


class Environment:
    """
        Class for describing the environments and how they handle states/actions/rewards/observations for the algorithms
        to learn from
    """

    def __init__(self):
        self._action_space = None
        self._observation_space = None

    @property
    def action_space(self):
        """

        :return:
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """

        :return:
        """
        raise NotImplementedError

    def step(self, action: Action) -> tuple:
        """
        Perform the action on the current model state. Return an observation and a corresponding reward
        :param action: The action to be performed
        :return: A three-tuple of
                        - an observation
                        - reward obtained from performing the action
                        - a boolean indicating whether the current environment state is terminal
        """
        raise NotImplementedError

    def reset(self) -> Observation:
        """
        Reset the internal model state
        :return: an initial observation
        """
        raise NotImplementedError
