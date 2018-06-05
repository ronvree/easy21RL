from version3.space import Space


class Policy:

    def __init__(self, observation_space: Space, action_map: callable):
        self.observation_space = observation_space
        self.action_map = action_map

    def sample(self, observation):
        raise NotImplementedError

    def __call__(self, observation, **kwargs):
        return self.sample(observation)


class QPolicy(Policy):

    def __init__(self, observation_space: Space, action_map: callable, q_source: callable):
        super().__init__(observation_space, action_map)
        self.q = q_source

    def sample(self, observation):
        raise NotImplementedError


class Greedy(QPolicy):

    def __init__(self, observation_space: Space, action_map: callable, q_source: callable):
        super().__init__(observation_space, action_map)
        self.q = q_source

    def sample(self, observation):

        pass

