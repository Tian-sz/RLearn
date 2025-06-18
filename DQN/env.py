from abc import ABC, abstractmethod

class EnvironmentWrapper(ABC):
    """环境接口"""

    @abstractmethod
    def __init__(self, env_name):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def render(self):
        pass

    @property
    @abstractmethod
    def state_dim(self):
        pass

    @property
    @abstractmethod
    def action_dim(self):
        pass