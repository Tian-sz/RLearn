from abc import ABC, abstractmethod
import torch
from utils import get_action_dim


class Algorithm(ABC):
    """强化学习算法基类"""

    @abstractmethod
    def __init__(self, state_dim, action_dim, config):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    @abstractmethod
    def select_action(self, state, eval_mode=False):
        """选择动作（训练/评估模式）"""
        pass

    @abstractmethod
    def update(self, buffer):
        """使用回放缓冲区更新网络"""
        pass

    def save(self, path):
        """保存模型"""
        pass

    def load(self, path):
        """加载模型"""
        pass


# 创建一个模型注册工厂
class ModelFactory:
    _models = {}

    @classmethod
    def register(cls, name):
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class

        return decorator

    @classmethod
    def create(cls, name, state_dim, action_dim, config):
        model_class = cls._models.get(name)
        if not model_class:
            raise ValueError(f"Model {name} not found")
        return model_class(state_dim, action_dim, config)
