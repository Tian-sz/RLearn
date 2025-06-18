import random
import numpy as np
import torch
import gymnasium as gym

from DQN import DQN
from trainer import Trainer
from utils import *
from config import *

# 通用参数
seed = 42


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def dqn_main():
    # 定义环境、模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('CartPole-v1')
    input_size = np.array(env.observation_space.shape).prod()
    hidden_dim = DQN_config.hidden_dim
    output_size = get_action_dim(env)
    agent = DQN(input_size, hidden_dim, output_size)

    # 训练主流程
    DQN_trainer = Trainer(env, agent, buffer, evaluator)


if __name__ == '__main__':
    dqn_main()