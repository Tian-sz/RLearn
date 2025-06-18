import numpy as np
import torch
import os
import types
from tqdm import trange
from stable_baselines3.common.buffers import ReplayBuffer

from agent import ModelFactory
from config import trainer_dict_1, env_config_1
from utils import *


class Trainer:
    def __init__(self, env_config, agent_config, trainer_config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env_config_dict = env_config
        self.env_name = self.env_config_dict.pop('env_name')
        self.env = make_env(self.env_name, **self.env_config_dict)

        self.agent_config_dict = agent_config
        self.trainer_config_dict = trainer_config

        self.env_config = types.SimpleNamespace(**self.env_config_dict)
        self.agent_config = types.SimpleNamespace(**self.agent_config_dict)
        self.trainer_config = types.SimpleNamespace(**self.trainer_config_dict)

        self.state_dim = get_state_dim(self.env)
        self.action_dim = get_action_dim(self.env)

        self.agent = ModelFactory.create(
            name=self.agent_config.name,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=self.agent_config_dict
        )

        self.buffer = ReplayBuffer(
            self.trainer_config.replay_buffer_config.buffer_size,
            self.state_dim,
            self.action_dim,
            self.device,
            handle_timeout_termination=False,
        )

        self.epsilon = self.trainer_config.start_e
        self.total_steps = 0
        self.exploration_steps = int(self.trainer_config.exploration_fraction * self.trainer_config.epochs)

        # 结果目录结构: results/<env>/<agent>/
        self.log_dir = os.path.join(
            env_config['name'],
            agent_config.name,
        )
        os.makedirs(self.log_dir, exist_ok=True)

    def train(self):
        results = {'steps': [], 'rewards': []}

        for epoch in trange(self.trainer_config.epochs, desc="Training"):
            state = self.env.reset()
            episode_reward = 0

            while True:
                # 动态epsilon衰减
                self.epsilon = linear_schedule(
                    self.trainer_config.start_e,
                    self.trainer_config.end_e,
                    self.trainer_config.exploration_fraction * self.trainer_config.total_timesteps,
                    epoch
                )

                action = self.agent.select_action(state, epsilon=self.epsilon, )
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # todo ...

                buffer.add(next_state, reward, terminated, truncated, info)

            # 定期更新智能体
            if len(buffer) > self.config.batch_size and self.total_steps % self.config.update_iter == 0:
                batch = buffer.sample(self.config.batch_size)
                self.agent.update(batch)

            state = next_state
            episode_reward += reward
            self.total_steps += 1

            if done:
                break

        # 定期评估和保存
        if epoch % self.config.eval_freq == 0:
            avg_reward = self.evaluate()
            results['steps'].append(self.total_steps)
            results['rewards'].append(avg_reward)
            self.save_checkpoint(epoch)

    return results


def evaluate(self, n_episodes=10):
    total_rewards = []
    for _ in range(n_episodes):
        state = self.env.reset()
        episode_reward = 0

        while True:
            action = self.agent.select_action(state, eval_mode=True)
            state, reward, done, _ = self.env.step(action)
            episode_reward += reward

            if done:
                total_rewards.append(episode_reward)
                break

    return np.mean(total_rewards)


def save_checkpoint(self, epoch):
    checkpoint = {
        'epoch': epoch,
        'agent_state': self.agent.state_dict(),
        'optimizer_state': self.agent.optimizer.state_dict(),
        'configs': {
            'env': self.env_config,
            'agent': self.agent_config,
            'trainer': vars(self.config)
        }
    }
    torch.save(checkpoint, os.path.join(self.log_dir, f'checkpoint_{epoch}.pth'))
