import os
import types
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from stable_baselines3.common.buffers import ReplayBuffer

from agent import ModelFactory
from evaluator import SimpleEvaluator
from DQN import DQN
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

        self.env_config = dict_to_namespace(self.env_config_dict)
        self.agent_config = dict_to_namespace(self.agent_config_dict)
        self.trainer_config = dict_to_namespace(self.trainer_config_dict)

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
            self.env.observation_space,
            self.env.action_space,
            handle_timeout_termination = True,
        )

        self.epsilon = self.trainer_config.start_e
        self.total_steps = 0
        self.exploration_steps = int(self.trainer_config.exploration_fraction * self.trainer_config.epochs)

        env_eval = make_env(self.env_name, **self.env_config_dict)
        self.evaluator = SimpleEvaluator(env_eval)

        # 日志及结果
        run_name = "{}__{}__{}".format(self.env_name, self.agent_config.name, time.strftime("%Y-%m-%d_%H-%M-%S"))
        self.log_dir = os.path.join(f"logs/{run_name}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(f"logs/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.trainer_config).items()])),
        )

        print("{} initialized.".format(time.time()))

    def train(self):
        epoch = 0
        total_step = 0
        for epoch in trange(self.trainer_config.epochs, desc="Training"):
            state, _ = self.env.reset()
            episode_reward = 0
            step = 0
            done = False

            # 动态epsilon衰减
            self.epsilon = linear_schedule(
                self.trainer_config.start_e,
                self.trainer_config.end_e,
                self.trainer_config.exploration_epoch,
                epoch
            )
            self.writer.add_scalar("exploration/epsilon", self.epsilon, self.total_steps)

            while True:
                self.total_steps += 1  # 记录总步数

                action = self.agent.select_action(state, epsilon=self.epsilon)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated and not truncated
                self.buffer.add(state, next_state, action, reward, done, [{"truncated": truncated, **info}])

                step += 1
                total_step += 1

                # 训练过程
                if (self.buffer.size() > self.trainer_config.minimal_replay_size) and (total_step % self.trainer_config.update_iter == 0):
                    batch = self.buffer.sample(self.trainer_config.batch_size)
                    self.agent.update(batch)

                if done:
                    self.writer.add_scalar("reward/episode", episode_reward, epoch) # 记录回合奖励
                    self.writer.add_scalar("metrics/episode_length", step, epoch) # 记录回合长度
                    break

            # 定期评估和保存
            if epoch % self.trainer_config.eval_freq == 0:
                eval_res = self.evaluator.evaluate(self.agent)
                for eval_metric, value in eval_res.items():
                    self.writer.add_scalar(eval_metric, value, epoch)


if __name__ == '__main__':
    trainer = Trainer(env_config_1, agent_config_1, trainer_dict_1)
    trainer.train()
