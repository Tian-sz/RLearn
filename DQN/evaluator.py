import gymnasium as gym
import numpy as np
import pandas as pd
import time
import seaborn as sns
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Optional, Any
from pathlib import Path

from utils import *
from config import evaluator_config


class BaseEvaluator:
    """评估器基类"""

    def __init__(self, env):
        self.env = env
        self.metrics = {}

    def evaluate(self, agent, n_episodes=10, **kwargs):
        """评估单个智能体"""
        raise NotImplementedError

    def compare(self, agents_dict, n_episodes=10, **kwargs):
        """比较多个智能体"""
        results = {}
        for name, agent in agents_dict.items():
            results[name] = self.evaluate(agent, n_episodes, **kwargs)
        return results


class SimpleEvaluator(BaseEvaluator):
    """轻量级训练过程评估器"""

    def __init__(self, env):
        super().__init__(env)
        self.default_metrics = [
            "episode_reward",
            "episode_length",
            "success_rate",
        ]

    def evaluate(self, agent, n_episodes=10, **kwargs):
        results = {metric: [] for metric in self.default_metrics}

        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            success = False

            while not done:
                action = agent.select_action(obs, eval_mode=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = (terminated or truncated)
                if "success" in info:
                    success = success or info["success"]
                episode_reward += reward
                step_count += 1

            results["episode_reward"].append(episode_reward)
            results["episode_length"].append(step_count)
            results["success_rate"].append(1 if success else 0)

        # 计算统计量
        stats = {}
        for metric, values in results.items():
            stats[f"{metric}_mean"] = np.mean(values)
            stats[f"{metric}_std"] = np.std(values)
            # stats[f"{metric}_min"] = np.min(values)
            # stats[f"{metric}_max"] = np.max(values)
        return stats

if __name__ == "__main__":
    pass
