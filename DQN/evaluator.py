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


class RLAgentEvaluator:
    """
    强化学习通用评估器
    用于评估不同智能体在不同环境中的表现
    """

    def __init__(
            self,
            env_name: str,
            agent_maker: Callable[[gym.Env], Any],
            eval_episodes: int = 10,
            log_dir: str = "eval_results",
            render: bool = False,
            render_mode: str = "human",
            track_history: bool = True
    ):
        """
        初始化评估器

        参数:
            env_name: gym环境名称
            agent_maker: 创建智能体的函数，接受环境实例作为参数
            eval_episodes: 每次评估运行的episode数量
            log_dir: 结果保存目录
            render: 是否在评估时渲染环境
            render_mode: 渲染模式 ('human', 'rgb_array'等)
            track_history: 是否跟踪完整的评估历史
        """
        self.env_name = env_name
        self.agent_maker = agent_maker
        self.eval_episodes = eval_episodes
        self.log_dir = Path(log_dir)
        self.render = render
        self.render_mode = render_mode
        self.track_history = track_history

        # 确保日志目录存在
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 初始化结果存储
        self.results = {
            "episode": [],
            "total_reward": [],
            "steps": [],
            "truncated": [],
            "terminated": [],
            "env_name": [],
            "agent_type": [],
            "param_name": [],
            "param_value": []
        }

        # 历史记录
        self.history = defaultdict(list)

        # 创建临时环境以获取信息
        self._create_temp_env()

    def build_agent(self, agent_name: str, config: dict) -> Any:
        """通过工厂方法创建代理"""
        env = gym.make(self.env_name)  # 临时环境
        agent = ModelFactory.create(agent_name, env, config)
        return agent

    def _create_temp_env(self) -> None:
        """创建临时环境用于获取空间信息"""
        with gym.make(self.env_name) as temp_env:

            self.state_dim = temp_env.observation_space.shape
            self.action_dim = get_action_dim(temp_env)

            self.is_discrete = isinstance(temp_env.action_space, gym.spaces.Discrete)

    def _create_agent_env(self) -> Tuple[gym.Env, Any]:
        """创建环境和新智能体实例"""
        env = gym.make(
            self.env_name,
            render_mode=self.render_mode if self.render else None
        )
        agent = self.agent_maker(env)
        return env, agent

    def evaluate_single_episode(
            self,
            env: gym.Env,
            agent: Any,
            max_steps: int = 10000,
            save_video: bool = False
    ) -> Dict[str, Any]:
        """
        评估单个episode

        参数:
            env: 环境实例
            agent: 智能体实例
            max_steps: episode最大步数
            save_video: 是否保存视频帧

        返回:
            episode结果字典
        """
        state, info = env.reset()
        done = False
        truncated = False
        terminated = False
        total_reward = 0.0
        steps = 0
        frames = []

        # 如果是需要保存视频的模式
        if save_video and self.render_mode == "rgb_array":
            frames.append(env.render())

        while not done and not truncated and steps < max_steps:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            state = next_state
            done = terminated or truncated
            steps += 1

            if save_video and self.render_mode == "rgb_array" and steps % 2 == 0:
                frames.append(env.render())

        return {
            "total_reward": total_reward,
            "steps": steps,
            "truncated": truncated,
            "terminated": terminated,
            "frames": frames,
            "info": info
        }

    def run_evaluation(
            self,
            tag: str = "default",
            max_steps: int = 10000,
            agent_params: Optional[Dict[str, Any]] = None,
            save_video: int = 0
    ) -> Dict[str, Any]:
        """
        运行完整评估流程

        参数:
            tag: 实验标记
            max_steps: 每个episode的最大步数
            agent_params: 传递给agent_maker的参数
            save_video: 保存多少个episode的视频 (0表示不保存)

        返回:
            评估结果字典
        """
        # 应用代理参数（如果提供）
        if agent_params:
            prev_agent_maker = self.agent_maker
            self.agent_maker = lambda env: prev_agent_maker(env, **agent_params)

        # 创建环境和新智能体实例
        env, agent = self._create_agent_env()

        # 获取代理类型名称
        agent_type = type(agent).__name__

        # 记录开始时间
        start_time = time.time()

        # 运行多个episode
        episode_results = []
        for ep in range(self.eval_episodes):
            save_episode_video = save_video > 0 and (ep % (self.eval_episodes // save_video) == 0)
            result = self.evaluate_single_episode(
                env, agent,
                max_steps=max_steps,
                save_video=save_episode_video
            )
            episode_results.append(result)

            # 打印进度
            print(
                f"Episode {ep + 1}/{self.eval_episodes} | Reward: {result['total_reward']:.2f} | Steps: {result['steps']}")

            # 保存结果
            self._save_result(
                ep, result, agent_type,
                agent_params=agent_params,
                tag=tag
            )

        # 关闭环境
        env.close()

        # 计算汇总统计信息
        rewards = [ep["total_reward"] for ep in episode_results]
        steps = [ep["steps"] for ep in episode_results]

        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)

        avg_steps = np.mean(steps)

        summary = {
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "success_rate": np.mean([not ep["truncated"] for ep in episode_results]),
            "avg_steps": avg_steps,
            "execution_time": time.time() - start_time,
            "episode_results": episode_results,
            "agent_params": agent_params,
            "tag": tag
        }

        # 记录历史
        if self.track_history:
            self.history[tag].append(summary)

        # 重置agent_maker
        if agent_params:
            self.agent_maker = prev_agent_maker

        return summary

    def _save_result(
            self,
            episode_idx: int,
            result: Dict[str, Any],
            agent_type: str,
            agent_params: Optional[Dict[str, Any]] = None,
            tag: str = "default"
    ) -> None:
        """保存单个episode的结果"""
        self.results["episode"].append(episode_idx)
        self.results["total_reward"].append(result["total_reward"])
        self.results["steps"].append(result["steps"])
        self.results["truncated"].append(result["truncated"])
        self.results["terminated"].append(result["terminated"])
        self.results["env_name"].append(self.env_name)
        self.results["agent_type"].append(agent_type)
        self.results["tag"].append(tag)

        # 记录参数（如果存在）
        if agent_params:
            for param_name, param_value in agent_params.items():
                self.results["param_name"].append(param_name)
                self.results["param_value"].append(param_value)
        else:
            self.results["param_name"].append(None)
            self.results["param_value"].append(None)

    def save_results(self, filename: str = "evaluation_results.csv") -> None:
        """保存所有评估结果到CSV文件"""
        df = pd.DataFrame(self.results)
        filepath = self.log_dir / filename
        df.to_csv(filepath, index=False)
        print(f"评估结果已保存到: {filepath}")

    def compare_agents(
            self,
            agent_makers: Dict[str, Callable],
            eval_episodes: int = 10,
            tags: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        比较多个智能体的表现

        参数:
            agent_makers: 智能体创建函数字典 {name: callable}
            eval_episodes: 每个agent的评估episode数
            tags: 每个智能体的标签列表

        返回:
            包含每个agent评估结果的字典
        """
        original_episodes = self.eval_episodes
        original_agent = self.agent_maker
        results = {}

        self.eval_episodes = eval_episodes
        agent_names = list(agent_makers.keys())

        if not tags:
            tags = agent_names

        for name, tag in zip(agent_names, tags):
            print(f"\n评估代理: {name} ({tag})")
            self.agent_maker = agent_makers[name]
            results[tag] = self.run_evaluation(tag=tag)

        # 恢复原始设置
        self.eval_episodes = original_episodes
        self.agent_maker = original_agent

        return results

    def hyperparameter_search(
            self,
            param_name: str,
            param_values: List[Any],
            eval_episodes: int = 5,
            prefix: str = "param_search"
    ) -> Dict[Any, Dict]:
        """
        对单个参数进行搜索和评估

        参数:
            param_name: 参数名称
            param_values: 参数值列表
            eval_episodes: 每个参数值的评估episode数
            prefix: 实验名称前缀

        返回:
            每个参数值的评估结果字典
        """
        results = {}

        for value in param_values:
            tag = f"{prefix}_{param_name}_{value}"
            print(f"\n评估 {param_name}={value}")
            results[value] = self.run_evaluation(
                tag=tag,
                agent_params={param_name: value},
                eval_episodes=eval_episodes
            )

        return results

    def plot_results(self, value_key: str = "total_reward", rolling_window: int = 5) -> None:
        """绘制评估结果"""
        if not self.results["episode"]:
            print("没有可用的评估结果")
            return

        df = pd.DataFrame(self.results)
        plt.figure(figsize=(12, 6))

        # 按标签分组（如果存在）
        if "tag" in df.columns and len(df["tag"].unique()) > 1:
            sns.lineplot(
                x="episode",
                y=value_key,
                hue="tag",
                data=df,
                ci="sd",
                estimator=np.mean
            )
            plt.title(f"{value_key} 随时间变化")
            plt.legend(title="实验标签")
        else:
            # 对单一实验应用滚动平均
            if rolling_window > 1:
                df["rolling_mean"] = df[value_key].rolling(rolling_window).mean()
                df["rolling_std"] = df[value_key].rolling(rolling_window).std()

                plt.plot(df["episode"], df["rolling_mean"], label=f"滑动平均 ({rolling_window}项)")
                plt.fill_between(
                    df["episode"],
                    df["rolling_mean"] - df["rolling_std"],
                    df["rolling_mean"] + df["rolling_std"],
                    alpha=0.2
                )
            else:
                plt.plot(df["episode"], df[value_key], label="原始数据")

            plt.title(f"{value_key} 随时间变化")
            plt.legend()

        plt.xlabel("Episode")
        plt.ylabel(value_key)
        plt.grid(True)

        # 保存图表
        plot_path = self.log_dir / f"{value_key}_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"结果图表已保存到: {plot_path}")

    def plot_param_search(self, param_name: str, value_key: str = "avg_reward") -> None:
        """绘制参数搜索结果"""
        if not self.history:
            print("没有可用的参数搜索数据")
            return

        # 筛选出参数搜索的历史记录
        param_results = {}
        for tag, summaries in self.history.items():
            if param_name in tag and summaries:
                param_values = []
                metric_values = []
                for summary in summaries:
                    if summary.get("agent_params") and param_name in summary["agent_params"]:
                        param_values.append(summary["agent_params"][param_name])
                        metric_values.append(summary[value_key])

                if param_values:
                    param_results[tag] = {
                        "params": param_values,
                        "values": metric_values
                    }

        if not param_results:
            print(f"未找到参数 {param_name} 的搜索结果")
            return

        plt.figure(figsize=(10, 6))

        for tag, data in param_results.items():
            # 按参数值排序
            sorted_indices = np.argsort(data["params"])
            sorted_params = np.array(data["params"])[sorted_indices]
            sorted_values = np.array(data["values"])[sorted_indices]

            plt.plot(sorted_params, sorted_values, 'o-', label=tag)

        plt.title(f"参数: {param_name} 对 {value_key} 的影响")
        plt.xlabel(param_name)
        plt.ylabel(value_key)
        plt.grid(True)
        plt.legend()

        # 保存图表
        plot_path = self.log_dir / f"param_search_{param_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"参数搜索图表已保存到: {plot_path}")

    def save_video(self, frames: List[np.ndarray], filename: str) -> None:
        """保存视频帧为GIF文件"""
        if not frames:
            print("没有视频帧可保存")
            return

        try:
            from PIL import Image
        except ImportError:
            print("无法保存视频: 需要安装PIL库")
            return

        filepath = self.log_dir / filename
        if not filepath.suffix.lower() == ".gif":
            filepath = filepath.with_suffix(".gif")

        # 保存为GIF
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(
            filepath,
            save_all=True,
            append_images=images[1:],
            duration=100,  # ms
            loop=0  # 无限循环
        )
        print(f"视频已保存到: {filepath}")

    def export_full_report(self) -> None:
        """导出完整评估报告"""
        # 保存结果CSV
        self.save_results()

        # 绘制结果
        self.plot_results()
        self.plot_results("steps")

        # 参数搜索图
        if self.history:
            for tag in self.history.keys():
                if tag.startswith("param_search"):
                    param_name = tag.split("_")[2]
                    self.plot_param_search(param_name)

        print(f"完整评估报告已生成到: {self.log_dir}")

if __name__ == "__main__":
    from agent import ModelFactory
    # 配置参数
    env_name = evaluator_config.env_name
    epochs = evaluator_config.epochs
    agent_list = evaluator_config.agents

    # v0.1 使用config配置对比实验的环境、模型，使用compare_agents对比具体模型的不同参数

    for env_name in evaluator_config.env_list:

        env = gym.make(env_name)
        input_size = np.array(env.observation_space.shape).prod()
        hidden_dim = DQN_config.hidden_dim
        output_size = get_action_dim(env)

        for agent_name in agent_list:

            agent = ModelFactory.create(agent_name, env, input_size, hidden_dim, output_size)

            eval = RLAgentEvaluator(
                env_name=env_name,
                eval_episodes=epochs,
                log_dir="results/"
            )

            # 运行基本评估
            print("运行基本评估...")
            summary = eval.run_evaluation()
            print(f"平均奖励: {summary['avg_reward']:.2f} ± {summary['std_reward']:.2f}")

            # 运行参数搜索
            print("\n运行epsilon参数搜索...")
            param_results = eval.hyperparameter_search(
                param_name="epsilon",
                param_values=[0.0, 0.1, 0.2, 0.3, 0.4],
                eval_episodes=10
            )

            # 比较不同代理
            print("\n比较不同epsilon的代理...")

