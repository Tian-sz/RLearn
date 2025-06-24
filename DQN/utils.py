from abc import ABC, abstractmethod
import numpy as np
import collections
import random
import gymnasium as gym

from config import *

def model_summary(model):
    """显示模型基本信息"""
    # 1. 基本信息
    print(f"模型名称: {model.__class__.__name__}")
    print(f"运行设备: {next(model.parameters()).device}")

    # 2. 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"总参数量: {total_params:,} | 可训练参数: {trainable_params:,} | 冻结参数: {total_params - trainable_params:,}")

    # 3. 内存占用
    param_mem = sum(p.element_size() * p.nelement() for p in model.parameters()) / (1024 ** 2)
    buffer_mem = sum(b.element_size() * b.nelement() for b in model.buffers()) / (1024 ** 2)
    print(f"参数内存: {param_mem:.2f} MB | 缓冲区内存: {buffer_mem:.2f} MB")

    # 4. 层级结构信息
    print("\n层级结构:")
    for name, layer in model.named_children():
        layer_params = sum(p.numel() for p in layer.parameters())
        print(f"├─ {name} ({layer.__class__.__name__}) - 参数: {layer_params:,}")

def make_env(env_name, **kwargs):
    env = gym.make(env_name, **kwargs)
    return env

def get_state_dim(env):
    return np.array(env.observation_space.shape).prod()

def get_action_dim(env):
    """获取动作空间的维度表示"""
    if isinstance(env.action_space, gym.spaces.Discrete):
        return env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        return env.action_space.shape[0]
    elif isinstance(env.action_space, gym.spaces.MultiBinary):
        return env.action_space.n
    elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
        return len(env.action_space.nvec)
    else:
        raise NotImplementedError(f"不支持的空间类型: {type(env.action_space)}")

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration  # 斜率
    epsilon = max(slope * t + start_e, end_e)
    return epsilon

def dict_to_namespace(data):
    """递归将嵌套字典转换为嵌套SimpleNamespace"""
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = dict_to_namespace(value)
        return types.SimpleNamespace(**data)
    elif isinstance(data, list):
        return [dict_to_namespace(item) for item in data]
    else:
        return data