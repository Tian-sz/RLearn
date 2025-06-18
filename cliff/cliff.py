import gymnasium as gym

import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod



class BaseAgent(ABC):
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率

        # 初始化Q表：状态 -> 动作价值
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))

        # 训练数据记录
        self.episode_rewards = []
        self.episode_steps = []

    def choose_action(self, state):
        """ ε-greedy动作选择 """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # 随机探索
        else:
            return np.argmax(self.q_table[state])  # 贪心利用

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """ 更新Q表（由子类实现） """
        pass

    def train(self, num_episodes=1000):
        """ 训练循环 """
        for ep in range(num_episodes):
            state, prob = self.env.reset()
            reward_list = []
            total_reward = 0
            steps = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, truncated, prob = self.env.step(action)

                # 核心更新逻辑（调用子类实现）
                self.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps += 1

            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)

            # 可选的ε衰减策略
            self.epsilon *= 0.95

            if ep % 100 == 0:
                print(f"Episode {ep}, Reward: {total_reward}, Steps: {steps}")
                print('average reward: {}, average step: {}'.format(sum(self.episode_rewards[:100])/100, sum(self.episode_steps[:100])/100))

        return self.episode_rewards, self.episode_steps


# ====================== 子类实现 ======================
class SarsaAgent(BaseAgent):
    def update(self, state, action, reward, next_state, done):
        current_q_a = self.q_table[state][action]

        if done:
            # 终止状态时后续Q值为0
            target = reward
        else:
            next_action = self.choose_action(next_state)
            next_q_a = self.q_table[next_state][next_action]
            target = reward + self.gamma * next_q_a

        # 更新公式（注意使用+=实现累积更新）
        self.q_table[state][action] += self.alpha * (target - current_q_a)


class QLearningAgent(BaseAgent):
    def update(self, state, action, reward, next_state, done):
        current_q_a = self.q_table[state][action]

        if done:
            target = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * max_next_q

        self.q_table[state][action] += self.alpha * (target - current_q_a)


class ExpectedSarsaAgent(BaseAgent):
    def update(self, state, action, reward, next_state, done):
        # TODO: 实现Expected SARSA更新规则
        pass


# ====================== 使用示例 ======================
if __name__ == "__main__":
    # env = gym.make('CliffWalking-v0', render_mode="human")
    env = gym.make('CliffWalking-v0')

    # 初始化Agent（选择其中一个）
    agent = SarsaAgent(env, alpha=0.5, epsilon=0.1)

    # 训练
    rewards, steps = agent.train(num_episodes=500)

    # 测试策略
    env = gym.make('CliffWalking-v0', render_mode="human")
    state, prob = env.reset()
    done = False
    while not done:
        action = np.argmax(agent.q_table[state])
        next_state, reward, done, truncated, prob = env.step(action)
        state = next_state
        env.render()
