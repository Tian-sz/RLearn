from abc import ABC, abstractmethod
import numpy as np
import random
from tqdm import tqdm
from env import WindyGridWorld as Environment
from matplotlib import pyplot as plt


class BasePolicy(ABC):
    def __init__(self, env, gamma=0.9, learning_rate=0.1, delta=1e-2):
        self.env = env
        self.gamma = gamma
        self.lr = learning_rate
        self.pi = {}
        self.delta = delta
        self.rewards_list = []

        # 初始化策略，例如等概率随机策略
        for state in self.env.states:
            num_actions = len(self.env.actions)
            self.pi[state] = {action: 1 / num_actions for action in self.env.actions}

    def choose_action(self, state):
        pass

    def update_policy(self):
        pass

    def evaluate_policy(self):
        pass

    def save_policy(self, filename):
        np.save(filename, self.pi)

    def load_policy(self, filename):
        self.pi = np.load(filename, allow_pickle=True).item()

    def test_policy(self):
        state = self.env.start
        rewards = 0
        i = 0
        while True:
            action = self.choose_action(state)
            next_state, reward, done = self.env.transition(state, action)
            rewards += reward
            if done or i == 10000:
                break
            state = next_state
            i += 1
        return rewards

    def plot(self):
        plt.plot(self.rewards_list)
        plt.show()


class PolicyIteration(BasePolicy):
    def __init__(self, env, gamma=0.9, learning_rate=0.1, delta=1e-4, estimate_times=10):
        super().__init__(env, gamma, learning_rate, delta=delta)
        self.v = {state: 0 for state in env.states}
        self.estimate_times = estimate_times  # 由于随机性的问题，多次采样取平均值，尽量保证稳定

    def choose_action(self, state):
        action_dict = self.pi[state]
        random_max_key = random.choice([k for k, v in action_dict.items() if v == max(action_dict.values())])
        max_value = max(action_dict.values())
        return random_max_key

    def evaluate_policy(self):
        episode = 0
        while True:
            delta = 0
            for state in self.env.states:
                if state == self.env.goal:
                    continue
                action_dict = self.pi[state]
                estimate_value = 0
                for i in range(self.estimate_times):
                    for action, prob in action_dict.items():
                        next_stat, reward, done = self.env.transition(state, action)
                        next_v = 0 if done else self.v[next_stat]
                        estimate_value += prob * (reward + self.gamma * next_v)
                estimate_value = estimate_value / self.estimate_times
                delta = max(delta, round(np.abs(estimate_value - self.v[state]), 5))
                self.v[state] = estimate_value
            if delta <= self.delta:
                print('{}次迭代后完成收敛'.format(episode))
                # print(self.v, '\n')
                break
            episode += 1
            if episode % 100 == 0:
                print(episode, delta)
        return delta

    def update_policy(self):
        for state in self.env.states:
            episode = 0
            action_dict = {'↑': 0,'↓': 0,'←': 0,'→': 0}
            while episode < self.estimate_times:
                episode += 1
                for action, prob in self.pi[state].items():
                    next_stat, reward, done = self.env.transition(state, action)
                    expected_value = reward + self.gamma * self.v[next_stat]
                    action_dict[action] += expected_value
            action_dict = {k:v/self.estimate_times for k,v in action_dict.items()}
            # if next_stat == self.env.goal:
            #     print()
            for action, action_value in action_dict.items():
                if action_value == max(action_dict.values()):
                    self.pi[state][action] = 1/len([v for v in action_dict.values() if v == max(action_dict.values())])
                else:
                    self.pi[state][action] = 0.0
        return None

    def train(self, episode=100):
        print('start training...')
        delta_list = []
        for i in tqdm(range(episode)):
            # print('evaluate policy...')
            delta = self.evaluate_policy()
            delta_list.append(delta)
            # print('update policy...')
            self.update_policy()
            # print('training finished.')
            rewards = self.test_policy()
            self.rewards_list.append(rewards)
        self.delta_list = delta_list


class ValueIteration(BasePolicy):
    def __init__(self, env, gamma=0.9, learning_rate=0.1, delta=1e-4, estimate_times=10):
        super().__init__(env, gamma, learning_rate, delta=delta)
        self.v = {state: 0 for state in env.states}
        self.estimate_times = estimate_times  # 由于随机性的问题，多次采样取平均值，尽量保证稳定

    def update_policy(self):
        episode = 0
        while True:
            delta = 0
            for state in self.env.states:
                if state == self.env.goal:
                    continue
                # if state == self.env.trap:
                #     print(None)
                action_dict = self.pi[state]
                qsa = {'↑': 0,'↓': 0,'←': 0,'→': 0}
                for i in range(self.estimate_times):
                    for action, prob in action_dict.items():
                        next_stat, reward, done = self.env.transition(state, action)
                        next_v = 0 if done else self.v[next_stat]
                        qsa[action] += (reward + self.gamma * next_v) # 无需乘以 prob !
                qsa = {k: v/self.estimate_times for k, v in qsa.items()}
                vs = max(qsa.values())
                delta = max(delta, round(np.abs(vs - self.v[state]), 5))
                self.v[state] = vs
            if delta <= self.delta:
                print('{}次迭代后完成收敛'.format(episode))
                # print(self.v, '\n')
                break
            episode += 1
            # if episode % 10 == 0:
            #     print(episode, delta)

        for state in self.env.states:
            if state == self.env.goal:
                continue
            episode = 0
            action_dict = {'↑': 0,'↓': 0,'←': 0,'→': 0}
            while episode < self.estimate_times:
                episode += 1
                for action, prob in self.pi[state].items():
                    next_stat, reward, done = self.env.transition(state, action)
                    next_v = 0 if done else self.v[next_stat]
                    expected_value = reward + self.gamma * next_v
                    action_dict[action] += expected_value
            action_dict = {k:v/self.estimate_times for k,v in action_dict.items()}
            for action, action_value in action_dict.items():
                if action_value == max(action_dict.values()):
                    self.pi[state][action] = 1/len([v for v in action_dict.values() if v == max(action_dict.values())])
                else:
                    self.pi[state][action] = 0.0
        return None

    def choose_action(self, state):
        action_dict = self.pi[state]
        random_max_key = random.choice([k for k, v in action_dict.items() if v == max(action_dict.values())])
        max_value = max(action_dict.values())
        return random_max_key

    def test_policy(self):
        state = self.env.start
        rewards = 0
        while True:
            action = self.choose_action(state)
            next_state, reward, done = self.env.transition(state, action)
            rewards += reward
            if done:
                break
            state = next_state
        return rewards


    def train(self, episode=100):
        print('start training...')
        delta_list = []
        for i in tqdm(range(episode)):
            # print('update policy...')
            self.update_policy()
            # print('training finished.')
            rewards = self.test_policy()
            self.rewards_list.append(rewards)
        self.delta_list = delta_list


if __name__ == '__main__':
    from env import WindyGridWorld as Environment
    env = Environment(random_transition=0.0, trap=(1,1))
    policy_agent = PolicyIteration(env, gamma=0.99, learning_rate=0.1, delta=1e-4, estimate_times=100)
    policy_agent.train(episode=10)
    value_agent = ValueIteration(env, gamma=0.99, learning_rate=0.1, estimate_times=100)
    value_agent.train(episode=10)
    print()