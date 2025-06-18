import numpy as np
import random
import gym

class WindyGridWorld:
    def __init__(self, dimension=5, start=(0, 0), goal=(4, 4), trap=None, random_transition=0.2):
        self.states = [(i, j) for i in range(dimension) for j in range(dimension)]
        self.actions = ['↑', '↓', '←', '→']
        self.start = (0, 0)
        self.goal = (4, 4)
        self.trap = trap
        self.random_transition = random_transition

    def transition(self, state, action):
        i, j = state
        # 定义动作对应的坐标变化
        move = {
            '↑': (-1, 0),
            '↓': (1, 0),
            '←': (0, -1),
            '→': (0, 1)
        }
        di, dj = move[action]

        # 80%的概率执行该动作
        if random.random() < self.random_transition:
            # 20%的概率随机选择其他三个动作，包括di, dj的可能
            possible_actions = [a for a in self.actions if a != action]
            random_action = random.choice(possible_actions)
            di, dj = move[random_action]
            ni, nj = i + di, j + dj
        else:
            # 可能移动的位置
            ni, nj = i + di, j + dj

        # 确保新位置在5x5网格内
        ni = max(0, min(ni, 4))
        nj = max(0, min(nj, 4))

        current_state = (ni, nj)
        reward, done = self.reward(current_state)
        return current_state, reward, done

    def reward(self, state):
        if state == self.goal:
            return 10.0, True
        elif state == self.trap:
            return -10.0, False
        else:
            return -0.1, False



if __name__ == '__main__':
    # env = WindyGridWorld()
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")
    env.reset()
    env.render()