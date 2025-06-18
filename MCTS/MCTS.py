# mcts_optimized.py
import math
import random
import numpy as np
from copy import deepcopy
from env import Gomoku


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # (board, current_player, last_move)
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_value = 0.0


class MCTS:
    def __init__(self, env, exploration_weight=1.414, n_simulations=1000):
        self.env = env
        self.exploration_weight = exploration_weight
        self.n_simulations = n_simulations
        self.max_path = 0

    def search(self, initial_state):
        root = MCTSNode(initial_state)
        self.max_path = 0
        for i in range(self.n_simulations):
            if i + 2 == self.n_simulations:
                print()

            node = self.select(root)  # agent的选择后状态
            reward = self.simulate(node)  # 人类视角的reward
            self.backpropagate(node, reward)
        res_dict = sorted({child.action: (child.visits, child.total_value) for child in root.children}.items(),
                          key=lambda x: x[1], reverse=True)
        # print(self.best_action(root))
        # sorted({child.action: (child.visits, child.total_value) for child in root.children}.items(), key= lambda x:x[1], reverse=True)
        return self.best_action(root)

    def select(self, node):
        while True:
            if not self.is_fully_expanded(node):
                return self.expand(node)
            else:
                node = self.best_child(node)

    def is_fully_expanded(self, node):
        board, player, _ = node.state
        legal_actions = list(zip(*np.where(board == 0)))
        return len(node.children) == len(legal_actions)

    def expand(self, node):
        board, player, last_move = node.state
        legal_actions = list(zip(*np.where(board == 0)))
        children_actions = [c.action for c in node.children]
        available_actions = [a for a in legal_actions if a not in children_actions]

        if not available_actions:
            return node  # 无可用动作时返回当前节点

        # 邻近优先排序策略
        if last_move:
            available_actions.sort(
                key=lambda a: self._distance(a, last_move)
            )
        else:  # 初始状态优先中心
            center = (self.env.board_size // 2, self.env.board_size // 2)
            available_actions.sort(
                key=lambda a: self._distance(a, center)
            )

        # 选择最近距离组中的一个随机动作
        min_dist = self._distance(available_actions[0], last_move) if last_move else 0
        candidates = [a for a in available_actions
                      if self._distance(a, last_move if last_move else center) == min_dist]
        action = random.choice(candidates)

        # 创建新节点
        new_env = self._create_sim_env(node.state)
        new_env.step(action)
        new_node = MCTSNode(new_env._get_state(), parent=node, action=action)
        node.children.append(new_node)
        return new_node

    def _distance(self, a, b):
        """曼哈顿距离"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def best_child(self, node):
        best_score = -math.inf
        best_children = []
        score_dict = {}
        for child in node.children:
            if child.visits == 0:
                score = math.inf
            else:
                exploitation = child.total_value / child.visits
                exploration = math.sqrt(math.log(node.visits) / child.visits)
                score = exploitation + self.exploration_weight * exploration

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)
            score_dict[child.action] = score
        score_dict = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        return random.choice(best_children)

    def simulate(self, node):
        current_env = self._create_sim_env(node.state)
        current_player = current_env.current_player

        while not current_env.done:
            legal_actions = current_env.get_legal_actions()
            if not legal_actions:
                break

            # 邻近优先模拟策略
            if current_env.last_move:
                legal_actions.sort(
                    key=lambda a: self._distance(a, current_env.last_move)
                )
                # 选择前30%的邻近位置
                selected_range = max(1, len(legal_actions) // 3)
                action = random.choice(legal_actions[:selected_range])
            else:
                action = random.choice(legal_actions)

            current_env.step(action)

        if current_env.winner == 1:  # 如果人类获胜
            return 1
        elif current_env.winner is None:
            return 0
        else:
            return -1

    def backpropagate(self, node, reward):
        path = []
        while node is not None:
            path.append(node)
            node.visits += 1
            node.total_value += reward
            reward = -reward  # 切换玩家视角
            # print('current_player:',node.state[1])
            node = node.parent
        len_path = len(path)
        if len_path > self.max_path:
            self.max_path = len_path
            print('max_path', self.max_path-1)

    def best_action(self, node):
        return max(node.children, key=lambda c: c.visits).action

    def _create_sim_env(self, state):
        """根据状态创建模拟环境"""
        board, player, last_move = state
        env = Gomoku(self.env.board_size, self.env.win_length)
        env.board = board.copy()
        env.current_player = player
        env.last_move = last_move
        env.done = False
        env.winner = None
        return env


if __name__ == "__main__":
    env = Gomoku(board_size=10)
    state = env.reset()
    mcts = MCTS(env, n_simulations=10000)

    while not env.done:
        env.render()
        if env.current_player == 1:
            # 人类玩家
            try:
                action = tuple(map(int, input("Enter row,col: ").split(',')))
            except:
                print("Invalid input! Try again.")
                continue
        else:
            # AI玩家
            print("AI is thinking...")
            action = mcts.search(state)
            print(f"AI plays: {action}")

        state, _, done, _ = env.step(action)

    env.render()
    print(f"Winner: Player {env.winner}" if env.winner else "Draw!")