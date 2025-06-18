import numpy as np


class Gomoku:
    def __init__(self, board_size=15, win_length=5):
        """
        初始化五子棋环境
        :param board_size: 棋盘尺寸（默认15x15）
        :param win_length: 胜利所需连子长度（默认5）
        """
        self.board_size = board_size
        self.win_length = win_length
        self.board = np.zeros((board_size, board_size), dtype=int)  # 棋盘状态（0:空, 1:黑棋, 2:白棋）
        self.current_player = 1  # 当前玩家（1或2）
        self.done = False  # 是否游戏结束
        self.winner = None  # 胜利者（1或2）
        self.last_move = None  # 新增：记录最后落子位置

    def reset(self):
        """重置游戏状态"""
        self.board.fill(0)
        self.current_player = 1
        self.done = False
        self.winner = None
        self.last_move = None  # 重置时清除最后落子位置
        return self._get_state()

    def get_legal_actions(self):
        """获取合法动作列表（返回所有空位的坐标）"""
        return list(zip(*np.where(self.board == 0)))

    def step(self, action):
        """
        执行动作
        :param action: 落子位置（行, 列）
        :return: (新状态, 奖励, 是否结束, 额外信息)
        """
        if self.done:
            raise ValueError("请重置环境！")

        row, col = action
        if self.board[row, col] != 0:
            print(ValueError("非法动作：该位置已被占据！"))
            return self._get_state(), 0, self.done, {}

        # 落子并切换玩家
        self.board[row, col] = self.current_player
        if self._check_win(row, col):
            self.done = True
            self.winner = self.current_player
            reward = 1  # 胜利奖励为1
        else:
            if np.all(self.board != 0):
                self.done = True
                self.winner = None  # 平局
                reward = 0
            else:
                self.current_player = 3 - self.current_player
                reward = 0
        self.last_move = (row, col)  # 更新最后落子位置
        return self._get_state(), reward, self.done, {}

    def _get_state(self):
        """返回当前状态（棋盘数组+当前玩家）"""
        return (self.board.copy(), self.current_player, self.last_move)

    def _check_win(self, row, col):
        """
        检查是否胜利（以新落子位置为中心，检查四个方向）
        :param row: 行坐标
        :param col: 列坐标
        :return: 是否胜利
        """
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 水平、垂直、正斜、反斜

        for dr, dc in directions:
            count = 1
            # 向两个方向延伸检查
            for step in [-1, 1]:
                r, c = row, col
                for _ in range(self.win_length - 1):
                    r += dr * step
                    c += dc * step
                    if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                        count += 1
                    else:
                        break
            if count >= self.win_length:
                return True
        return False

    def render(self):
        """打印棋盘（文本可视化）"""
        symbols = {0: '·', 1: '●', 2: '○'}
        print("  " + "".join(f"{i:2d}" for i in range(self.board_size)))
        for i in range(self.board_size):
            row = [symbols[self.board[i, j]] for j in range(self.board_size)]
            print(f"{i:2d} " + " ".join(row))


if __name__ == "__main__":
    env = Gomoku()
    state = env.reset()

    # 简单测试：人类玩家与AI交替落子（AI随机选择）
    while not env.done:
        env.render()
        legal_actions = env.get_legal_actions()
        if env.current_player == 1:
            # 人类玩家输入（示例：输入坐标 7,7）
            action = tuple(map(int, input("输入落子位置（行 列）：").split(',')))
        else:
            # AI随机选择动作
            action = legal_actions[np.random.choice(len(legal_actions))]
            print(f"AI选择动作：{action}")

        state, reward, done, _ = env.step(action)

    env.render()
    print(f"游戏结束，胜利者：玩家{env.winner}")