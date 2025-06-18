import torch
import torch.nn.functional as F
import types

from agent import *
from utils import *
from config import default_DQN_config_dict

class QNetwork(torch.nn.Module):
    """ 只有一层隐藏层的Q网络 """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)

@ModelFactory.register('DQN')
class DQN(Algorithm):
    def __init__(self, state_dim, action_dim, config:dict=None):
        super().__init__(state_dim, action_dim, config)

        self.config_dict = default_DQN_config_dict
        if config:
            self.config_dict.update(config)
        self.config = types.SimpleNamespace(**self.config_dict)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_net = QNetwork(self.state_dim, self.config.hidden_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.config.hidden_dim, self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.config.lr)

        self.update_count = 0

    def select_action(self, state, epsilon=None, eval_mode=False):
        if epsilon is None:
            epsilon = self.config.epsilon
        if eval_mode or np.random.rand() > epsilon:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_net(state_tensor).argmax().item()
        else:
            return np.random.randint(self.action_dim)

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        # DQN特有的更新逻辑
        current_q = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.config.gamma * next_q

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 定期更新目标网络
        if self.update_count % self.config.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.update_count += 1

if __name__ == '__main__':
    agent = DQN(128, 128)
    model_summary(agent.q_net)
    print('*****')
    model_summary(agent.target_net)