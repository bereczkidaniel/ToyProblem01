import torch
import torch.nn.functional as F
from toyproblem03.config import EnvConfig, State, Action, QValues
from toyproblem03.q_model import DeepQ

class Agent:
    def __init__(self, conf: EnvConfig):
        self.conf = conf
        self.dev = conf.device

        self.A = conf.nn.n_actions
        self.B = conf.batch_size
        self.T = conf.T

        self.gamma = conf.agent.gamma

        self.epsilon = conf.agent.epsilon
        self.eps_min = 0.05
        self.eps_dec = conf.agent.epsilon_decay
        self.lr = conf.agent.lr

        self.online_model = DeepQ(conf)
        self.target_model = DeepQ(conf)
        self.update_target_model()


        self.optimizer = torch.optim.AdamW(
            params=self.online_model.parameters(),
            lr=self.lr,
            weight_decay=1e-3
        )

    def select_action(self, state: State, greedy: bool = False):
        q_values = self.online_model(state)
        if greedy:
            action = q_values.argmax(dim=-1)
        else:
            action = self.soft_action(q_values)
        return action

    @torch.no_grad()
    def soft_action(self, q_values: QValues):

        greedy = torch.argmax(q_values, dim=1)
        B = greedy.shape[0]
        #per-row exploration: some envs explore, others exploit
        explore_mask = torch.rand(B,device=self.dev) < self.epsilon
        random_acts = torch.randint(self.A, (B,), device=self.dev)
        action = torch.where(explore_mask,random_acts,greedy)

        return action

    def train(self, batch):
        prices, positions, actions, rewards, next_prices, next_positions = batch

        with torch.no_grad():
            next_q_online = self.online_model((next_prices, next_positions))
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            # action evaluation: target model
            next_q_target = self.target_model((next_prices, next_positions))
            max_next_q = next_q_target.gather(1, next_actions)
            target = rewards.unsqueeze(1) + self.gamma * max_next_q

        q_values = self.online_model((prices, positions))
        chosen_q = q_values.gather(1, actions.unsqueeze(1))  # shape (B,1)
        loss = F.mse_loss(chosen_q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), 10.0)
        self.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

        metrics = {
            "loss": float(loss.item()),
            "epsilon": float(self.epsilon),
        }
        return metrics


    def update_target_model(self):
        self.target_model.load_state_dict(self.online_model.state_dict())


