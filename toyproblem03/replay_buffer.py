import torch
from torch.utils.data import Dataset
from toyproblem03.config import EnvConfig, State, Action, Reward

class ReplayBuffer(Dataset):
    def __init__(self, conf: EnvConfig):
        self.capacity = conf.buffer_capacity
        self.ptr = 0
        self.size = 0
        self.T = conf.T
        self.dev = conf.device

        self.prices = torch.zeros((self.capacity, self.T), dtype=torch.float32, device=self.dev)
        self.positions = torch.zeros(self.capacity, dtype=torch.float32, device=self.dev)
        self.actions = torch.zeros(self.capacity, dtype=torch.long, device=self.dev)
        self.rewards = torch.zeros(self.capacity, dtype=torch.float32, device=self.dev)
        self.next_prices = torch.zeros((self.capacity, self.T), dtype=torch.float32, device=self.dev)
        self.next_positions = torch.zeros(self.capacity, dtype=torch.float32, device=self.dev)

    def push(self, state: State, action: Action, reward: Reward, next_state: State):
        price_seq, pos = state
        next_price_seq, next_pos = next_state
        B = price_seq.shape[0]

        write_end = self.ptr + B
        if write_end <= self.capacity:
            self._write_slice(self.ptr, write_end,
                              price_seq, pos, action, reward, next_price_seq, next_pos)
            self.ptr = write_end % self.capacity
        else:
            first = self.capacity - self.ptr
            second = B - first
            self._write_slice(self.ptr, self.capacity,
                              price_seq[:first], pos[:first],
                              action[:first], reward[:first],
                              next_price_seq[:first], next_pos[:first])
            self._write_slice(0, second,
                              price_seq[first:], pos[first:],
                              action[first:], reward[first:],
                              next_price_seq[first:], next_pos[first:])
            self.ptr = second
        self.size = min(self.size + B, self.capacity)

    def _write_slice(self, start, end, prices, positions, actions, rewards, next_prices, next_positions):
        self.prices[start:end] = prices
        self.positions[start:end] = positions
        self.actions[start:end] = actions
        self.rewards[start:end] = rewards
        self.next_prices[start:end] = next_prices
        self.next_positions[start:end] = next_positions

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        return (self.prices[idx],
                self.positions[idx],
                self.actions[idx],
                self.rewards[idx],
                self.next_prices[idx],
                self.next_positions[idx])
