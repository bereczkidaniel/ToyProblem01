from dataclasses import dataclass, field
import torch
from torch import Tensor

# ===== Aliases to make type-setting more expressive =====
PriceSeq = Tensor       # shape: (B, T)
Position = Tensor       # shape: (B, )
State = tuple[PriceSeq, Position]
QValues = Tensor        # shape: (B, A)
Action = Tensor         # shape: (B,)
Reward = Tensor         # shape: (B,)

def to_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def action_2_pos(action: torch.Tensor) -> torch.Tensor:
    """ Convert action [0, 1, 2] to position [-1.0, 0.0, 1.0] """
    return (action - 1).to(dtype=torch.float32, device=to_device())


def pos_2_action(pos: torch.Tensor) -> torch.Tensor:
    """ Convert position [-1.0, 0.0, 1.0] to action [0, 1, 2] """
    return pos.to(dtype=torch.long, device=to_device()) + 1

@dataclass
class MarketParams:
    S_min: int = 80
    S_m: int = 100
    S_max: int = 120
    sigma: float = 2.0
    kappa: float = 0.1

@dataclass
class RewardParams:
    beta: float = 0.1          # trading friction coefficient (half-spread * 2, linear)
    gamma_risk: float = 0.01   # variance (risk) penalty coefficient
    discount: float = 0.999    # RL discount for infinite-horizon objective

@dataclass
class NNParams:
    fc1_dims = 128
    fc2_dims = 128
    n_actions = 3

@dataclass
class AgentParams:
    gamma: float = 0.99
    epsilon = 1
    epsilon_min = 0.1
    epsilon_decay = 5e-5
    lr = 3e-4

@dataclass
class TrainerParams:
    num_epochs: int = 50
    minibatch_size: int = 128
    target_update: int = 1



@dataclass
class EnvConfig:
    batch_size: int = 64
    T: int = 10                 # history length for price and position
    buffer_mult: int = 5
    rollout_steps: int = 20
    buffer_capacity = buffer_mult * rollout_steps * batch_size
    market: MarketParams = field(default_factory=MarketParams)
    reward: RewardParams = field(default_factory=RewardParams)
    device: torch.device = field(default_factory=to_device)
    nn: NNParams = field(default_factory=NNParams)
    agent: AgentParams = field(default_factory=AgentParams)
    trainer: TrainerParams = field(default_factory=TrainerParams)

