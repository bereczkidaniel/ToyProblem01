import torch
import torch.nn as nn
import torch.nn.functional as F
from toyproblem03.config import EnvConfig



class DeepQ(nn.Module):
    def __init__(self, conf: EnvConfig):
        super(DeepQ, self).__init__()

        self.input_dims = conf.T + 1
        self.fc1_dims = conf.nn.fc1_dims
        self.fc2_dims = conf.nn.fc2_dims
        self.n_actions = conf.nn.n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)


        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    #as mentioned before, defining a backpropagation is not needed
    #but, we DO need forward propagation (/forward pass)
    def forward(self,state):
        price_seq, pos = state  # prices: [B,T], pos: [B]
        price_seq = price_seq.to(self.device, dtype=torch.float32)  # [B,T]
        pos = pos.to(self.device, dtype=torch.float32)  # [B]

        x = torch.cat([price_seq, pos.unsqueeze(-1)], dim=1)  # [B, T+1]
        x = x.to(dtype=torch.float32, device=self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #in the last layer we do not want to use the activation function:
        #we want the agent's raw estimate;
        #the PV of the future rewards could be negative or could be greater than 1
        #this means we do not want ReLU or Sigmoid, we get rid of the activation altogether
        action = self.fc3(x)
        return action