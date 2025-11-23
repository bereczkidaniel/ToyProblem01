import torch
from toyproblem03.config import EnvConfig, action_2_pos, Action

class Environment:
    def __init__(self, conf: EnvConfig):
        #getting data from the object
        self.conf =conf
        self.dev =conf.device
        self.B = conf.batch_size
        self.T = conf.T
        self.marketparams = conf.market
        self.rewardparams = conf.reward

        #setting up (empty) price and position tensors; also data
        self.price_seq = torch.zeros(self.B, self.T, dtype=torch.int32, device=self.dev)
        self.pos = torch.zeros(self.B, dtype=torch.int32, device=self.dev)



    # defines how to price process evolves
    def _price_step(self, current_price: torch.Tensor):
        noise = torch.randn(self.B, device=self.dev)
        next_price = current_price + self.marketparams.sigma * noise - self.marketparams.kappa * (current_price - self.marketparams.S_m)
        next_price = torch.round(next_price).clamp_(self.marketparams.S_min, self.marketparams.S_max)
        return next_price

    def _initial_price(self):
        #start with the mean value for all batches
        price_seq = torch.full((self.B,self.T), self.marketparams.S_m, dtype=torch.float32, device=self.dev)


        #generate a T long history for all batches that the Agent can initially see
        for t in range(self.T - 1):
            price_seq[:, t+1] = self._price_step(price_seq[:, t])
        return price_seq


    def get_state(self):
        return (
            self.price_seq.detach().clone(),
            self.pos.detach().clone()
        )

    #env to initial state
    def reset(self):
        self.price_seq = self._initial_price()
        self.pos = torch.zeros(self.B, dtype=torch.int32, device=self.dev)
        return self.get_state()


    # defines the reward based on the action taken
    @torch.no_grad()
    def step(self, action: Action):
        """
        Advance the environment one time-step given an action.
        Args:
            action: long Tensor (B,)
        Returns:
            (new_price_seq, new_pos_seq): float32 tensors (B, T)
            reward: float32 tensor (B,)
        """
        action = action.to(self.dev, dtype=torch.long)
        new_pos = action_2_pos(action)

        new_price = self._price_step(self.price_seq[:,-1])
        d_price = new_price - self.price_seq[:,-1]

        pnl = new_pos * d_price
        friction = self.rewardparams.beta * (new_pos - self.pos).abs()
        vol_pen = self.rewardparams.gamma_risk * (self.marketparams.sigma ** 2) * (new_pos ** 2)
        reward = pnl - friction - vol_pen


        #roll windows
        # === Compute *functional* new state = (new_price_seq, new_pos)
        new_price_seq = torch.roll(self.price_seq, shifts=-1, dims=1)
        new_price_seq[:, -1] = new_price

        # new_pos = self.pos
        # new_pos_seq[:, -1] = new_pos
        # === in-place commit
        self.price_seq.copy_(new_price_seq)
        self.pos.copy_(new_pos)
        # === return safe, detached output tensors
        state = new_price_seq.detach(), new_pos.detach()
        return state, reward.detach()





