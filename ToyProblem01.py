import torch
#for layers
import torch.nn as nn
#for ReLU
import torch.nn.functional as F
#for adam
import torch.optim as optim
import numpy as np
from dataclasses import dataclass, field


def to_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
class EnvConfig:
    batch_size: int = 256
    T: int = 10                 # history length for price and position
    market: MarketParams = field(default_factory=MarketParams)
    reward: RewardParams = field(default_factory=RewardParams)
    device: torch.device = field(default_factory=to_device)


#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------Environment---------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

class Environment:
    def __init__(self, cfg: EnvConfig):
        #getting data from the object
        self.cfg =cfg
        self.dev =cfg.device
        self.B = cfg.batch_size
        self.T = cfg.T
        self.marketparams = cfg.market
        self.rewardparams = cfg.reward

        #mapping
        self._idx2val = torch.tensor([-1, 0, 1], dtype=torch.int32, device=self.dev)

        #setting up (empty) price and position tensors; also data
        self.price = torch.zeros(self.B, self.T, dtype=torch.int32, device=self.dev)
        self.pos = torch.zeros(self.B, self.T, dtype=torch.int32, device=self.dev)

        #generates a T long price process of the given Environment for all Batches
        self._initial_state()

    def _initial_state(self):
        #start with the mean value for all batches
        S = torch.full((self.B,), self.marketparams.S_m, dtype=torch.float32, device=self.dev)
        history = []

        #generate a T long history for all batches that the Agent can initially see
        for _ in range(self.T):
            eps = torch.randn(self.B, device=self.dev)
            S = S + self.marketparams.sigma * eps - self.marketparams.kappa * (S - self.marketparams.S_m)
            S = torch.round(S).clamp_(self.marketparams.S_min, self.marketparams.S_max)
            history.append(S.clone())
        P = torch.stack(history, dim=1) # [B,T] floats, összecsatol tensorokat egy adott dim mentén

        #price history up to T
        self.price = P.to(torch.int32)

        #no positions taken yet, so its all zero
        self.pos.zero_()  # start flat

        self._update_state()
        return self.x

    def _update_state(self):
        #standardizing the price process
        z_price = (self.price - float(self.marketparams.S_m)) / (float(self.marketparams.sigma))
        z_pos= self.pos
        self.x = torch.cat([z_price, z_pos], dim = 1).contiguous()



    # defines how to price process evolves
    def _price_step(self, prev_price: torch.Tensor):
        eps = torch.randn(self.B, device=self.dev)
        S = prev_price + self.marketparams.sigma * eps - self.marketparams.kappa * (prev_price - self.marketparams.S_m)
        S = torch.round(S).clamp_(self.marketparams.S_min, self.marketparams.S_max)
        return S


    # defines the reward based on the action taken
    def step(self, action_indices):
        if isinstance(action_indices, np.ndarray):
            action_indices = torch.from_numpy(action_indices)
        action_indices = action_indices.to(self.dev, dtype=torch.long)
        a_t = self._idx2val[action_indices]

        prev_price = self.price[:, -1].to(torch.float32)
        prev_action = self.pos[:, -1].to(torch.float32)

        next_price = self._price_step(prev_price)
        dS = next_price - prev_price



        #reward calculation
        pnl = a_t * dS
        friction = self.rewardparams.beta * (a_t - prev_action).abs()
        vol_pen = self.rewardparams.gamma_risk * (self.marketparams.sigma ** 2) * (a_t ** 2)
        reward = pnl - friction - vol_pen


        #roll windows
        self.price = torch.cat([self.price[:, 1:], next_price[:,None]], dim = 1)
        self.pos = torch.cat([self.pos[:, 1:], a_t.unsqueeze(1)], dim = 1)

        #update state
        self._update_state()

        done = torch.zeros(self.B, dtype=torch.bool ,device=self.dev)

        return self.x, reward, done, {}

    #env to initial state
    def reset(self):
        return self._initial_state()





#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------NeuralNetwork-------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------



#in practice one should have a Targer Network and a Replay Buffer

#the Replay Buffer is a FIFO memory that stores (s(t),a(t),r(t),s(t+1),done(t))
#and lets you sample minibatches for learning (1;breaks correlation 2;Improves sample effciency 3;Stabilizes Learning).

#the Target Network is a second Q-network that lags behind the online Q-network.
#without a Target Network, your targets depend on the same rapidly changing parameters you’re updating.
#This “chasing a moving target” can cause divergence.
#a slowly updated target makes the Bellman target more stationary.

#this model has only Replay Buffer

class DeepQ(nn.Module):
    # inheriting from nn.Modul gives access to the parameters for the optimization
    # as well as the backpropagation function
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,n_actions):
        #calls the constructor of the base class
        super(DeepQ, self).__init__()
        #save the needed variables
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # a DQL network is an estimator of the value of each action given some set of states
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        #its kinda like a lin. reg. : fitting a line to the Delta between the target value and output of the NN
        self.loss = nn.MSELoss()
        #later for GPU use
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    #as mentioned before, defining a backpropagation is not needed
    #but, we DO need forward propagation (/forward pass)
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        #in the last layer we do not want to use the activation function:
        #we want the agent's raw estimate;
        #the PV of the future rewards could be negative or could be greater than 1
        #this means we do not want ReLU or Sigmoid, we get rid of the activation altogether
        action = self.fc3(x)
        return action




#-----------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------Agent----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


#the main functionality lives in the Agent class
class Agent:
    #defining the constructor
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size = 100000,
                 eps_end = 0.01, eps_decay = 5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        #by what to decrement Epsilon with each time step
        #it can be any kind of dependence, i.e. a multiplicative.
        #in this case its linear: we subtract a constant factor with each time step
        #until it reaches eps_end (as it should still explore but to a lesser extent)
        self.eps_dec = eps_decay
        self.lr = lr
        #interger representation of the available actions
        #we use this later at the epsilon-greedy action selection
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.input_dims = input_dims
        self.n_actions = n_actions

        #evaluation network
        self.Q_eval = DeepQ(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)

        #mechanism for storing memory/Memory buffers
        #we save them as named arrays but there are several working options to consider

        self.state_memory = np.zeros((self.mem_size, self.input_dims),dtype = np.float32)
        #the agent want to know for the temporal difference update rule is:
        # 1; the value of the current state
        # 2; the value of the next state
        # 3; the reward recieved
        # 4; the action it took
        #to get that we have to pass in a memory of the states that resulted from its actions
        #Because DQL is a model-free, bootsrapped, off-policy learning method:
            #1; We dont need to know anything of the dynamics of the environment
            #   (the Agent figures it out by interacting with it)
            #2; We are going to constuct estimates of action value functions:
            #   estimating the value of each action given the Agent is in some state is based on earlier estimates
            #   the Agent is using one estimate to update another (its pulling itself up by the bootsraps)
            #3; Using a policy that used to generate actions which are epsilon-greedy to generate data for updating the purely greedy policy
            #   (the agent's estimate of the action value function)
        self.new_state_memory = np.zeros((self.mem_size, self.input_dims),dtype = np.float32)
        #(DQL does not work for continuous actions spaced, but we have a discrete one)
        self.action_memory = np.zeros(self.mem_size, dtype = np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
        #the value of a terminal state is always 0
        #because if we encounter the terminal state the game is done
        #there are no future actions until you reset the episode but when you do that you are in the initial state not the terminal state.
        #The future value of the terminal state is identically 0; we need some way of capturing that when we tell the agent to update its estimate of the Q-value (action value function Q)
        #this is facilitated through the terminal memory: we will be passing in the done flags from the Environment, that is why it is a boolian
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.float32)

    #interface function to store the transition in the Agent's memory
    #the Done parameter is a flag
    #(filling up the previously defined arrays?)
    def store_transition(self, state, action, reward, next_state, done):
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().numpy()
        if isinstance(done, torch.Tensor):
            done = done.cpu().numpy().astype(np.float32)
        #the position of the first unoccupied memory
        #using the modulus here has the property that this will wrap around:
        # the 100.000th memory we want to store will go all the way back to position 0
        #so we rewrite the agents earliest memories with new memories
        B = action.shape[0]
        index = (np.arange(self.mem_cntr, self.mem_cntr + B) % self.mem_size)
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        #increment the memory counter to signal that we filled up a memory
        self.mem_cntr += B

#function for choosing an action
    def choose_action(self, observation):

        if isinstance(observation, np.ndarray):
            state = torch.tensor(observation, dtype=torch.float32, device=self.Q_eval.device)
        else:
            state = observation.to(self.Q_eval.device, dtype=torch.float32)

        with torch.no_grad():
            q = self.Q_eval.forward(state)
            greedy = torch.argmax(q, dim=1)

        B = greedy.shape[0]
        #per-row exploration: some envs explore, others exploit
        explore_mask = torch.rand(B,device=self.Q_eval.device) < self.epsilon
        n_action = getattr(self, "n_actions", len(self.action_space))
        random_acts = torch.randint(n_action, (B,), device=self.Q_eval.device)
        acts = torch.where(explore_mask,random_acts,greedy)

        return acts.detach().cpu().numpy()



    #learning from experience
    #dilemma: the memory is filled up with 0s which we cannot use for learning (useless)
    #   1; Let the agent act randomly until it fills up its memory, then it can start learning
    #   2; Start learning as soon as the agent filled up the batch size of the memory
    #We are using the second method

    #we will call this function at every iteration of our Environment
    def learn(self):
        #if the memory counter is less than the batch size just return; there is no point in learning
        #for previously discussed purposes we want to use the Replay Buffer framework so we choose past experiences randomly
        #the arrays of past memories are preallocated with zeros
        if self.mem_cntr < self.batch_size:
            return

        #if we do learn we first zero the gradient of the optimizer
        #(if i understand correctly this is unique in Pytorch, you do not have to do it Keras or TF)
        self.Q_eval.optimizer.zero_grad()
        dev = self.Q_eval.device

        #calculate the position of the maximum memory
        #we need a subset of the available memory, and we only want to select up to the last filled memory
        #this can be achieved my taking the min. of the memory counter and memory size
        max_mem = min(self.mem_size, self.mem_cntr)
        #we want the replace to be False because we do not want to select the same memory more than once
        #only matters if the SGD minibatch is small
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        #Getting ordered indices from 0 to batch_size (size of SGD minibatch)
        batch_index = np.arange(self.batch_size,dtype=np.int32)

        #converting the Numpy array subset of the Agent's memory into a Pytorch tensor
        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = torch.tensor(self.action_memory[batch]).to(self.Q_eval.device)


        #performing the forward pass through the NN to get the parameters for calculating the loss
        #we want to be moving the Agent's estimate for the value of the current state towards the maximal value for the next state
        #in other words: tilt it towards selecting maximal actions

        #to do that we have to do the dereferencing (array slicing) because we want to got the values of the actions
        #the Agent actually took
        #we can not update the values of actions the agent did not take (we did not sample those)
        #so we want the actions we took in each set of our memory batch
        #q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        #we want the Agent's estimate of the next state as well (we do not need to do any dereferencing here, we are going to get the max. actions)
        #if we would use a Target Network, this is where we would use it
        #q_next = self.Q_eval.forward(new_state_batch)
        #the values of the terminal states are identically 0
        #q_next[terminal_batch] = 0.0

        #this is where we want to update our estimates towards
        #the max. function returns the values as well as the index (its a tuple), thats why we need [0]
        #q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0] #* terminal_batch

        q_values = self.Q_eval.forward(state_batch)  # [B, A]
        row_idx = torch.arange(self.batch_size, dtype=torch.long, device=dev)  # FIXED: torch indices
        q_eval = q_values[row_idx, action_batch]

        with torch.no_grad():
            q_next = self.Q_eval.forward(new_state_batch)
            q_next_max = torch.max(q_next, dim=1)[0]
            q_target = reward_batch + self.gamma * (1.0 - terminal_batch) * q_next_max

        loss = self.Q_eval.loss(q_eval, q_target)
        #back propagating the loss
        loss.backward()
        #step the optimizer
        self.Q_eval.optimizer.step()

        #each time we learn we decrease the epsilon
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min







#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------Logger----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------



from collections import deque
import time
from datetime import datetime
import matplotlib.pyplot as plt

class TrainingLogger:
    """Track and log training metrics"""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.q_values = []
        self.epsilons = []
        self.actions_taken = {0: 0, 1: 0, 2: 0}  # -1, 0, +1 positions

        # Rolling windows for smoothed metrics
        self.reward_window = deque(maxlen=window_size)
        self.q_window = deque(maxlen=window_size)

    def log_step(self, reward, q_value, action, epsilon):
        """Log single step metrics"""
        self.reward_window.append(reward)
        self.q_window.append(q_value)
        self.epsilons.append(epsilon)
        self.actions_taken[action] += 1

    def log_episode(self, total_reward, length):
        """Log episode-level metrics"""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(length)

    def get_stats(self):
        """Get current statistics"""
        return {
            'avg_reward': np.mean(self.reward_window) if self.reward_window else 0,
            'avg_q': np.mean(self.q_window) if self.q_window else 0,
            'epsilon': self.epsilons[-1] if self.epsilons else 1.0,
            'total_steps': len(self.epsilons),
            'action_dist': self.actions_taken.copy()
        }

    def print_stats(self, episode, steps):
        """Pretty print current statistics"""
        stats = self.get_stats()
        action_total = sum(stats['action_dist'].values())
        action_pct = {k: v / action_total * 100 if action_total > 0 else 0
                      for k, v in stats['action_dist'].items()}

        print(f"\n{'=' * 70}")
        print(f"Episode {episode} | Steps: {steps} | Total Steps: {stats['total_steps']}")
        print(f"{'=' * 70}")
        print(f"Avg Reward (last {self.window_size}): {stats['avg_reward']:>8.4f}")
        print(f"Avg Q-value            : {stats['avg_q']:>8.4f}")
        print(f"Epsilon                : {stats['epsilon']:>8.4f}")
        print(f"Actions: Sell={action_pct[0]:.1f}% | Hold={action_pct[1]:.1f}% | Buy={action_pct[2]:.1f}%")
        print(f"{'=' * 70}\n")

    def plot_training(self, save_path=None):
        """Create training visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
        if len(self.episode_rewards) > self.window_size:
            smoothed = np.convolve(self.episode_rewards,
                                   np.ones(self.window_size) / self.window_size,
                                   mode='valid')
            axes[0, 0].plot(smoothed, label=f'{self.window_size}-episode MA', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Q-values
        if self.q_values:
            axes[0, 1].plot(self.q_values, alpha=0.5)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Avg Q-value')
            axes[0, 1].set_title('Q-value Estimates')
            axes[0, 1].grid(True, alpha=0.3)

        # Epsilon decay
        axes[1, 0].plot(self.epsilons)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].set_title('Exploration Rate')
        axes[1, 0].grid(True, alpha=0.3)

        # Action distribution
        actions = list(self.actions_taken.keys())
        counts = list(self.actions_taken.values())
        action_labels = ['Sell (-1)', 'Hold (0)', 'Buy (+1)']
        axes[1, 1].bar(action_labels, counts)
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Action Distribution')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training plot saved to {save_path}")

        plt.show()



#-----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------Deployment---------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------



def train_dqn_agent(
        n_episodes=1000,
        steps_per_episode=200,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=5e-5,
        history_length=10,
        log_interval=10,
        save_interval=100,
        model_save_path='checkpoints/dqn_trader.pt'
):
    """
    Main training loop for DQN trading agent

    Args:
        n_episodes: Number of episodes to train
        steps_per_episode: Steps per episode
        batch_size: Batch size for environment (parallel envs)
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Epsilon decay rate per step
        history_length: Length of price/position history
        log_interval: Print stats every N episodes
        save_interval: Save model every N episodes
        model_save_path: Path to save model checkpoints
    """

    # Initialize environment
    print("Initializing environment...")
    cfg = EnvConfig(
        batch_size=batch_size,
        T=history_length
    )
    env = Environment(cfg)

    # Initialize agent
    print("Initializing agent...")
    input_dims = 2 * history_length  # price + position history
    agent = Agent(
        gamma=gamma,
        epsilon=epsilon_start,
        lr=learning_rate,
        input_dims=input_dims,
        batch_size=128,  # SGD minibatch size (different from env batch_size)
        n_actions=3,
        max_mem_size=100000,
        eps_end=epsilon_end,
        eps_decay=epsilon_decay
    )

    # Initialize logger
    logger = TrainingLogger(window_size=100)

    # Training info
    print(f"\n{'=' * 70}")
    print(f"Training Configuration")
    print(f"{'=' * 70}")
    print(f"Device: {env.dev}")
    print(f"Environment batches: {batch_size}")
    print(f"History length: {history_length}")
    print(f"Input dimensions: {input_dims}")
    print(f"Episodes: {n_episodes}")
    print(f"Steps per episode: {steps_per_episode}")
    print(f"Total steps: {n_episodes * steps_per_episode}")
    print(f"Learning rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Epsilon: {epsilon_start} → {epsilon_end} (decay: {epsilon_decay})")
    print(f"{'=' * 70}\n")

    start_time = time.time()
    total_steps = 0

    try:
        for episode in range(1, n_episodes + 1):
            # Reset environment
            state = env.reset()
            episode_reward = 0
            episode_q_values = []

            for step in range(steps_per_episode):
                # Choose action
                actions = agent.choose_action(state)

                # Get Q-values for logging
                with torch.no_grad():
                    state_tensor = state.to(agent.Q_eval.device, dtype=torch.float32)
                    q_vals = agent.Q_eval(state_tensor)
                    avg_q = q_vals.mean().item()
                    episode_q_values.append(avg_q)

                # Environment step
                next_state, rewards, dones, _ = env.step(actions)

                # Store transitions
                agent.store_transition(state, actions, rewards, next_state, dones)

                # Learn
                agent.learn()

                # Log metrics (use first environment's data for logging)
                avg_reward = rewards.mean().item()
                logger.log_step(avg_reward, avg_q, actions[0], agent.epsilon)

                episode_reward += avg_reward
                state = next_state
                total_steps += 1

            # Log episode
            logger.log_episode(episode_reward, steps_per_episode)
            logger.q_values.extend(episode_q_values)

            # Print statistics
            if episode % log_interval == 0:
                logger.print_stats(episode, steps_per_episode)

                # Estimate time remaining
                elapsed = time.time() - start_time
                steps_per_sec = total_steps / elapsed
                remaining_steps = (n_episodes - episode) * steps_per_episode
                eta_seconds = remaining_steps / steps_per_sec
                eta_mins = eta_seconds / 60
                print(f"Speed: {steps_per_sec:.1f} steps/sec | ETA: {eta_mins:.1f} minutes\n")

            # Save checkpoint
            if episode % save_interval == 0:
                checkpoint = {
                    'episode': episode,
                    'model_state_dict': agent.Q_eval.state_dict(),
                    'optimizer_state_dict': agent.Q_eval.optimizer.state_dict(),
                    'epsilon': agent.epsilon,
                    'mem_cntr': agent.mem_cntr,
                    'episode_rewards': logger.episode_rewards,
                }
                torch.save(checkpoint, model_save_path)
                print(f"✓ Checkpoint saved to {model_save_path}\n")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    # Training complete
    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Training Complete!")
    print(f"{'=' * 70}")
    print(f"Total time: {total_time / 60:.2f} minutes")
    print(f"Total steps: {total_steps}")
    print(f"Average reward (last 100): {np.mean(logger.episode_rewards[-100:]):.4f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"{'=' * 70}\n")

    # Plot results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"training_results_{timestamp}.png"
    logger.plot_training(save_path=plot_path)

    return agent, logger


def evaluate_agent(agent, env, n_episodes=10):
    """
    Evaluate trained agent performance

    Args:
        agent: Trained Agent instance
        env: Environment instance
        n_episodes: Number of episodes to evaluate
    """
    print(f"\n{'=' * 70}")
    print(f"Evaluating Agent (Greedy Policy)")
    print(f"{'=' * 70}\n")

    # Store original epsilon and set to 0 (pure exploitation)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    episode_rewards = []
    episode_positions = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        positions = []

        for step in range(100):  # Fixed evaluation length
            actions = agent.choose_action(state)
            next_state, rewards, dones, _ = env.step(actions)

            total_reward += rewards.mean().item()
            positions.append(actions[0])  # Track first env's positions

            state = next_state

        episode_rewards.append(total_reward)
        episode_positions.append(positions)
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.4f}")

    # Restore original epsilon
    agent.epsilon = original_epsilon

    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"\n{'=' * 70}")
    print(f"Evaluation Results")
    print(f"{'=' * 70}")
    print(f"Average Reward: {avg_reward:.4f} ± {std_reward:.4f}")
    print(f"Min Reward: {np.min(episode_rewards):.4f}")
    print(f"Max Reward: {np.max(episode_rewards):.4f}")
    print(f"{'=' * 70}\n")

    return episode_rewards, episode_positions


if __name__ == "__main__":
    # Train the agent
    agent, logger = train_dqn_agent(
        n_episodes=500,
        steps_per_episode=200,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=1e-5,
        history_length=10,
        log_interval=10,
        save_interval=50
    )

    # Evaluate the trained agent
    cfg = EnvConfig(batch_size=256, T=10)
    eval_env = Environment(cfg)
    eval_rewards, eval_positions = evaluate_agent(agent, eval_env, n_episodes=20)

