from torch.utils.data import DataLoader
from toyproblem03.config import EnvConfig
from toyproblem03.environment import Environment
from toyproblem03.agent import Agent
from toyproblem03.replay_buffer import ReplayBuffer
from toyproblem03.logger import TrainingLogger
from line_profiler_pycharm import profile
import torch
#import time
from datetime import datetime


class Trainer:
    def __init__(self, conf: EnvConfig, env: Environment, agent: Agent, buffer: ReplayBuffer, logger: TrainingLogger | None = None):
        self.conf = conf
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.logger = logger or TrainingLogger(window_size=100)

        # Extract config parameters
        self.rollout_steps = conf.rollout_steps
        self.num_epochs = conf.trainer.num_epochs
        self.minibatch_size = conf.trainer.minibatch_size
        self.target_update = conf.trainer.target_update

    # ------------------------------------------------------------
    # Rollout (collect experience)
    # ------------------------------------------------------------
    @profile
    def rollout(self) -> float:
        """Collect experience by rolling out the policy"""
        self.agent.online_model.eval()
        sum_reward = 0.0
        state = self.env.get_state()

        for _ in range(self.rollout_steps):
            # 1) Select action
            action = self.agent.select_action(state)

            # 2) Compute avg_q for logging
            with torch.no_grad():
                if isinstance(state, tuple):
                    st = (state[0].to(self.agent.online_model.device, dtype=torch.float32),
                          state[1].to(self.agent.online_model.device, dtype=torch.float32))
                else:
                    st = state.to(self.agent.online_model.device, dtype=torch.float32)
                q_vals = self.agent.online_model(st)
                avg_q = q_vals.mean().item()

            # 3) Step environment
            next_state, reward = self.env.step(action)

            # 4) Store transition
            self.buffer.push(state, action, reward, next_state)

            # 5) Log step (FIXED: removed duplicate sum_reward)
            self.logger.log_step(
                reward=reward.mean().item(),
                q_value=avg_q,
                action=action,
                epsilon=self.agent.epsilon,
            )

            # 6) Accumulate reward
            sum_reward += reward.mean().item()

            # 7) Update state
            state = next_state

        # Log episode
        self.logger.log_episode(sum_reward, self.rollout_steps)
        return sum_reward / self.rollout_steps
    # ------------------------------------------------------------
    # Training (optimize Q network)
    # ------------------------------------------------------------

    @profile
    def train(self):
        if len(self.buffer) == 0:
            return 0.0
        self.agent.online_model.train()
        self.agent.target_model.eval()

        dataloader = DataLoader(
            self.buffer,
            batch_size=self.minibatch_size,
            shuffle=True
        )

        losses = []
        for _ in range(self.num_epochs):
            for batch in dataloader:
                metrics = self.agent.train(batch)
                losses.append(metrics['loss'])

        return sum(losses) / len(losses) if losses else 0.0

    # ------------------------------------------------------------
    # One full cycle of rollout + training
    # ------------------------------------------------------------
    @profile
    def cycle(self, cycle_idx):
        avg_reward = self.rollout()
        avg_loss = self.train()

        if cycle_idx % self.target_update == 0:
            self.agent.update_target_model()

        return avg_loss, avg_reward

    # ------------------------------------------------------------
    # Train for N cycles
    # ------------------------------------------------------------
    @profile
    def run(self, n_cycles, verbose=True, log_every=10):
        """Train for N cycles"""
        print(f"\n{'=' * 70}")
        print(f"Starting Training")
        print(f"{'=' * 70}")
        print(f"Device: {self.conf.device}")
        print(f"Batch size: {self.conf.batch_size}")
        print(f"Rollout steps: {self.rollout_steps}")
        print(f"Buffer capacity: {self.buffer.capacity}")
        print(f"Total cycles: {n_cycles}")
        print(f"{'=' * 70}\n")

        for i in range(1, n_cycles + 1):
            avg_loss, avg_reward = self.cycle(i)

            if verbose and (i % log_every == 0):
                print(f"[Cycle {i}/{n_cycles}] "
                      f"Buffer: {len(self.buffer)}/{self.buffer.capacity} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Reward: {avg_reward:.4f}")
                self.logger.print_stats(episode=i, steps=self.rollout_steps)

        # Save final plot
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"training_results_{ts}.png"
        self.logger.plot_training(save_path=save_path)

        print(f"\n{'=' * 70}")
        print(f"Training Complete!")
        print(f"{'=' * 70}")
        print(f"Final epsilon: {self.agent.epsilon:.4f}")
        print(f"Buffer size: {len(self.buffer)}/{self.buffer.capacity}")
        print(f"Total steps: {len(self.logger.epsilons)}")
        print(f"{'=' * 70}\n")


