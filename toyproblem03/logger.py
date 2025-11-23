from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import torch


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
        # Convert to float if tensor
        if isinstance(reward, torch.Tensor):
            reward = float(reward.item() if reward.numel() == 1 else reward.mean().item())
        if isinstance(q_value, torch.Tensor):
            q_value = float(q_value.item() if q_value.numel() == 1 else q_value.mean().item())
        if isinstance(epsilon, torch.Tensor):
            epsilon = float(epsilon.item())

        self.reward_window.append(float(reward))
        self.q_window.append(float(q_value))
        self.q_values.append(float(q_value))
        self.epsilons.append(float(epsilon))

        # Handle action counting - action can be tensor or numpy array
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        elif not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.int64)

        # Count actions (ensure we handle batch of actions)
        if action.ndim == 0:  # Single action
            action = np.array([action])

        counts = np.bincount(action.flatten(), minlength=3)
        for i in range(3):
            self.actions_taken[i] += int(counts[i])

    def log_episode(self, total_reward, length):
        """Log episode-level metrics"""
        if isinstance(total_reward, torch.Tensor):
            total_reward = float(total_reward.item() if total_reward.numel() == 1 else total_reward.mean().item())
        if isinstance(length, torch.Tensor):
            length = int(length.item())

        self.episode_rewards.append(float(total_reward))
        self.episode_lengths.append(int(length))

    def get_stats(self):
        """Get current statistics"""
        action_total = sum(self.actions_taken.values())
        action_pct = {k: (v / action_total * 100.0 if action_total > 0 else 0.0)
                      for k, v in self.actions_taken.items()}

        return {
            'avg_reward': np.mean(self.reward_window) if self.reward_window else 0,
            'avg_q': np.mean(self.q_window) if self.q_window else 0,
            'epsilon': self.epsilons[-1] if self.epsilons else 1.0,
            'total_steps': len(self.epsilons),
            'action_dist': self.actions_taken.copy(),
            'action_pct': action_pct
        }

    def print_stats(self, episode, steps):
        """Pretty print current statistics"""
        stats = self.get_stats()

        print(f"\n{'=' * 70}")
        print(f"Episode {episode} | Steps: {steps} | Total Steps: {stats['total_steps']}")
        print(f"{'=' * 70}")
        print(f"Avg Reward (last {self.window_size}): {stats['avg_reward']:>8.4f}")
        print(f"Avg Q-value            : {stats['avg_q']:>8.4f}")
        print(f"Epsilon                : {stats['epsilon']:>8.4f}")
        ap = stats["action_pct"]
        print(f"Actions %: Sell={ap[0]:5.1f}% | Hold={ap[1]:5.1f}% | Buy={ap[2]:5.1f}%")
        print(f"{'=' * 70}\n")

    def plot_training(self, save_path=None):
        """Create training visualization"""
        # Create 2x2 subplot (4 panels total)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Episode Rewards (top-left)
        ax = axes[0, 0]
        if self.episode_rewards:
            ax.plot(self.episode_rewards, alpha=0.4, label="Episode reward", color='blue')
            if len(self.episode_rewards) > self.window_size:
                kernel = np.ones(self.window_size) / self.window_size
                ma = np.convolve(self.episode_rewards, kernel, mode="valid")
                x_ma = range(self.window_size - 1, self.window_size - 1 + len(ma))
                ax.plot(x_ma, ma, lw=2, label=f"{self.window_size}-ep MA", color='red')
            ax.set_title("Episode Rewards", fontsize=12, fontweight='bold')
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data yet', ha='center', va='center', transform=ax.transAxes)

        # Plot 2: Q-values (top-right)
        ax = axes[0, 1]
        if self.q_values:
            ax.plot(self.q_values, alpha=0.7, color='green')
            ax.set_title("Average Q-values per Step", fontsize=12, fontweight='bold')
            ax.set_xlabel("Step")
            ax.set_ylabel("Avg Q")
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data yet', ha='center', va='center', transform=ax.transAxes)

        # Plot 3: Epsilon decay (bottom-left)
        ax = axes[1, 0]
        if self.epsilons:
            ax.plot(self.epsilons, color='orange')
            ax.set_title("Epsilon Decay", fontsize=12, fontweight='bold')
            ax.set_xlabel("Step")
            ax.set_ylabel("Epsilon")
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data yet', ha='center', va='center', transform=ax.transAxes)

        # Plot 4: Action distribution (bottom-right)
        ax = axes[1, 1]
        actions = ['Sell (-1)', 'Hold (0)', 'Buy (+1)']
        counts = [self.actions_taken.get(i, 0) for i in range(3)]
        colors = ['red', 'gray', 'green']
        bars = ax.bar(actions, counts, color=colors, alpha=0.7, edgecolor='black')

        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=10)

        ax.set_title("Action Distribution", fontsize=12, fontweight='bold')
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Training plot saved to {save_path}")

        plt.show()