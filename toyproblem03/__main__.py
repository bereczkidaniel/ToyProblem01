import torch
import numpy as np
from toyproblem03.config import EnvConfig
from toyproblem03.environment import Environment
from toyproblem03.agent import Agent
from toyproblem03.replay_buffer import ReplayBuffer
from toyproblem03.trainer import Trainer
from toyproblem03.logger import TrainingLogger
from line_profiler_pycharm import profile

@profile
def evaluate_agent(agent: Agent, env: Environment, n_episodes=20, steps_per_episode=100):
    """
    Evaluate the trained agent with greedy policy (no exploration)
    """
    print(f"\n{'=' * 70}")
    print(f"Evaluating Agent (Greedy Policy)")
    print(f"{'=' * 70}\n")

    # Save original epsilon and set to 0 for greedy evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    agent.online_model.eval()

    episode_rewards = []

    with torch.no_grad():
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0.0

            for step in range(steps_per_episode):
                # Greedy action selection
                action = agent.select_action(state, greedy=True)
                next_state, reward = env.step(action)

                total_reward += reward.mean().item()
                state = next_state

            episode_rewards.append(total_reward)
            print(f"Episode {episode + 1:2d}: Total Reward = {total_reward:8.4f}")

    # Restore original epsilon
    agent.epsilon = original_epsilon
    agent.online_model.train()

    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)

    print(f"\n{'=' * 70}")
    print(f"Evaluation Results")
    print(f"{'=' * 70}")
    print(f"Average Reward: {avg_reward:8.4f} ± {std_reward:.4f}")
    print(f"Min Reward:     {min_reward:8.4f}")
    print(f"Max Reward:     {max_reward:8.4f}")
    print(f"{'=' * 70}\n")

    return episode_rewards

@profile
def main():
    """Main training script"""

    # Configuration
    conf = EnvConfig()

    # Initialize components
    print("Initializing components...")
    env = Environment(conf)
    agent = Agent(conf)
    buffer = ReplayBuffer(conf)
    logger = TrainingLogger(window_size=100)
    trainer = Trainer(conf, env, agent, buffer, logger)

    # Training
    print("\nStarting training...")
    trainer.run(n_cycles=10, verbose=True, log_every=1)

    # Evaluation
    print("\nEvaluating trained agent...")
    eval_rewards = evaluate_agent(agent, env, n_episodes=20, steps_per_episode=100)

    # Save model
    checkpoint = {
        'model_state_dict': agent.online_model.state_dict(),
        'target_state_dict': agent.target_model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'config': conf,
        'eval_rewards': eval_rewards
    }

    save_path = 'trained_agent.pt'
    torch.save(checkpoint, save_path)
    print(f"✓ Model saved to {save_path}")


if __name__ == "__main__":
    main()