import gymnasium as gym

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import argparse

class DQNConfig:
    
    lr: float = 5e-4
    hidden_dim: int = 64
    batch_size: int = 256
    network_sync_rate: int = 1000
    replay_memory_size: int = 10000
    gamma: float = 0.99
    n_training_episodes: int = 150000
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay_episodes = 70000
    
class ActorNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hl_dim: int, lr: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hl_dim),
            nn.ReLU(),
            nn.Linear(hl_dim, hl_dim),
            nn.ReLU(),
            nn.Linear(hl_dim, output_dim)
        )

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.network(state)
        return logits
    
class ReplayMemory:
    def __init__(self, max_size=1e6):
        self.buffer = deque(maxlen=int(max_size))
    
    def add(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)

class DQN:
    def __init__(self, config: DQNConfig, env_name: str = "FrozenLake-v1", 
                 render_mode: str = "human"):
        self.config = config
        self.env_name = env_name

        self.env = gym.make(env_name, render_mode=render_mode)
        
        if hasattr(self.env.observation_space, 'n'):
            self.state_dim = self.env.observation_space.n
        else:
            self.state_dim = self.env.observation_space.shape[0]  # Continuous (MountainCar)

        self.action_dim = self.env.action_space.n

        print(f"State dim {self.state_dim}, action dim {self.action_dim}")

        self.memory = ReplayMemory(self.config.replay_memory_size)

        self.policy = ActorNetwork(self.state_dim, self.action_dim, self.config.hidden_dim, self.config.lr)
        self.target = ActorNetwork(self.state_dim, self.action_dim, self.config.hidden_dim, self.config.lr)
        self.target.load_state_dict(self.policy.state_dict())
        for param in self.target.parameters():
            param.requires_grad = False


    def train(self):

        epsilon = self.config.epsilon_start
        step_count = 0
        loss_log = []
        epsilon_log = []
        reward_per_episode_log = []
        q_value_log = []
        q_value_target_log = []
        grad_norm_log = []

        for n_episode in range(config.n_training_episodes):
            state, _ = self.env.reset()
            done = False
            reward_per_episode = 0

            while not done:

                if random.random() < epsilon:                    
                    action = self.env.action_space.sample()
                else:
                    state_tensor = torch.tensor([state]).to(self.policy.device)
                    with torch.no_grad():
                        actions = self.policy(state_tensor)
                    action = actions.argmax().item()

                
                new_state, reward, terminated, truncated,_ = self.env.step(action)
                if "MountainCar-v0" in self.env_name:
                    reward = reward + 10 * abs(new_state[1]) + (new_state[0] if new_state[0] > 0.25 else 0)
                reward_per_episode += reward
                done = terminated or truncated

                transition = (state, action, reward, new_state, done)
                self.memory.add(transition)

                state = new_state
                step_count += 1

            reward_per_episode_log.append(reward_per_episode)
            if n_episode % 100 == 0:
                print(f"Episode {n_episode} Reward {np.mean(reward_per_episode_log[-100:])} +- {np.std(reward_per_episode_log[-100:])}")

            if len(reward_per_episode_log) > 1000:
                recent_avg = np.mean(reward_per_episode_log[-100:])
                if recent_avg > -110:
                    print(f"Converged at episode {n_episode}")
                    break
          
            if len(self.memory) > self.config.batch_size:

                mini_batch = self.memory.sample(self.config.batch_size)
                loss, q_value, q_target_value = self.optimize(mini_batch, self.policy, self.target)
                loss_log.append(loss)
                q_value_log.append(q_value)
                q_value_target_log.append(q_target_value)

                epsilon = max(self.config.epsilon_start - (self.config.epsilon_start - self.config.epsilon_end) * n_episode / self.config.epsilon_decay_episodes, self.config.epsilon_end)
                epsilon_log.append(epsilon)

                if step_count % config.network_sync_rate == 0:
                    self.target.load_state_dict(self.policy.state_dict())

                grad_norm = sum(p.grad.norm().item() for p in self.policy.parameters() if p.grad is not None)
                grad_norm_log.append(grad_norm)


        self.env.close()
        torch.save(self.policy.state_dict(), "dql.pt")

        plt.figure(figsize=(15, 10))

        # 1. Loss
        plt.subplot(3, 2, 1)
        plt.plot(loss_log, label='Loss')
        window_size = 10
        moving_avg = np.convolve(loss_log, np.ones(window_size)/window_size, mode='valid')
        plt.plot(moving_avg, label='Smooth Loss')
        plt.legend()
        plt.title("Losses")
        plt.grid()

        plt.subplot(3, 2, 2)
        plt.plot(reward_per_episode_log, label='Reward')
        moving_avg = np.convolve(reward_per_episode_log, np.ones(window_size)/window_size, mode='valid')
        plt.plot(moving_avg, label='Smooth Reward')
        plt.legend()
        plt.title("Reward per Episode")
        plt.grid()

        plt.subplot(3, 2, 3)
        plt.plot(epsilon_log, label='Epsilon')
        plt.legend()
        plt.title("Epsilon")
        plt.grid()

        plt.subplot(3, 2, 4)
        plt.plot(q_value_log, label='Q Policy')
        plt.plot(q_value_target_log, label='Q Target')
        plt.legend()
        plt.title("Q Values")
        plt.grid()

        plt.subplot(3, 2, 5)
        plt.plot(grad_norm_log, label='Norm')
        plt.legend()
        plt.title("Gradient Norms")
        plt.grid()

        plt.savefig("log_dql.png")

    def optimize(self, mini_batch, policy_net, target_net):

        state, action, reward, new_state, done = zip(*mini_batch)

        state = torch.tensor(state, dtype=torch.float32).to(self.policy.device)

        action = torch.tensor(action, dtype=torch.float32).to(self.policy.device)
        new_state = torch.tensor(new_state, dtype=torch.float32).to(self.policy.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.policy.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.policy.device)

        q_target = reward + (torch.ones(self.config.batch_size).to(self.policy.device) - done) * self.config.gamma * target_net(new_state).max(1)[0].to(dtype=torch.float32)
        q_policy = policy_net(state).gather(1, action.long().unsqueeze(1)).squeeze(1)
        
        avg_q = q_policy.mean().item()
        avg_q_target = q_target.mean().item()

        loss = F.smooth_l1_loss(q_policy, q_target)

        self.policy.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)

        self.policy.optimizer.step()

        return loss.item(), avg_q, avg_q_target

    def test(self, n_test_episodes: int = 5, model_file: str = "dql.pt"):
        self.policy.load_state_dict(torch.load(model_file))
        self.policy.eval()

        for episode in range(n_test_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                state_tensor = torch.tensor([state]).to(self.policy.device)

                with torch.no_grad():
                    action = self.policy(state_tensor).argmax().item()
    
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

            print(f"Episode {episode} reward: {episode_reward:.4f}")

        self.env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DQN algorithm implementation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--train", action="store_true", help="Run training")
    group.add_argument("-i", "--infer", action="store_true", help="Run inference")
    parser.add_argument("--model_file", type=str, default="dql.pt", help="The model weigths")
    parser.add_argument("--env_name", type=str, default="MountainCar-v0", help="The environment name")

    args = parser.parse_args()

    config = DQNConfig()

    if args.train:
        print("Running in training mode.")
        dqn = DQN(config, args.env_name, None)
        dqn.train()
    elif args.infer:
        print("Running in inference mode.")
        print(f"Loading {args.model_file} model")
        dqn = DQN(config, args.env_name, "human")
        dqn.test(5, args.model_file)
    else:
        print("Please specify a mode with -t (train) or -i (infer).")
