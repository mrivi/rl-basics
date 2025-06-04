import gymnasium as gym

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, out_dim, hl_dim, lr):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hl_dim),
            nn.ReLU(),
            nn.Linear(hl_dim, hl_dim),
            nn.ReLU(),
            nn.Linear(hl_dim, out_dim)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        action = self.policy(state)
        return action


def expert_policy(state):
    x, x_dot, theta, theta_dot = state
    action = 0 if (theta + 0.1 * theta_dot + 0.01 * x + 0.1 * x_dot) < 0 else 1
    return action


def collect_expert_trajectories(env_name="CartPole-v1", num_episodes=20):
    env = gym.make(env_name)
    trajectories = []

    for _ in range(num_episodes):
        episode = []
        state, _ = env.reset()
        done = False

        while not done:
            action = expert_policy(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action))
            state = new_state
            done = terminated or truncated

        trajectories.append(episode)

    env.close()
    return trajectories

def flatten_trajectories(trajectories):
    states = []
    actions = []

    for episode in trajectories:
        for state, action in episode:
            states.append(state)
            actions.append(action)

    return np.array(states), np.array(actions)

def train_bc_policy(actor, traj, n_epochs=10):

    loss_fn = nn.CrossEntropyLoss()

    states, actions = flatten_trajectories(traj)
    X = torch.tensor(states, dtype=torch.float32).to(actor.device)
    y = torch.tensor(actions, dtype=torch.long).to(actor.device)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    loss_log = []

    for epoch in range(n_epochs):
        n_batch = 0
        for xb, yb in loader:
            n_batch += 1
            logits = actor(xb)
            loss = loss_fn(logits, yb)

            actor.optimizer.zero_grad()
            loss.backward()
            actor.optimizer.step()

            loss_log.append(loss)
        
        print(f"Loss = {loss.item():.4f}")


    return loss_log


def evaluate_policy(env_name, policy_net, episodes=5):
    env = gym.make(env_name, render_mode="human")

    policy_net.eval()

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()
            state_tensor = torch.tensor(state, dtype=torch.float32).to(policy_net.device)
            with torch.no_grad():
                logits = policy_net(state_tensor)
                action = torch.argmax(logits).item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {ep+1}: total reward = {total_reward}")

    env.close()


traj = collect_expert_trajectories()

env = gym.make("CartPole-v1")
num_actions = env.action_space.n
states_space = env.observation_space.shape[0]

print(f"input dim {states_space} out dim {num_actions}")
env.close()

actor = PolicyNetwork(states_space, num_actions, 64, 1e-3)

loss_log = train_bc_policy(actor, traj, 10)

evaluate_policy("CartPole-v1", actor)
