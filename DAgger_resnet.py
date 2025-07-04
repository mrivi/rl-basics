import gymnasium as gym

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import gc

import ppo as ppo
from stable_baselines3 import PPO


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

class PolicyNetworkCarRacingLight(nn.Module):
    def __init__(self, out_dim, lr):
        super().__init__()
        
        # Use ResNet-18 without pretrained weights, modify first conv layer
        self.backbone = models.resnet18(weights=None)
        
        # Modify first conv layer to accept 96x96 input instead of 224x224
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass for 96x96 images
        
        Args:
            state: Image tensor of shape (batch, 3, 96, 96) or (3, 96, 96)
        """
        # Handle single image
        if len(state.shape) == 3:
            state = state.unsqueeze(0)

        state = state.permute(0, 3, 1, 2)  # Change to (3, 96, 96)

        # Normalize to [0,1] if needed
        if state.max() > 1.0:
            state = state / 255.0
            
        # Pass through ResNet (no resizing needed)
        action_logits = self.backbone(state)
        
        return action_logits


class MemoryEfficientDataset:
    """Custom dataset that keeps data on CPU and only moves batches to GPU"""
    def __init__(self, states, actions, max_size=None):
        # Keep everything on CPU
        self.states = np.array(states)
        self.actions = np.array(actions)
        
        # Limit dataset size if specified
        if max_size and len(self.states) > max_size:
            indices = np.random.choice(len(self.states), max_size, replace=False)
            self.states = self.states[indices]
            self.actions = self.actions[indices]
    
    def __len__(self):
        return len(self.states)
    
    def get_batch(self, indices, device):
        """Get a batch and move to device"""
        batch_states = torch.tensor(self.states[indices], dtype=torch.float32).to(device)
        batch_actions = torch.tensor(self.actions[indices], dtype=torch.long).to(device)
        return batch_states, batch_actions
    
    def add_data(self, new_states, new_actions, max_total_size=50000):
        """Add new data and optionally limit total size"""
        self.states = np.concatenate([self.states, new_states])
        self.actions = np.concatenate([self.actions, new_actions])
        
        # Keep only most recent data if too large
        if len(self.states) > max_total_size:
            keep_size = max_total_size
            self.states = self.states[-keep_size:]
            self.actions = self.actions[-keep_size:]


def expert_policy(state, env_name, expert_network):
    action = 0
    if env_name == "CartPole-v1":
        action = expert_policy_cartpole(state)
    elif env_name == "LunarLander-v3":
        action = expert_policy_lunarlander(state, expert_network)
    elif env_name == "CarRacing-v3":
        action = expert_policy_continuoscar(state, expert_network)
    else:
        print("Error, no known expert policy")
        return -1

    return action

def expert_policy_cartpole(state):
    x, x_dot, theta, theta_dot = state
    action = 0 if (theta + 0.1 * theta_dot + 0.01 * x + 0.1 * x_dot) < 0 else 1
    return action

def expert_policy_lunarlander(state, expert_network):    
    state_tensor = torch.tensor(state, dtype=torch.float32).to(expert_network.actor_network.device)
    with torch.no_grad():
        action = expert_network.actor_network(state_tensor)
    
    return action.sample().item()

def expert_policy_continuoscar(state, expert_network):
    state_tensor = torch.tensor(state, dtype=torch.float32).to(expert_network.device)

    with torch.no_grad():
        action, _ = expert_network.predict(state) 

    return action


def collect_expert_trajectories(env, env_name, expert_network, num_episodes=20):
    trajectories = []
    reward_per_episode_log = []
    
    while len(trajectories) < num_episodes:
        print(f"collecting expert trajectories episode {len(trajectories)}")
        episode = []
        state, _ = env.reset()
        done = False
        reward_episode = 0

        while not done:
            action = expert_policy(state, env_name, expert_network)
            new_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action))
            state = new_state
            done = terminated or truncated
            reward_episode += reward

        if reward_episode > 150:
            reward_per_episode_log.append(reward_episode)
            trajectories.append(episode)

    print(f"Collected {len(trajectories)} expert episode. Reward mean {np.mean(reward_per_episode_log):.4f} std {np.std(reward_per_episode_log):.4f}")
    return trajectories

def save_trajectories_npz(trajectories, save_path="expert_trajectories.npz"):
    states = []
    actions = []
    episode_lengths = []

    for episode in trajectories:
        episode_lengths.append(len(episode))
        for state, action in episode:
            states.append(state)
            actions.append(action)

    np.savez_compressed(save_path,
                        states=np.array(states),
                        actions=np.array(actions),
                        episode_lengths=np.array(episode_lengths))
    print(f"Saved {len(trajectories)} expert episodes to {save_path}")

def load_trajectories_npz(load_path="expert_trajectories.npz"):
    """Load expert trajectories from npz file"""
    if not os.path.exists(load_path):
        print(f"File {load_path} not found!")
        return None, None
    
    data = np.load(load_path)
    states = data['states']
    actions = data['actions']
    episode_lengths = data['episode_lengths']
    
    print(f"Loaded {len(states)} state-action pairs from {len(episode_lengths)} episodes")
    print(f"Episode lengths: min={min(episode_lengths)}, max={max(episode_lengths)}, mean={np.mean(episode_lengths):.1f}")
    
    return states, actions


def train(env_name, expert_file, load_trajectories=True, trajectories_path="expert_trajectories.npz"):
    env = gym.make(env_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Clear GPU cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if env_name == "CarRacing-v3":
        num_actions = env.action_space.shape[0]
        actor = PolicyNetworkCarRacingLight(num_actions, 1e-4)
        loss_fn = nn.MSELoss()
        
        ppo_2m_path = "/home/martina/src/Training/Saved Models/PPO_Driving_Model"
        expert_network = PPO.load(ppo_2m_path, device=device)
        print("loaded expert model")
    else:
        num_actions = env.action_space.n
        states_space = env.observation_space.shape[0]
        actor = PolicyNetwork(states_space, num_actions, 64, 1e-3)
        loss_fn = nn.CrossEntropyLoss()

        expert_network = ppo.Agent(
            input_dim=states_space,
            out_dim=num_actions)

        expert_network.actor_network.load_state_dict(torch.load(expert_file))
        expert_network.actor_network.eval()

    # Load or collect expert trajectories
    if load_trajectories and os.path.exists(trajectories_path):
        print(f"Loading existing trajectories from {trajectories_path}")
        states, actions = load_trajectories_npz(trajectories_path)
        if states is None:
            print("Failed to load trajectories, collecting new ones...")
            traj = collect_expert_trajectories(env, env_name, expert_network, num_episodes=50)
            save_trajectories_npz(traj, trajectories_path)
            states, actions = flatten_trajectories(traj)
    else:
        print("Collecting new expert trajectories...")
        traj = collect_expert_trajectories(env, env_name, expert_network, num_episodes=50)
        save_trajectories_npz(traj, trajectories_path)
        states, actions = flatten_trajectories(traj)

    # Initialize memory-efficient dataset
    dataset = MemoryEfficientDataset(states, actions, max_size=30000)  # Limit initial size
    
    print(f"Initial dataset size: {len(dataset)} samples")

    n_iterations = 20
    n_episodes = 10
    batch_size = 16  # Reduced batch size

    loss_over_all_batches_log = []
    loss_mean_per_iteration_log = []
    loss_std_per_iteration_log = []
    new_policy_episode_mean_reward_log = []
    new_policy_episode_std_reward_log = []

    for iter in range(n_iterations):
        print(f"Starting iteration {iter}")
        
        # Clear GPU cache before each iteration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Training phase - process data in batches
        actor.train()
        loss_per_iteration = []
        
        # Create batches manually to control memory usage
        dataset_size = len(dataset)
        indices = np.random.permutation(dataset_size)
        
        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i + batch_size]
            
            # Get batch on GPU
            if env_name == "CarRacing-v3":
                xb, yb = dataset.get_batch(batch_indices, device)
                yb = yb.float()  # For MSE loss
            else:
                xb, yb = dataset.get_batch(batch_indices, device)

            logits = actor(xb)
            loss = loss_fn(logits, yb)

            actor.optimizer.zero_grad()
            loss.backward()
            actor.optimizer.step()

            loss_val = loss.detach().cpu().item()
            loss_over_all_batches_log.append(loss_val)
            loss_per_iteration.append(loss_val)
            
            # Clear intermediate tensors
            del xb, yb, logits, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Iteration: {iter} Loss mean {np.mean(loss_per_iteration):.4f} std {np.std(loss_per_iteration):.4f}")
        loss_mean_per_iteration_log.append(np.mean(loss_per_iteration))
        loss_std_per_iteration_log.append(np.std(loss_per_iteration))

        # Evaluation phase
        actor.eval()
        new_policy_episode_reward = []
        new_states_list = []
        new_actions_list = []

        for episode in range(n_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Get action from current policy
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)

                    if env_name == "CarRacing-v3":
                        if len(state_tensor.shape) == 3:
                            state_tensor = state_tensor.unsqueeze(0)
                        action_values = actor(state_tensor)
                        action = action_values.squeeze().cpu().numpy()
                    else:
                        logits = actor(state_tensor)
                        action = torch.argmax(logits).item()

                new_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

                # Get expert action for this state
                action_expert = expert_policy(state, env_name, expert_network)
                
                # Store new data (keep on CPU)
                new_states_list.append(state.copy())
                new_actions_list.append(action_expert)
                
                state = new_state
                done = terminated or truncated

            new_policy_episode_reward.append(episode_reward)

        # Add new data to dataset (with size limiting)
        if new_states_list:
            dataset.add_data(np.array(new_states_list), np.array(new_actions_list), max_total_size=50000)
            print(f"Dataset size after adding new data: {len(dataset)}")

        print(f"New policy reward mean {np.mean(new_policy_episode_reward):.4f} std {np.std(new_policy_episode_reward):.4f}")
        new_policy_episode_mean_reward_log.append(np.mean(new_policy_episode_reward))
        new_policy_episode_std_reward_log.append(np.std(new_policy_episode_reward))

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    env.close()
    torch.save(actor.state_dict(), "dagger_resnet2.pt")

    # Plotting code remains the same
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(loss_over_all_batches_log, label='Loss')
    window_size = 10
    moving_avg = np.convolve(loss_over_all_batches_log, np.ones(window_size)/window_size, mode='valid')
    plt.plot(moving_avg, label='Smooth Loss')
    plt.legend()
    plt.title("Losses")
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(loss_mean_per_iteration_log, label='Mean')
    plt.plot(loss_std_per_iteration_log, label='Std')
    plt.legend()
    plt.title("Loss per iteration")
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(new_policy_episode_mean_reward_log, label='Mean')
    plt.plot(new_policy_episode_std_reward_log, label='Std')
    plt.legend()
    plt.title("New Policy Reward per episode")
    plt.grid()
    
    plt.tight_layout()
    plt.savefig("dagger_reset2.png")


def flatten_trajectories(trajectories):
    states = []
    actions = []

    for episode in trajectories:
        for state, action in episode:
            states.append(state)
            actions.append(action)

    return np.array(states), np.array(actions)


def test_expert(env_name, expert_file, episodes=5):
    env = gym.make(env_name, render_mode="human")

    if env_name == "CarRacing-v3":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ppo_2m_path = "/home/martina/src/Training/Saved Models/PPO_Driving_Model"
        expert_network = PPO.load(ppo_2m_path, device=device)
    else:
        num_actions = env.action_space.n
        states_space = env.observation_space.shape[0]

        expert_network = ppo.Agent(
            input_dim=states_space,
            out_dim=num_actions)

        expert_network.actor_network.load_state_dict(torch.load(expert_file))
        expert_network.actor_network.eval()

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()
            action = expert_policy(state, env_name, expert_network)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {ep+1}: total reward = {total_reward}")

    env.close()


def test(env_name, episodes=5):
    env = gym.make(env_name, render_mode="human")

    if env_name == "CarRacing-v3":
        num_actions = env.action_space.shape[0]
        actor = PolicyNetworkCarRacingLight(num_actions, 1e-4)
        
    else:
        num_actions = env.action_space.n
        states_space = env.observation_space.shape[0]
        actor = PolicyNetwork(states_space, num_actions, 64, 1e-3)

    actor.load_state_dict(torch.load("dagger_resnet2.pt"))
    actor.eval()

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()
            state_tensor = torch.tensor(state, dtype=torch.float32).to(actor.device)
            with torch.no_grad():
                if env_name == "CarRacing-v3":
                    if len(state_tensor.shape) == 3:
                        state_tensor = state_tensor.unsqueeze(0)
                    action_values = actor(state_tensor)
                    action = action_values.squeeze().cpu().numpy()
                else:
                    logits = actor(state_tensor)
                    action = torch.argmax(logits).item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {ep+1}: total reward = {total_reward}")

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DAgger algorithm implementation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--train", action="store_true", help="Run training")
    group.add_argument("-i", "--infer", action="store_true", help="Run inference")
    group.add_argument("-e", "--expert", action="store_true", help="Test expert policy")
    parser.add_argument("--env", type=str, default="CarRacing-v3", help="Environment name")
    parser.add_argument("--expert_file", type=str, default="ppo_seed_789_2025_06_02_23_46.pt", help="Expert model file")
    parser.add_argument("--trajectories_path", type=str, default="expert_trajectories.npz", help="Path to saved expert trajectories")
    parser.add_argument("--collect_new", action="store_true", help="Collect new trajectories instead of loading existing ones")

    args = parser.parse_args()

    if args.train:
        print("Running in training mode.")
        load_existing = not args.collect_new
        train(args.env, args.expert_file, load_trajectories=load_existing, trajectories_path=args.trajectories_path)
    elif args.infer:
        print("Running in inference mode.")
        test(args.env)
    elif args.expert:
        print("Testing expert policy.")
        print(f"Environment {args.env}")
        test_expert(args.env, args.expert_file)
    else:
        print("Please specify a mode with -t (train) or -i (infer).")