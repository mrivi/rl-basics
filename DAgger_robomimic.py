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
import h5py

import ppo as ppo
from stable_baselines3 import PPO

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
import h5py

import expert_oracle as eo

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, hl1_dim: int, hl2_dim: int, lr: float):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hl1_dim),
            nn.ReLU(),
            nn.Linear(hl1_dim, hl1_dim),
            nn.ReLU(),
            nn.Linear(hl1_dim, hl2_dim),
            nn.ReLU(),
            nn.Linear(hl2_dim, out_dim)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        action = self.policy(state)
        return action 


class MemoryEfficientDataset:
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

def load_expert_dataset(expert_trajectories_files : str) -> tuple:

    dataset = {}
    with h5py.File(expert_trajectories_files, 'r') as f:
        # Load observations
        obs_keys = list(f['data/demo_0/obs'].keys())
        dataset['observations'] = []
        dataset['actions'] = []
        dataset['done'] = []


        # Iterate through all demonstrations
        for demo_key in f['data'].keys():
            demo = f['data'][demo_key]
            
            # Load observations for this demo and concatenate them
            demo_obs_list = []
            for obs_key in sorted(obs_keys):
                obs_data = demo['obs'][obs_key][:]
                demo_obs_list.append(obs_data)
            
            # Concatenate all observations for this demo along the last axis
            demo_obs_concatenated = np.concatenate(demo_obs_list, axis=-1)
            dataset['observations'].append(demo_obs_concatenated)
            
            # Load actions for this demo
            actions = demo['actions'][:]
            dataset['actions'].append(actions)
            
            # Create done tensor for this demo (True at the end of trajectory)
            demo_length = len(actions)
            done = np.zeros(demo_length, dtype=bool)
            done[-1] = True  # Mark last step as done
            dataset['done'].append(done)

    # Concatenate all demonstrations
    dataset['observations'] = np.concatenate(dataset['observations'], axis=0)
    dataset['actions'] = np.concatenate(dataset['actions'], axis=0)
    dataset['done'] = np.concatenate(dataset['done'], axis=0)
    
    # Convert to tensors
    observations_tensor = torch.tensor(dataset['observations'], dtype=torch.float32)
    actions_tensor = torch.tensor(dataset['actions'], dtype=torch.float32)
    done_tensor = torch.tensor(dataset['done'], dtype=torch.bool)
    
    return observations_tensor, actions_tensor, done_tensor

def get_expert_action(obs, expert_obs, expert_actions, k=5):
    """Get expert action using k-nearest neighbors and averaging"""
    flatten_obs = flatten_observation(obs)
    # Find k nearest neighbors
    distances = np.linalg.norm(expert_obs - flatten_obs, axis=1)
    k_nearest_idx = np.argpartition(distances, k)[:k]
    
    # Weight actions by inverse distance
    weights = 1 / (distances[k_nearest_idx] + 1e-8)  # avoid division by zero
    weights = weights / np.sum(weights)  # normalize weights

    
    # Weighted average of k nearest actions
    weighted_action = np.sum(expert_actions[k_nearest_idx].cpu().numpy() * weights[:, None], axis=0)
    return weighted_action

def flatten_observation(obs: dict):
    if isinstance(obs, dict):
        # Handle dictionary observations
        flat_obs = []
        for key in sorted(obs.keys()):
            # Filter out robot0_joint_acc key
            if key == 'robot0_joint_acc':
                continue
            if 'image' in key:
                # For images, you might want to use features from your ResNet
                # or flatten the image
                flat_obs.append(obs[key].flatten())
            else:
                flat_obs.append(obs[key].flatten())
        return np.concatenate(flat_obs)
    else:
        return obs.flatten()

def train(expert_trajectories_file : str):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = config_factory(algo_name="bc")
    ObsUtils.initialize_obs_utils_with_config(config)

    env_meta = FileUtils.get_env_metadata_from_dataset(expert_trajectories_file)
    env_meta['env_kwargs']['ignore_done'] = False
    env_meta['env_kwargs']['reward_shaping'] = True

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta['env_name'], 
        render=False, 
        render_offscreen=False,
        use_image_obs=False, 
    )

    states, actions, dones = load_expert_dataset(expert_trajectories_file)
    print(f"Expert dataset size: {states.shape[0]} {actions.shape[0]}")
    num_actions = actions.shape[1]
    num_states = states.shape[1]
    print(f"Number of states: {num_states}, number of actions: {num_actions}")
    actor = PolicyNetwork(num_states, num_actions, 512, 256, 1e-4)
    loss_fn = nn.MSELoss()

    # Initialize memory-efficient dataset
    dataset = MemoryEfficientDataset(states, actions, max_size=30000)  # Limit initial size
    print(f"Initial dataset size: {len(dataset)} samples")

    oracle = eo.setup_expert_oracle(states, actions)

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

            xb, yb = dataset.get_batch(batch_indices, device)
            yb = yb.float()  # For MSE loss

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
            state = env.reset()
            done = False
            episode_reward = 0
            episode_step_count = 0
            success = False

            while not done:
                # Get action from current policy
                with torch.no_grad():
                    state_tensor = torch.tensor(flatten_observation(state), dtype=torch.float32).to(device)

                    action_values = actor(state_tensor)
                    action = action_values.squeeze().cpu().numpy()

                new_state, reward, terminated, _ = env.step(action)
                if reward >= 1.0:
                    reward = 100.0
                    success = True
                episode_reward += reward

                # Get expert action for this state
                action_expert, _ = oracle.get_expert_action(state)
                
                # Store new data (keep on CPU)
                new_states_list.append(flatten_observation(state).copy())
                new_actions_list.append(action_expert)
                
                state = new_state
                episode_step_count += 1

                truncated = (episode_step_count > 200)
                done = terminated or success or truncated

            print(f"Episode {episode} terminated after {episode_step_count} steps")

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


def test_expert(expert_trajectories_file: str, n_episodes: int = 5):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = config_factory(algo_name="bc")
    ObsUtils.initialize_obs_utils_with_config(config)

    env_meta = FileUtils.get_env_metadata_from_dataset(expert_trajectories_file)
    env_meta['env_kwargs']['ignore_done'] = False
    env_meta['env_kwargs']['reward_shaping'] = True

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta['env_name'], 
        render=True, 
        render_offscreen=False,
        use_image_obs=False, 
    )


    states, actions, dones = load_expert_dataset(expert_trajectories_file)
    print(f"Expert dataset size: {states.shape[0]} {actions.shape[0]}")

    oracle = eo.setup_expert_oracle(states, actions)

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        success = False
        total_reward = 0
        episode_step_count = 0

        while not done:
            env.render()
            action, _ = oracle.get_expert_action(state)
            state, reward, terminated, _ = env.step(action)
            
            episode_step_count += 1
            truncated = (episode_step_count > 200)
            if reward >= 1.0:
                success = True
                reward = 100.0
            total_reward += reward

            if terminated or success or truncated:
                done = True

        print(f"Episode {ep+1}: total reward = {total_reward}")



def test(model_file: str, expert_trajectories_file: str, n_episodes: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = config_factory(algo_name="bc")
    ObsUtils.initialize_obs_utils_with_config(config)

    env_meta = FileUtils.get_env_metadata_from_dataset(expert_trajectories_file)
    env_meta['env_kwargs']['ignore_done'] = False
    env_meta['env_kwargs']['reward_shaping'] = True

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta['env_name'], 
        render=True, 
        render_offscreen=False,
        use_image_obs=False, 
    )

    states, actions, dones = load_expert_dataset(expert_trajectories_file)
    print(f"Expert dataset size: {states.shape[0]} {actions.shape[0]}")
    num_actions = actions.shape[1]
    num_states = states.shape[1]
    print(f"Number of states: {num_states}, number of actions: {num_actions}")
    actor = PolicyNetwork(num_states, num_actions, 512, 256, 1e-4)

    actor.load_state_dict(torch.load(model_file))
    actor.eval()

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()
            state_tensor = torch.tensor(state, dtype=torch.float32).to(actor.device)
            with torch.no_grad():
                action_values = actor(state_tensor)
                action = action_values.squeeze().cpu().numpy()

            state, reward, done, _ = env.step(action)
            total_reward += reward

        print(f"Episode {ep+1}: total reward = {total_reward}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DAgger algorithm implementation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--train", action="store_true", help="Run training")
    group.add_argument("-i", "--infer", action="store_true", help="Run inference")
    group.add_argument("-e", "--expert", action="store_true", help="Test expert policy")
    parser.add_argument("--env", type=str, default="CarRacing-v3", help="Environment name")
    parser.add_argument("--expert_file", type=str, default="/home/martina/src/robomimic/data/lift/ph/low_dim_v15.hdf5", help="Expert trajectories")
    parser.add_argument("--model_file", type=str, default="dagger_resnet2.pt", help="Model file")

    args = parser.parse_args()

    if args.train:
        print("Running in training mode.")
        train(args.expert_file)
    elif args.infer:
        print("Running in inference mode.")
        test(args.model_file, args.expert_file)
    elif args.expert:
        print("Testing expert policy.")
        print(f"Environment {args.env}")
        test_expert(args.expert_file)
    else:
        print("Please specify a mode with -t (train) or -i (infer).")