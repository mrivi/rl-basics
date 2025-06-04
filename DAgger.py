import gymnasium as gym

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import argparse

import ppo_better as ppo

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



def expert_policy(state, env_name, expert_network):

    action = 0
    if env_name == "CartPole-v1":
        action = expert_policy_cartpole(state)
    elif env_name == "LunarLander-v3":
        action = expert_policy_lunarlander(state, expert_network)
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


def collect_expert_trajectories(env, env_name, expert_network, num_episodes=20):
    trajectories = []

    reward_per_episode_log = []
    while len(trajectories) < num_episodes:
    # for _ in range(num_episodes):
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

        # print(f"Expert trajectory episode reward {reward_episode}")
        if reward_episode > 150:
            reward_per_episode_log.append(reward_episode)
            trajectories.append(episode)

    print(f"Collected {len(trajectories)} expert episode. Reward mean {np.mean(reward_per_episode_log):.4f} std {np.std(reward_per_episode_log):.4f}")
    return trajectories

def flatten_trajectories(trajectories):
    states = []
    actions = []

    for episode in trajectories:
        for state, action in episode:
            states.append(state)
            actions.append(action)

    return np.array(states), np.array(actions)


def train(env_name, expert_file):

    env = gym.make(env_name)
    num_actions = env.action_space.n
    states_space = env.observation_space.shape[0]
    
    expert_network = ppo.Agent(
        input_dim   = states_space,
        out_dim     = num_actions)

    expert_network.actor_network.load_state_dict(torch.load(expert_file))
    expert_network.actor_network.eval()

    traj = collect_expert_trajectories(env, env_name, expert_network, num_episodes=100)

    actor = PolicyNetwork(states_space, num_actions, 64, 1e-3)

    n_iterations = 20
    n_episodes = 10

    loss_fn = nn.CrossEntropyLoss()

    states, actions = flatten_trajectories(traj)

    print(f"inital shape states {states.shape} and actions {actions.shape}")
    print(states[0])


    loss_over_all_batches_log = []
    loss_mean_per_iteration_log = []
    loss_std_per_iteration_log = []
    new_policy_episode_mean_reward_log = []
    new_policy_episode_std_reward_log = []

    for iter in range(n_iterations):

        X = torch.tensor(states, dtype=torch.float32).to(actor.device)
        y = torch.tensor(actions, dtype=torch.long).to(actor.device)

        # print(f"X {X.shape} y {y.shape}")

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        loss_per_iteration = []

        for xb, yb in loader:

            logits = actor(xb)
            loss = loss_fn(logits, yb)

            actor.optimizer.zero_grad()
            loss.backward()
            actor.optimizer.step()

            loss_over_all_batches_log.append(loss.detach().cpu())
            loss_per_iteration.append(loss.detach().cpu())
        
        print(f"Iteration: {iter} Loss mean {np.mean(loss_per_iteration):.4f} std {np.std(loss_per_iteration):.4f}")
        loss_mean_per_iteration_log.append(np.mean(loss_per_iteration))
        loss_std_per_iteration_log.append(np.std(loss_per_iteration))


        new_policy_episode_reward = []
        for episode in range(n_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:

                state_tensor = torch.tensor(state, dtype=torch.float32).to(actor.device)

                logits = actor(state_tensor)
                action = torch.argmax(logits).item()

                new_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

                action_expert = expert_policy(state, env_name, expert_network)
                states = np.concatenate((states, state.reshape(1, states_space)), axis=0)
                actions = np.concatenate((actions, np.array([action_expert])), axis=0)
            
                state = new_state
                done = terminated or truncated

            new_policy_episode_reward.append(episode_reward)

        print(f"New policy reward mean {np.mean(new_policy_episode_reward):.4f} std {np.std(new_policy_episode_reward):.4f}")
        new_policy_episode_mean_reward_log.append(np.mean(new_policy_episode_reward))
        new_policy_episode_std_reward_log.append(np.std(new_policy_episode_reward))

    env.close()
    torch.save(actor.state_dict(), "dagger.pt")

    plt.figure(figsize=(15, 10))

    # 1. Loss
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
    plt.title("New Policy Reward per episde")
    plt.grid()
    
    plt.tight_layout()
    plt.savefig("dagger.png")


def test_expert(env_name, expert_file, episodes=5):
    env = gym.make(env_name, render_mode="human")

    num_actions = env.action_space.n
    states_space = env.observation_space.shape[0]

    expert_network = ppo.Agent(
        input_dim   = states_space,
        out_dim     = num_actions)

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
    num_actions = env.action_space.n
    states_space = env.observation_space.shape[0]

    actor = PolicyNetwork(states_space, num_actions, 64, 1e-3)
    actor.load_state_dict(torch.load("dagger.pt"))

    actor.eval()

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()
            state_tensor = torch.tensor(state, dtype=torch.float32).to(actor.device)
            with torch.no_grad():
                logits = actor(state_tensor)
                action = torch.argmax(logits).item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {ep+1}: total reward = {total_reward}")

    env.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PPO algorithm implementation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--train", action="store_true", help="Run training")
    group.add_argument("-i", "--infer", action="store_true", help="Run inference")
    group.add_argument("-e", "--expert", action="store_true", help="Test expert policy")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--expert_file", type=str, default="ppo_seed_789_2025_06_02_23_46.pt", help="Expert model file")

    args = parser.parse_args()

    if args.train:
        print("Running in training mode.")
        train(args.env, args.expert_file)
    elif args.infer:
        print("Running in inference mode.")
        test(args.env)
    elif args.expert:
        print("Tesing expert policy.")
        test_expert(args.env, args.expert_file)
    else:
        print("Please specify a mode with -t (train) or -i (infer).")
