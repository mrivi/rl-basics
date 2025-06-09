import torch
from torch import nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import argparse
from datetime import datetime
import random
import time
from datetime import timedelta

class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.buffer = deque(maxlen=int(max_size))
    
    def add(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def clear(self):
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hl1_dim, hl2_dim, out_dim, lr):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hl1_dim),
            nn.Tanh(),
            nn.Linear(hl1_dim, hl2_dim),
            nn.Tanh(),
            nn.Linear(hl2_dim, out_dim),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, state):
        probs = self.actor(state)
        distribution = Categorical(probs)
        return distribution, probs
    
    def act(self, state):
        distribution, probs = self.forward(state)
        action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), probs
    
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hl1_dim, hl2_dim, lr):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hl1_dim),
            nn.Tanh(),
            nn.Linear(hl1_dim, hl2_dim),
            nn.Tanh(),
            nn.Linear(hl2_dim, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

def set_seed(seed=789):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  
class Agent:
    def __init__(self, input_dim, out_dim, gamma=0.99, gae_lambda=0.95, clip=0.1, lr=0.0003,
            n_epochs=10, batch_size=64, entropy_coef=0.01, rollout_length=2048):
    
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef

        set_seed()

        self.actor_network = ActorNetwork(input_dim, 128, 128, out_dim, lr)
        self.critic_network = CriticNetwork(input_dim, 128, 128, lr)

        self.memory = ReplayBuffer(rollout_length)

        self.loss_mse = nn.MSELoss()

    def store_memory(self, transition):
        self.memory.add(transition)

    def clear_memory(self):
        self.memory.clear()

    def generate_batches(self, states):
        indices = torch.randperm(states.size(0))
        batches = [indices[i:i + self.batch_size] for i in range(0, len(states), self.batch_size)]
        return batches

    def compute_advantage(self, state, reward, value, done):
        print(state.size(0))
        advantages = np.zeros(state.size(0), dtype=np.float32)
        for t in range(state.size(0)-1):
            discount = 1
            a_t = 0
            for k in range(t, state.size(0)-1):
                a_t += discount * (reward[k] + (self.gamma * value[k+1]) * (1 - int(done[k])) - value[k])
                discount *= self.gamma * self.gae_lambda
            advantages[t] = a_t
        advantages = torch.tensor(advantages, dtype=torch.float32 ).to(self.actor_network.device)

        return advantages
    
    def compute_adv_opt(self, state, reward, value, done, next_value):
        T = len(reward)
        advantages = torch.zeros(T, dtype=torch.float32, device=self.actor_network.device)

        value_flat = value.squeeze(-1)
        next_value = next_value.view(-1)
        value_extended = torch.cat([value_flat, next_value], dim=0) 

        last_advantage = 0
        for t in reversed(range(T)):
            delta = reward[t] + self.gamma * value_extended[t + 1] * (1 - done[t]) - value_extended[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - done[t]) * last_advantage
            last_advantage = advantages[t]

        return advantages

    
    def train(self, next_value):

        state, action, reward, log_prob, value, done, old_probs = zip(*list(self.memory.buffer) )
        state = torch.stack(state)
        action = torch.stack(action)
        if action.shape != torch.Size([2048]):
            print(f"action shape {action.shape}")
        reward = torch.tensor(reward, dtype=torch.float32).to(self.actor_network.device)
        log_prob = torch.stack(log_prob)
        value = torch.stack(value)
        done = torch.tensor(done, dtype=torch.float32).to(self.actor_network.device)
        old_probs = torch.stack(old_probs)

        advantage = self.compute_adv_opt(state, reward, value, done, next_value)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        returns = advantage.unsqueeze(1) + value #.detach()

        ratio_log = []
        new_value_log = []
        entropy_log = []

        for epoch in range(self.n_epochs):
            batches = self.generate_batches(state)

            for batch in batches:    

                new_prob_distrib, _ = self.actor_network(state[batch])
                new_value = self.critic_network(state[batch])
                new_value_log.append(new_value.detach().cpu())

                old_dist = Categorical(probs=old_probs[batch])
                kl_div_batch = torch.distributions.kl_divergence(old_dist, new_prob_distrib).mean().item()

                if kl_div_batch > 0.01:
                    # print(f"Early stopping at epoch {epoch}, batch due to KL {kl_div_batch:.4f}")
                    break

                ratio = torch.exp(new_prob_distrib.log_prob(action[batch]) - log_prob[batch])
                ratio_log.append(ratio.detach().cpu())
                clipped_ratio = torch.clip(ratio, 1 - self.clip, 1 + self.clip)
                loss_actor = -(torch.min(ratio * advantage[batch], clipped_ratio * advantage[batch])).mean()
                
                kl_div = (log_prob[batch] - new_prob_distrib.log_prob(action[batch])).mean().item()

                loss_critic = self.loss_mse(new_value, returns[batch])

                entropy = new_prob_distrib.entropy().mean()
                entropy_log.append(entropy.detach().cpu())

                loss = loss_actor + 0.5 * loss_critic - self.entropy_coef * entropy

                self.actor_network.optimizer.zero_grad()
                self.critic_network.optimizer.zero_grad()
                loss.backward()
                
                grad_norm_actor = torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 0.5)
                grad_norm_critic = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), 0.5)

                self.actor_network.optimizer.step()
                self.critic_network.optimizer.step()


        return {
            "tot_loss": loss.item(), 
            "loss_actor": loss_actor.item(), 
            "loss_critic": loss_critic.item(), 
            "advantage": advantage.detach().cpu().mean(), 
            "kl_divergence": kl_div,
            "ratio": np.mean(ratio_log),
            "new_value": np.mean(new_value_log), 
            "entropy": np.mean(entropy_log),
            "grad_norm_actor": grad_norm_actor.item(),
            "grad_norm_critic": grad_norm_critic.item()
        }

def train_model():

    env = gym.make('LunarLander-v3')

    num_actions = env.action_space.n
    states_space = env.observation_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_steps = 1000000
    rollout_length = 2048
    initial_lr = 2.5e-4 #3e-4
    clip = 0.25 #0.2

    agent = Agent(
        input_dim   = states_space,
        out_dim     = num_actions,
        gamma       = 0.99,
        gae_lambda  = 0.95,
        clip        = clip,
        lr          = initial_lr,
        n_epochs    = 10, #5,
        batch_size  = 256, #128,
        entropy_coef = 0.01, #0.02,
        rollout_length = rollout_length
    )

    state = env.reset()[0]
    total_steps = 0
    episode_count = 0
    episode_reward = 0
    
    rollout_reward_log = []

    logs = {
        "episode_reward_log": [],
        "tot_loss": [],
        "loss_actor": [],
        "loss_critic": [],
        "advantage": [],
        "kl_divergence" : [],
        "ratio": [],
        "critic_value": [],
        "entropy": [],
        "learning_rate": [],
        "grad_norm_actor_log": [],
        "grad_norm_critic_log": []
    }

    start_time = time.time()

    while total_steps < max_steps:
        rollout_reward = 0
        for _ in range(rollout_length):
        
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            action, log_prob, _ , probs = agent.actor_network.act(state_tensor)

            value = agent.critic_network(state_tensor)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            agent.store_memory((state_tensor, action.detach(), reward, log_prob.detach(), value.detach(), done, probs.detach()))

            episode_reward += reward
            rollout_reward += reward
            state = next_state
            total_steps +=1

            if done:
                logs["episode_reward_log"].append(episode_reward)
                state = env.reset()[0]
                episode_reward = 0
                episode_count += 1
        
        rollout_reward_log.append(rollout_reward)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            next_value = agent.critic_network(state_tensor)
    
        train_log_output = agent.train(next_value)

        logs["tot_loss"].append(train_log_output["tot_loss"])
        logs["loss_actor"].append(train_log_output["loss_actor"])
        logs["loss_critic"].append(train_log_output["loss_critic"])
        logs["advantage"].append(train_log_output["advantage"])
        logs["kl_divergence"].append(train_log_output["kl_divergence"])
        logs["ratio"].append(train_log_output["ratio"])
        logs["critic_value"].append(train_log_output["new_value"])
        logs["entropy"].append(train_log_output["entropy"])
        logs["grad_norm_actor_log"].append(train_log_output["grad_norm_actor"])
        logs["grad_norm_critic_log"].append(train_log_output["grad_norm_critic"])

        if len(logs["episode_reward_log"]) > 0:
            print(f"Steps: {total_steps}, Episode: {episode_count} Reward avg: {np.mean(logs['episode_reward_log'][-10:]):.2f}, Loss {train_log_output['tot_loss']}")

        frac = 1.0 - (total_steps / max_steps)
        lr_now = initial_lr * frac
        logs["learning_rate"].append(lr_now)
        for param_group in agent.actor_network.optimizer.param_groups:
            param_group['lr'] = lr_now
        for param_group in agent.critic_network.optimizer.param_groups:
            param_group['lr'] = lr_now

        agent.clear_memory()

    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training took {elapsed_time:.2f} seconds.")
    print(f"Training time {timedelta(seconds=int(elapsed_time))}")

    env.close()

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    filename = f"ppo_{timestamp}.pt"
    torch.save(agent.actor_network.state_dict(), filename)

    return logs

def test_model(model, n_episodes=5):

    env = gym.make('LunarLander-v3', render_mode="human")

    num_actions = env.action_space.n
    states_space = env.observation_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    initial_lr = 3e-4
    clip = 0.2

    agent = Agent(
        input_dim   = states_space,
        out_dim     = num_actions,
        gamma       = 0.99,
        gae_lambda  = 0.95,
        clip        = clip,
        lr          = initial_lr,
        n_epochs    = 3,
        batch_size  = 128,
        entropy_coef = 0.02,
        rollout_length = 2048
    )

    if isinstance(model, str):
        state_dict = torch.load(model, map_location=device)
        agent.actor_network.load_state_dict(state_dict)
    else:
        agent.actor_network.load_state_dict(model.state_dict())

    agent.actor_network.eval()

    total_eval_reward = 0

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            with torch.no_grad():
                action, _, _, _ = agent.actor_network.act(state_tensor)
            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward
            total_eval_reward += reward
            env.render()

        print(f"Episode {episode} total reward: {total_reward}. Last reward: {reward}")

    env.close()
    
    return total_eval_reward / n_episodes


def plot_debug_info(logs, clip):
    plt.figure(figsize=(15, 10))

    # 1. Losses
    plt.subplot(4, 2, 1)
    plt.plot(logs["tot_loss"], label='Total Loss')
    plt.plot(logs["loss_actor"], label='Actor Loss')
    plt.plot(logs["loss_critic"], label='Critic Loss')
    plt.title("Losses")
    plt.legend()
    plt.grid()

    plt.subplot(4, 2, 2)
    window_size = 100
    rewards = logs["episode_reward_log"]
    plt.plot(rewards, label='Episode Reward')
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, len(rewards)), moving_avg, label=f'{window_size}-ep Moving Avg')
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.legend()
    plt.grid()

    # 3. Critic Value
    plt.subplot(4, 2, 3)
    plt.plot(logs["critic_value"], label='Critic Value')
    plt.title("Critic Value")
    plt.legend()
    plt.grid()

    # 4. Entropy
    plt.subplot(4, 2, 4)
    plt.plot(logs["entropy"], label='Entropy')
    plt.title("Entropy")
    plt.legend()
    plt.grid()

    # 5. Advantage
    plt.subplot(4, 2, 5)
    plt.plot(logs["advantage"], label='Advantage')
    plt.title("Advantage")
    plt.legend()
    plt.grid()

    # 6. Ratio with clipping bounds
    plt.subplot(4, 2, 6)
    ratios = logs["ratio"]
    steps = range(len(ratios))
    plt.plot(steps, ratios, label='Ratio')
    plt.hlines([1 - clip, 1 + clip], xmin=0, xmax=len(ratios), colors='red', linestyles='dashed', label='Clip Bounds')
    plt.title("Ratio")
    plt.legend()
    plt.grid()

    # 7. Gradient Norms
    plt.subplot(4, 2, 7)
    plt.plot(logs["grad_norm_actor_log"], label='Actor')
    plt.plot(logs["grad_norm_critic_log"], label='Critic')
    plt.title("Gradient Norms")
    plt.legend()
    plt.grid()

    # 7. Gradient Norms
    plt.subplot(4, 2, 8)
    plt.plot(logs["kl_divergence"], label='Div')
    plt.title("KL Divergence")
    plt.legend()
    plt.grid()


    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    save_path = f"ppo_graphs_{timestamp}.png"
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PPO algorithm implementation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--train", action="store_true", help="Run training")
    group.add_argument("-i", "--infer", action="store_true", help="Run inference")
    parser.add_argument("--model_file", type=str, default="ppo.pt", help="The model weigths")
    args = parser.parse_args()

    clip = 0.2

    if args.train:
        print("Running in training mode.")
        logs = train_model()
        plot_debug_info(logs, clip)
    elif args.infer:
        print("Running in inference mode.")
        print(f"Loading {args.model_file} model")
        avg_reward = test_model(args.model_file)
        print(f"Average reward during test {avg_reward}")
    else:
        print("Please specify a mode with -t (train) or -i (infer).")
