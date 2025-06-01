import torch
from torch import nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

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
        return distribution
    
    def act(self, state):
        distribution = self.forward(state)
        action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy()
    
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

        self.actor_network = ActorNetwork(input_dim, 64, 64, out_dim, lr)
        self.critic_network = CriticNetwork(input_dim, 64, 64, lr)

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

        state, action, reward, log_prob, value, done = zip(*list(self.memory.buffer) )
        state = torch.stack(state)
        action = torch.stack(action)
        if action.shape != torch.Size([2048]):
            print(f"action shape {action.shape}")
        reward = torch.tensor(reward, dtype=torch.float32).to(self.actor_network.device)
        log_prob = torch.stack(log_prob)
        value = torch.stack(value)
        done = torch.tensor(done, dtype=torch.float32).to(self.actor_network.device)

        advantage = self.compute_adv_opt(state, reward, value, done, next_value)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        returns = advantage.unsqueeze(1) + value.detach()

        ratio_log = []
        new_value_log = []
        entropy_log = []

        for epoch in range(self.n_epochs):
            batches = self.generate_batches(state)

            for batch in batches:    

                new_prob_distrib = self.actor_network(state[batch])
                new_value = self.critic_network(state[batch])
                new_value_log.append(new_value.detach().cpu())


                ratio = torch.exp(new_prob_distrib.log_prob(action[batch]) - log_prob[batch])
                ratio_log.append(ratio.detach().cpu())
                clipped_ratio = torch.clip(ratio, 1 - self.clip, 1 + self.clip)
                loss_actor = -(torch.min(ratio, clipped_ratio) * advantage[batch]).mean()
                
                # returns = advantage.unsqueeze(1) + value
                loss_critic = self.loss_mse(new_value, returns[batch])

                entropy = new_prob_distrib.entropy().mean()
                entropy_log.append(entropy.detach().cpu())

                loss = loss_actor + 0.5 * loss_critic - self.entropy_coef * entropy

                self.actor_network.optimizer.zero_grad()
                self.critic_network.optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), 0.5)

                self.actor_network.optimizer.step()
                self.critic_network.optimizer.step()

        

        return {
            "tot_loss": loss.item(), 
            "loss_actor": loss_actor.item(), 
            "loss_critic": loss_critic.item(), 
            "advantage": advantage.detach().cpu().mean(), 
            "ratio": np.mean(ratio_log),
            "new_value": np.mean(new_value_log), 
            "entropy": np.mean(entropy_log)
        }

if __name__ == '__main__':
    # env = gym.make('CartPole-v1')
    env = gym.make('LunarLander-v3')
    num_actions = env.action_space.n
    states_space = env.observation_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_steps = 800000
    rollout_length = 2048
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

    state = env.reset()[0]
    total_steps = 0
    episode_count = 0
    episode_reward = 0
    
    episode_reward_log = []
    rollout_reward_log = []
    tot_loss_log = []
    loss_actor_log = []
    loss_critic_log = []
    advantage_log = []
    critic_value_log = []
    entropy_log = []
    ratio_log = []
    lr_log = []

    while total_steps < max_steps:
        rollout_reward = 0
        for _ in range(rollout_length):
        
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            action, log_prob, _ = agent.actor_network.act(state_tensor)

            value = agent.critic_network(state_tensor)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            agent.store_memory((state_tensor, action.detach(), reward, log_prob.detach(), value.detach(), done))

            episode_reward += reward
            rollout_reward += reward
            state = next_state
            total_steps +=1

            if done:
                episode_reward_log.append(episode_reward)
                state = env.reset()[0]
                episode_reward = 0
                episode_count += 1
        
        rollout_reward_log.append(rollout_reward)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            next_value = agent.critic_network(state_tensor)
    
        train_log_output = agent.train(next_value)

        tot_loss_log.append(train_log_output["tot_loss"])
        loss_actor_log.append(train_log_output["loss_actor"])
        loss_critic_log.append(train_log_output["loss_critic"])
        advantage_log.append(train_log_output["advantage"])
        ratio_log.append(train_log_output["ratio"])
        critic_value_log.append(train_log_output["new_value"])
        entropy_log.append(train_log_output["entropy"])

        if len(episode_reward_log) > 0:
            print(f"Steps: {total_steps}, Episode: {episode_count} Reward avg: {np.mean(episode_reward_log[-10:]):.2f}, Loss {train_log_output['tot_loss']}")

        frac = 1.0 - (total_steps / max_steps)
        lr_now = initial_lr * frac
        lr_log.append(lr_now)
        for param_group in agent.actor_network.optimizer.param_groups:
            param_group['lr'] = lr_now
        for param_group in agent.critic_network.optimizer.param_groups:
            param_group['lr'] = lr_now

        agent.clear_memory()

    env.close()
    torch.save(agent.actor_network.state_dict(), "ppo_lr_batchs.pt")

    plt.figure(figsize=(15, 10))

    # 1. Loss
    plt.subplot(3, 2, 1)
    plt.plot(tot_loss_log, label='Total Loss')
    plt.plot(loss_actor_log, label='Actor Loss')
    plt.plot(loss_critic_log, label='Critic Loss')
    plt.legend()
    plt.title("Losses")
    plt.grid()

    # 2. Episode Reward
    window_size = 100
    plt.subplot(3, 2, 2)
    plt.plot(episode_reward_log, label='Episode Reward')
    # plt.plot(rollout_reward_log, label='Rollout Reard')
    moving_avg = np.convolve(episode_reward_log, np.ones(window_size)/window_size, mode='valid')
    plt.plot(moving_avg, label='Smoothed Episode Reward')
    plt.legend()
    plt.title("Reward")
    plt.xlabel("Episode")
    plt.grid()

    # 3. Value
    plt.subplot(3, 2, 3)
    plt.plot(critic_value_log, label='Value')
    plt.legend()
    plt.title("Critic Value")
    plt.grid()

    # 4. Value
    plt.subplot(3, 2, 4)
    plt.plot(entropy_log, label='Entropy')
    plt.legend()
    plt.title("Entropy")
    plt.grid()

    # 4. Advantage
    plt.subplot(3, 2, 5)
    plt.plot(advantage_log, label='Advantage')
    plt.legend()
    plt.title("Advantage")
    plt.grid()

    # 4. Ratio
    plt.subplot(3, 2, 6)
    lower_bound = (1 - clip) * np.ones(len(ratio_log))
    upper_bound = (1 + clip) * np.ones(len(ratio_log))

    plt.plot(ratio_log, label='Ratio')
    plt.plot(lower_bound, label='Lower Clip')
    plt.plot(upper_bound, label='Upper Clip')
    plt.plot()
    plt.legend()
    plt.title("Ratio")
    plt.grid()



    plt.tight_layout()
    plt.savefig("ppo_lr_batches.png")
