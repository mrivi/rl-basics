import torch
from torch import nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import gymnasium as gym
import copy
from collections import deque
import matplotlib.pyplot as plt

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.buffer = deque(maxlen=int(max_size))
    
    def add(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hl_dim, out_dim, lr):
        super().__init__()
        self.actor = nn.Sequential(
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
        out = self.actor(state)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp() 

        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z) #action selected by adding noise for exploration

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True) 

        return action, log_prob, torch.tanh(mean) #best action for eval
    
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hl_dim, lr):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hl_dim),
            nn.ReLU(),
            nn.Linear(hl_dim, hl_dim),
            nn.ReLU(),
            nn.Linear(hl_dim, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, state_action):
        q_value = self.critic(state_action) 
        return q_value
    
class Agent:
    def __init__(self, input_dim, out_dim, learning_rate=3e-4, batch_size=256, gamma=0.99, alpha=0.2, tau=0.005):
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.memory = ReplayBuffer(1e6)

        self.actor = ActorNetwork(input_dim, 256, out_dim * 2, learning_rate)
        self.critic1 = CriticNetwork((input_dim + out_dim), 256, learning_rate)
        self.critic2 = CriticNetwork((input_dim + out_dim), 256, learning_rate)
        self.target1 = CriticNetwork((input_dim + out_dim), 256, learning_rate)
        self.target1.load_state_dict(copy.deepcopy(self.critic1.state_dict()))
        self.target2 = CriticNetwork((input_dim + out_dim), 256, learning_rate)
        self.target2.load_state_dict(copy.deepcopy(self.critic2.state_dict()))

        for param in self.target1.parameters():
            param.requires_grad = False
        for param in self.target2.parameters():
            param.requires_grad = False


        self.target_entropy = -float(out_dim)
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, 
                                        requires_grad=True, device=self.actor.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        self.alpha = self.log_alpha.exp()

        self.mse_loss = nn.MSELoss()

    def store_transition(self, transition):
        self.memory.add(transition)

    def generate_batches(self, states):
        indices = torch.randperm(states.size(0))
        batches = [indices[i:i + self.batch_size] for i in range(0, len(states), self.batch_size)]
        return batches
    
    def soft_update(self, target_net, source_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    
    def train(self):

        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        state, action, reward, new_state, done = zip(*batch)

        state = torch.stack(state)
        action = torch.stack(action)
        new_state = torch.stack(new_state)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.actor.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.actor.device)

        with torch.no_grad():
            new_action, new_action_log_prob, _ = self.actor(new_state)
            new_state_action = torch.cat([new_state, new_action], dim=1)

            q_target1 = self.target1(new_state_action)                    
            q_target2 = self.target2(new_state_action)

            min_target = torch.min(q_target1, q_target2)
            scaled_new_action_log_prob = self.alpha * new_action_log_prob
            q_value_target = reward.unsqueeze(dim=1) + self.gamma * (1 - done.float().unsqueeze(dim=1)) * \
                (min_target - scaled_new_action_log_prob)

        state_action = torch.cat([state, action], dim=1)

        q1 = self.critic1(state_action)
        q2 = self.critic2(state_action)

        loss_q1 = self.mse_loss(q1, q_value_target)
        loss_q2 = self.mse_loss(q2, q_value_target)

        self.critic1.optimizer.zero_grad()
        loss_q1.backward()
        self.critic1.optimizer.step()

        self.critic2.optimizer.zero_grad()
        loss_q2.backward()
        self.critic2.optimizer.step()

        new_action_pi, new_action_log_prob_pi, _ = self.actor(state)
        state_action_pi = torch.cat([state, new_action_pi], dim=1)

        q1_pi = self.critic1(state_action_pi)
        q2_pi = self.critic2(state_action_pi)

        loss_actor = (self.alpha.detach() * new_action_log_prob_pi - torch.min(q1_pi, q2_pi)).mean()

        self.actor.optimizer.zero_grad()
        loss_actor.backward()
        self.actor.optimizer.step()

        alpha_loss = -(self.log_alpha * (new_action_log_prob_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        self.soft_update(self.target1, self.critic1)
        self.soft_update(self.target2, self.critic2)

        return loss_actor.item(), loss_q1.item(), loss_q1.item(), q1.mean().item(), q2.mean().item(), self.alpha.detach().item(), -new_action_log_prob_pi.mean().item()


def evaluate_agent(agent, env, device, num_episodes=5):
    total_reward = 0
    
    for _ in range(num_episodes):
        state = env.reset()[0]
        state = torch.tensor(state, dtype=torch.float32).to(device)
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                _, _, action = agent.actor(state)
            
            action_np = action.cpu().detach().numpy().squeeze()
            state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            episode_reward += reward
            state = torch.tensor(state, dtype=torch.float32).to(device)
        
        total_reward += episode_reward
    
    return total_reward / num_episodes

                


if __name__ == '__main__':
    env = gym.make("LunarLanderContinuous-v3")
    num_actions = env.action_space.shape[0]
    state_space = env.observation_space.shape[0]

    print(f"num_actions {num_actions}")
    print(f"state_space {state_space}")
    print(f"action space min {env.action_space.low} max {env.action_space.high}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = Agent(state_space, num_actions)
    state = env.reset()[0]
    state = torch.tensor(state, dtype=torch.float32).to(device)

    print(state)

    total_steps = 0
    reward_log = []
    max_num_steps = 300000
    warmup_steps = 10000
    eval_freq = 5000
    print(f"max_num_steps {max_num_steps}")
    episode_reward = 0
    episode_count = 0

    loss_actor_history = []
    loss_q1_history = []
    loss_q2_history = []
    q1_history = []
    q2_history = []
    alpha_history = []
    entropy_history = []

    while total_steps < (max_num_steps):
        
        if total_steps > warmup_steps:
            action, action_log_prob, action_mean = agent.actor(state)
            action_np = action.cpu().detach().numpy().squeeze()
            if (action_np.any() < -1 or action_np.any() > 1):
                print(f"Action out of bounds: {action_np}")
        else:
            action_np = env.action_space.sample()
            action = torch.tensor(action_np, dtype=torch.float32).to(agent.actor.device)


        new_state, reward, terminated, truncated,_ = env.step(action_np)
        done = terminated or truncated

        new_state = torch.tensor(new_state, dtype=torch.float32).to(agent.actor.device)
        agent.store_transition((state.detach(), action.detach(), reward, new_state.detach(), done))

        episode_reward += reward
        state = new_state
        total_steps +=1

        if total_steps > warmup_steps:
            loss_actor, loss_q1, loss_q2, q1, q2, alpha, entropy = agent.train()
            loss_actor_history.append(loss_actor)
            loss_q1_history.append(loss_q1)
            loss_q2_history.append(loss_q2)
            q1_history.append(q1)
            q2_history.append(q2)
            alpha_history.append(alpha)
            entropy_history.append(entropy)

        if done:
            episode_count +=1 
            reward_log.append(episode_reward)
            state = env.reset()[0]
            state = torch.tensor(state, dtype=torch.float32).to(device)
            episode_reward = 0

        
            if len(reward_log) > 0:
                print(f"Episode: {episode_count}, Steps: {total_steps}, Reward avg: {np.mean(reward_log[-10:]):.2f}, Reward max: {np.max(reward_log[-10:]):.2f} ")

        if total_steps % eval_freq == 0 and total_steps > warmup_steps:
            eval_reward = evaluate_agent(agent, env, device)
            print(f"Evaluation at step {total_steps}: {eval_reward:.2f}")

    env.close()
    torch.save(agent.actor.state_dict(), "sac_lunar_lander_fix_alpha.pt")


    plt.figure(figsize=(15, 10))

    # 1. Alpha (Entropy Coefficient)
    plt.subplot(3, 2, 1)
    plt.plot(alpha_history)
    plt.title("Alpha (Entropy Coefficient)")
    plt.grid()

    # 2. Policy Entropy
    plt.subplot(3, 2, 2)
    plt.plot(entropy_history)
    plt.title("Policy Entropy")
    plt.grid()

    # 3. Q-values
    plt.subplot(3, 2, 3)
    plt.plot(q1_history, label='Q1')
    plt.plot(q2_history, label='Q2')
    plt.title("Q-values")
    plt.legend()
    plt.grid()

    # 4. Losses
    plt.subplot(3, 2, 4)
    plt.plot(loss_actor_history, label='Actor Loss')
    plt.plot(loss_q1_history, label='Critic1 Loss')
    plt.plot(loss_q2_history, label='Critic2 Loss')
    plt.title("Losses")
    plt.legend()
    plt.grid()

    # 5. Episode Reward
    window_size = 100
    plt.subplot(3, 2, 5)
    plt.plot(reward_log, label='Reward')
    moving_avg = np.convolve(reward_log, np.ones(window_size)/window_size, mode='valid')
    plt.plot(moving_avg, label='Smoothed Reward')
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.grid()

    plt.tight_layout()
    plt.savefig("sac_debug_metrics_fix_alpha.png")