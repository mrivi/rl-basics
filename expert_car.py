import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

environment_name = 'CarRacing-v3'

# env = gym.make(environment_name)
# env = DummyVecEnv([lambda: env])

# log_path = os.path.join('Training', 'Logs')
# model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path)

# model.learn(total_timesteps=100000)

# ppo_path = os.path.join('Training', 'Saved Models', 'PPO_Driving_Model')
# model.save(ppo_path)


ppo_2m_path = "/home/martina/src/Training/Saved Models/PPO_Driving_Model"
model = PPO.load(ppo_2m_path)
# evaluate_policy(model, env, n_eval_episodes=10, render=True)

env = gym.make(environment_name, render_mode="human")
env.reset()

episodes = 5
for episode in range(1, episodes+1):
    obs = env.reset()[0]  # Unpack the observation from the tuple
    print(f"Observation shape: {obs.shape}")
    done = False
    score = 0

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        score += reward
    print(f"Episode: {episode} Score: {score}")
env.close()