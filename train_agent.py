import gymnasium as gym
import numpy as np
import pickle
import os
from collections import defaultdict
from gymenv import ShooterEnv

# Create folder to save q-tables and folder for training logs
os.makedirs("q_tables", exist_ok=True)

os.makedirs("logs", exist_ok=True)
# Discretization function
def discretize(state):
    bins = [200, 200, 10, 200, 200, 1, 1, 1, 1, 50, 50, 1, 1]
    return tuple(int(state[i] // bins[i]) for i in range(len(state)))

# Initialize Q-table (defaultdict for dynamic state space)
q_table = defaultdict(lambda: np.zeros(7))  # 7 discrete actions

# Hyperparameters
alpha = 0.5             #Learning rate: how much new information overrides old information
gamma = 0.95            #Discount factor: importance of future rewards
epsilon = 1.0           #Exploration rate: probability of choosing a random action
epsilon_decay = 0.998   #Rate at which exploration rate decreases
epsilon_min = 0.01      #Minimum exploration rate to ensure some randomness
episodes = 1000         #Number of episodes for training
max_steps = 1000

# Training the agent
env = ShooterEnv(render_mode=None)

# Log file path
log_path = "logs/training_log.txt"
with open(log_path, "w") as f:
    f.write("Episode,Reward,Epsilon,Steps\n")

print("Beggining agent training...")
for episode in range(episodes):
    state, _ = env.reset()
    state = discretize(state)
    done = False
    step_count = 0
    total_reward = 0

    while not done and step_count < max_steps:
        # Choose action using epsilon-greedy policy
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit
            
        #Perform action and observe next state and reward
        next_state_raw, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize(next_state_raw)
        done = terminated or truncated
        

        #Q-learning formula to update Q-table
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action] += alpha * (reward + gamma * q_table[next_state][best_next_action] - q_table[state][action])

        state = next_state
        total_reward += reward
        step_count += 1

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Save every 100 episodes
    print(f"Episode: {episode} | Total reward: {total_reward:.1f} | Epsilon: {epsilon:.3f}")
    with open(log_path, "a") as log_file:
       log_file.write(f"{episode},{total_reward:.1f},{epsilon:.3f},{step_count}\n")
    if episode % 100 == 0:
        with open(f"q_tables/q_{episode}.pkl", "wb") as f:
            pickle.dump(dict(q_table), f)

# Save final Q-table
with open("q_tables/q_final.pkl", "wb") as f:
    pickle.dump(dict(q_table), f)

print("Training finished. Q-table saved.")
