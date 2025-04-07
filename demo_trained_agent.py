import gymnasium as gym
import numpy as np
import pickle
from gymenv import ShooterEnv

# Load trained Q-table
with open("q_tables/q_final.pkl", "rb") as f:
    q_table = pickle.load(f)

def discretize(state):
    bins = [200, 200, 10, 200, 200, 1, 1, 1, 1, 50, 50, 1, 1]
    return tuple(int(state[i] // bins[i]) for i in range(len(state)))

# Create environment 
env = ShooterEnv(render_mode="human")
state_raw, info = env.reset()
state = discretize(state_raw)

# Setup episode loop 
done = False
total_reward = 0
step_count = 0
max_steps = 1000

action_names = [
    "Run Left", "Run Right", "Jump", 
    "Jump Left", "Jump Right", "Shoot", "Grenade"
]

while not done and step_count < max_steps:
    action = np.argmax(q_table.get(state, np.zeros(7)))
    next_state_raw, reward, terminated, truncated, info = env.step(action)
    env.render()  # update screen

    state = discretize(next_state_raw)
    total_reward += reward
    step_count += 1
    done = terminated or truncated

    print(f"Step {step_count:3} | Action: {action_names[action]:<10} "
          f"| Reward: {reward:6.2f} | Health: {info['player_health']:3} "
          f"| Pos: {info['player_distance'][0]:4} "
          f"| Exit: {info['exit_distance'][0]:4} | Done: {done}")

# summary
print(f"\nDemo finished in {step_count} steps. Total reward: {total_reward:.2f}")
env.close()
