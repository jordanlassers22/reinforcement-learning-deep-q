
import numpy as np
import torch
from dqn_model import DQN, ReplayBuffer
from dqn_render import visualize_episode
from gymenv import ShooterEnv
import os
import glob
import re
import warnings



GAMMA = 0.99             # Discount factor highly values future reward
EPSILON_START = 0.8     # Begin with 100% random exploration 1.0
EPSILON_DECAY = 0.995    # Decay slowly
EPSILON_MIN = 0.05       #  Never fully stop exploring
TARGET_UPDATE_FREQ = 10  # How often to reload stable_net from online_net
TARGET_RENDER_FREQ = 10  # How often to render the agent during training
BATCH_SIZE = 64          # Balance learning with efficiency
MAX_EPISODES = 500

# Global variable holding CPU/GPU status
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_episode(env:ShooterEnv, 
                  online_net:DQN, 
                  stable_net:DQN, 
                  epsilon:float, 
                  replay:ReplayBuffer):
    '''
    Runs a single episode of interaction with the environment and trains the 
    Q-network using experience replay and target Q-value updates.

    Parameters:
    - env (ShooterEnv): The game environment.
    - train_net (DQN): The Q-network being actively trained.
    - stable_net (DQN): The target Q-network used for stable Q-value estimation.
    - epsilon (float): Current epsilon value for epsilon-greedy exploration.
    - replay (ReplayBuffer): Memory buffer storing past experiences.

    Returns:
    - float: Total reward accumulated during the episode.
    '''

    # Initialize this episode with the observation (state) being the input
    # to the neural network (a tensor).
    state, info = env.reset()
    state = torch.from_numpy(state).unsqueeze(0).float().to(device)
    done = False
    total_reward = 0

    # Play this episode out and train the model as you go.
    while not done:

        # Choose the next action (get an action, don't track gradients).
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_vals = online_net(state)
                action = q_vals.argmax().item()

        # Take a step forward in the game environment.
        nstate, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Add this step to our replay buffer and convert to a torch object.
        #replay.push((state.numpy(), action, reward, next_state, done))
        replay.push((state, action, reward, nstate, done))
        state = torch.from_numpy(nstate).unsqueeze(0).float().to(device)

        # Don't train the network until we have enough experiences.
        if len(replay) < BATCH_SIZE:
            continue

        # Take a 'replay buffer' worth of states and run through NN.
        states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)
        states = states.to(device)
        actions = actions.unsqueeze(1).to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        qvals = online_net(states).gather(1, actions).squeeze()

        # Compute the current Q-Values using stable network as 
        # (but don't track gradients yet)
        with torch.no_grad():
            next_qvals = stable_net(next_states).max(1)[0]
            targets = rewards + GAMMA * next_qvals * (1 - dones)

        # Train online NN (q_net) using gradient descent
        online_net.backpropagate(qvals, targets)

    # Total reward helps monitor progress
    return total_reward, env.furthest_x


def main():
    '''
    Trains a Deep Q-Learning neural network to play the custom shooter game
    using experience replay and a target network for stability.
    '''
    os.makedirs("models", exist_ok=True) # Make models folder
    epsilon = EPSILON_START
    training_level = 5  # <<< Change training level here
    env_headless = ShooterEnv(render_mode=None, level=training_level)
    obs_size = env_headless.observation_space.shape[0]
    n_actions = env_headless.action_space.n
    print(f"Creating online and stable neural networks "
          f"({n_actions} actions / {obs_size} observation vars)")

    # Online and stable networks with the stable version being a copy of the
    # online version that is refreshed slowly to maintain learning stability.
    online_net = DQN(obs_size, n_actions)
    stable_net = DQN(obs_size, n_actions)
    online_net.to(device)
    stable_net.to(device)
    stable_net.load_state_dict(online_net.state_dict())
    replay = ReplayBuffer()
    
    

    #Resume the next training episode or start at 1
    start_episode = 1
    model_files = glob.glob("models/dqn_shooter_*.pt")
    if model_files:
        latest_model = max(model_files, key=os.path.getctime)
        print(f"Loading latest model: {latest_model}")
        online_net.load_state_dict(torch.load(latest_model, map_location=device))
        stable_net.load_state_dict(online_net.state_dict())
        
        #Hide warning messages in console
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            online_net.load_state_dict(torch.load(latest_model, map_location=device))
    
        # Extract episode number from filename
        match = re.search(r'dqn_shooter_(\d+)\.pt', latest_model)
        if match:
            start_episode = int(match.group(1)) + 1
            print(f"Resuming training at episode {start_episode}")
    else:
        print("No existing model found. Starting from scratch.")
    
    
    print("Training started...")
    for episode in range(start_episode, MAX_EPISODES):
        
        # Train the neural networks.
        online_net.train()
        stable_net.train()
        reward, furthest_x_travelled = train_episode(env_headless, 
                               online_net, 
                               stable_net, 
                               epsilon, 
                               replay)

        # Update target network (the stable network) every X episodes.
        if episode > 0 and episode % TARGET_UPDATE_FREQ == 0:
            print("(updated stable network with latest from online network)")
            stable_net.load_state_dict(online_net.state_dict())

        # Report on the learning progress (and render agent every x episodes).
        if episode > 0 and episode % TARGET_RENDER_FREQ == 0:
            print(f"\n>> Visualizing agent at episode {episode} <<\n")
            render_reward, render_max_x = visualize_episode(online_net, device, level=training_level)
            print(f"Rendering reward: {reward:>10.2f} | Rendering Max X: {render_max_x}")
            torch.save(online_net.state_dict(), f'models/dqn_shooter_{episode}.pt')
        print(f"Episode {episode:>6} | "
              f"Reward: {reward:>10.2f} | "
              f"Epsilon: {epsilon:.3f} | "
              f"Max X coordinate: {furthest_x_travelled}")

        # Epsilon greedy decay
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # All done, close out the environment
    filename = 'models/dqn_shooter_final.pt'
    torch.save(online_net.state_dict(), filename)

    print(f"\nTraining complete. Final model saved as '{filename}'\n")
    env_headless.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCtrl+C detected... bye for now.\n")
