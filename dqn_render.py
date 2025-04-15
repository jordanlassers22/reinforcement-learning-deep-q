
import sys
import time
import torch
import pygame
from gymenv import ShooterEnv
from dqn_model import DQN
from settings import FPS


def visualize_episode(model:DQN=None, 
                      #env:ShooterEnv, 
                      device:str='cpu', level = 1):
    '''
    Renders a single episode of the custom Shooter environment using a 
    pre-trained DQN agent.

    Parameters:
    - model (torch.nn.Module): The trained DQN model used to select actions.
    - env (gym.Env): The Shooter environment instance to interact with.
    - device (torch.device): The device (CPU or GPU) on which to run the model.

    Returns:
    - float: The total accumulated reward from the episode.
    '''

    # We need pygame so that we can handle keyboard events and, at the end of
    # the episode, close the display window. The keyboard is *not* to interact
    # with the game, but we might want to close early.
    clock = pygame.time.Clock()
    env = ShooterEnv(render_mode="human", level=level)
    if not pygame.get_init():
        pygame.init()
        pygame.mixer.init()

    if model is None:
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n
        model = DQN(obs_size, n_actions)

    # Take the model out of training mode
    model.eval()

    # Initialize this episode and setup the 'state' to be a tensor since our
    # DQN model is expecting all inputs to be tensors.
    state, info = env.reset()
    state = torch.from_numpy(state).unsqueeze(0).float().to(device)
    total_reward = 0
    done = False

    # Loop until this episode is complete.
    furthest_x = 0 #Keep track of furthest x reached.
    while not done:

        # Allow user to quit early.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print(">> Window closed. Ending visualization early. <<\n")
                done = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                print(">> Escape pressed. Ending visualization early. <<\n")
                done = True

        # Display the current frame
        env.render()

        # Choose the next best thing (being careful not to accidentally train).
        with torch.no_grad():
            q_vals = model(state)
            action = q_vals.argmax().item()

        # Perform the next action in the environment.
        next_state, reward, terminated, truncated, info = env.step(action)
        done = done or terminated or truncated
        total_reward += reward
        state = torch.from_numpy(next_state).unsqueeze(0).float().to(device)
        clock.tick(FPS)
        
        curr_x = env.game.player.rect.centerx
        furthest_x = max(furthest_x, curr_x)

    # Gracefuly close PyGame environment and return.
    time.sleep(1)
    pygame.mixer.music.stop()
    pygame.display.quit()
    return total_reward, furthest_x


def main(model_path: str, level: int = 1):
    '''
    Loads a pre-trained DQN agent and runs a single episode of the Shooter game
    with rendering enabled.

    Parameters:
    - model_path (str): Path to the saved model file (.pt) containing the 
    trained weights.
    
    - level (int): integer 1-6 for the level that should be played

    Returns:
    - float: The total accumulated reward from the episode.
    '''

    # Create a fresh instance of the model for the appropriate environment.
    env = ShooterEnv(render_mode="human")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    model = DQN(obs_size, n_actions)

    # Load the saved model for CPU or GPU execution.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Run this episode.
    reward, furthest_x  = visualize_episode(model, device, level)
    env.close()
    print(f"Episode finished with {reward:.2f} reward and max X: {furthest_x}")
    return reward

if __name__ == '__main__':
    """If main params are path to model and level number"""
    # python dqn_render.py "models/dqn_shooter_600.pt" 5 
    # We must read in a saved model to play the game.
    if len(sys.argv) < 2:
        print(f"Usage: python {__file__} <model_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    if len(sys.argv) > 2:
        level = int(sys.argv[2]) 
    else:
        level = 1

    total_reward = main(model_path, level)
    print(f"Episode finished with {total_reward} reward\n")
    pygame.quit()

