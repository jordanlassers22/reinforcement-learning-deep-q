# Todo
# Reward for killing / damaging enemies
# Penalty for falling in water


import numpy as np
import pygame

pygame.init()
pygame.mixer.init()

import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from gymnasium.envs.registration import register
from engine import GameEngine
from controller import GameController
from settings import SCREEN_WIDTH, SCREEN_HEIGHT

# Not required unless we want to provide traditional gym.make capabilities
register(id='Sidescroller-v0',
         entry_point='gymenv:ShooterEnv',
         max_episode_steps=5000)


class ShooterEnv(gym.Env):
    '''
    Wrapper class that creates a gym interface to the original game engine.
    '''

    # Hints for registered environments; ignored otherwise
    metadata = {
        'render_modes': ['human'],
        'render_fps': 60
    }

    def __init__(self, render_mode=None):
        '''
        Initializes a new Gymnasium environment for the Shooter Game. Loads the
        game engine into the background and defines the action space and the
        observation space.
        '''

        super().__init__()
        self.render_mode = render_mode
        pygame.display.init()
        if self.render_mode == 'human':
            pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('Shooter')
            self.screen = pygame.display.get_surface()
            self.game = GameEngine(self.screen, False)
            self.clock = pygame.time.Clock()

        else:
            pygame.display.set_mode((1, 1), pygame.HIDDEN)
            self.game = GameEngine(None, False)

        # Discrete action space: 7 possible moves
        self.action_space = Discrete(7)
        # Observation: [dx, dy, health, exit_dx, exit_dy, ammo, grenades, in_air, enemy_in_range, enemy_dx, enemy_dy, able_to_shoot, able_to_throw_grenade]
        low = np.array([-10000, -1000, 0, -10000, -10000, 0, 0, 0, 0, -1500, -1500, 0, 0], dtype=np.float32)
        high = np.array([10000, 1000, 100, 10000, 10000, 50,20, 1, 1, 1500, 1500, 1, 1], dtype=np.float32)
        self.observation_space = Box(low, high, dtype=np.float32)


    def reset(self, seed=None, options=None):   
        '''
        Resets the game environment for the beginning of another episode.
        '''
        self.last_action = None
        self.step_count = 0
        self.game.reset_world()
        self.game.load_current_level()
        
        self.prev_enemy_health = {} #Dictionary to track enemy health
        for enemy in self.game.groups['enemy']:
            self.prev_enemy_health[id(enemy)] = enemy.health


        # Tracks observation and reward values across steps
        self.start_x = self.game.player.rect.centerx
        self.start_y = self.game.player.rect.centery
        self.prev_x = self.start_x  # Track previous horizontal position
        self.stuck_counter = 0 # Counter to determine amount of frames an agent is stuck for
        self.furthest_x = self.game.player.rect.centerx # Variable to keep track of record x distance



        
        # Store starting ammo and health for reference in get_reward
        self.prev_ammo = self.game.player.ammo
        self.prev_grenades = self.game.player.grenades
        self.prev_health = self.game.player.health

        # Return the initial game state
        observation, debug_info = self._get_observation()
        return observation, debug_info


    def step(self, action):
        '''
        Agent performs a single action in the game environemnt.
        '''
        controller = self._action_to_controller(action)
        self.game.update(controller)
        self.step_count += 1

        observation, debug_info = self._get_observation()
        self.last_action = action
        reward = self._get_reward()
        terminated = not self.game.player.alive or self.game.level_complete
        truncated = self.step_count >= 1000

        return observation, reward, terminated, truncated, debug_info


    def render(self):
        ''' 
        Visually renders the game so that viewers can watch the agent play. The
        first time this function is called, it initializes the PyGame display
        and mixer (just like a real game). If the self. Every time that it is called, this
        function draws the game.
        '''
        # Do nothing if rendering is disabled
        if self.render_mode != "human":
            return

        # Draw the screen        
        self.game.draw()
        pygame.display.update()
        self.clock.tick(self.metadata['render_fps'])


    def _get_observation(self):
        p = self.game.player

        # Distance from start
        p_dx = p.rect.centerx - self.start_x
        p_dy = p.rect.centery - self.start_y

        # Exit distance
        exit_dx, exit_dy = self._get_exit_distance(p)
        
        # Closest enemy position to player
        enemy_dx, enemy_dy, enemy_in_range = self._get_closest_enemy_offset(p)

        # Create an observation (13 values)
        obs = [
            p_dx,
            p_dy,
            p.health,
            exit_dx,
            exit_dy,
            p.ammo,
            p.grenades,
            int(p.in_air),  # 0 = on the ground 1 = in air
            enemy_in_range, # 0 if enemy farther than 1500, 1 if closer
            enemy_dx, # X-distance to closest enemy (or 9999 if none in range)
            enemy_dy, # Y-distance to closest enemy (or 9999 if none in range)
            int(p.shoot_time + p.shoot_delay < pygame.time.get_ticks()), # 1 if able to shoot, 0 if on cooldown
            int(p.throw_time + p.throw_delay < pygame.time.get_ticks()) # 1 if able to throw, 0 if on cooldown
        ]

        # Create debug information
        debug_info = {
            'player_health': p.health,
            'player_distance': (p_dx, p_dy),
            'exit_distance': (exit_dx, exit_dy),
            'closest_enemy_distance': (enemy_dx, enemy_dy),
        }

        return np.array(obs, dtype=np.float32), debug_info
    

    def _get_exit_distance(self, player):
        min_dist = float('inf')
        closest_dx, closest_dy = 9999, 9999

        for tile in self.game.groups['exit']:
            dx = tile.rect.centerx - player.rect.centerx
            dy = tile.rect.centery - player.rect.centery
            dist = abs(dx) + abs(dy)
            if dist < min_dist:
                min_dist = dist
                closest_dx = dx
                closest_dy = dy

        return closest_dx, closest_dy


    def _get_reward(self):
        p = self.game.player
        action = self.last_action
        reward = 0
        
        if not p.alive: # Punishment for dying
            return -300

        if p.rect.centerx > self.prev_x:
            reward += 5  # reward for moving forward 
        elif p.rect.centerx < self.prev_x:
            reward -= 2  # penalize for moving backward
        
        # Track if agent is stuck and not moving right or left
        if p.rect.centerx == self.prev_x:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        # Penalize if stuck too long > 10 frames
        if self.stuck_counter >= 10:
            reward -= 10
        
        # Bonus reward for beating level
        if self.game.level_complete:
            reward += 1000
            
        # penalty for using ammo when empty    
        if action == 5 and p.ammo == 0:
            reward -= 5
        if action == 6 and p.grenades == 0:
            reward -= 5
        
        # Penalty for jump spamming
        if action == 2 and p.in_air:
            reward -= 5

    
            # Ammo reward 
        if p.ammo > self.prev_ammo:
            reward += 10
        elif p.ammo < self.prev_ammo:
            reward -= 4
    
        # Grenade reward 
        if p.grenades > self.prev_grenades:
            reward += 2
        elif p.grenades < self.prev_grenades:
            reward -= 1.5
    
        # Health reward 
        if p.health > self.prev_health:
            reward += 20  # Healed
        elif p.health < self.prev_health:
            reward -= 10  # Took damage
            
        # Reward for damaging enemies
        for enemy in self.game.groups['enemy']:
            enemy_id = id(enemy)
            prev_health = self.prev_enemy_health.get(enemy_id, enemy.health)
            damage = prev_health - enemy.health
            if damage > 0:
                reward += damage * 0.5  
            self.prev_enemy_health[enemy_id] = enemy.health
            
        # Reward for killing enemy
        if prev_health > 0 and enemy.health <= 0:
            reward += 30

        curr_x = self.game.player.rect.centerx
        if curr_x > self.furthest_x:
            reward += (curr_x - self.furthest_x) * 0.01 #Scale reward based on distance traveled, rewarding more for further explortation
            reward = min(reward, 10) #Cap at a reward of 10

            self.furthest_x = curr_x

    
        # Update trackers
        self.prev_ammo = p.ammo
        self.prev_grenades = p.grenades
        self.prev_health = p.health
        self.prev_x = p.rect.centerx
        
        
        return reward


    def _action_to_controller(self, action):
        '''
        Converts an action (just an integer) to a game controller object.
        '''
        ctrl = GameController()
        if action == 0: ctrl.mleft = True
        elif action == 1: ctrl.mright = True
        elif action == 2: ctrl.jump = True
        elif action == 3: ctrl.jump = ctrl.mleft = True
        elif action == 4: ctrl.jump = ctrl.mright = True
        elif action == 5: ctrl.shoot = True
        elif action == 6: ctrl.throw = True
        return ctrl
    
    def _get_closest_enemy_offset(self, player):
        """Method that returns the closest enemies distance to the player, as well as a boolean flag enemy_in_range, if the enemy is within 1500 pixels"""
        vision_distance = 1500
        min_dist = vision_distance #Minimum distance for it to care about an enemy
        dx, dy = 1500, 1500 #Returns large value if no enemies nearby
        for enemy in self.game.groups['enemy']: #Go through all enemies in game
            #Get enemies coordinates
            ex = enemy.rect.centerx - player.rect.centerx
            ey = enemy.rect.centery - player.rect.centery
            dist = abs(ex) + abs(ey) #Calculate distance away from player
            if dist < min_dist: #If that enemy is closer to the player than the previous min distance, update that enemies position.
                dx, dy = ex, ey
                min_dist = dist
        enemy_in_range = int(min_dist < vision_distance) # Flag for whether there is an enemy in range
        
        return dx, dy, enemy_in_range
    
    def close(self):
        pygame.quit()
