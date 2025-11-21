import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import pickle
from config import STEP_REWARD, WALL_REWARD, HOLE_REWARD, GOAL_REWARD
from config import ALPHA, GAMMA, EPSILON, SIMMULATION_NUMBER, ALPHA_DECAY, EPSILON_DECAY, DECAY_STEP, TRAIN, RENDERS
from stable_baselines3 import DQN


class MazeEnv(gym.Env):
    def __init__(self):
        super(MazeEnv, self).__init__()

        # Define grid size
        self.height = 15
        self.width = 15

        self.turn = 0

        # Define elements
        self.EMPTY = 0
        self.WALL = 1
        self.HOLE = 2
        self.AGENT = 3
        self.GOAL = 4
        self.agent_pos = None
        self.goal_pos = None
        self.goal_id = None
        
        # Initialize visualization
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        plt.ion()

        # Action space: up, right, down, left
        self.action_space = spaces.Discrete(4)
        # Observation space: single number representing state
        self.observation_space = spaces.MultiDiscrete([5, 5, 5, 5, 4, 15*15])

        # Create grid
        self.grid = None
        self.grid_initialization()

    def grid_initialization(self):
        self.turn = 0
        
        # Create grid
        self.grid = np.zeros((self.height, self.width))
        
        # Set agent, enemy and goal position
        self.agent_pos = [random.randint(2, self.height - 3), random.randint(2, self.width - 3)]

        self.goal_id = random.randint(1,4)
        if self.goal_id == 1:
            self.goal_pos = [1, 1]
        elif self.goal_id == 2:
            self.goal_pos = [1, self.width - 2]
        elif self.goal_id == 3:
            self.goal_pos = [self.height - 2,1]
        else:
            self.goal_pos = [self.height - 2, self.width - 2]
        
        # Add border walls
        self.grid[0:self.height, 0] = self.WALL
        self.grid[0, 0:self.width ] = self.WALL
        self.grid[self.height-1, 0:self.width] = self.WALL
        self.grid[0:self.height , self.width-1] = self.WALL
        # Set initial positions
        self.grid[self.agent_pos[0], self.agent_pos[1]] = self.AGENT
        self.grid[self.goal_pos[0], self.goal_pos[1]] = self.GOAL
        # Add inner walls
        for i in range(1, self.height):
            for j in range(1, self.width):
                if self.grid[i][j] == self.EMPTY:
                    r = random.random() 
                    if r < 0.2:
                        self.grid[i][j] = self.WALL
                    elif r < 0.25:
                        self.grid[i][j] = self.HOLE       

    def get_state(self, isnumpy=True):
        up_view = self.grid[self.agent_pos[0] - 1][self.agent_pos[1]]
        down_view = self.grid[self.agent_pos[0] + 1][self.agent_pos[1]]
        left_view = self.grid[self.agent_pos[0]][self.agent_pos[1] - 1]
        right_view = self.grid[self.agent_pos[0]][self.agent_pos[1] + 1]
        agent_position = self.agent_pos[0] * self.width + self.agent_pos[1]
        observation = (up_view, down_view, left_view, right_view, self.goal_id - 1, agent_position)
        if isnumpy:
            return np.array(observation)
        else:
            return observation

    def step(self, action, isnumpy = True):
        # Add 1 turn
        self.turn += 1

        # Save previous position
        prev_pos = self.agent_pos.copy()

        # Move agent
        if action == 0:  # up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # right
            self.agent_pos[1] = min(self.width - 1, self.agent_pos[1] + 1)
        elif action == 2:  # down
            self.agent_pos[0] = min(self.height - 1, self.agent_pos[0] + 1)
        elif action == 3:  # left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)

        # Check if new position is valid
        new_pos_value = self.grid[self.agent_pos[0], self.agent_pos[1]]

        # Define rewards and terminal states
        done = False
        reward = STEP_REWARD()  # small negative reward for each step

        if new_pos_value == self.WALL:
            self.agent_pos = prev_pos  # revert move
            reward = WALL_REWARD()
        elif new_pos_value == self.HOLE:
            done = True
            reward = HOLE_REWARD()
        elif self.agent_pos == self.goal_pos:
            done = True
            reward = GOAL_REWARD()

        # Update grid
        if new_pos_value != self.WALL:
            self.grid[prev_pos[0], prev_pos[1]] = self.EMPTY
            self.grid[self.agent_pos[0], self.agent_pos[1]] = self.AGENT

        if self.turn == 200:
            return self.get_state(isnumpy), reward, True, False, {}

        return self.get_state(isnumpy), reward, done, False, {}

    def reset(self, *, seed = None, options = None, return_info = False, isnumpy = True):
        super().reset(seed=seed)
        # Reset game to initial state
        self.turn = 0
        self.grid_initialization()
        return self.get_state(isnumpy), {}

    def render(self):
        self.ax.clear()
        # Define colors for each element
        colors = {
            self.EMPTY: 'white',
            self.WALL: 'gray',
            self.HOLE: 'black',
            self.AGENT: 'blue',
            self.GOAL: 'green',
        }

        name = {
            self.EMPTY: 'Vazio',
            self.WALL: 'Parede',
            self.HOLE: 'Buraco',
            self.AGENT: 'Agente',
            self.GOAL: 'Objetivo',
        }

        # Create color map
        cmap = plt.cm.colors.ListedColormap(list(colors.values()))
        # Plot the grid
        self.ax.imshow(self.grid, cmap=cmap)

        # Add legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, label=name[key])
                           for key, color in colors.items()]
        self.ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.axis('off')
        plt.pause(0.1)
        self.fig.canvas.draw()


class QLearningAgent:

    def __init__(self, action_space, learning_rate, discount_factor, epsilon):
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.action_space = action_space

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = defaultdict(lambda: np.zeros(self.action_space.n), pickle.load(f))

    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.action_space.sample()

        q_values = self.q_table[state]
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(len(q_values), p=probs)

    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value


def run():
# Training Model/Agent
    if TRAIN():
        env = MazeEnv()
        agent = QLearningAgent(env.action_space, ALPHA(), GAMMA(), EPSILON())

        episodes = SIMMULATION_NUMBER()
        total_reward = 0
        sucess = 0
        for episode in range(1, episodes+1):
            state, _ = env.reset(isnumpy = False)
            done = False
            if RENDERS():
                env.render()

            while not done:
                action = agent.get_action(state)
                next_state, reward, done, _, _ = env.step(action, isnumpy = False)

                agent.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                if reward == GOAL_REWARD():
                    sucess += 1
                if RENDERS():
                    env.render()

            # Parameter Decay
            if episode % DECAY_STEP() == 0:
                agent.epsilon *= EPSILON_DECAY()
                agent.alpha *= ALPHA_DECAY()
                print(f"Episode {episode}, Mean Reward: {(total_reward/DECAY_STEP()):.2f}, Success Rate: {(sucess/DECAY_STEP()):.2f}")
                print("Explore Chance (epsilon): ", agent.epsilon)
                print("Exploit Chance (1-epsilon): ", 1-agent.epsilon)
                print("Learning Rate (alpha): ", agent.alpha)
                total_reward = 0
                sucess = 0

        # Save the trained agent
        agent.save_model('./models/model_parameter_19.pkl')

    # Test the trained agent
    else:
        env = MazeEnv()
        state, _ = env.reset(isnumpy = False)
        done = False
        if RENDERS():
            env.render()
        # Load the trained agent
        agent = QLearningAgent(env.action_space, ALPHA(), GAMMA(), 0)
        agent.load_model('./models/model_parameter_19.pkl')

        while not done:
            action = agent.get_action(state)
            state, reward, done, _, _ = env.step(action, isnumpy = False)
            print(f"Action: {action}, Reward: {reward}")
            if RENDERS():
                env.render()
                
if __name__ == "__main__":
    run()