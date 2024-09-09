# Import necessary libraries: numpy for numerical computations and gym for the game environment
import numpy as np
import gym

# Game environment constants
ENV_NAME = 'FrozenLake-v0'
NUM_EPISODES = 1000

# Create a simple game environment using the FrozenLake-v0 game from the gym library
env = gym.make(ENV_NAME)

# Q-learning hyperparameters
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1

# Initialize the Q-table with zeros, where the number of rows is the number of states and the number of columns is the number of actions
q_table = np.zeros([env.observation_space.n, env.action_space.n])

def choose_action(state, epsilon):
    """Choose an action using epsilon-greedy policy"""
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

def update_q_table(state, action, next_state, reward):
    """Update the Q-table"""
    q_table[state, action] += ALPHA * (reward + GAMMA * np.max(q_table[next_state]) - q_table[state, action])

def train_agent():
    """Train the agent using Q-learning"""
    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        rewards = 0
        while not done:
            action = choose_action(state, EPSILON)
            next_state, reward, done, _ = env.step(action)
            rewards += reward
            update_q_table(state, action, next_state, reward)
            state = next_state
        print(f'Episode {episode+1}, Total Rewards: {rewards}')

def play_game():
    """Use the trained Q-table to play the game"""
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state])
        state, _, done, _ = env.step(action)
        env.render()

train_agent()
play_game()

# Use the trained Q-table to play the game
state = env.reset()
done = False
while not done:
    action = np.argmax(q_table[state])
    state, _, done, _ = env.step(action)
    env.render()
