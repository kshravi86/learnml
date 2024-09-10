# Import necessary libraries: numpy for numerical computations and gym for the game environment
# numpy is used for numerical computations
import numpy as np
# gym is used for the game environment
import gym

# Game environment constants
# Game environment constants
# ENV_NAME is the name of the game environment
ENV_NAME = 'FrozenLake-v0'
# NUM_EPISODES is the number of episodes to train the agent
NUM_EPISODES = 1000

# Create a simple game environment using the FrozenLake-v0 game from the gym library
env = gym.make(ENV_NAME)

# Q-learning hyperparameters
# Q-learning hyperparameters
# ALPHA is the learning rate
ALPHA = 0.1
# GAMMA is the discount factor
GAMMA = 0.9
# EPSILON is the exploration rate
EPSILON = 0.1

# Initialize the Q-table with zeros, where the number of rows is the number of states and the number of columns is the number of actions
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Choose an action using epsilon-greedy policy
def choose_action(state, epsilon):
    """Choose an action using epsilon-greedy policy"""
    # Randomly choose an action with probability epsilon
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        # Choose the action with the highest Q-value
        return np.argmax(q_table[state])

# Update the Q-table
def update_q_table(state, action, next_state, reward):
    """Update the Q-table"""
    # Update the Q-value using the Q-learning update rule
    q_table[state, action] += ALPHA * (reward + GAMMA * np.max(q_table[next_state]) - q_table[state, action])

# Train the agent using Q-learning
def train_agent():
    """Train the agent using Q-learning"""
    # Train the agent for NUM_EPISODES episodes
    for episode in range(NUM_EPISODES):
        # Reset the environment and initialize the rewards
        state = env.reset()
        done = False
        rewards = 0
        while not done:
            # Choose an action using epsilon-greedy policy
            action = choose_action(state, EPSILON)
            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)
            # Update the rewards
            rewards += reward
            # Update the Q-table
            update_q_table(state, action, next_state, reward)
            # Update the state
            state = next_state
        # Print the episode and total rewards
        print(f'Episode {episode+1}, Total Rewards: {rewards}')

# Use the trained Q-table to play the game
def play_game():
    """Use the trained Q-table to play the game"""
    # Reset the environment
    state = env.reset()
    done = False
    while not done:
        # Choose the action with the highest Q-value
        action = np.argmax(q_table[state])
        # Take a step in the environment
        state, _, done, _ = env.step(action)
        # Render the environment
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
