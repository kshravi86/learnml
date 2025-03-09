# Import necessary libraries: numpy for numerical computations and gym for the game environment
# numpy is used for numerical computations
import numpy as np
# gym is used for the game environment
import gym

# Game environment constants
# Game environment constants
# ENV_NAME is the name of the game environment
ENV_NAME = 'FrozenLake-v1'
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
    """
    Epsilon-greedy action selection:
    P(random action) = ε
    P(greedy action) = 1 - ε
    """
    # Randomly choose an action with probability epsilon
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        # Choose the action with the highest Q-value
        return np.argmax(q_table[state])

# Q-learning Mathematical Foundations:
# Q(s,a) = Q(s,a) + α[R + γ * max(Q(s',a')) - Q(s,a)]
# where:
# - Q(s,a) is the Q-value for state s and action a
# - α (alpha) is the learning rate (0 < α ≤ 1)
# - R is the immediate reward
# - γ (gamma) is the discount factor (0 ≤ γ ≤ 1)
# - max(Q(s',a')) is the maximum Q-value for the next state
#
# The equation can be broken down into:
# 1. Q(s,a): Current Q-value
# 2. α: Learning rate that determines how much new information overrides old
# 3. R + γ * max(Q(s',a')): Target Q-value
#    - R: Immediate reward
#    - γ * max(Q(s',a')): Discounted future reward
# 4. [R + γ * max(Q(s',a')) - Q(s,a)]: Temporal Difference Error

# Update the Q-table
def update_q_table(state, action, next_state, reward):
    """
    Update Q-table using the Q-learning formula:
    Q(s,a) = Q(s,a) + α[R + γ * max(Q(s',a')) - Q(s,a)]
    """
    current_q = q_table[state, action]  # Current Q-value: Q(s,a)
    next_max_q = np.max(q_table[next_state])  # Future maximum Q-value: max(Q(s',a'))
    td_error = reward + GAMMA * next_max_q - current_q  # Temporal Difference Error
    q_table[state, action] = current_q + ALPHA * td_error  # Update Q-value

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
