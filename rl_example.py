# Import necessary libraries: numpy for numerical computations and gym for the game environment
import numpy as np
import gym

# Create a simple game environment using the FrozenLake-v0 game from the gym library
env = gym.make('FrozenLake-v0')

# Initialize the Q-table with zeros, where the number of rows is the number of states and the number of columns is the number of actions
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Set the learning rate (alpha), discount factor (gamma), and exploration rate (epsilon) for the Q-learning algorithm
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Train the agent using Q-learning
for episode in range(1000):
    state = env.reset()
    done = False
    rewards = 0
    while not done:
        # Choose an action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Take the action and observe the next state and reward
        next_state, reward, done, _ = env.step(action)
        rewards += reward

        # Update the Q-table
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

        # Move to the next state
        state = next_state

    # Print the total rewards for each episode
    print(f'Episode {episode+1}, Total Rewards: {rewards}')

# Use the trained Q-table to play the game
state = env.reset()
done = False
while not done:
    action = np.argmax(q_table[state])
    state, _, done, _ = env.step(action)
    env.render()
