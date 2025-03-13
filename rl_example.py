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
    # Convert state to integer if it's a tuple
    state = state[0] if isinstance(state, tuple) else state
    
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
    print("\n=== Starting Q-Learning Training ===")
    print(f"Training for {NUM_EPISODES} episodes in {ENV_NAME} environment")
    print("Training progress:")
    
    # Track successful episodes
    successes = 0
    
    for episode in range(NUM_EPISODES):
        state = env.reset()[0]  # Add [0] to get the state value
        done = False
        rewards = 0
        steps = 0
        
        while not done:
            action = choose_action(state, EPSILON)
            # Handle the new gym API return format
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = next_state[0] if isinstance(next_state, tuple) else next_state
            rewards += reward
            steps += 1
            update_q_table(state, action, next_state, reward)
            state = next_state
        
        # Update success count
        if rewards > 0:
            successes += 1
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            success_rate = (successes / (episode + 1)) * 100
            print(f"Episode {episode + 1}/{NUM_EPISODES} | Success Rate: {success_rate:.1f}% | Last Episode Steps: {steps}")
    
    print("\n=== Training Complete ===")
    print(f"Final Success Rate: {(successes/NUM_EPISODES)*100:.1f}%")

def play_game():
    """Use the trained Q-table to play the game"""
    print("\n=== Starting Game with Trained Agent ===")
    try:
        # Initialize state and validate
        initial_state = env.reset()
        state = initial_state[0] if isinstance(initial_state, tuple) else initial_state
        if state is None:
            raise ValueError("Invalid initial state received from environment")

        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 1000:  # Add step limit to prevent infinite loops
            steps += 1
            # Ensure state is within valid range
            state = min(max(state, 0), env.observation_space.n - 1)
            action = np.argmax(q_table[state])
            
            try:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                # Validate and process next state
                state = next_state[0] if isinstance(next_state, tuple) else next_state
                state = min(max(state, 0), env.observation_space.n - 1)
                total_reward += reward
                
                try:
                    env.render()
                except Exception as e:
                    print(f"Warning: Render failed but continuing execution: {e}")
                
                print(f"Step {steps}: State {state}, Action {action}")
                
            except Exception as e:
                print(f"Error during step execution: {e}")
                break
        
        print("\n=== Game Finished ===")
        print(f"Total Steps: {steps}")
        print(f"Total Reward: {total_reward}")
        print("Result: " + ("Success!" if total_reward > 0 else "Failure"))
        
    except Exception as e:
        print(f"Fatal error during game execution: {e}")
        print("Attempting to close environment gracefully...")
    finally:
        try:
            env.close()
        except:
            pass

# Main execution
print("\n=== Q-Learning Agent for FrozenLake ===")
print("Training agent...")
train_agent()
print("\nStarting gameplay demonstration...")
play_game()
