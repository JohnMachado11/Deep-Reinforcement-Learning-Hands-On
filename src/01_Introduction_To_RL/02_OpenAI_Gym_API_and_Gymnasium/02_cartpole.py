import gymnasium as gym


# Create the CartPole environment instance.
# "CartPole-v1" is a classic control problem where a pole is attached to a cart moving along a track.
# The goal is to balance the pole by moving the cart left or right.
e = gym.make("CartPole-v1")

# Reset the environment to its initial state.
# This returns two values:
#   obs  -> the initial observation (state vector)
#   info -> an empty dict with optional metadata
obs, info = e.reset()


# Print the initial observation (4 floating-point values representing the environment's state)
# These correspond to:
#   1. Cart position (meters)
#   2. Cart velocity (m/s)
#   3. Pole angle (radians)
#   4. Pole angular velocity (radians/s)
print("Observation:", obs)

# Print the info dictionary (empty here, but can include debug data for some envs)
print("Info:", info)

# Print the action space:
#   Discrete(2) → only 2 possible actions:
#     0 = push cart to the left
#     1 = push cart to the right
print("Action Space:", e.action_space)

# Print the observation space:
#   A Box() space with 4 continuous values, representing the valid range for each element of obs.
#   The extreme values show min and max limits for position, velocity, angle, and angular velocity.
print("Observation Space:", e.observation_space)

# Take one step in the environment using action 0 (push cart left)
# e.step(action) returns a tuple:
#   (new_obs, reward, terminated, truncated, info)
#   - new_obs: next observation (the new state after the action)
#   - reward: immediate reward (here always 1.0 per step until termination)
#   - terminated: whether the episode ended due to failure (pole fell, etc.)
#   - truncated: whether the episode ended due to time limit
#   - info: optional metadata
print("\nSending Action to Environment:", e.step(0))

# Randomly sample a valid action (either 0 or 1) from the action space.
# Useful for exploring the environment randomly.
print("Sample of Action Space:", e.action_space.sample())

# Randomly sample a valid observation (state vector) from the observation space.
# This shows a random point within the allowed numerical range of the state space,
# though it doesn’t necessarily correspond to a physically realistic state.
print("Sample of Observation Space:", e.observation_space.sample())