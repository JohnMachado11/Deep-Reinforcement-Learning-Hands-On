import gymnasium as gym

if __name__ == "__main__":
    num_episodes = 10
    total_rewards = []

    # create environment with video recording enabled
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.HumanRendering(env)
    # env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix="cartpole")

    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        total_steps = 0

        while True:
            # random action for demonstration
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            total_steps += 1

            # end if done (either pole fell or episode limit reached)
            if terminated or truncated:
                print(f"Episode {episode + 1}: {total_steps} steps, reward={total_reward:.2f}")
                total_rewards.append(total_reward)
                break

    env.close()
    print(f"\nAverage reward over {num_episodes} episodes: {sum(total_rewards)/num_episodes:.2f}")
