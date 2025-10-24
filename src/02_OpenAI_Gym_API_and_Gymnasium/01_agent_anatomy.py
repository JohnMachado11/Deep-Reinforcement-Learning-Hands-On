from typing import List
import random


class Environment:
    def __init__(self):
        self.steps_left = 10
    
    def get_observation(self) -> List[float]:
        return [0.0, 0.0, 0.0]
    
    def get_actions(self) -> List[float]:
        return [0, 1]
    
    def is_done(self) -> bool:
        return self.steps_left == 0
    
    def action(self, action: int) -> float:
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1
        return random.random()


class Agent:
    def __init__(self):
        self.total_reward = 0.0
    
    def step(self, env: Environment, debug=False):
        current_obs = env.get_observation()
        actions = env.get_actions()
        chosen_action = random.choice(actions)
        # reward = env.action(random.choice(actions))
        reward = env.action(chosen_action)
        self.total_reward += reward
        if debug:
            self.debug(env, current_obs, actions, reward, chosen_action)
    
    def debug(self, env, current_obs, actions, reward, chosen_action):
        print("\n", "-" * 50)
        print("Steps completed:", 10 - env.steps_left)
        print("Current obs:", current_obs)
        print("Actions:", actions)
        print("Chosen Action:", chosen_action)
        print("Reward:", reward)
        print("Reward Accumulated:", self.total_reward)


if __name__ == "__main__":
    env = Environment()
    agent = Agent()
    while not env.is_done():
        agent.step(env, debug=True)
    print("\nTotal reward got: %.4f" % agent.total_reward)