import numpy as np
import gym
from gym.spaces import Discrete, Dict, Box
from Game2048 import GameEnv
from copy import deepcopy

class Game2048:
    """
    Wrapper for gym CartPole environment where the reward
    is accumulated to the end
    """
    def __init__(self, config=None):
        self.env = GameEnv()
        self.action_space = self.env.action_space
        self.observation_space = Dict({
            "obs": self.env.observation_space,
            "action_mask": Box(low=self.action_space.low, high=self.action_space.high, shape=(self.action_space.n,), dtype=np.uint8)
        })
        self.running_reward = 0.0

    def reset(self):
        self.running_reward = 0.0
        return {"obs": self.env.reset(), "action_mask": np.ones((self.action_space.n,), dtype=np.uint8)}

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.running_reward += rew
        score = self.running_reward if done else 0.0
        return {"obs": obs, "action_mask": np.ones((self.action_space.n,), dtype=np.uint8)}, score, done, info

    def set_state(self, state):
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        obs = self.env.state.reshape((16,)) # I do not like this !!!
        return {"obs": obs, "action_mask": np.ones((self.action_space.n,), dtype=np.uint8)}

    def get_state(self):
        return deepcopy(self.env), self.running_reward
