import gym
from gym.spaces import Box
import numpy as np


class History(gym.Wrapper):

    def __init__(self, env, history=2):
        super(History, self).__init__(env)
        self.history = history

        low = env.observation_space.low.repeat(history, 0)
        high = env.observation_space.high.repeat(history, 0)
        self.observation_space = Box(low=low, high=high)

    def reset(self):
        obs = super(History, self).reset()
        # repeat first frame n times
        obs = obs.repeat(self.history, 0)
        # remember history
        self._obs = obs
        return obs.copy()

    def step(self, action):
        obs, rew, done, info = super(History, self).step(action)
        # shift history
        self._obs = np.roll(self._obs, -1, axis=0)
        # replace last obs with new obs
        self._obs[-1] = obs
        return self._obs.copy(), rew, done, info
