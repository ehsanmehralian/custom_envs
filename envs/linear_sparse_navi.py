import gym
from gym import spaces
import numpy as np

'''
---- DESCRIPTION ----

Navigation problem with linear dynamics (s' = s + a) and sparse reward.
There are multiple goals, with the furthest yielding the highest reward.
The initial position is fixed and the episode ends when a reward is collected.

With the default implementation, the highest reward is located in [20,20] and
needs 20 steps to be collected. To make it challenging, set max_episode_steps=25.
'''


class SparseNaviEnv(gym.Env):
    def __init__(self):
        self.size = 2  # dimensionality of state and action
        # self.action_space = spaces.Box(low=-1., high=1., shape=(self.size,))#, dtype=np.float32)
        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0., high=20., shape=(self.size,))#, dtype=np.float32)
        self.rwd_radius = 1.  # the reward is collected if the distance of the agent from the goal is within this radius
        # self.rwd_states = np.array([[1, 1], [-2, 3], [10, -2], [20, 20]])
        self.rwd_states = np.array([[0, 20], [20, 0]])
        self.rwd_magnitude = [50, 1]
        self.A = np.array([[0.9, 0.4], [-0.4, 0.9]])

    def step(self, u):
        # u = np.clip(u, self.action_space.low, self.action_space.high)
        u = self.directions[u]
        # self.state = self.A.dot(self.state)
        # rwd -= np.sum(0.01 * u ** 2)  # penalty on the action
        self.state = np.clip(self.state + u, self.observation_space.low, self.observation_space.high)
        dist = np.sqrt(np.sum((self.state - self.rwd_states) ** 2, 1))
        is_close = np.where(dist < self.rwd_radius)[0]
        if is_close.size == 0:
            rwd = 0.
            done = False
        else:
            rwd = self.rwd_magnitude[is_close[0]]
            done = True
        # u = np.clip(u, self.action_space.low, self.action_space.high)

        return self._get_obs(), rwd, done, {}

    def reset(self):
        self.state = np.array([0, 0])
        return self._get_obs()

    def _get_obs(self):
        return self.state

    def set_rewards(self, rewards):
        for i in range(len(self.rwd_magnitude)):
            self.rwd_magnitude[i] = rewards[i]
