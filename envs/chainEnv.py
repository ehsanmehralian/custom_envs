from gym.envs.registration import register
from gym import spaces
import numpy as np
import gym


class Chain(gym.Env):
    def __init__(self, n=10, slip=0.2, left=2, right=10):
        self.n = n
        self.slip = slip  # probability of 'slipping' an action
        self.left = left  # payout for 'backwards' action
        self.right = right  # payout at end of chain for 'forwards' action
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2 * self.n + 1)
        self.rng = np.random.RandomState(1234)
        self.AcReward = 0
        self.goals = [[0, right], [20, left]]

    def step(self, action):
        assert self.action_space.contains(action)
        if self.rng.uniform() < self.slip:
            action = not action  # agent slipped, reverse action taken
        reward = 0
        done = False
        if action:  # '1: backwards'
            # if self.state > self.n:
            #     self.state = self.n
            if self.state == 0:
                self.state = 0
            else:
                self.state -= 1
        else:
            # if self.state < self.n:
            #     self.state = self.n
            if self.state == 2 * self.n:
                self.state = 2 * self.n
            else:
                self.state += 1

        if self.state == 0:
            reward = self.goals[0][1]
        elif self.state == 2 * self.n:
            reward = self.goals[1][1]
        self.AcReward += reward

        # if self.right > self.left:
        #     if self.state == 2 * self.n:
        #         done = True
        # elif self.right <= self.left:
        #     if self.state == 0:
        #         done = True
        # if self.AcReward >= 10:
        #     done = True
        # else:
        #     done = False
        return self.state, reward, done, {}

    def reset(self):
        self.state = self.n
        self.AcReward = 0
        return self.state

    def reset_random_start(self):
        state = self.rng.randint(0, 2 * self.n)
        return state

    def set_rewards(self, rewards):
        self.goals[0][1] = rewards[0]
        self.goals[1][1] = rewards[1]

    def set_agent_state(self, input_state):
        self.state = input_state
        return self.state



# register(
#     id='chain-v0',
#     entry_point='chainEnv:Chain',
#     reward_threshold=1,
# )

