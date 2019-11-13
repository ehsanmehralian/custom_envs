import numpy as np
from gym import core, spaces
from gym.envs.registration import register
import gym


# layout = """\
# wwwwwwwwwwwww
# wr    w    rw
# w     w     w
# w           w
# w     w     w
# w     w     w
# ww wwww     w
# w     www www
# w     w     w
# w     w     w
# w           w
# wr    w    Gw
# wwwwwwwwwwwww
# """
class Fourrooms(gym.Env):
    def __init__(self):
        layout = """\
wwwwwwwwwwwww
w     w     w
w           w
w           w
w     w     w
w     w     w
ww  www     w
w     www  ww
w     w     w
w           w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.occupancy = np.array([list(map(lambda c: 1 if c == 'w' else 0, line)) for line in layout.splitlines()])
        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))

        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        self.rng = np.random.RandomState(1234)

        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i, j)] = statenum
                    statenum += 1
        self.tocell = {v: k for k, v in self.tostate.items()}

        # self.goal = [[103, 1]]
        # self.goals = []
        # for i in range(self.observation_space.n):
        #     self.goals.append([i, 0])
        self.goals = [[26, 0.5], [107, 0.5]]
        # self.goals = [[0, 0.5], [26, 0.5], [64, 0.5], [107, 0.5], [9, 1]]
        self.init_states = list(range(self.observation_space.n))
        for goal in self.goals:
            self.init_states.remove(goal[0])

    def set_rewards(self, rewards):
        fixed_goals = [26, 107]
        for i in range(len(fixed_goals)):
            self.goals[i][1] = rewards[i]

    def set_random_rewards(self, rewards):
        goal = self.rng.choice(self.init_states)
        # for i in fixed_goals:
        self.goals[goal][1] = rewards[0]

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):
        # state = self.rng.choice(self.init_states)
        state = 98
        self.currentcell = self.tocell[state]
        return state

    def set_agent_state(self, location):
        state = location
        self.currentcell = self.tocell[state]
        return state

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.

        We consider a case in which rewards are zero on all state transitions.
        """
        nextcell = tuple(self.currentcell + self.directions[action])
        if not self.occupancy[nextcell]:
            if self.rng.uniform() < 1 / 3.:
                empty_cells = self.empty_around(self.currentcell)
                self.currentcell = empty_cells[self.rng.randint(len(empty_cells))]
            else:
                self.currentcell = nextcell

        state = self.tostate[self.currentcell]
        done = reward = 0
        for i in range(len(self.goals)):
            if state == self.goals[i][0]:
                reward = self.goals[i][1]
                # self.goals[i][1] = 0
                # if state == 9:
                #     done = 1
        return state, reward, done, None
