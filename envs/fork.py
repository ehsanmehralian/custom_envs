import numpy as np
from gym import core, spaces
from gym.envs.registration import register
import gym

class Fork(gym.Env):
    def __init__(self):
        layout = """\
wwwwwwwwwwwwwwwwwwwwwww
wwwwwwww              w
wwwwwwww wwwwwwwwwwwwww
w                     w
wwwwwwww wwwwwwwwwwwwww
wwwwwwww              w
wwwwwwwwwwwwwwwwwwwwwww
"""
        self.occupancy = np.array([list(map(lambda c: 1 if c == 'w' else 0, line)) for line in layout.splitlines()])
        print(self.occupancy)
        print(len(self.occupancy))
        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))
        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        self.rng = np.random.RandomState(1234)

        self.tostate = {}
        statenum = 0
        for i in range(7):
            for j in range(23):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i, j)] = statenum
                    statenum += 1
        self.tocell = {v: k for k, v in self.tostate.items()}

        self.goals = [[13, 0.5], [35, 0.5], [50, 0.5]]
        # self.init_states = list(range(self.observation_space.n))
        # for goal in self.goals:
        #     self.init_states.remove(goal[0])

    def set_rewards(self, rewards):
        for i in range(len(self.goals)):
            self.goals[i][1] = rewards[i]

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):
        # state = self.rng.choice(self.init_states)
        state = 15
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
        reward = 0
        done = 0
        for i in range(len(self.goals)):
            if state == self.goals[i][0]:
                reward = self.goals[i][1]
        return state, reward, done, None


# register(
#     id='narrow_hallway-v0',
#     entry_point='narrow_hallway:Narrow_Hallway',
#     reward_threshold=1,
# )
