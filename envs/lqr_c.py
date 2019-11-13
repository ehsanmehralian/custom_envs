import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from numpy.linalg import inv

'''
---- DESCRIPTION ----

Very simple linear-quadratic regulator, with dynamics
s' = As + Ba

and reward
r = - s'Qs - a'Ru

In this simple task, A, B, Q, R are all identity matrices.
The initial state is drawn from a uniform distribution in [-s0,s0].
All value functions and average return can be computed in closed form if the
policy is linear in the state, i.e., a = Ks.
'''


class LqrEnv(gym.Env):
    def __init__(self, size=2, init_state=[0, 0], state_bound=1):
        self.init_state = init_state
        self.state_bound = state_bound
        self.desired = np.array([0, 0])
        self.size = size  # dimensionality of state and action
        self.action_bound = 1
        self.action_space = spaces.Box(low=-self.action_bound, high=self.action_bound,
                                       shape=(size,))  # , dtype=np.float32)

        # self.action_space = spaces.Discrete(4)
        # self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        self.observation_space = spaces.Box(low=-state_bound, high=state_bound, shape=(size,))  # , dtype=np.float32)
        self._seed()
        A_all = {}

        A_all[10] = np.array([[0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                              [0.01, 0.2, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                              [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                              [0.01, 0.01, 0.01, 0.6, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                              [0.01, 0.01, 0.01, 0.01, 0.8, 0.01, 0.01, 0.01, 0.01, 0.01],
                              [0.01, 0.01, 0.01, 0.01, 0.01, 0.7, 0.01, 0.01, 0.01, 0.01],
                              [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01, 0.01, 0.01],
                              [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.4, 0.01, 0.01],
                              [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.3, 0.01],
                              [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.5]])

        A_all[5] = np.array([[-0.2, 0.1, 0.1, 0.1, 0.1],
                             [0.1, 0.1, 0.1, 0.1, 0.1],
                             [0.1, 0.1, 0.5, 0.1, 0.1],
                             [0.1, 0.1, 0.1, 0.8, 0.1],
                             [0.1, 0.1, 0.1, 0.1, -0.9]])

        A_all[4] = np.array([[-0.2, 0.3, 0.3, 0.3],
                             [0.3, -0.4, 0.3, 0.3],
                             [0.3, 0.3, 0.3, 0.3],
                             [0.3, 0.3, 0.3, -0.1]])

        A_all[3] = np.array([[-0.5, -0.5, -0.5],
                             [0.3, -0.2, 0.3],
                             [0.3, 0.3, 0.4]])

        A_all[2] = np.array([[0.9, 0.4], [-0.4, 0.9]])

        A_all[1] = np.array([0.4])

        self.A = A_all[size]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        # u = np.clip(u, self.action_space.low, self.action_space.high)
        # costs = -(np.sum(u**2) + np.sum(self.state**2))
        # costs = -(np.dot(u.T, u) + np.dot((self.state - self.desired).T, (self.state - self.desired)))
        costs = -(np.dot(u.T, u) + np.dot((self.state - self.desired).T, (self.state - self.desired)))
        self.state = self.A.dot(self.state)
        self.state += u
        # self.state = np.clip(self.state + u, self.observation_space.low, self.observation_space.high)
        return self._get_obs(), costs, False, {}

    def reset(self):
        self.state = np.random.random(size=(self.size,)) * (2 * self.state_bound) - self.state_bound
        return self.state

        # high = self.init_state*np.ones((self.size,))
        # self.state = self.np_random.uniform(low=-high, high=high)
        # self.last_u = None
        # return self._get_obs()

    def _get_obs(self):
        return self.state

    def k_i(self, last_P):
        I = np.eye(self.size)
        return - np.dot(inv(I + last_P), last_P).dot(self.A)

    def p_i(self, k_i, last_P):
        I = np.eye(self.size)
        return I + np.dot(k_i.T, k_i) + np.dot((self.A + k_i).T, last_P).dot((self.A + k_i))

    def riccati_matrix(self, K, gamma=1.):
        tolerance = 0.0001
        converged = False
        itr = 0
        maxitr = 500
        I = np.eye(self.size)
        P = I
        Pnew = I + gamma * P + gamma * np.dot(K.T, P) + gamma * np.dot(P, K) + gamma * np.dot(K.T, P).dot(K) + np.dot(
            K.T, K)
        while not converged or itr < maxitr:
            P = Pnew
            Pnew = I + gamma * P + gamma * np.dot(K.T, P) + gamma * np.dot(P, K) + gamma * np.dot(K.T, P).dot(
                K) + np.dot(K.T, K)
            P_diff = P - Pnew
            if np.any(np.isnan(P_diff)) or np.any(np.isinf(P_diff)):
                break
            converged = np.max(P_diff) < tolerance
            itr += 1
        return P

    def v_function(self, K, state, gamma=1.):
        return - np.sum(np.dot(np.square(state), self.riccati_matrix(K, gamma)), axis=1)

    def q_function(self, K, state, action, gamma=1.):
        I = np.eye(self.size)
        tmp = state + action
        return - np.sum(np.square(state) + np.square(action), axis=1, keepdims=True) - gamma * np.dot(np.square(tmp),
                                                                                                      self.riccati_matrix(
                                                                                                          K, gamma))

    def avg_return(self, K, gamma=1.):
        P = self.riccati_matrix(K, gamma)
        high = self.init_state * np.ones((self.size,))
        Sigma_s = np.diag(high + high) ** 2 / 12.
        return - np.trace(Sigma_s * P)


