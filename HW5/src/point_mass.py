import numpy as np
from gym import spaces
from gym import Env


class PointEnv(Env):
    """
    point mass on a 2-D plane
    goals are sampled randomly from a square
    """

    def __init__(self, num_tasks=1):
        self.reset_task()
        self.reset()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, is_evaluation=False, square_size=-1):
        '''
        sample a new task randomly

        Problem 3: make training and evaluation goals disjoint sets
        if `is_evaluation` is true, sample from the evaluation set,
        otherwise sample from the training set
        '''
        #====================================================================================#
        #                           ----------PROBLEM 3----------
        #====================================================================================#
        # YOUR CODE HERE
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        
        # If square size is provided, sample from checkboard
        if square_size != -1:
            sq_per_row = (20 / square_size)
            sq_per_row_set = sq_per_row / 2
            
            # Checkboard coordinates
            idx = np.random.randint(low=0, high=sq_per_row_set, size=1)
            idy = np.random.randint(low=0, high=sq_per_row_set, size=1)

            # Sample from evaluation set (even positions top left)
            if is_evaluation:
                # Take location from evaluation positions
                x = np.random.uniform(-10 + (idx % 2) * square_size  + square_size * 2 * idy, 
                                      -10 + square_size + idy * square_size * 2 + square_size * (idx % 2))
                y = np.random.uniform(-10 + idx * square_size, 
                                      -10 + square_size * (idx + 1) )

            # Sample from training set (odd positions top left)
            else:
                # Take location from training positions
                x = np.random.uniform(-10 + ((idx % 2) + 1) * square_size  + square_size * 2 * idy, 
                                      -10 + square_size + idy * square_size * 2 + square_size * (1 + (idx % 2)))
                y = np.random.uniform(-10 + idx * square_size, 
                                      -10 + square_size * (idx + 1) )
                
        self._goal = np.array([x, y])

    def reset(self):
        self._state = np.array([0, 0], dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        return np.copy(self._state)

    def reward_function(self, x, y):
        return - (x ** 2 + y ** 2) ** 0.5

    def step(self, action):
        x, y = self._state
        # compute reward, add penalty for large actions instead of clipping them
        x -= self._goal[0]
        y -= self._goal[1]
        # check if task is complete
        done = abs(x) < .01 and abs(y) < .01
        reward = self.reward_function(x, y)
        # move to next state
        self._state = self._state + action
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

    def seed(self, seed):
        np.random.seed = seed
