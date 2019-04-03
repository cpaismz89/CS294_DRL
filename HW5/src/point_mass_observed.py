import numpy as np
from gym import spaces
from gym import Env


class ObservedPointEnv(Env):
    """
    point mass on a 2-D plane
    four tasks: move to (-10, -10), (-10, 10), (10, -10), (10, 10)

    Problem 1: augment the observation with a one-hot vector encoding the task ID
     - change the dimension of the observation space
     - augment the observation with a one-hot vector that encodes the task ID
    """
    #====================================================================================#
    #                           ----------PROBLEM 1----------
    #====================================================================================#
    # YOUR CODE SOMEWHERE HERE
    def __init__(self, num_tasks=1):
        self.tasks = [0, 1, 2, 3][:num_tasks]
        self.task_idx = -1
        self.reset_task()
        self.reset()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        #print("Original Observation space:", self.observation_space)
        
        # Modify the observation space
        # Low and high space mod for one hot
        taskSpaceLow = np.zeros(num_tasks)
        taskSpaceHigh = np.ones(num_tasks)
        #print("\nSpaceLow:", taskSpaceLow, 
        #      "\nSpaceHigh:", taskSpaceHigh)
        
        # Full low and high spaces
        lowSpace = np.concatenate(([-np.inf, -np.inf], taskSpaceLow), None)
        HighSpace = np.concatenate(([np.inf, np.inf], taskSpaceHigh), None)    
        #print("\nlowSpace:", lowSpace, 
        #      "\nHighSpace:", HighSpace)
        
        self.observation_space = spaces.Box(low=lowSpace, high=HighSpace)
        #print("New Observation space:", self.observation_space)
        
        
        #print("Observation sample:", self.observation_space.sample())
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))
        #print("Action space:", self.action_space)
        
    def reset_task(self, is_evaluation=False):
        # for evaluation, cycle deterministically through all tasks
        if is_evaluation:
            self.task_idx = (self.task_idx + 1) % len(self.tasks)
        # during training, sample tasks randomly
        else:
            self.task_idx = np.random.randint(len(self.tasks))
        self._task = self.tasks[self.task_idx]
        goals = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        self._goal = np.array(goals[self.task_idx])*10

    def reset(self):
        self._state = np.array([0, 0], dtype=np.float32)
        
        # One hot codification using numpy
        #OneHot = tf.one_hot(tasks, depth=len(tasks), axis=0,dtype=tf.int16)[taskidx]
        #print("Tasks:", self.tasks)
        n_values = np.max(self.tasks) + 1
        oneHot = np.eye(n_values)[self.task_idx].astype(np.float32)
                
        self._state = np.concatenate((self._state, oneHot), None)
        
        #print("Reseted state:", self._state)
        
        return self._get_obs()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        # Extract only the coordinates, not the one hot vector
        x, y = self._state[:2]
        # compute reward, add penalty for large actions instead of clipping them
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        # check if task is complete
        done = abs(x) < 0.01 and abs(y) < 0.01
        # move to next state
        self._state[:2] = self._state[:2] + action
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

    def seed(self, seed):
        np.random.seed = seed
