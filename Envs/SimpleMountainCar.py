# -*- coding: utf-8 -*-
"""
Implementation of the Simple MountainCar Preference Environment, as described 
in "EPMC: Every Visit Preference Monte Carlo for Reinforcement Learning" by C. 
Wirth and J. Furnkranz (2013) and in "Efficient Preference-based Reinforcement 
Learning" by C. Wirth (2017).
"""

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import math

class SimpleMountainCarEnv(gym.Env):
    """
    This class defines the Simple MountainCar environment.
    """

    def __init__(self):

        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.horizon = 500
        self.num_step = 0
        self.done = False

        self.force = 0.001
        self.gravity = 0.0023

        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.nA = self.action_space.n

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        """
        Action: 0 left 1 stay 2 right
        """

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state

        velocity += (action-1+np.random.uniform(-0.2,0.2))*self.force + math.cos(3*position)*(-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0

        self.num_step += 1
        
        reward = -1

        if position >= self.goal_position:
            self.done = True
        else:
            self.done = False

        if not self.done:
            self.state = (position, velocity)
        # otherwise state does not change
        return np.array(self.state), reward


    def reset(self, start_state = None):
        """
        Starting space is randomly sampled from the complete state space
        """
        self.num_step = 0
        self.done = False
        
        if start_state is None:    # By default, randomly-sample start state
            self.state = np.array([self.np_random.uniform(low=self.min_position, 
                                                          high=self.max_position),\
                                   self.np_random.uniform(low=-self.max_speed, 
                                                          high=self.max_speed)])
        else:   # Otherwise, set to user-defined value
            self.state = start_state
            
        return np.array(self.state)


class SimpleMountainCarDiscEnv(SimpleMountainCarEnv):
    """
    This is a wrapper for the SimpleMountainCarEnv class, which discretizes
    the state space into 100 bins (10 in the position dimension and 10 in the
    velocity dimension).
    """
    
    def __init__(self):

        super().__init__()
        
        self.states_per_dim = [10, 10]
        self.nS = np.prod(self.states_per_dim)

        num_state_dims = len(self.states_per_dim)
    
        self.thresholds = []  # Thresholds to use for discretizing state space
    
        for i in range(num_state_dims):
            self.thresholds.append(np.linspace(self.low[i], self.high[i], 
                                    num = self.states_per_dim[i] + 1)[:-1])
            
        self.store_episode_reward = True


    def reset(self, start_state = None):
        """
        We will keep track of the total reward for the current episode and the
        previous one; these are both initialized here.
        """

        state = super().reset(start_state)[0]

        return self.convert_state_values_to_state_idx(state, self.thresholds)


    def step(self, action):
        """
        For one step, reward is 1 if reaching the goal state, otherwise 0
        Action: 0 left 1 stay 2 right
        """        
        state, reward = super().step(action)
        state = self.convert_state_values_to_state_idx(state[0], self.thresholds)
        
        return state, reward


    def convert_state_values_to_state_idx(state, thresholds):
        """
        This function converts the state into a discretized state (with bins 
        defined according to the given thresholds). Then, it returns the 
        state's scalar index, which is used for updating the dynamics and 
        reward models.
        """
    
        # Convert to integer bins:
        num_dims = len(thresholds)
        state_bins = np.empty(num_dims)
    
        for i in range(num_dims):
    
            state_bins[i] = np.where(state[i] >= thresholds[i])[0][-1]
    
        # Convert to the state's scalar index:
        state_idx = 0
        prod = 1
    
        for i in range(num_dims - 1, -1, -1):
    
            state_idx += state_bins[i] * prod
            prod *= len(thresholds[i])
    
        return int(state_idx)



class SimpleMountainCarPreferenceEnv(SimpleMountainCarDiscEnv):
    """
    Extends SimpleMountainCarDiscEnv (defined above) to give feedback as
    preferences instead of numerical rewards.
    """

    def __init__(self, user_noise_model):
        """
        user_noise_model specifies the degree of noisiness in the generated 
        preferences. See description of the function get_trajectory_preference
        for details.
        """
        
        super().__init__()
        
        self.user_noise_model = user_noise_model

    def step(self, action):

        _, reward = super().step(action)

        self.episode_return += reward

        return self.state

    def reset(self, start_state = None):
        """
        We will keep track of the total reward for the current episode and the
        previous one; these are both initialized here.
        """

        super().reset(start_state)

        self.episode_return = 0

        return self.state


    def get_trajectory_return(self, tr):
        """
        Return the total reward accrued in a particular trajectory. For this 
        environment, the total reward is the negative of the number of actions
        taken in the trajectory.
        """
        return -len(tr[1])     # Negative length of action sequence
    

    def get_trajectory_preference(self, tr1, tr2):
        """
        Return a preference between two given state-action trajectories, tr1 and 
        tr2.
        
        Format of inputted trajectories: [[s1, s2, ..., sH], [a1, a2, ..., aH]]
        
        Preference information: 0 = trajectory 1 preferred; 1 = trajectory 2 
        preferred; 0.5 = trajectories preferred equally.
        
        Preferences are determined by comparing the rewards accrued in the 2 
        trajectories.
        
        self.user_noise_model takes the form [noise_type, noise_param].
        
        noise_type should be equal to 0, 1, or 2.
        noise_type = 0: deterministic preference; return 0.5 if tie.
        noise_type = 1: logistic noise model; user_noise parameter determines degree
        of noisiness.
        noise_type = 2: linear noise model; user_noise parameter determines degree
        of noisiness
        
        noise_param is not used if noise_type = 0. Otherwise, smaller values
        correspond to noisier preferences.
        """          
        
        # Unpack self.user_noise_model:
        noise_type, noise_param = self.user_noise_model

        assert (noise_type in [0,1,2]), "noise_type %i invalid" % noise_type
        
        trajectories = [tr1, tr2]
        num_traj = len(trajectories)
        
        returns = np.empty(num_traj)  # Will store cumulative returns for the 2 trajectories
        
        # Get cumulative reward for each trajectory
        for i in range(num_traj):
            
            returns[i] = self.get_trajectory_return(trajectories[i])
            
        if noise_type == 0:  # Deterministic preference:
            
            if returns[0] == returns[1]:  # Compare returns to determine the preference
                preference = 0.5
            elif returns[0] > returns[1]:
                preference = 0
            else:
                preference = 1
                
        elif noise_type == 1:   # Logistic noise model
            
            # Probability of preferring the 2nd trajectory:
            prob = 1 / (1 + np.exp(-noise_param * (returns[1] - returns[0])))
            
            preference = np.random.choice([0, 1], p = [1 - prob, prob])

        elif noise_type == 2:   # Linear noise model
            
            # Probability of preferring the 2nd trajectory:
            prob = noise_param * (returns[1] - returns[0]) + 0.5
            
            # Clip to ensure it's a valid probability:
            prob = np.clip(prob, 0, 1)

            preference = np.random.choice([0, 1], p = [1 - prob, prob])                
        
        return preference

