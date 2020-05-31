# -*- coding: utf-8 -*-
"""
Implementation of the Simple MountainCar Environment and the Simple Mountain
Car Preference Environment; these are as described in "EPMC: Every Visit 
Preference Monte Carlo for Reinforcement Learning" by C. Wirth and J. Furnkranz 
(2013) and in "Efficient Preference-based Reinforcement Learning" by C. Wirth
(2017).

This, in turn, is a simplification of the OpenAI Gym Mountain car environment,
which can be found at
https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py,
and comes with the following license:
    
    The MIT License

Copyright (c) 2016 OpenAI (https://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
        """
        Constructor for the SimpleMountainCar environment class.
        """
        
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.horizon = 500
        self.num_step = 0
        self.done = False     # Keeps track of whether episode has finished

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
        Take a step in the environment, given an action.
        
        Input: the action can have 3 possible values. 0 = left, 1 = stay,
               2 = right
        Returns: 1) state, 2) reward
        """

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state

        velocity += (action-1+np.random.uniform(-0.2,0.2))*self.force + math.cos(3*position)*(-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0

        self.num_step += 1
        
        reward = -1    # Reward is always -1

        # Check if agent reached the goal:
        if position >= self.goal_position:
            self.done = True
        else:
            self.done = False

        # Update state:
        self.state = (position, velocity)

        return np.array(self.state), reward


    def reset(self, start_state = None):
        """
        Reset initial state, so that we can start a new episode. Starting spate 
        is uniformly-randomly sampled from the complete state space.
        
        Input (optional): if start_state is passed in, then use this as the
                          starting state instead of sampling it randomly.
        Returns: current state
        """
        self.num_step = 0
        self.done = False
        
        if start_state is None:   # By default, randomly-sample start state
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
        
        # Discretize into 10 bins in each dimension:
        self.states_per_dim = [10, 10]
        self.nS = np.prod(self.states_per_dim)    # Number of states

        num_state_dims = len(self.states_per_dim) # Dimensionality of state space
    
        self.thresholds = []  # Thresholds to use for discretizing state space
    
        for i in range(num_state_dims):
            self.thresholds.append(np.linspace(self.low[i], self.high[i], 
                                    num = self.states_per_dim[i] + 1)[:-1])
            
        self.store_episode_reward = True   # Track rewards over whole episode,
                                           # not at each step


    def reset(self, start_state = None):
        """
        Calls reset function of parent class. Only difference: converts the
        state to a discretized state.
        """

        state = super().reset(start_state)[0]

        # Convert to discretized state:
        return self.convert_state_values_to_state_idx(state, self.thresholds)


    def step(self, action):
        """
        Take a step in the environment, given an action.

        Calls step function of parent class. Only difference: converts the
        state to a discretized state.
        
        Input: the action can have 3 possible values. 0 = left, 1 = stay,
               2 = right
        Returns: 1) discretized state, 2) reward

        """   
        state, reward = super().step(action)

        # Convert to discretized state:
        state = self.convert_state_values_to_state_idx(state[0], self.thresholds)
        
        return state, reward


    def convert_state_values_to_state_idx(state, thresholds):
        """
        This function converts the state into a discretized state (with bins 
        defined according to the given thresholds). Then, it returns the 
        scalar index associated with that state.
        
        Inputs: 1) state: [position, velocity]
                2) thresholds: [position_thresholds, velocity_thresholds],
                   where position_thresholds and velocity_thresholds are both
                   arrays of thresholds specifying the bins for discretizing
                   the state space along the given parameter. 
                   E.g., position_thresholds[0] is the lower bound of the 
                   position range and position_thresholds[1] is the divide
                   between the 1st and 2nd position bins.
            
        Returns: integer index of the current discretized state.
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
    preferences over trajectories instead of numerical rewards at each step.
    """

    def __init__(self, user_noise_model):
        """
        Arguments:
            1) user_noise_model: specifies the degree of noisiness in the 
                   generated preferences. See description of the function 
                   get_trajectory_preference for details.
        """
        
        super().__init__()
        
        self.user_noise_model = user_noise_model

    def step(self, action):
        """
        Take a step in the environment given the specified action. This is 
        identical to the parent class, except that now, we no longer return the 
        reward.
        """
        _, reward = super().step(action)

        return self.state

    def reset(self, start_state = None):

        super().reset(start_state)
        return self.state


    def get_trajectory_return(self, tr):
        """
        Return the total reward accrued in a particular trajectory. For this 
        environment, the total reward is the negative of the number of actions
        taken in the trajectory.
        
        Input: a trajectory.        
        Format of inputted trajectory: [[s1, s2, ..., sH], [a1, a2, ..., aH]]
        """
        return -len(tr[1])     # Negative length of action sequence
    

    def get_trajectory_preference(self, tr1, tr2):
        """
        Return a preference between two given trajectories of states and 
        actions, tr1 and tr2.
        
        Format of inputted trajectories: [[s1, s2, ..., sH], [a1, a2, ..., aH]]
        
        Preference information: 0 = trajectory 1 preferred; 1 = trajectory 2 
        preferred; 0.5 = trajectories preferred equally (i.e., a tie).
        
        Preferences are determined by comparing the rewards accrued in the 2 
        trajectories.
        
        self.user_noise_model takes the form [noise_type, noise_param].
        
        noise_type should be equal to 0, 1, or 2.
        noise_type = 0: deterministic preference; return 0.5 if tie.
        noise_type = 1: logistic noise model; user_noise parameter determines 
        degree of noisiness.
        noise_type = 2: linear noise model; user_noise parameter determines 
        degree of noisiness
        
        noise_param is not used if noise_type = 0. Otherwise, smaller values
        correspond to noisier preferences.
        """          
        
        # Unpack self.user_noise_model:
        noise_type, noise_param = self.user_noise_model

        assert (noise_type in [0,1,2]), "noise_type %i invalid" % noise_type
        
        trajectories = [tr1, tr2]
        num_traj = len(trajectories)
        
        # For both trajectories, determine cumulative reward / total return:
        returns = np.empty(num_traj)
        
        # Get cumulative reward for each trajectory
        for i in range(num_traj):
            
            returns[i] = self.get_trajectory_return(trajectories[i])
            
        if noise_type == 0:  # Deterministic preference:
            
            if returns[0] == returns[1]:  # Compare returns to determine preference
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

