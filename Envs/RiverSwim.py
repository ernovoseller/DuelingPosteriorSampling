# -*- coding: utf-8 -*-
"""
This file defines the RiverSwim environment and RiverSwim preference environment.

The RiverSwim environment is as described in "(More) Efficient Reinforcement 
Learning via Posterior Sampling" by Ian Osband, Benjamin Van Roy, and Daniel
Russo (2013).
"""

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

class RiverSwimEnv(gym.Env):
    """
    This class defines the RiverSwim MDP environment.
    """
    
    def __init__(self, num_states = 6):
        """
        Input: num_states = number of states in the MDP.
        """
        
        # Initialize state and action spaces.
        self.nA = 2     # Actions: left and right
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(num_states)
        self.nS = num_states
        
        self.states_per_dim = [num_states]  # Only one dimension
        self.store_episode_reward = False   # Track rewards at each step, not
                                            # over whole episode
        self.done = False                   # This stays false, since an episode
                                            # only finishes at its time horizon.
        
        self._seed()
        
        # Store transition probability matrix and rewards.
        # self.P[s][a] is a list of transition tuples (prob, next_state, reward)
        self.P = {}
        
        for s in range(self.nS):
            
            self.P[s] = {a : [] for a in range(self.nA)}
            
            for a in range(self.nA):
                
                if a == 0:  # Left action
                    
                    next_state = np.max([s - 1, 0])
                    
                    reward = 5/1000 if (s == 0 and next_state == 0) else 0
                    
                    self.P[s][a] = [(1, next_state, reward)]
                    
                elif s == 0:  # Leftmost state, and right action
                    
                    self.P[s][a] = [(0.4, s, 0), (0.6, s + 1, 0)]
                    
                elif s == self.nS - 1:   # Rightmost state, and right action
                    
                    self.P[s][a] = [(0.4, s - 1, 0), (0.6, s, 1)]
                    
                else:   # Intermediate state, and right action
                    
                    self.P[s][a] = [(0.05, s - 1, 0), (0.6, s, 0), 
                          (0.35, s + 1, 0)]

        # Reset the starting state:
        self._reset()
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        return self._reset()
    
    def step(self, action):
        return self._step(action)
    
    def _reset(self): # We always start at the leftmost state.
        
        self.state = 0
        return self.state
    
    def _step(self, action):
        """
        Take a step using the transition probability matrix specified in the 
        constructor.
        
        Input: action = 0 (left action) or 1 (right action).
        """        
        transition_probs = self.P[self.state][action]
        
        num_next_states = len(transition_probs)

        next_state_probs = [transition_probs[i][0] for i in range(num_next_states)]
            
        outcome = np.random.choice(np.arange(num_next_states), p = next_state_probs)
        
        self.state = transition_probs[outcome][1]    # Update state
        reward = transition_probs[outcome][2]
        
        return self.state, reward, self.done    # done = False always
    
    
    def get_step_reward(self, state, action, next_state):
        """
        Return the reward associated with a specific action from a state-action 
        pair, and the resulting next state.
        """
    
        transition_probs = self.P[state][action]
        
        reward = 0
        
        for i in range(len(transition_probs)):
            
            if transition_probs[i][1] == next_state:
                reward = transition_probs[i][2]
                break
            
        return reward
    
        
    def get_trajectory_return(self, tr):
        """
        Return the total reward accrued in a particular trajectory.
        """        
        states = tr[0]
        actions = tr[1]
        
        # Sanity check:        
        if not len(states) == len(actions) + 1:
            print('Invalid input given to get_trajectory_return.')
            print('State sequence expected to be one element longer than corresponding action sequence.')      
        
        total_return = 0
        
        for i in range(len(actions)):
            
            total_return += self.get_step_reward(states[i], actions[i], \
                                         states[i + 1])
            
        return total_return

         
      
class RiverSwimPreferenceEnv(RiverSwimEnv):
    """
    This class extends the RiverSwim environment to handle trajectory-
    preference feedback.
    
    The following changes are made to the RiverSwimEnv class defined above:
        1) Step function no longer returns reward feedback.
        2) Add a function that calculates a preference between 2 inputted
            trajectories.
    """

    def __init__(self, num_states = 6, user_noise_model):
        """
        user_noise_model specifies the degree of noisiness in the generated 
        preferences. See description of the function get_trajectory_preference
        for details.
        """
        
        self.user_noise_model = user_noise_model
        
        super().__init__(num_states)        # RiverSwimPreferenceEnv, self

    def _step(self, action):
        """
        Take a step using the transition probability matrix specified in the 
        constructor. This is identical to the RiverSwim class, except that now 
        we no longer return the reward.
        """
        state, _, done = super()._step(action)    # RiverSwimPreferenceEnv, self
        return state, done
       
   
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

