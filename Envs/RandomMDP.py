# -*- coding: utf-8 -*-
"""
This script defines the Random MDP environment. This is based on the description
in "(More) Efficient Reinforcement Learning via Posterior Sampling" by Ian 
Osband, Benjamin Van Roy, and Daniel Russo (2013), though with modifications 
to the distributions from which the random MDP parameters are generated.
"""

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np


class RandomMDPEnv(gym.Env):
    """
    This class defines the Random MDP environment.
    """
    
    def __init__(self, num_states = 10, num_actions = 5,
                 lambd = 5, diri_prior = 1, transition_arg = [], 
                 reward_arg = [], P0_arg = []):
        """
        The constructor samples a random MDP (rewards, dynamics, and initial
        state distribution) from the prior to define the environment.
        Alternatively, the MDP parameters can be passed in as arguments,
        rather than sampled randomly; this is meant to allow for sampling the
        random MDPs and saving them, so that the same set of random MDPs can
        be repeatedly re-used.
        
        Arguments:
            1) num_states: number of states in the MDP
            2) num_actions: number of actions in the MDP
            3) lambd: parameter for exponential distribution; this is used for
               sampling the rewards
            4) diri_prior: Dirichlet prior parameter for sampling the 
               state/action transition probabilities and initial state 
               distribution.
            5) transition_arg: transition probabilities; if passed in, then
                use these instead of generating them randomly. Matrix of size
               (num_states, num_actions, num_states), in which element
               [s, a, s_next] is the probability of transitioning to state 
               s_next when taking action a in state s.
            6) reward_arg: reward matrices; if passed in, then use this instead
               of generating them randomly. Matrix of size (num_states, 
               num_actions, num_states), in which rewards[s, a, s_next] is 
               the reward when taking action a in state s and transitioning to
               state s_next.
            7) P0_arg: initial state probability vector; if passed in, then use 
               this instead of generating it randomly. Vector of length 
               num_states.
        """
        
        # Initialize state and action spaces.
        self.nA = num_actions
        self.action_space = spaces.Discrete(self.nA)
        self.nS = num_states
        self.observation_space = spaces.Discrete(self.nS)
        
        self.states_per_dim = [num_states]  # State space has only one dimension
        self.store_episode_reward = False   # Track rewards at each step, not
                                            # over whole episode
        self.done = False                   # This stays false: in this 
        # environment, an episode can only finish at the episode time horizon.
        
        self._seed()
        
        # Check if valid transition and reward information has been passed in:
        transition_passed = False
        reward_passed = False
        init_prob_passed = False
        
        if transition_arg != []:
            
            if transition_arg.shape == (num_states, num_actions, num_states):
                transition_passed = True
            else:
                print('Error: not using inputted transition information.')

        if P0_arg != []:
            
            if P0_arg.shape == (num_states, ):
                init_prob_passed = True
            else:
                print('Error: not using inputted initial state probabilities.')

        if reward_arg != []:
            
            if reward_arg.shape == (num_states, num_actions, num_states):
                reward_passed = True
            else:
                print('Error: not using inputted reward information.')
        
        # Sample rewards from exponential distribution, if not inputted:
        if not reward_passed:
            
            rewards = np.random.exponential(lambd, 
                                    size = (num_states, num_actions, num_states))
    
            rewards = rewards - np.min(rewards)
            rewards = rewards / np.mean(rewards)

                        
        else:
            rewards = reward_arg
                    
        # Construct transition probability matrix and rewards. Format:
        # self.P[s][a] is a list of transition tuples (prob, next_state, reward).
        
        # Dirichlet distribution to use for sampling the dynamics parameters:
        dirichlet_prior = diri_prior * np.ones(num_states)

        self.P = {}
        
        for s in range(self.nS):
            
            self.P[s] = {a : [] for a in range(self.nA)}
            
            for a in range(self.nA):
                
                # If transition dynamics were passed in as an argument, then 
                # use those:
                if transition_passed:
                    transition_probs = transition_arg[s, a, :]
                else:     # Otherwise, sample transition dynamics:
                    transition_probs = np.random.dirichlet(dirichlet_prior)
                
                for s_next in range(self.nS):
                    
                    self.P[s][a].append((transition_probs[s_next], 
                            s_next, rewards[s, a, s_next]))

        # Sample initial state distribution (or use inputted values, if given):
        if not init_prob_passed:
            self.P0 = np.random.dirichlet(dirichlet_prior)
        else:
            self.P0 = P0_arg

        # Reset the starting state:
        self._reset()
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        return self._reset()
    
    def step(self, action):
        return self._step(action)
    
    def _reset(self):   # Draw a sample from the initial state distribution.
        """
        Reset initial state, so that we can start a new episode. Sample from
        the initial state distribution.
        """
        
        outcome = np.random.choice(np.arange(self.nS), p = self.P0)
        
        self.state = outcome
        return self.state
    

    def _step(self, action):
        """
        Take a step using the transition probability matrix specified in the 
        constructor.
        """
        
        transition_probs = self.P[self.state][action]
        
        num_next_states = len(transition_probs)

        next_state_probs = [transition_probs[i][0] for i in range(num_next_states)]
            
        outcome = np.random.choice(np.arange(num_next_states), p = next_state_probs)
        
        self.state = transition_probs[outcome][1]    # Update state
        reward = transition_probs[outcome][2]
        
        return self.state, reward, False   # done = False, since finite-horizon setting


    def get_step_reward(self, state, action, next_state):
        """
        Return the reward corresponding to the given state, action and 
        subsequent state.
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
        Format of inputted trajectory: [[s1, s2, ..., sH], [a1, a2, ..., aH]]
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


class RandomMDPPreferenceEnv(RandomMDPEnv):
    """
    This class is a wrapper for the Random MDP environment, which gives
    preferences over trajectories instead of absolute feedback.
    
    The following changes are made to the RandomMDPEnv class defined above:
        1) The step function no longer returns reward feedback.
        2) We add a function that calculates a preference between 2 inputted
            trajectories.

    """

    def __init__(self, user_noise_model, num_states = 10, num_actions = 5,
                 lambd = 5, diri_prior = 1, transition_arg = [], 
                 reward_arg = [], P0_arg = []):

        """       
        Arguments:
            1) user_noise_model: specifies the degree of noisiness in the 
                   generated preferences. See description of the function 
                   get_trajectory_preference for details.
            2-8) Identical to the arguments in the constructor of 
                 RandomMDPEnv above.
        """
    
        super().__init__(num_states, num_actions, lambd, diri_prior, 
             transition_arg, reward_arg, P0_arg)
        
        self.user_noise_model = user_noise_model


    def _step(self, action):
        """
        Take a step using the transition probability matrix specified in the 
        constructor. This is identical to the RandomMDP class, except that now 
        we no longer return the reward.
        """
        state, _, done = super()._step(action)    # RandomMDPPreferenceEnv, self
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

