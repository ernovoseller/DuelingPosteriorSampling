# -*- coding: utf-8 -*-
"""
Implementation of the Every-Visit Preference Monte Carlo (EPMC) algorithm with 
probabilistic temporal credit assignment.

This algorithm is described in "A Policy Iteration Algorithm for Learning
from Preference-Based Feedback" by C. Wirth and J. Furnkranz (2013) and in 
"Efficient Preference-based Reinforcement Learning" by C. Wirth (2017).
"""

import numpy as np
from collections import defaultdict


def initialize_policy(num_states, num_actions):
    """
    Generate a deterministic policy by selecting a (uniformly-random) action
    to take in each state.
    
    Inputs: 1) num_states: number of states in the MDP.
            2) num_actions: number of actions in the MDP.
    
    Output: a policy, specified by a num_states x num_actions matrix, in which
            policy[s][a] is the probability of taking action a in state s.
    """
    
    policy = np.zeros((num_states, num_actions))

    # Random actions to take in each state:
    actions = np.random.choice(num_actions, size = num_states)
    
    for state in range(num_states):
        
        policy[state, actions[state]] = 1
        
    return policy
    

def get_deterministic_policy(Q):
    """
    Return a deterministic policy that acts greedily with respect to a given
    Q-function.

    Inputs: 1) Q: a num_states x num_actions matrix, in which Q[s][a] specifies
               the Q-function in state s and action a.    
    
    Output: a policy, specified by a num_states x num_actions matrix, in which
            policy[s][a] is the probability of taking action a in state s.
    """
    
    [num_states, num_actions] = Q.shape
    
    policy = np.zeros((num_states, num_actions))
    
    # Select action in each state that has the highest Q-value. Break ties
    # randomly.
    for state in range(num_states):
        
        max_idxs = np.where(Q[state, :] == np.max(Q[state, :]))[0]
        
        policy[state, np.random.choice(max_idxs)] = 1
        
    return policy


def get_EXP3_policy(Q, eta, G_previous):
    """
    Obtain EXP-3 policy based on a given Q-function. Also, return updated 
    values of G, to be used in future calls to this function.
    
    Inputs:
        1) Q:  a num_states x num_actions matrix, in which Q[s][a] specifies
               the Q-function in state s and action a.   
        2) eta: a scalar; this is the eta parameter defined in the EPMC algorithm.
        3) G_previous: num_states x num_actions matrix; this is a matrix of the
           G-values defined in the EPMC algorithm. These values are from the 
           previous iteration.
           
    Outputs:
        1) policy: a policy, specified by a num_states x num_actions matrix, in 
           which policy[s][a] is the probability of taking action a in state s.
        2) G: num_states x num_actions updated G matrix, as defined in the EPMC
           algorithm.
    """
    
    num_actions = Q.shape[1]
    
    # Update the policy:
    policy = np.exp((eta / num_actions) * G_previous)    
    policy = (policy.T / policy.sum(axis=1)).T
    policy = eta / num_actions + (1 - eta) * policy
    
    # Update G:
    G = G_previous + Q / policy

    return policy, G


def prob_to_Q(preference_probs):
    """
    Convert from estimates of the probabilities P(a' > a | s) to a Q-function.

    Input:
        1) preference_probs: a num_states x num_actions x num_states matrix.
           If a > a', then preference_probs[s, a, a'] gives an estimate of the 
           probability P(a > a' | s). If a < a', then preference_probs[s, a, a'] 
           gives an estimate of 1 - P(a > a' | s).

    Output: 
        1) Q:  a num_states x num_actions matrix, in which Q[s][a] specifies
               the Q-function in state s and action a.   
    """
        
    [num_states, num_actions, _] = preference_probs.shape
    
    Q = np.ones((num_states, num_actions))
    
    for state in range(num_states):
        for action in range(num_actions):
            
            denom = 2 - num_actions
            
            for action_ in range(num_actions):
                
                if action_ != action:
                    
                    if action > action_:
                        prob = preference_probs[state, action, action_]
                    else:
                        prob = 1 - preference_probs[state, action_, action]
                    
                    denom += 1 / prob
            
            Q[state, action] /= denom
    
    return Q
    
    
def policy_improvement(P_prev, P_sampled, alpha):
    """
    Perform policy improvement step. This function interpolates between the
    previous and newly sampled estimates for the probabilities P(a' > a | s).

    Please see the description of the prob_to_Q function above for a more 
    detailed description of the format of the estimates P(a' > a | s). This 
    applies to P_prev, P_sampled, and the function's output.
    
    Inputs: 1) P_prev: previous estimates of P(a' > a | s); this is a 
               num_states x num_actions x num_states matrix.
            2) P_sampled:  newly-sampled estimates for P(a' > a | s); this is a 
               num_states x num_actions x num_states matrix.
            3) alpha: number between 0 and 1; this is the learning rate 
               hyperparameter.
    
    Output: num_states x num_actions x num_states interpolation between P_prev
            and P_sampled. This will be used as a new estimate of P(a' > a | s).
    """
    
    return (1 - alpha) * P_prev + alpha * P_sampled
    

def sampled_state_action_preference_probs(T1, T2, preference, P_prev):
    """
    Determine new state/action preference estimates given a newly-sampled 
    preference.

    Please see the description of the prob_to_Q function above for a more 
    detailed description of the format of the estimates P(a' > a | s). This 
    applies to P_prev and the function's output.
    
    Inputs:
        1) T1: first trajectory in the comparison.
           Format of inputted trajectories: [[s1, s2, ..., sH], 
           [a1, a2, ..., aH]]
        2) T2: second trajectory in the comparison.
        3) preference: 0 if trajectory T1 preferred to T2; 1 if T2 preferred
           to T1.
        4) P_prev: previous estimates of P(a' > a | s); this is a 
               num_states x num_actions x num_states matrix.
               
    Output:
        1) P: newly-sampled estimate of P(a' > a | s); this is a 
               num_states x num_actions x num_states matrix.
    """
    
    [num_states, num_actions, _] = P_prev.shape
    
    # Unpack the data:
    states_T1 = T1[0][:-1]
    actions_T1 = T1[1]
    states_T2 = T2[0][:-1]
    actions_T2 = T2[1]
    
    # Set of overlapping states:
    S = np.intersect1d(states_T1, states_T2)
    
    n = len(S)     # Number of overlapping states
    
    # Values to use as samples:
    sample_vals = [(n + 1) / (2*n), (n - 1) / (2*n)]

    # Use a defaultdict object to keep track of all preference samples:
    samples = defaultdict(lambda: [])
    
    for overlapping_state in S:  # Consider each overlapping state.
        
        # Indices where this overlapping state occurs:
        T1_idxs = np.where(states_T1 == overlapping_state)[0]
        T2_idxs = np.where(states_T2 == overlapping_state)[0]
            
        actions_1 = np.unique(actions_T1[T1_idxs])
        actions_2 = np.unique(actions_T2[T2_idxs])
        
        for a1 in actions_1:
            for a2 in actions_2:
                
                if a1 > a2:
                    
                    samples[overlapping_state, a1, a2].append(\
                           sample_vals[preference])
                    
                elif a1 < a2:
                    
                    samples[overlapping_state, a2, a1].append(\
                           sample_vals[1 - preference])                   
                    
    # Initialize probabilities to their previous values, so that for non-
    # updated values, the probabilities will not change in the policy 
    # improvement step:
    P = np.copy(P_prev)            
            
    # Take mean of sampled values in each case.
    for s in range(num_states):
        for a1 in range(num_actions):
            for a2 in range(num_actions):
                
                vals = samples[s, a1, a2]
                
                if len(vals) > 0:
                    P[s, a1, a2] = np.mean(vals)
    
    return P


def EPMC(time_horizon, hyper_params, env, num_iter, run_str = '', seed = None):
    """
    Implementation of the EPMC algorithm.
    
    Inputs:
        1) time_horizon: episode horizon; this is the number of state/action 
           pairs in each learning episode.
        2) hyper_params: a list [alpha, eta], where alpha and eta are the 
           alpha and eta hyperparameters defined in the EPMC algorithm.
        3) env: the RL environment.
        4) num_iter: the number of iterations of the learning algorithm to run.
           Note that two trajectory rollouts occur per iteration of learning.
        5) run_str: if desired, a string with information about the current
           call to EPMC (e.g. hyperparameter values or repetition number), 
           which can be useful for print statements to track progress.
        6) seed: seed for random number generation.
 
    
    Returns: a vector of rewards received as the algorithm runs. This is either
             a) the total rewards from each trajectory rollout, or b) the 
             rewards at every step taken in the environment (the environment
             determines whether a) or b) is used).
    """    
 
    if not seed is None:
        np.random.seed(seed)

    # Unpack hyperparameters:
    [alpha, eta] = hyper_params
    
    # Numbers of states and actions in the environment:
    num_states = env.nS
    num_actions = env.nA
    
    # Sample initial deterministic policy:
    determ_policy = initialize_policy(num_states, num_actions)
    
    # Initialize G values for use with EXP3:
    G = np.zeros((num_states, num_actions))
    
    # Initialize state-action preference information:
    pref_probs = 0.5 * np.tril(np.ones((num_states, num_actions, num_actions)))
    
    # Initial EXP3 policy prefers all actions equally:
    EXP3_policy = (1/num_actions) * np.ones((num_states, num_actions))

    # To store results (for evaluation purposes only):
    num_policies = 2     # Number of policies to roll out per learning iteration
    
    if env.store_episode_reward:   # Store total reward for each trajectory
        rewards = np.empty(num_iter * num_policies)
    else:    # Store reward at each step within each trajectory
        rewards = np.empty(num_iter * time_horizon * num_policies)
    
    reward_count = 0   # Counts how many values in the "rewards" variable 
                       # defined above have been populated

    """
    Here is where the learning algorithm begins.
    """    
    for i in range(num_iter):
        
        trajectories = []    # Trajectories to be sampled in this iteration
        
        # To keep track of which states have already been visited:
        already_visited_states = np.empty(0)
        
        for j in range(num_policies):   # Roll out 2 trajectories per iteration

            # State and action sequence to be sampled in this trajectory:
            state_sequence = np.empty(time_horizon + 1)
            action_sequence = np.empty(time_horizon)
        
            state = env.reset()
            
            for t in range(time_horizon):        
                
                # Select next action via EXP3 policy if 1st trajectory sample
                # or visiting an overlapping state:
                if j == 0 or state in already_visited_states:
                    
                    action = np.random.choice(num_actions, p = EXP3_policy[state, :])
                    
                else:    # Otherwise, use deterministic policy.
                    
                    action = np.argmax(determ_policy[state, :])
                
                next_state, _, _ = env.step(action)
                
                state_sequence[t] = state
                action_sequence[t] = action
                
                # Tracking rewards for evaluation purposes (in case of 
                # tracking rewards at every single step):
                if not env.store_episode_reward:
                    rewards[reward_count] = env.get_step_reward(state, 
                                action, next_state)
                    reward_count += 1

                # Terminate trajectory if environment turns on "done" flag.
                if env.done:
                    state_sequence = state_sequence[: t + 2]
                    action_sequence = action_sequence[: t + 1]
                    
                    break

                state = next_state
            
            state_sequence[-1] = next_state
            trajectories.append([state_sequence, action_sequence])  

            # Tracking rewards for evaluation purposes (in case of tracking
            # rewards just over entire episodes):
            if env.store_episode_reward:
    
                rewards[reward_count] = env.get_episode_return()
                reward_count += 1
            
            # Update list of visited states:
            already_visited_states = np.unique(state_sequence[:-1])
            
        # Obtain a preference between the two trajectories:
        preference = env.get_trajectory_preference(trajectories[0], 
                                            trajectories[1])
        
        # Determine new state-action preference information given newly-
        # sampled preference.
        pref_probs_sampled = sampled_state_action_preference_probs(trajectories[0], 
                                    trajectories[1], preference, pref_probs) 
        
        # Policy improvement step:
        pref_probs = policy_improvement(pref_probs, pref_probs_sampled, alpha)
    
        # Convert from probabilities P(a' > a | s) to Q-function.
        Q = prob_to_Q(pref_probs)    

        # Deterministic policy for next iteration:
        determ_policy = get_deterministic_policy(Q)
        
        # EXP3 policy for next iteration:
        EXP3_policy, G = get_EXP3_policy(Q, eta, G)
        
    # Return performance results:
    return rewards
