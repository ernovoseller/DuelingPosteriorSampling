# Dueling Posterior Sampling for Preference-Based Reinforcement Learning

This repository contains code for reproducing the simulation results in the paper:

**Dueling Posterior Sampling for Preference-Based Reinforcement Learning**<br/>
Conference on Uncertainty in Artificial Intelligence (UAI), 2020<br/>
Ellen Novoseller, Yibing Wei, Yanan Sui, Yisong Yue, and Joel W. Burdick<br/>
[PDF](https://arxiv.org/abs/1908.01289)

In preference-based reinforcement learning (RL), an agent interacts with the environment
while receiving preferences instead of absolute feedback. This work presents Dueling Posterior Sampling (DPS), a posterior
sampling-based framework to learn both the system dynamics and the underlying
utility function that governs the preference feedback. For learning the utility function from preference feedback, we propose several 
approaches for solving the credit assignment problem,
including via Gaussian process regression, Bayesian linear regression, and a Gaussian process preference
model (for which Bayesian logistic regression is a special case).

In the paper, we prove an asymptotic Bayesian no-regret rate for DPS with a Bayesian linear regression
credit assignment model. To our knowledge, this is the first regret guarantee for preference-based RL.

## Implementation Notes

The DPS code and baseline algorithms can all be found in the "Learning_algorithms/" folder. Simulation environments are located in the "Envs/" folder. Finally, the scripts DPS_GPR_in_RiverSwim.py, and DPS_linear_in_RiverSwim.py demonstrate executing the DPS algorithm (with the Gaussian process regression and Bayesian linear regression credit assignment models, respectively) in the RiverSwim environment.

 
