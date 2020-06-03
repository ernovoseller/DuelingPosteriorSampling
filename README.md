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

The DPS [1] code and baseline algorithms can all be found in the "Learning_algorithms/" folder. Simulation environments are located in the "Envs/" folder. Finally, the following scripts demonstrate executing the DPS algorithm (with several different credit assignment models) and two baseline algorithms (EPMC [2, 3] and PSRL [4]) in the RiverSwim environment, to reproduce the corresponding simulation results in the paper: DPS_GPR_in_RiverSwim.py, DPS_GP_preference_in_RiverSwim.py, DPS_linear_in_RiverSwim.py, DPS_logistic_in_RiverSwim.py, EPMC_in_RiverSwim.py, and PSRL_in_RiverSwim.py.

### References

[1] E. Novoseller, Y. Wei, Y. Sui, Y. Yue, and J. W. Burdick. Dueling Posterior Sampling for Preference-Based Reinforcement Learning. *arXiv preprint arXiv:1908.01289*, 2020. Accepted to *Conference on Uncertainty in Artificial Intelligence (UAI)*, 2020. <br/>
[2] C. Wirth and J. Fürnkranz. A policy iteration algorithm for learning from preference-based feedback. In *International Symposium on Intelligent Data Analysis*, pages 427–437. Springer, 2013. <br/>
[3] C. Wirth. *Efficient Preference-based Reinforcement Learning*. PhD thesis, Technische Universität, 2017. <br/>
[4] I. Osband, D. Russo, and B. Van Roy. (More) efficient reinforcement learning via posterior sampling. In *Advances in Neural Information Processing Systems*, pages 3003–3011, 2013.
