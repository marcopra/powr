# POWR: Operator World Models for Reinforcement Learning

[Paper]() / [Website]() 

##### name1, name2, name3, name4

This repository contains the code for the paper "Operator World Models for Reinforcement Learning".

*Abstract:* Policy Mirror Descent (PMD) is a powerful and theoretically sound methodology for sequential decision-making. However, it is not directly applicable to Reinforcement Learning (RL) due to the inaccessibility of explicit action-value functions. We address this challenge by introducing a novel approach based on learning a world model of the environment using conditional mean embeddings (CME). We then leverage the operatorial formulation of RL to express the action-value function in terms of this quantity in closed form via matrix operations. Combining these estimators with PMD leads to POWR, a new RL algorithm for which we prove convergence rates to the global optimum. Preliminary experiments in both finite and infinite state settings support the effectiveness of our method, making this the first concrete implementation of PMD in RL to our knowledge.


Our release is **under construction**, you can track its progress below:

- [ ] Installation instructions
- [ ] Code implementation
	- [ ] Optimization
	- [ ] training
	- [ ] testing
- [ ] Reproducing Results
- [ ] Hyperparameters
- [ ] Trained models

## Installation

1. Install POWR dependences:
```
conda create -n powr python=3.8
conda activate powr 
pip install -r requirements.txt
```

2. (optional) set up `wandb login` with your WeightsAndBiases account. If you do not wish to use wandb to track the experiment results, ...  

## Getting started

### Quick test
- `python3 train_powr.py`

### Reproduce paper results


#### POWR
- MountainCar: ``
- Taxi: ``
- FrozenLake: ``

## Cite us
If you use this repository, please consider citing
```

```