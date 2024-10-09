# POWR: Operator World Models for Reinforcement Learning

[Paper](https://arxiv.org/pdf/2406.19861)

##### [Pietro Novelli](), [Marco Pratticò](), [Carlo Ciliberto](), [Massimiliano Pontil]()

This repository contains the code for the paper "Operator World Models for Reinforcement Learning".

*Abstract:* Policy Mirror Descent (PMD) is a powerful and theoretically sound methodology for sequential decision-making. However, it is not directly applicable to Reinforcement Learning (RL) due to the inaccessibility of explicit action-value functions. We address this challenge by introducing a novel approach based on learning a world model of the environment using conditional mean embeddings (CME). We then leverage the operatorial formulation of RL to express the action-value function in terms of this quantity in closed form via matrix operations. Combining these estimators with PMD leads to POWR, a new RL algorithm for which we prove convergence rates to the global optimum. Preliminary experiments in both finite and infinite state settings support the effectiveness of our method, making this the first concrete implementation of PMD in RL to our knowledge.


Our release is **under construction**, you can track its progress below:

- [x] Installation instructions
- [ ] Code implementation
	- [x] Training
	- [x] Testing
	- [x] Optimization
	- [x] Model saving and loading
	- [] Cleaning
- [ ] Reproducing Results Scripts
- [ ] Hyperparameters for each env
- [ ] Trained models
- [ ] Complete the README

## Installation

1. Install POWR dependences:
```
conda create -n powr python=3.11
conda activate powr 
pip install -r requirements.txt
```

2. (optional) set up `wandb login` with your WeightsAndBiases account. If you do not wish to use wandb to track the experiment results, run it offline adding the following arg `--offline`. For example, `python3 train.py --offline`

## Getting started

### Quick test
- `python3 train.py`

## Cite us
If you use this repository, please consider citing
```
@misc{novelli2024operatorworldmodelsreinforcement,
      title={Operator World Models for Reinforcement Learning}, 
      author={Pietro Novelli and Marco Pratticò and Massimiliano Pontil and Carlo Ciliberto},
      year={2024},
      eprint={2406.19861},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.19861}, 
}
```