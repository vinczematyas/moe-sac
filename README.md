<div align="center">
    <h1>I(MoE)-SAC: Interpretable Mixture of Interpretable Experts via
Soft Actor-Critic for Inverse Scaling in Continuous Control</h1>
</div>

## Control policy score, explanation, and pipeline description
Find the 2 page document `IMoE_SAC_GECCO.pdf` explaining our methodology, performance and interpretation of the policy

## Optimization Log
Find the optimization logs in `runs/gecco/log.log` containing <global_step, episodic_reward, episode_length> triplets

## Setup Environment
Updated environment file - `environment.yml`

The code is tested on CPU only.

```
conda env create -f environment.yml --name ic38
conda activate ic38
export PYTHONPATH=`pwd`
```

## Evaluate Policy
Python script, from which the submitted policy can be assessed on the environment - `evaluation.py`

```
python evaluation.py --path="runs/gecco" --checkpoint="checkpoint_final" --n_episodes=100
```

## Training New Policy
Python script, from which the optimization process can be reproduced - `main.py`

```
python main.py --log_local --save_models --wandb --run_name=<RUN_NAME> --seed=<SEED>
```

