<div align="center">
    <h1>MoE-SAC: Mixture of Interpretable Experts for Inverse Scaling in Continuous Control</h1>
</div>

This repository contains the code for the paper "Mixture of Interpretable Experts for Inverse Scaling in Continuous Control".

## Setup Environment

The code is tested on CPU only.

```
conda env create -f environment.yml --name moe-sac
conda activate moe-sac
export PYTHONPATH=`pwd`
```

## Evaluate Policy

```
python evaluation.py --path="runs/gecco" --checkpoint="checkpoint_final" --n_episodes=100
```

## Training New Policy

```
python main.py --log_local --save_models --wandb --run_name=<RUN_NAME> --seed=<SEED>
```

