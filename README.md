# MARL-RWARE: Multi-Agent RL for Warehouse Robotics

A streamlined multi-agent reinforcement learning framework focused on training cooperative robotic policies in warehouse environments using the Robot Warehouse (RWARE) environment.

## Overview

This codebase provides clean implementations of four key MARL algorithms:

- **MAPPO**: Multi-Agent PPO with centralized critic
- **IPPO**: Independent PPO with decentralized critics
- **MAA2C**: Multi-Agent A2C with centralized critic
- **IA2C**: Independent A2C with decentralized critics

The implementation is optimized for the RWARE environment, which simulates robots working together in warehouse settings where they need to pick up and deliver items efficiently.

## Key Features

- **Centralized vs Decentralized Training**: Mix and match critic architectures with policy update methods
- **Parallel Execution**: Train with multiple environment instances in parallel
- **RNN-Based Policies**: Agents use GRU cells to handle partial observability
- **Flexible Reward Structure**: Support for both common reward and individual agent reward functions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/marl-robotics-warehouse.git
cd marl-robotics-warehouse
```

2. Install dependencies:
```bash
pip install gymnasium rware torch numpy
```

## Quick Start

Train using MAPPO (centralized critic with PPO updates):
```bash
python src/main.py --config=mappo --env-config=gymma --env_args.key=rware:rware-tiny-2ag-v1
```

Train using IPPO (independent critics with PPO updates):
```bash
python src/main.py --config=ippo --env-config=gymma --env_args.key=rware:rware-tiny-2ag-v1
```

Train using MAA2C (centralized critic with A2C updates):
```bash
python src/main.py --config=maa2c --env-config=gymma --env_args.key=rware:rware-tiny-2ag-v1
```

Train using IA2C (independent critics with A2C updates):
```bash
python src/main.py --config=ia2c --env-config=gymma --env_args.key=rware:rware-tiny-2ag-v1
```

## Core Components

- **Agents**: RNN-based policy networks defined in `controller.py`
- **Critics**: Both centralized (CentralVCritic) and decentralized (ACCritic) value functions in `critics.py`
- **Learners**: Policy optimization algorithms in `learners/`
- **Runner**: Parallel environment execution for efficient data collection in `runner.py`
- **Environment**: `GymMAWrapper` in `envs/gymma_wrapper.py` provides a consistent interface to the RWARE environment

## Configuration

Algorithm configuration files are in `src/config/algs/`:
- `mappo.yaml`: Multi-Agent PPO with centralized critic
- `ippo.yaml`: Independent PPO with decentralized critics
- `maa2c.yaml`: Multi-Agent A2C with centralized critic  
- `ia2c.yaml`: Independent A2C with decentralized critics

Environment configuration is in `src/config/envs/gymma.yaml` with RWARE-specific settings.

## Citation

If you use this codebase in your research, please cite:
```
@misc{marl-rware,
  author = {Your Name},
  title = {MARL-RWARE: Multi-Agent Reinforcement Learning for Warehouse Robotics},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/yourusername/marl-robotics-warehouse}}
}
```