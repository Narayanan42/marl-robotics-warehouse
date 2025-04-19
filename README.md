# Multi-Agent RL for Warehouse Robotics
## Overview

This codebase provides implementations of four key MARL algorithms:

- **MAPPO**: Multi-Agent PPO with centralized critic
- **IPPO**: Independent PPO with decentralized critics
- **MAA2C**: Multi-Agent A2C with centralized critic
- **IA2C**: Independent A2C with decentralized critics

The implementation is optimized for the RWARE environment, which simulates robots working together in warehouse settings where they need to pick up and deliver items efficiently.

## Key Features

- **Centralized vs Decentralized Training**: centralize and decentralized critic architectures with policy update methods
- **Parallel Execution**: Train with multiple environment instances in parallel

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Narayanan42/marl-robotics-warehouse.git
cd marl-robotics-warehouse
```

2. Install dependencies:
```bash
pip install gymnasium rware torch numpy
```

## Quick Start

Train using different alg(mappo,maa2c,ippo,ia2c):
```bash
python src/main.py --config=mappo --alg mappo --env rware
```

## Configuration

Algorithm configuration files are in `src/config/algs/`:
- `mappo.yaml`: Multi-Agent PPO with centralized critic
- `ippo.yaml`: Independent PPO with decentralized critics
- `maa2c.yaml`: Multi-Agent A2C with centralized critic  
- `ia2c.yaml`: Independent A2C with decentralized critics
