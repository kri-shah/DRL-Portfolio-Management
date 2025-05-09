# Deep Reinforcement Learning for Portfolio Management

**Author**: Krish Shah  
**Affiliation**: University of Connecticut, School of Computing  
**Email**: krish.shah@uconn.edu  

## Overview

This project explores the application of Deep Reinforcement Learning (DRL) to long-only portfolio management using sector-level S&P 500 ETFs. Two agents—Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC)—are benchmarked against classical strategies like Mean-Variance Optimization (MVO) and Dollar-Cost Averaging (DCA). The environment models realistic constraints such as transaction costs and market volatility.

## Performance Metrics (on Test Set (2023-12-01 to 2025-04-15))

| Strategy | Sharpe Ratio | Max Drawdown | CAGR     | Return (%) |
|----------|--------------|---------------|----------|------------|
| **PPO**  | **1.01**     | **8.90%**      | 17.52%   | 21.43%     |
| SPY      | 0.60         | 20.23%        | 14.91%   | 16.50%     |
| SAC      | 0.53         | 16.02%        | 11.35%   | 13.10%     |
| MVO      | 0.39         | 18.06%        | 9.94%    | 10.49%     |


## Features

- **Custom Gym Environment** simulating daily sector ETF trading
- **Realistic Dataset** from June 2018 to April 2025 (via `yfinance`)
- **Differentiable Reward Function** using the Differential Sharpe Ratio
- **Benchmarks**: MVO (via PyPortfolioOpt) and DCA
- **Stable-Baselines3** integration for DRL agent training

## Data

- **Sector ETFs**: XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLK, XLB, XLRE, XLU
- **Market Index**: SPY (S&P 500)
- **Volatility**: VIX, 20D/60D rolling volatility
- **Transaction Cost**: 0.1% per trade
- **Train/Val/Test Split**:
  - Train: 2018-06-19 to 2022-10-26
  - Val: 2022-10-27 to 2023-11-30
  - Test: 2023-12-01 to 2025-04-15

## State and Action Space

- **State**: 59-day log returns + volatility indicators (VIX, 20D, 60D)
- **Action**: 11-dimensional portfolio allocation vector (long-only, sum=1)

## Algorithms

- **PPO** (on-policy, clipped surrogate loss, stable training)
- **SAC** (off-policy, entropy regularization, sample efficient)

## Reward Function

- Custom **Differential Sharpe Ratio** using exponential moving averages to ensure differentiability and stability during training.

## Requirements

- Python 3.10+
- `stable-baselines3`
- `gymnasium`
- `yfinance`
- `PyPortfolioOpt`
- `matplotlib`, `pandas`, `numpy`, `tensorboard`

## How to Run

1. Clone the repository
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run drl_train.ipynb 
4. Evaluate by running drl_inference.ipynb

## Future Work

- Extend to full S&P 500 or U.S. equities universe
- Incorporate Group Relative Policy Optimization (GRPO)
- Live trading simulation using Alpaca API

## Acknowledgements

Inspired by [Sood et al. (2023)](https://icaps23.icaps-conference.org/papers/finplan/FinPlan23_paper_4.pdf) and supported by guidance from Prof. Ji. 
