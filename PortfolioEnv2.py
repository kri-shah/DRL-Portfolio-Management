import gym
import numpy as np
import pandas as pd
from gym import spaces


class PortfolioEnv2(gym.Env):
    """
    N-asset portfolio optimisation environment with Differential Sharpe Ratio reward.
    Action  : ℝⁿ, portfolio weights, forced to sum to 1 and clipped to [0,1]
    State   : log-returns for the last `lookback`-1 days + todays volatility features (replicated across time axis)
    Reward  : Differential Sharpe 
    Episode : runs to the end of `df`
    """
    metadata = {"render.modes": []}

    def __init__(self, df: pd.DataFrame, asset_cols: list[str],
        vol_cols: list[str],
        *,
        lookback: int = 60,
        initial_cash: float = 100_000.0,
        transaction_cost_rate: float = 1e-3,
        warmup_steps: int = 20,
        seed: int | None = 42,):
        
        super().__init__()

        self.df = df.sort_index()
        self.asset_cols = asset_cols
        self.vol_cols = vol_cols
        self.lookback = lookback
        self.initial_cash = float(initial_cash)
        self.transaction_cost_rate = float(transaction_cost_rate)
        self.warmup_steps = warmup_steps

        self.n_assets = len(asset_cols)
        self.dates = self.df.index
        
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

        obs_shape = (self.n_assets + len(vol_cols), self.lookback - 1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        self.eta = 1.0 / 252.0
        self.A = 0.0
        self.B = 0.0

        self.seed(seed)
        self.reset()

    def seed(self, seed: int | None = None):
        self.np_random, seed_ = gym.utils.seeding.np_random(seed)
        return [seed_]

    def reset(self):
        self.current_step = self.lookback
        self.portfolio_value = self.initial_cash
        self.weights = np.zeros(self.n_assets, dtype=np.float32)

        self.A = 0.0
        self.B = 0.0

        return self._get_observation()

    def _get_observation(self):

        window = self.df.iloc[self.current_step - self.lookback : self.current_step]

        log_returns = np.log(window[self.asset_cols] / window[self.asset_cols].shift(1)).dropna().values.T

        current_vol = self.df.iloc[self.current_step][self.vol_cols].values.astype(np.float32)
        vol_features = np.tile(current_vol.reshape(-1, 1), (1, log_returns.shape[1]))

        obs = np.vstack([log_returns, vol_features])
        return obs.astype(np.float32)


    def step(self, action):
        w = np.maximum(np.asarray(action, dtype=np.float32).flatten(), 0.0)
        w = w / w.sum() if w.sum() > 0.0 else np.full_like(w, 1.0 / self.n_assets)

        turnover = 0.5 * np.sum(np.abs(w - self.weights))

        prev_prices = self.df.iloc[self.current_step - 1][self.asset_cols].values
        curr_prices = self.df.iloc[self.current_step][self.asset_cols].values
        gross_ret = np.dot(w, curr_prices / prev_prices)
        net_ret = gross_ret * (1.0 - self.transaction_cost_rate * turnover)

        R_t = np.log(net_ret)

        A_prev, B_prev = self.A, self.B
        deltaA, deltaB = R_t - A_prev, R_t**2 - B_prev

        var_prev = max(B_prev - A_prev**2, 1e-8)           
        numer = var_prev * deltaA - 0.5 * A_prev * deltaB
        denom = var_prev ** 1.5
        dSharpe = numer / denom

        if self.current_step < self.lookback + self.warmup_steps:
            dSharpe = 0.0

        self.A += self.eta * deltaA
        self.B += self.eta * deltaB

        self.portfolio_value *= net_ret
        self.weights = w
        self.current_step += 1

        done = self.current_step >= len(self.df)
        obs  = (np.zeros(self.observation_space.shape, dtype=np.float32) if done else self._get_observation())
        
        info = {"portfolio_value": self.portfolio_value}
        return obs, float(dSharpe), done, info