#!/usr/bin/env python3
"""
DiscoEnv - Custom Gymnasium Environment for TradeGPT Scalper
Environment for training Disco57 RL model to decide ALLOW or BLOCK trades
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class DiscoEnv(gym.Env):
    """
    Custom Environment for Disco57 RL Model
    Observation: Last 10 candles with 8 normalized features each
    Action: ALLOW (1) or BLOCK (0) trade
    """
    def __init__(self, data, lookback=10, max_steps=1000):
        super(DiscoEnv, self).__init__()
        
        self.data = data  # Historical data with features
        self.lookback = lookback  # Number of candles to look back (10)
        self.max_steps = max_steps  # Max steps per episode
        self.current_step = 0
        self.episode_return = 0.0
        
        # Observation space: 10 candles x 8 features = 80 values
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.lookback * 8,), dtype=np.float32
        )
        
        # Action space: 0 = BLOCK, 1 = ALLOW
        self.action_space = spaces.Discrete(2)
        
        # Track trade outcomes for reward calculation
        self.trade_outcome = None
        self.steps_since_trade = 0
        self.trade_active = False
        
    def _get_observation(self):
        """Get observation at current step"""
        start_idx = max(0, self.current_step - self.lookback)
        end_idx = self.current_step
        
        # Get slice of data
        data_slice = self.data[start_idx:end_idx]
        
        # If less than lookback, pad with zeros
        if len(data_slice) < self.lookback:
            padding = np.zeros((self.lookback - len(data_slice), 8), dtype=np.float32)
            data_slice = np.vstack([padding, data_slice])
        
        # Flatten to 1D array
        return data_slice.flatten()
    
    def _calculate_reward(self, action):
        """Calculate reward based on action and trade outcome"""
        if action == 0:  # BLOCK
            return 0.0  # Neutral reward for blocking
        
        if action == 1:  # ALLOW
            if not self.trade_active:
                # Check future outcome (simulating trade)
                future_steps = min(15, len(self.data) - self.current_step - 1)
                if future_steps > 3:  # Need at least 3 steps to evaluate
                    for i in range(future_steps):
                        future_price = self.data[self.current_step + i + 1, 0]  # close_zscore
                        current_price = self.data[self.current_step, 0]
                        
                        # Calculate simulated P&L (assuming LONG for simplicity)
                        # In real scenario, direction would be determined by signal
                        pnl_usd = (future_price - current_price) * 20  # $20 exposure
                        
                        if pnl_usd >= 0.50:  # Profit >= $0.50
                            self.trade_outcome = 'profit'
                            self.trade_active = True
                            self.steps_since_trade = i + 1
                            return 1.0  # Big reward for good trade
                        elif pnl_usd <= -0.15:  # Loss <= -$0.15
                            self.trade_outcome = 'loss'
                            self.trade_active = True
                            self.steps_since_trade = i + 1
                            return -2.0  # Big penalty for loss
                
                # If no clear outcome yet, small penalty for unclear trade
                return -0.5  # Penalty for ALLOW with unclear result
            else:
                # Trade already active, no additional reward until close
                return 0.0
        return 0.0
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        self.current_step = self.lookback
        self.episode_return = 0.0
        self.trade_active = False
        self.trade_outcome = None
        self.steps_since_trade = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Take a step in the environment"""
        reward = self._calculate_reward(action)
        self.episode_return += reward
        
        # Update trade status if active
        if self.trade_active:
            self.steps_since_trade -= 1
            if self.steps_since_trade <= 0:
                self.trade_active = False
                self.trade_outcome = None
        
        self.current_step += 1
        
        # Check if episode should terminate
        terminated = self.current_step >= len(self.data) - 1 or self.current_step >= self.max_steps
        truncated = False
        
        obs = self._get_observation()
        info = {
            'step': self.current_step,
            'episode_return': self.episode_return,
            'action': action
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render environment (for debugging)"""
        return f"Step: {self.current_step}, Return: {self.episode_return:.2f}"


# Example usage
if __name__ == '__main__':
    # Generate dummy data for testing
    np.random.seed(42)
    n_samples = 1000
    dummy_data = np.random.randn(n_samples, 8)  # 8 features
    
    env = DiscoEnv(dummy_data, lookback=10, max_steps=500)
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    done = False
    total_reward = 0
    step = 0
    
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        if step % 100 == 0:
            print(f"Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
        
        done = terminated or truncated
    
    print(f"Episode finished after {step} steps, Total Reward: {total_reward:.2f}")
