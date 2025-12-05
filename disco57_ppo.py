#!/usr/bin/env python3
"""
Disco57 PPO - Упрощенная RL модель для TradeGPT Scalper
Бинарное решение: ALLOW или BLOCK trade
Использует PPO из stable-baselines3
"""

import logging
import os
from typing import Dict, Any
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn

from disco_env import DiscoEnv

logger = logging.getLogger(__name__)


class CustomMLPExtractor(BaseFeaturesExtractor):
    """
    Custom MLP Feature Extractor for Disco57
    Small network with 2-3 layers for low resource usage
    """
    def __init__(self, observation_space, features_dim: int = 128):
        super(CustomMLPExtractor, self).__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]  # 80 features (10 candles x 8 features)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.net.parameters())
        logger.info(f"Disco57 MLP Extractor: {total_params} параметров")
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)


class Disco57PPO:
    """
    Disco57 PPO Model for ALLOW/BLOCK trade decisions
    Uses a small MLP network for low resource usage
    Loads only pre-trained model, no online training during trading
    Prediction speed < 5ms, defaults to BLOCK on load failure
    """
    def __init__(self, model_path: str = 'disco57_best.zip', device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
        self.model = None
        
        # Load pre-trained model
        self.load_model()
    
    def load_model(self):
        """Load pre-trained model"""
        if os.path.exists(self.model_path):
            try:
                self.model = PPO.load(self.model_path, device=self.device)
                logger.info(f"Disco57 PPO модель загружена из {self.model_path}")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели: {e}")
                self.model = None
        else:
            logger.error(f"Файл модели {self.model_path} не найден")
            self.model = None
    
    def predict(self, observation: np.ndarray) -> str:
        """
        Predict ALLOW or BLOCK for a trade
        Ensures prediction in < 5ms by using small MLP and CPU if needed
        
        Args:
            observation: Numpy array of shape (80,) with 10 candles x 8 features
        
        Returns:
            'ALLOW' or 'BLOCK', defaults to 'BLOCK' if model not loaded or prediction timeout
        """
        if self.model is None:
            logger.warning("Модель не загружена, используется безопасный дефолт BLOCK")
            return 'BLOCK'
        
        try:
            import time
            start_time = time.time()
            action, _ = self.model.predict(observation, deterministic=True)
            end_time = time.time()
            pred_time_ms = (end_time - start_time) * 1000
            if pred_time_ms > 5:
                logger.warning(f"Предсказание заняло {pred_time_ms:.2f} мс, превышает лимит 5 мс, возвращаем BLOCK")
                return 'BLOCK'
            return 'ALLOW' if action == 1 else 'BLOCK'
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return 'BLOCK'  # Default to BLOCK on error
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics - placeholder since no online learning"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'win_rate': 0.0
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Test model
    disco = Disco57PPO()
    
    # Dummy observation (10 candles x 8 features = 80 values)
    dummy_obs = np.random.randn(80).astype(np.float32)
    
    # Test prediction
    decision = disco.predict(dummy_obs)
    print(f"Prediction for dummy data: {decision}")
    
    # Print stats
    stats = disco.get_stats()
    print(f"Stats: {stats}")
