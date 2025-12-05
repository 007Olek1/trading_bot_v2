#!/usr/bin/env python3
"""
Train Disco57 PPO Model for TradeGPT Scalper
Trains the RL model on historical data
"""

import logging
import os
import numpy as np
import argparse
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import components
try:
    from disco_env import DiscoEnv
    from disco57_ppo import Disco57PPO
    from data_collector import DataCollector
except ImportError as e:
    logger.error(f"Ошибка импорта модулей: {e}")
    logger.error("Убедитесь, что все зависимости установлены: pip install -r requirements.txt")
    exit(1)

# Trading symbols (limited set for training)
TRAINING_SYMBOLS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT',
    'ADA/USDT:USDT', 'DOGE/USDT:USDT', 'SOL/USDT:USDT', 'DOT/USDT:USDT',
    'MATIC/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT', 'UNI/USDT:USDT'
]


def load_data(data_file: str) -> np.ndarray:
    """
    Load training data from file
    
    Args:
        data_file: Path to .npz file with training data
    
    Returns:
        Numpy array with observations
    """
    if not os.path.exists(data_file):
        logger.error(f"Файл данных {data_file} не найден")
        logger.error("Сначала соберите данные с помощью: python data_collector.py")
        exit(1)
    
    data = np.load(data_file)
    observations = data['observations']
    logger.info(f"Загружено {len(observations)} записей из {data_file}")
    return observations


def split_data(data: np.ndarray, train_ratio: float = 0.8) -> tuple:
    """
    Split data into training and evaluation sets
    
    Args:
        data: Full dataset
        train_ratio: Ratio of data for training
    
    Returns:
        Tuple of (train_data, eval_data)
    """
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    logger.info(f"Данные разделены: {len(train_data)} для обучения, {len(eval_data)} для оценки")
    return train_data, eval_data


def train_model(data_file: str, timesteps: int = 100000, model_path: str = 'disco57_ppo_model.zip'):
    """
    Train Disco57 PPO model
    
    Args:
        data_file: Path to training data file
        timesteps: Number of training timesteps
        model_path: Path to save the model
    """
    # Load data
    full_data = load_data(data_file)
    train_data, eval_data = split_data(full_data)
    
    # Create environments
    train_env = DiscoEnv(train_data, lookback=10, max_steps=5000)
    eval_env = DiscoEnv(eval_data, lookback=10, max_steps=1000)
    
    # Create or load model
    model = Disco57PPO(model_path=model_path)
    
    # Train model
    model.train(train_env, total_timesteps=timesteps, eval_env=eval_env, eval_freq=10000)
    
    logger.info("Обучение завершено")
    stats = model.get_stats()
    logger.info(f"Статистика: {stats['total_trades']} сделок, Win Rate: {stats['win_rate']:.1f}%")


def collect_data(symbols: List[str] = TRAINING_SYMBOLS, days: int = 30, output_file: str = 'training_data.npz'):
    """
    Collect training data
    
    Args:
        symbols: List of trading symbols
        days: Number of days to collect data for
        output_file: Output file for collected data
    """
    import asyncio
    
    collector = DataCollector(symbols, timeframe='1m', days=days)
    asyncio.run(collector.collect_data(output_file))
    asyncio.run(collector.close())
    logger.info("Сбор данных завершен")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Disco57 PPO Model for TradeGPT Scalper')
    parser.add_argument('--mode', choices=['collect', 'train', 'both'], default='both',
                        help='Mode: collect data, train model, or both')
    parser.add_argument('--data-file', type=str, default='training_data.npz',
                        help='Path to training data file')
    parser.add_argument('--model-path', type=str, default='disco57_ppo_model.zip',
                        help='Path to save/load model')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Number of training timesteps')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days of data to collect')
    parser.add_argument('--symbols', type=str, nargs='*',
                        help='List of symbols for data collection (default: predefined list)')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.basicConfig(level=logging.INFO)
    
    # Determine symbols
    symbols = args.symbols if args.symbols else TRAINING_SYMBOLS
    logger.info(f"Используемые символы: {len(symbols)} пар")
    
    if args.mode in ['collect', 'both']:
        logger.info("Запуск сбора данных...")
        collect_data(symbols=symbols, days=args.days, output_file=args.data_file)
    
    if args.mode in ['train', 'both']:
        logger.info("Запуск обучения модели...")
        train_model(data_file=args.data_file, timesteps=args.timesteps, model_path=args.model_path)
