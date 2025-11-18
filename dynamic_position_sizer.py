"""
💰 ДИНАМИЧЕСКИЙ РАЗМЕР ПОЗИЦИИ
Адаптивное управление размером позиции на основе уверенности и риска
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging


class DynamicPositionSizer:
    """Динамическое определение размера позиции"""
    
    def __init__(self, base_size_usd: float = 1.0, max_size_usd: float = 3.0, logger: logging.Logger = None):
        """
        Args:
            base_size_usd: Базовый размер позиции в USD
            max_size_usd: Максимальный размер позиции в USD
            logger: Логгер
        """
        self.base_size = base_size_usd
        self.max_size = max_size_usd
        self.logger = logger or logging.getLogger(__name__)
        
        # Пороги уверенности для масштабирования
        self.confidence_tiers = {
            'weak': (0.60, 0.70, 0.7),      # 60-70%: 0.7x размер
            'normal': (0.70, 0.80, 1.0),    # 70-80%: 1.0x размер
            'strong': (0.80, 0.90, 1.3),    # 80-90%: 1.3x размер
            'very_strong': (0.90, 1.00, 1.5), # 90-100%: 1.5x размер
        }
        
        # История для Kelly Criterion
        self.trade_history = []
    
    def calculate_size_by_confidence(self, confidence: float) -> float:
        """
        Рассчитывает размер на основе уверенности
        
        Args:
            confidence: Уверенность сигнала (0-1)
            
        Returns:
            Размер позиции в USD
        """
        # Определяем tier
        multiplier = 1.0
        for tier_name, (min_conf, max_conf, tier_mult) in self.confidence_tiers.items():
            if min_conf <= confidence < max_conf:
                multiplier = tier_mult
                break
        
        size = self.base_size * multiplier
        
        # Ограничиваем максимумом
        size = min(size, self.max_size)
        
        return size
    
    def calculate_size_by_volatility(self, volatility: float, base_size: float) -> float:
        """
        Корректирует размер на основе волатильности
        
        Args:
            volatility: Волатильность (ATR в %)
            base_size: Базовый размер
            
        Returns:
            Скорректированный размер
        """
        # Чем выше волатильность, тем меньше размер
        if volatility < 0.02:  # Низкая
            multiplier = 1.2
        elif volatility < 0.04:  # Средняя
            multiplier = 1.0
        elif volatility < 0.06:  # Высокая
            multiplier = 0.8
        else:  # Очень высокая
            multiplier = 0.6
        
        return base_size * multiplier
    
    def calculate_size_by_kelly(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Рассчитывает размер по Kelly Criterion
        
        Kelly % = (W * R - L) / R
        где:
        W = вероятность выигрыша
        L = вероятность проигрыша
        R = отношение среднего выигрыша к среднему проигрышу
        
        Args:
            win_rate: Win rate (0-1)
            avg_win: Средний выигрыш в USD
            avg_loss: Средний проигрыш в USD (положительное число)
            
        Returns:
            Оптимальный размер позиции
        """
        if avg_loss == 0 or win_rate == 0:
            return self.base_size
        
        # Kelly Criterion
        R = avg_win / avg_loss  # Reward/Risk ratio
        W = win_rate
        L = 1 - win_rate
        
        kelly_percent = (W * R - L) / R
        
        # Используем половину Kelly (более консервативно)
        kelly_percent = kelly_percent * 0.5
        
        # Ограничиваем
        kelly_percent = np.clip(kelly_percent, 0.5, 2.0)
        
        size = self.base_size * kelly_percent
        size = min(size, self.max_size)
        
        return size
    
    def calculate_optimal_size(
        self,
        confidence: float,
        volatility: float = None,
        win_rate: float = None,
        avg_win: float = None,
        avg_loss: float = None
    ) -> Dict:
        """
        Рассчитывает оптимальный размер позиции
        
        Args:
            confidence: Уверенность сигнала
            volatility: Волатильность (опционально)
            win_rate: Win rate (опционально)
            avg_win: Средний выигрыш (опционально)
            avg_loss: Средний проигрыш (опционально)
            
        Returns:
            Словарь с размером и деталями расчёта
        """
        # 1. Базовый размер по уверенности
        size_by_confidence = self.calculate_size_by_confidence(confidence)
        
        # 2. Корректировка по волатильности
        if volatility is not None:
            size_by_volatility = self.calculate_size_by_volatility(volatility, size_by_confidence)
        else:
            size_by_volatility = size_by_confidence
        
        # 3. Kelly Criterion (если есть история)
        if win_rate and avg_win and avg_loss:
            size_by_kelly = self.calculate_size_by_kelly(win_rate, avg_win, avg_loss)
            # Среднее между volatility-adjusted и Kelly
            final_size = (size_by_volatility + size_by_kelly) / 2
        else:
            final_size = size_by_volatility
        
        # Округляем до 2 знаков
        final_size = round(final_size, 2)
        
        # Ограничиваем
        final_size = max(self.base_size * 0.5, min(final_size, self.max_size))
        
        result = {
            'size_usd': final_size,
            'size_by_confidence': round(size_by_confidence, 2),
            'size_by_volatility': round(size_by_volatility, 2),
            'size_by_kelly': round(size_by_kelly, 2) if win_rate else None,
            'multiplier': round(final_size / self.base_size, 2),
        }
        
        self.logger.info(
            f"💰 Размер позиции: ${final_size} "
            f"(Confidence: ${size_by_confidence} | "
            f"Volatility: ${size_by_volatility} | "
            f"Multiplier: {result['multiplier']}x)"
        )
        
        return result
    
    def add_trade_result(self, pnl_usd: float, success: bool):
        """
        Добавляет результат сделки для Kelly Criterion
        
        Args:
            pnl_usd: Прибыль/убыток в USD
            success: Успешна ли сделка
        """
        self.trade_history.append({
            'pnl': pnl_usd,
            'success': success,
            'timestamp': pd.Timestamp.now()
        })
        
        # Храним последние 100 сделок
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
    
    def get_statistics(self) -> Dict:
        """
        Получает статистику для Kelly Criterion
        
        Returns:
            Словарь со статистикой
        """
        if not self.trade_history:
            return {
                'win_rate': 0.5,
                'avg_win': 1.0,
                'avg_loss': 1.0,
                'total_trades': 0
            }
        
        df = pd.DataFrame(self.trade_history)
        
        wins = df[df['success'] == True]
        losses = df[df['success'] == False]
        
        win_rate = len(wins) / len(df) if len(df) > 0 else 0.5
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 1.0
        avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 1.0
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(df),
            'total_pnl': df['pnl'].sum()
        }


if __name__ == "__main__":
    # Тест
    print("🧪 Тестирование динамического размера позиции\n")
    
    sizer = DynamicPositionSizer(base_size_usd=1.0, max_size_usd=3.0)
    
    # Тест 1: Разные уровни уверенности
    print("📊 Тест 1: Размер по уверенности")
    for conf in [0.65, 0.75, 0.85, 0.95]:
        result = sizer.calculate_optimal_size(confidence=conf)
        print(f"   Confidence {conf:.0%}: ${result['size_usd']} ({result['multiplier']}x)")
    print()
    
    # Тест 2: С учётом волатильности
    print("📊 Тест 2: С учётом волатильности")
    result_low = sizer.calculate_optimal_size(confidence=0.80, volatility=0.01)
    result_high = sizer.calculate_optimal_size(confidence=0.80, volatility=0.08)
    print(f"   Низкая волатильность: ${result_low['size_usd']}")
    print(f"   Высокая волатильность: ${result_high['size_usd']}")
    print()
    
    # Тест 3: С Kelly Criterion
    print("📊 Тест 3: С Kelly Criterion")
    
    # Добавляем историю сделок
    for i in range(20):
        pnl = np.random.choice([1.5, -1.0], p=[0.6, 0.4])  # 60% win rate
        sizer.add_trade_result(pnl, pnl > 0)
    
    stats = sizer.get_statistics()
    print(f"   Win Rate: {stats['win_rate']:.1%}")
    print(f"   Avg Win: ${stats['avg_win']:.2f}")
    print(f"   Avg Loss: ${stats['avg_loss']:.2f}")
    
    result_kelly = sizer.calculate_optimal_size(
        confidence=0.80,
        volatility=0.03,
        win_rate=stats['win_rate'],
        avg_win=stats['avg_win'],
        avg_loss=stats['avg_loss']
    )
    print(f"   Оптимальный размер: ${result_kelly['size_usd']}")
    print()
    
    print("✅ Тест завершён!")
