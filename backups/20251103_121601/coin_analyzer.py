
            if market_data.get('market_cap', 0) < self.min_market_cap:
                return {'suitable': False, 'reason': 'Малая капитализация'}

            # Расчет волатильности
            volatility = self._calculate_volatility(market_data['price_history'])
            if volatility > self.max_volatility:
                return {'suitable': False, 'reason': 'Высокая волатильность'}

            # Анализ тренда
            trend = self._analyze_trend(market_data['price_history'])
            if trend == 'undefined':
                return {'suitable': False, 'reason': 'Нет четкого тренда'}

            # Проверка манипуляций
            if self._detect_manipulation(market_data):
                return {'suitable': False, 'reason': 'Признаки манипуляций'}

            score = self._calculate_coin_score(market_data)

            return {
                'suitable': True,
                'score': score,
                'trend': trend,
                'volatility': volatility,
                'volume_24h': market_data['volume_24h'],
                'market_cap': market_data.get('market_cap', 0),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Ошибка при анализе {symbol}: {str(e)}")
            return {'suitable': False, 'reason': f'Ошибка анализа: {str(e)}'}

    def _calculate_volatility(self, price_history: List[float]) -> float:
        """Расчет волатильности"""
        returns = np.diff(price_history) / price_history[:-1]
        return np.std(returns)

    def _analyze_trend(self, price_history: List[float]) -> str:
        """Анализ тренда"""
        # Используем EMA для определения тренда
        ema_short = pd.Series(price_history).ewm(span=12).mean().iloc[-1]
        ema_long = pd.Series(price_history).ewm(span=26).mean().iloc[-1]

        if ema_short > ema_long * 1.005:  # 0.5% буфер
            return 'bullish'
        elif ema_short < ema_long * 0.995:  # 0.5% буфер
            return 'bearish'
        return 'undefined'

    def _detect_manipulation(self, market_data: Dict) -> bool:
        """Обнаружение манипуляций"""
        volume_history = market_data.get('volume_history', [])
        if not volume_history:
            return False

        # Резкие скачки объема
        avg_volume = np.mean(volume_history)
        max_volume = np.max(volume_history)
        if max_volume > avg_volume * 5:  # 5x от среднего
            return True

        # Другие проверки манипуляций...
        return False

    def _calculate_coin_score(self, market_data: Dict) -> float:
        """Расчет рейтинга монеты"""
        score = 0.0

        # Объем (40% веса)
        volume_score = min(market_data['volume_24h'] / self.min_volume_usd, 10.0) * 4

        # Волатильность (30% веса)
        volatility = self._calculate_volatility(market_data['price_history'])
        volatility_score = (self.max_volatility - volatility) / self.max_volatility * 3

        # Капитализация (20% веса)
        cap_score = min(market_data.get('market_cap', 0) / self.min_market_cap, 10.0) * 2

        # Тренд (10% веса)
        trend = self._analyze_trend(market_data['price_history'])
        trend_score = 1.0 if trend == 'bullish' else 0.5 if trend == 'undefined' else 0.0

        return (volume_score + volatility_score + cap_score + trend_score) / 10.0  # 0-10 шкала

    def get_best_coins(self, all_market_data: Dict[str, Dict], max_coins: int = 5) -> List[Dict]:
        """Получение лучших монет для торговли"""
        analyzed_coins = []

        for symbol, market_data in all_market_data.items():
            analysis = self.analyze_coin(symbol, market_data)
            if analysis['suitable']:
                analyzed_coins.append({
                    'symbol': symbol,
                    **analysis
                })

        # Сортировка по рейтингу
        analyzed_coins.sort(key=lambda x: x['score'], reverse=True)

        # Сохраняем результаты анализа
        self.analyzed_coins = {
            coin['symbol']: coin for coin in analyzed_coins[:max_coins]
        }

        return analyzed_coins[:max_coins]

    def is_coin_suitable(self, symbol: str) -> bool:
        """Проверка пригодности монеты для торговли"""
        if symbol not in self.analyzed_coins:
            return False

        analysis = self.analyzed_coins[symbol]
        # Проверяем давность анализа (не старше 1 часа)
        analysis_time = datetime.fromisoformat(analysis['timestamp'])
        if datetime.now() - analysis_time > timedelta(hours=1):
            return False

        return analysis['suitable']
#!/usr/bin/env python3
"""
🔍 Анализатор монет
==================
- Поиск перспективных монет
- Анализ объемов и волатильности
- Фильтрация по критериям
"""

import logging
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CoinAnalyzer:
    def __init__(self):
        self.min_volume_usd = 1000000  # Минимальный объем $1M
        self.min_market_cap = 5000000  # Минимальная капитализация $5M
        self.max_volatility = 0.15     # Максимальная волатильность 15%
        self.analyzed_coins = {}       # История анализа монет

    def analyze_coin(self, symbol: str, market_data: Dict) -> Dict:
        """Анализ отдельной монеты"""
        try:
            # Базовые проверки
            if market_data['volume_24h'] < self.min_volume_usd:
                return {'suitable': False, 'reason': 'Недостаточный объем'}
