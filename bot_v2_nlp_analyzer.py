#!/usr/bin/env python3
"""
🤖 NLP Market Analyzer
Анализирует рынок с использованием NLP и ML моделей
- Генерация естественных описаний рынка
- Классификация тренда (РОСТ/ПАДЕНИЕ/БОКОВИК)
- Мультимодальный анализ
"""

import asyncio
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    # Попытка импорта трансформеров
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        pipeline
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("⚠️ transformers не установлен. NLP функции будут ограничены.")


class NLPMarketAnalyzer:
    """
    Анализатор рынка на основе NLP/ML
    """
    
    def __init__(self):
        self.model_name = "distilbert-base-uncased"
        self.classifier = None
        self.tokenizer = None
        
        # Кэш для оптимизации
        self.cache = {}
        
        logger.info("🤖 NLP Market Analyzer инициализирован")
        
        if TRANSFORMERS_AVAILABLE:
            self._load_models()
        else:
            logger.warning("⚠️ NLP модели недоступны - используется rule-based подход")
    
    def _load_models(self):
        """Загрузка предобученных моделей"""
        try:
            # Для начала используем zero-shot классификацию
            # Позже можно обучить свою модель на исторических данных
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("✅ Zero-shot classifier загружен")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки моделей: {e}")
            self.classifier = None
    
    def generate_market_description(
        self,
        candles: List[Dict],
        indicators: Dict[str, float],
        current_price: float,
        support: float = None,
        resistance: float = None
    ) -> str:
        """
        Генерирует естественное описание рынка на английском
        
        Пример выхода:
        "price rising strongly, volume increasing, near resistance"
        """
        
        if len(candles) < 2:
            return "insufficient data"
        
        # Извлекаем данные
        current_candle = candles[-1]
        prev_candle = candles[-2]
        
        close_prices = [c['close'] for c in candles[-10:]]
        volumes = [c['volume'] for c in candles[-10:]]
        
        # Анализируем компоненты
        price_movement = self._analyze_price_movement(close_prices)
        volume_trend = self._analyze_volume_trend(volumes)
        position_vs_levels = self._analyze_position(
            current_price, support, resistance
        )
        momentum = self._analyze_momentum(indicators)
        
        # Собираем описание
        parts = []
        
        # 1. Price movement
        parts.append(price_movement)
        
        # 2. Volume
        if volume_trend:
            parts.append(volume_trend)
        
        # 3. Position relative to levels
        if position_vs_levels:
            parts.append(position_vs_levels)
        
        # 4. Momentum indicators
        if momentum:
            parts.append(momentum)
        
        description = ", ".join(parts)
        
        logger.debug(f"📝 Market description: {description}")
        return description
    
    def _analyze_price_movement(self, prices: List[float]) -> str:
        """Анализирует движение цены"""
        if len(prices) < 3:
            return "price stable"
        
        recent = prices[-3:]
        overall = prices
        
        # Краткосрочный тренд (последние 3 свечи)
        short_trend = (recent[-1] - recent[0]) / recent[0] * 100
        
        # Долгосрочный тренд (все свечи)
        long_trend = (overall[-1] - overall[0]) / overall[0] * 100
        
        # Волатильность
        volatility = np.std(prices) / np.mean(prices) * 100
        
        # Определяем силу движения
        if abs(short_trend) < 0.5:
            strength = "stable"
        elif abs(short_trend) < 1.5:
            strength = "moving"
        elif abs(short_trend) < 3.0:
            strength = "rising" if short_trend > 0 else "falling"
        else:
            strength = "rising strongly" if short_trend > 0 else "falling sharply"
        
        return f"price {strength}"
    
    def _analyze_volume_trend(self, volumes: List[float]) -> str:
        """Анализирует тренд объёма"""
        if len(volumes) < 3:
            return ""
        
        recent_avg = np.mean(volumes[-3:])
        overall_avg = np.mean(volumes[:-3]) if len(volumes) > 3 else recent_avg
        
        change = (recent_avg - overall_avg) / overall_avg * 100
        
        if change > 50:
            return "volume surging"
        elif change > 20:
            return "volume increasing"
        elif change < -20:
            return "volume declining"
        
        return ""
    
    def _analyze_position(
        self, 
        price: float, 
        support: float = None, 
        resistance: float = None
    ) -> str:
        """Анализирует позицию цены относительно уровней"""
        parts = []
        
        if resistance:
            dist_to_resistance = (resistance - price) / price * 100
            if dist_to_resistance < 1:
                parts.append("at resistance")
            elif dist_to_resistance < 2:
                parts.append("near resistance")
        
        if support:
            dist_to_support = (price - support) / price * 100
            if dist_to_support < 1:
                parts.append("at support")
            elif dist_to_support < 2:
                parts.append("near support")
        
        return ", ".join(parts) if parts else ""
    
    def _analyze_momentum(self, indicators: Dict[str, float]) -> str:
        """Анализирует индикаторы моментума"""
        parts = []
        
        # RSI
        rsi = indicators.get('rsi')
        if rsi:
            if rsi > 80:
                parts.append("extremely overbought")
            elif rsi > 70:
                parts.append("overbought")
            elif rsi < 20:
                parts.append("extremely oversold")
            elif rsi < 30:
                parts.append("oversold")
        
        # MACD
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        if macd and macd_signal:
            if macd > macd_signal and macd > 0:
                parts.append("strong bullish momentum")
            elif macd > macd_signal:
                parts.append("bullish momentum")
            elif macd < macd_signal and macd < 0:
                parts.append("strong bearish momentum")
            elif macd < macd_signal:
                parts.append("bearish momentum")
        
        return ", ".join(parts[:2])  # Ограничиваем 2 признаками
    
    def classify_market_state(
        self,
        description: str,
        indicators: Dict[str, float] = None
    ) -> Tuple[str, float]:
        """
        Классифицирует состояние рынка: РОСТ / ПАДЕНИЕ / БОКОВИК
        
        Returns:
            (state, confidence) - состояние и уверенность (0-1)
        """
        
        if self.classifier and TRANSFORMERS_AVAILABLE:
            return self._classify_with_model(description)
        else:
            return self._classify_rule_based(description, indicators)
    
    def _classify_with_model(self, description: str) -> Tuple[str, float]:
        """Классификация с помощью ML модели"""
        try:
            candidate_labels = ["bullish trend", "bearish trend", "sideways movement"]
            
            result = self.classifier(
                description,
                candidate_labels=candidate_labels
            )
            
            # Получаем лучший результат
            best_label = result['labels'][0]
            confidence = result['scores'][0]
            
            # Маппинг на русские термины
            mapping = {
                "bullish trend": "РОСТ",
                "bearish trend": "ПАДЕНИЕ",
                "sideways movement": "БОКОВИК"
            }
            
            state = mapping.get(best_label, "БОКОВИК")
            
            logger.debug(f"🎯 Classification: {state} ({confidence:.2%})")
            return state, confidence
            
        except Exception as e:
            logger.error(f"❌ Ошибка ML классификации: {e}")
            return self._classify_rule_based(description, None)
    
    def _classify_rule_based(
        self, 
        description: str, 
        indicators: Dict[str, float] = None
    ) -> Tuple[str, float]:
        """Rule-based классификация (fallback)"""
        
        description_lower = description.lower()
        
        # Подсчет бычьих/медвежьих сигналов
        bullish_score = 0
        bearish_score = 0
        
        # Ключевые слова
        bullish_keywords = [
            'rising', 'increasing', 'surging', 'bullish', 
            'overbought', 'strong', 'breakout'
        ]
        bearish_keywords = [
            'falling', 'declining', 'dropping', 'bearish',
            'oversold', 'weakness', 'breakdown'
        ]
        
        for word in bullish_keywords:
            if word in description_lower:
                bullish_score += 1
        
        for word in bearish_keywords:
            if word in description_lower:
                bearish_score += 1
        
        # Анализ индикаторов
        if indicators:
            rsi = indicators.get('rsi', 50)
            if rsi > 55:
                bullish_score += 1
            elif rsi < 45:
                bearish_score += 1
        
        # Определяем состояние
        total_score = bullish_score + bearish_score
        
        if total_score == 0:
            return "БОКОВИК", 0.5
        
        if bullish_score > bearish_score:
            confidence = bullish_score / (total_score + 1)
            return "РОСТ", confidence
        elif bearish_score > bullish_score:
            confidence = bearish_score / (total_score + 1)
            return "ПАДЕНИЕ", confidence
        else:
            return "БОКОВИК", 0.6
    
    def extract_ternary_features(
        self,
        candles: List[Dict],
        indicators: Dict[str, float]
    ) -> Dict[str, int]:
        """
        Извлекает тройственные признаки: -1 (падение), 0 (боковик), +1 (рост)
        
        Returns:
            Dict с признаками для ML обучения
        """
        
        if len(candles) < 10:
            return {
                'price_trend': 0,
                'volume_trend': 0,
                'momentum': 0,
                'volatility': 0
            }
        
        close_prices = [c['close'] for c in candles[-10:]]
        volumes = [c['volume'] for c in candles[-10:]]
        
        # 1. Price trend
        price_change = (close_prices[-1] - close_prices[0]) / close_prices[0] * 100
        price_trend = 1 if price_change > 1 else (-1 if price_change < -1 else 0)
        
        # 2. Volume trend
        volume_change = (np.mean(volumes[-3:]) - np.mean(volumes[:3])) / np.mean(volumes[:3]) * 100
        volume_trend = 1 if volume_change > 20 else (-1 if volume_change < -20 else 0)
        
        # 3. Momentum (RSI based)
        rsi = indicators.get('rsi', 50)
        momentum = 1 if rsi > 60 else (-1 if rsi < 40 else 0)
        
        # 4. Volatility (ATR based)
        highs = [c['high'] for c in candles[-10:]]
        lows = [c['low'] for c in candles[-10:]]
        volatility_pct = (np.mean(highs) - np.mean(lows)) / np.mean(close_prices) * 100
        volatility = 1 if volatility_pct > 3 else (-1 if volatility_pct < 1 else 0)
        
        return {
            'price_trend': price_trend,
            'volume_trend': volume_trend,
            'momentum': momentum,
            'volatility': volatility
        }
    
    async def analyze_market_nlp(
        self,
        symbol: str,
        candles: List[Dict],
        indicators: Dict[str, float],
        support: float = None,
        resistance: float = None
    ) -> Dict[str, Any]:
        """
        Полный NLP анализ рынка
        
        Returns:
            {
                'description': str,  # Естественное описание
                'state': str,        # РОСТ/ПАДЕНИЕ/БОКОВИК
                'confidence': float, # Уверенность
                'features': dict     # Тройственные признаки
            }
        """
        
        try:
            current_price = candles[-1]['close']
            
            # 1. Генерируем описание
            description = self.generate_market_description(
                candles, indicators, current_price, support, resistance
            )
            
            # 2. Классифицируем
            state, confidence = self.classify_market_state(description, indicators)
            
            # 3. Извлекаем признаки
            features = self.extract_ternary_features(candles, indicators)
            
            result = {
                'symbol': symbol,
                'description': description,
                'state': state,
                'confidence': confidence,
                'features': features,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(
                f"📊 {symbol} NLP: {description[:50]}... | "
                f"State: {state} ({confidence:.0%})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка NLP анализа {symbol}: {e}")
            return {
                'symbol': symbol,
                'description': 'error',
                'state': 'БОКОВИК',
                'confidence': 0.0,
                'features': {},
                'error': str(e)
            }


# Singleton
nlp_analyzer = NLPMarketAnalyzer()


async def main():
    """Тестирование NLP анализатора"""
    
    # Пример данных
    test_candles = [
        {'open': 100, 'high': 102, 'low': 99, 'close': 101, 'volume': 1000},
        {'open': 101, 'high': 103, 'low': 100, 'close': 102.5, 'volume': 1200},
        {'open': 102.5, 'high': 105, 'low': 102, 'close': 104, 'volume': 1500},
        {'open': 104, 'high': 106, 'low': 103.5, 'close': 105.5, 'volume': 1800},
        {'open': 105.5, 'high': 107, 'low': 105, 'close': 106.5, 'volume': 2000},
    ]
    
    test_indicators = {
        'rsi': 65.5,
        'macd': 0.5,
        'macd_signal': 0.3
    }
    
    analyzer = NLPMarketAnalyzer()
    
    result = await analyzer.analyze_market_nlp(
        symbol="BTC/USDT",
        candles=test_candles,
        indicators=test_indicators,
        resistance=108.0
    )
    
    print("\n" + "="*60)
    print("🤖 NLP MARKET ANALYSIS")
    print("="*60)
    print(f"📝 Description: {result['description']}")
    print(f"🎯 State: {result['state']} ({result['confidence']:.0%} confidence)")
    print(f"📊 Features: {result['features']}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())


