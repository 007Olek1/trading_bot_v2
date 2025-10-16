"""
🕯️ АНАЛИЗАТОР СВЕЧЕЙ V3.0
✅ Анализ закрытия свечи перед входом
✅ Паттерны свечей
✅ Форматирование сигналов в стиле профессиональных каналов
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class CandleAnalyzer:
    """Анализатор свечных паттернов"""
    
    @staticmethod
    def analyze_candle_close(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ закрытия последней свечи
        
        Returns:
            {
                'bullish': bool,
                'bearish': bool,
                'strong': bool,
                'pattern': str,
                'body_size': float,
                'wick_ratio': float
            }
        """
        try:
            if len(df) < 2:
                return {}
            
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Размер тела свечи
            body = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            body_percentage = (body / total_range * 100) if total_range > 0 else 0
            
            # Размеры фитилей
            if current['close'] > current['open']:  # Бычья свеча
                upper_wick = current['high'] - current['close']
                lower_wick = current['open'] - current['low']
                bullish = True
                bearish = False
            else:  # Медвежья свеча
                upper_wick = current['high'] - current['open']
                lower_wick = current['close'] - current['low']
                bullish = False
                bearish = True
            
            wick_ratio = (upper_wick + lower_wick) / body if body > 0 else 0
            
            # Определение паттерна
            pattern = "Обычная свеча"
            strong = False
            
            # Сильная бычья свеча
            if bullish and body_percentage > 70 and lower_wick < upper_wick:
                pattern = "Сильная бычья свеча"
                strong = True
            
            # Сильная медвежья свеча
            elif bearish and body_percentage > 70 and upper_wick < lower_wick:
                pattern = "Сильная медвежья свеча"
                strong = True
            
            # Молот (Hammer) - бычий разворот
            elif lower_wick > body * 2 and upper_wick < body * 0.5:
                pattern = "Молот (бычий разворот)"
                bullish = True
                strong = True
            
            # Падающая звезда (Shooting Star) - медвежий разворот
            elif upper_wick > body * 2 and lower_wick < body * 0.5:
                pattern = "Падающая звезда (медвежий разворот)"
                bearish = True
                strong = True
            
            # Доджи - неопределенность
            elif body_percentage < 10:
                pattern = "Доджи (неопределенность)"
                strong = False
            
            # Поглощение (Engulfing)
            prev_body = abs(prev['close'] - prev['open'])
            if body > prev_body * 1.2:
                if bullish and prev['close'] < prev['open']:
                    pattern = "Бычье поглощение"
                    strong = True
                elif bearish and prev['close'] > prev['open']:
                    pattern = "Медвежье поглощение"
                    strong = True
            
            return {
                'bullish': bullish,
                'bearish': bearish,
                'strong': strong,
                'pattern': pattern,
                'body_percentage': body_percentage,
                'wick_ratio': wick_ratio,
                'close_price': float(current['close'])
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа свечи: {e}")
            return {}
    
    @staticmethod
    def format_signal_report(
        symbol: str,
        signal_data: Dict[str, Any],
        candle_data: Dict[str, Any]
    ) -> str:
        """
        Форматирование сигнала в стиле профессиональных каналов
        
        Пример:
        Coin #ETH/USDT 
        Position: LONG 
        Leverage: Cross 5X
        Entries: 3900 - 3875
        Targets: 🎯 3925, 3950, 3975, 4000, 4025
        Stop Loss: 3850
        """
        side = signal_data.get('signal', '').upper()
        position_type = "LONG" if side == "BUY" else "SHORT"
        
        entry_price = signal_data.get('entry_price', 0)
        sl_price = signal_data.get('sl_price', 0)
        tp_levels = signal_data.get('tp_levels', [])
        
        # Диапазон входа (entry_price ± 0.3%)
        entry_high = entry_price * 1.003
        entry_low = entry_price * 0.997
        
        # Форматирование TP уровней
        tp_formatted = ", ".join([f"{tp:.4f}" for tp in tp_levels[:5]])
        
        # Информация о свече
        candle_pattern = candle_data.get('pattern', 'Неизвестно')
        candle_strength = "💪 Сильная" if candle_data.get('strong') else "📊 Обычная"
        
        # Индикаторы
        indicators = signal_data.get('indicators', {})
        rsi = indicators.get('rsi', 0)
        adx = indicators.get('adx', 0)
        
        # Уровни поддержки/сопротивления
        levels = signal_data.get('levels', {})
        support_levels = levels.get('support', [])
        resistance_levels = levels.get('resistance', [])
        
        report = (
            f"🎯 *ТОРГОВЫЙ СИГНАЛ*\n\n"
            f"*Coin* #{symbol.replace('/USDT:USDT', '/USDT')}\n"
            f"*Position:* {position_type}\n"
            f"*Leverage:* Cross {signal_data.get('leverage', 5)}X\n\n"
            f"*Entries:* {entry_high:.4f} - {entry_low:.4f}\n"
            f"*Targets:* 🎯 {tp_formatted}\n"
            f"*Stop Loss:* {sl_price:.4f}\n\n"
            f"📊 *Анализ свечи:*\n"
            f"   {candle_strength}\n"
            f"   Паттерн: {candle_pattern}\n\n"
            f"📈 *Индикаторы:*\n"
            f"   RSI: {rsi:.1f}\n"
            f"   ADX: {adx:.1f} (сила тренда)\n"
            f"   Уверенность: {signal_data.get('confidence', 0):.0f}%\n\n"
        )
        
        if support_levels:
            report += f"🟢 *Поддержка:* {', '.join([f'{s:.4f}' for s in support_levels[:3]])}\n"
        
        if resistance_levels:
            report += f"🔴 *Сопротивление:* {', '.join([f'{r:.4f}' for r in resistance_levels[:3]])}\n"
        
        report += f"\n💡 *Причина:* {signal_data.get('reason', 'N/A')}"
        
        return report


# Глобальный экземпляр
candle_analyzer = CandleAnalyzer()

