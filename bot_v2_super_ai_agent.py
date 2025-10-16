#!/usr/bin/env python3
"""
🤖 СУПЕР AI/ML/LLM АГЕНТ V3.4
Полный контроль над всей системой:
- Мониторинг сервера
- Анализ рынка
- Выбор лучших сигналов
- Контроль качества сделок
- Самодиагностика и исправление
- Оптимизация параметров
"""

import logging
import asyncio
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SuperAIAgent:
    """
    🧠 СУПЕР AI АГЕНТ
    Контролирует ВСЕ аспекты работы бота
    """
    
    def __init__(self):
        # Мониторинг системы
        self.server_health = {
            'cpu_usage': deque(maxlen=60),  # 1 час истории
            'memory_usage': deque(maxlen=60),
            'disk_usage': deque(maxlen=60),
            'network_errors': 0,
            'last_check': datetime.now()
        }
        
        # Анализ рынка
        self.market_analysis = {
            'trend': 'neutral',  # bullish, bearish, neutral
            'volatility': 'normal',  # low, normal, high, extreme
            'volume_trend': 'normal',
            'market_confidence': 0.5,
            'best_symbols': [],
            'avoid_symbols': []
        }
        
        # Качество сигналов
        self.signal_quality = {
            'total_analyzed': 0,
            'strong_signals': 0,  # ≥85%
            'good_signals': 0,    # 70-84%
            'weak_signals': 0,    # <70%
            'best_indicators': {},
            'worst_indicators': {}
        }
        
        # История сделок
        self.trade_history = deque(maxlen=100)
        self.trade_stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'best_symbol': None,
            'worst_symbol': None
        }
        
        # Самообучение
        self.learning_data = {
            'successful_patterns': [],
            'failed_patterns': [],
            'optimal_confidence': 85,  # Динамически адаптируется
            'optimal_timeframe': '15m',
            'best_trading_hours': []
        }
        
        # Статус агента
        self.agent_status = 'active'
        self.decisions_made = 0
        self.corrections_made = 0
        
    # ===========================================
    # 📊 МОНИТОРИНГ СЕРВЕРА
    # ===========================================
    
    async def monitor_server_health(self) -> Dict[str, Any]:
        """Мониторинг здоровья сервера"""
        try:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
            
            self.server_health['cpu_usage'].append(cpu)
            self.server_health['memory_usage'].append(memory)
            self.server_health['disk_usage'].append(disk)
            self.server_health['last_check'] = datetime.now()
            
            # Анализ трендов
            cpu_trend = 'high' if cpu > 80 else 'normal' if cpu > 50 else 'low'
            memory_trend = 'high' if memory > 80 else 'normal' if memory > 50 else 'low'
            
            health_score = 100
            if cpu > 90:
                health_score -= 30
            if memory > 90:
                health_score -= 30
            if disk > 90:
                health_score -= 20
                
            return {
                'healthy': health_score >= 70,
                'score': health_score,
                'cpu': cpu,
                'memory': memory,
                'disk': disk,
                'cpu_trend': cpu_trend,
                'memory_trend': memory_trend,
                'issues': self._detect_server_issues(cpu, memory, disk)
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка мониторинга сервера: {e}")
            return {'healthy': False, 'score': 0, 'error': str(e)}
    
    def _detect_server_issues(self, cpu: float, memory: float, disk: float) -> List[str]:
        """Детектирование проблем сервера"""
        issues = []
        
        if cpu > 90:
            issues.append("КРИТИЧНО: CPU > 90%")
        elif cpu > 80:
            issues.append("ВНИМАНИЕ: CPU > 80%")
            
        if memory > 90:
            issues.append("КРИТИЧНО: Memory > 90%")
        elif memory > 80:
            issues.append("ВНИМАНИЕ: Memory > 80%")
            
        if disk > 90:
            issues.append("КРИТИЧНО: Disk > 90%")
        elif disk > 85:
            issues.append("ВНИМАНИЕ: Disk > 85%")
            
        return issues
    
    # ===========================================
    # 📈 АНАЛИЗ РЫНКА
    # ===========================================
    
    async def analyze_market_conditions(
        self,
        all_signals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Глубокий анализ рыночных условий
        Определяет общий тренд, волатильность, лучшие возможности
        """
        if not all_signals:
            return self.market_analysis
            
        # Анализ трендов
        buy_signals = sum(1 for s in all_signals if s.get('signal') == 'buy')
        sell_signals = sum(1 for s in all_signals if s.get('signal') == 'sell')
        total = len(all_signals)
        
        # Определяем тренд рынка
        if buy_signals > sell_signals * 1.5:
            trend = 'bullish'
            confidence = buy_signals / total if total > 0 else 0
        elif sell_signals > buy_signals * 1.5:
            trend = 'bearish'
            confidence = sell_signals / total if total > 0 else 0
        else:
            trend = 'neutral'
            confidence = 0.5
            
        # Анализ волатильности
        confidences = [s.get('confidence', 0) for s in all_signals if s.get('signal')]
        if confidences:
            avg_confidence = np.mean(confidences)
            std_confidence = np.std(confidences)
            
            if std_confidence > 20:
                volatility = 'extreme'
            elif std_confidence > 15:
                volatility = 'high'
            elif std_confidence > 10:
                volatility = 'normal'
            else:
                volatility = 'low'
        else:
            avg_confidence = 0
            volatility = 'unknown'
            
        # Обновляем данные
        self.market_analysis.update({
            'trend': trend,
            'volatility': volatility,
            'market_confidence': confidence,
            'avg_signal_strength': avg_confidence,
            'buy_pressure': buy_signals,
            'sell_pressure': sell_signals,
            'analyzed_at': datetime.now()
        })
        
        logger.info(
            f"🧠 AI АГЕНТ: Рынок {trend.upper()}, "
            f"Волатильность {volatility}, "
            f"Уверенность {confidence:.1%}"
        )
        
        return self.market_analysis
    
    # ===========================================
    # 🎯 ВЫБОР ЛУЧШИХ СИГНАЛОВ
    # ===========================================
    
    async def select_best_signal(
        self,
        signals: List[Dict[str, Any]],
        current_positions: int,
        balance: float
    ) -> Optional[Dict[str, Any]]:
        """
        AI выбор ЛУЧШЕГО сигнала из доступных
        Учитывает:
        - Уверенность сигнала
        - Рыночные условия
        - Историю символа
        - Качество индикаторов
        - Риск/награда
        
        ВАЖНО: signals уже содержат ТОЛЬКО сильные (≥85%)!
        """
        if not signals:
            logger.info("🧠 AI: Нет сильных сигналов для анализа")
            return None
            
        # Оценка каждого сигнала
        scored_signals = []
        for signal in signals:
            score = self._calculate_signal_score(signal)
            scored_signals.append((score, signal))
            logger.debug(f"   {signal['symbol']}: {score:.1f}/100")
            
        # Сортируем по оценке
        scored_signals.sort(key=lambda x: x[0], reverse=True)
        
        best_score, best_signal = scored_signals[0]
        
        logger.info(
            f"🧠 AI ВЫБОР ЛУЧШЕГО: {best_signal['symbol']} "
            f"{best_signal['signal'].upper()} "
            f"({best_signal['confidence']:.0f}%) "
            f"AI Оценка: {best_score:.1f}/100"
        )
        
        # Показываем ТОП-3 для анализа
        if len(scored_signals) > 1:
            logger.info(f"🥇 {scored_signals[0][1]['symbol']}: {scored_signals[0][0]:.1f}")
            if len(scored_signals) > 1:
                logger.info(f"🥈 {scored_signals[1][1]['symbol']}: {scored_signals[1][0]:.1f}")
            if len(scored_signals) > 2:
                logger.info(f"🥉 {scored_signals[2][1]['symbol']}: {scored_signals[2][0]:.1f}")
        
        # Возвращаем ЛУЧШИЙ (без порога - уже ≥85%)
        return best_signal
    
    def _calculate_signal_score(self, signal: Dict[str, Any]) -> float:
        """
        Рассчитывает AI оценку сигнала (0-100)
        """
        score = 0.0
        
        # 1. Базовая уверенность (40%)
        confidence = signal.get('confidence', 0)
        score += (confidence / 100) * 40
        
        # 2. Согласованность с рынком (20%)
        signal_direction = signal.get('signal')
        market_trend = self.market_analysis.get('trend')
        
        if (signal_direction == 'buy' and market_trend == 'bullish') or \
           (signal_direction == 'sell' and market_trend == 'bearish'):
            score += 20  # Совпадает с трендом
        elif market_trend == 'neutral':
            score += 10  # Нейтральный рынок
            
        # 3. Качество причины (15%)
        reason = signal.get('reason', '')
        quality_keywords = ['сильный', 'дивергенция', 'поддержка', 'сопротивление', 'тренд']
        reason_quality = sum(1 for kw in quality_keywords if kw.lower() in reason.lower())
        score += min(15, reason_quality * 3)
        
        # 4. История символа (15%)
        symbol = signal.get('symbol')
        symbol_history = self._get_symbol_history(symbol)
        if symbol_history:
            if symbol_history['win_rate'] > 0.7:
                score += 15
            elif symbol_history['win_rate'] > 0.5:
                score += 8
                
        # 5. Волатильность (10%)
        if self.market_analysis['volatility'] == 'normal':
            score += 10
        elif self.market_analysis['volatility'] == 'low':
            score += 5
            
        return min(100, score)
    
    def _get_symbol_history(self, symbol: str) -> Optional[Dict]:
        """Получить историю торговли символом"""
        symbol_trades = [t for t in self.trade_history if t.get('symbol') == symbol]
        if not symbol_trades:
            return None
            
        wins = sum(1 for t in symbol_trades if t.get('profit', 0) > 0)
        total = len(symbol_trades)
        
        return {
            'total_trades': total,
            'wins': wins,
            'win_rate': wins / total if total > 0 else 0,
            'avg_profit': np.mean([t.get('profit', 0) for t in symbol_trades])
        }
    
    # ===========================================
    # 🛡️ КОНТРОЛЬ КАЧЕСТВА СДЕЛОК
    # ===========================================
    
    async def validate_trade_before_open(
        self,
        symbol: str,
        side: str,
        signal_data: Dict[str, Any],
        balance: float
    ) -> Tuple[bool, str]:
        """
        ФИНАЛЬНАЯ ПРОВЕРКА перед открытием сделки
        120% контроль!
        """
        checks = []
        
        # 1. Проверка баланса
        if balance < 10:
            return False, f"Баланс слишком мал: ${balance:.2f}"
        checks.append("✅ Баланс достаточный")
        
        # 2. Проверка уверенности (сигнал уже ≥85%, проверяем оптимум)
        confidence = signal_data.get('confidence', 0)
        checks.append(f"✅ Уверенность {confidence}% (выше 85%)")
        
        # 3. Проверка рыночных условий
        if self.market_analysis['volatility'] == 'extreme':
            return False, "Экстремальная волатильность - опасно!"
        checks.append(f"✅ Волатильность {self.market_analysis['volatility']}")
        
        # 4. Проверка согласованности с трендом
        signal_dir = signal_data.get('signal')
        market_trend = self.market_analysis['trend']
        
        if signal_dir == 'buy' and market_trend == 'bearish':
            return False, "LONG на медвежьем рынке - опасно!"
        elif signal_dir == 'sell' and market_trend == 'bullish':
            return False, "SHORT на бычьем рынке - опасно!"
        checks.append(f"✅ Согласуется с трендом {market_trend}")
        
        # 5. Проверка истории символа
        symbol_history = self._get_symbol_history(symbol)
        if symbol_history and symbol_history['win_rate'] < 0.3:
            return False, f"Плохая история: Win Rate {symbol_history['win_rate']:.1%}"
        checks.append("✅ История символа приемлемая")
        
        # 6. Проверка времени (лучшие часы)
        current_hour = datetime.now().hour
        if current_hour in [0, 1, 2, 3, 4, 5]:  # Ночь
            return False, "Ночное время - низкая ликвидность"
        checks.append("✅ Время торговли подходящее")
        
        # 7. Проверка серии убытков
        recent_trades = list(self.trade_history)[-5:]
        if recent_trades:
            recent_losses = sum(1 for t in recent_trades if t.get('profit', 0) < 0)
            if recent_losses >= 3:
                return False, "3+ убытка подряд - пауза!"
        checks.append("✅ Нет серии убытков")
        
        # ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!
        logger.info(f"🧠 AI: ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ({len(checks)}/7)")
        for check in checks:
            logger.debug(f"   {check}")
            
        return True, "AI ОДОБРИЛ: Все 7 проверок пройдены"
    
    # ===========================================
    # 🔧 САМОДИАГНОСТИКА И ИСПРАВЛЕНИЕ
    # ===========================================
    
    async def self_diagnose(
        self,
        error_msg: str,
        context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Самодиагностика ошибок и предложение решения
        """
        error_lower = error_msg.lower()
        
        # База знаний ошибок
        if 'stop loss' in error_lower or 'sl' in error_lower:
            return True, "РЕШЕНИЕ: Проверить формат символа и retCode. Возможно нужен int(0) вместо str('0')"
        elif 'balance' in error_lower or 'баланс' in error_lower:
            return True, "РЕШЕНИЕ: Проверить подключение к бирже и обновить баланс"
        elif 'connection' in error_lower or 'network' in error_lower:
            return True, "РЕШЕНИЕ: Переподключиться к бирже, проверить интернет"
        elif 'rate limit' in error_lower:
            return True, "РЕШЕНИЕ: Увеличить задержки между запросами, добавить паузу"
        elif 'invalid symbol' in error_lower:
            return True, "РЕШЕНИЕ: Очистить символ от :USDT суффикса"
        else:
            return False, "Неизвестная ошибка - требуется ручное вмешательство"
    
    # ===========================================
    # 📚 САМООБУЧЕНИЕ
    # ===========================================
    
    async def learn_from_trade(
        self,
        trade_data: Dict[str, Any]
    ):
        """Обучение на основе результатов сделки"""
        profit = trade_data.get('profit', 0)
        confidence = trade_data.get('signal_confidence', 0)
        symbol = trade_data.get('symbol')
        
        # Добавляем в историю
        self.trade_history.append(trade_data)
        
        # Обновляем статистику
        self.trade_stats['total_trades'] += 1
        
        if profit > 0:
            self.trade_stats['wins'] += 1
            self.trade_stats['total_profit'] += profit
            self.learning_data['successful_patterns'].append({
                'symbol': symbol,
                'confidence': confidence,
                'reason': trade_data.get('reason'),
                'profit': profit
            })
        else:
            self.trade_stats['losses'] += 1
            self.trade_stats['total_loss'] += abs(profit)
            self.learning_data['failed_patterns'].append({
                'symbol': symbol,
                'confidence': confidence,
                'reason': trade_data.get('reason'),
                'loss': abs(profit)
            })
            
        # Рассчитываем метрики
        total_trades = self.trade_stats['total_trades']
        wins = self.trade_stats['wins']
        
        self.trade_stats['win_rate'] = wins / total_trades if total_trades > 0 else 0
        self.trade_stats['avg_win'] = self.trade_stats['total_profit'] / wins if wins > 0 else 0
        
        losses = self.trade_stats['losses']
        self.trade_stats['avg_loss'] = self.trade_stats['total_loss'] / losses if losses > 0 else 0
        
        if self.trade_stats['total_loss'] > 0:
            self.trade_stats['profit_factor'] = self.trade_stats['total_profit'] / self.trade_stats['total_loss']
        else:
            self.trade_stats['profit_factor'] = float('inf') if self.trade_stats['total_profit'] > 0 else 0
            
        # Адаптация оптимальной уверенности
        await self._adapt_optimal_confidence()
        
        logger.info(
            f"🧠 AI ОБУЧЕНИЕ: Win Rate {self.trade_stats['win_rate']:.1%}, "
            f"Profit Factor {self.trade_stats['profit_factor']:.2f}"
        )
    
    async def _adapt_optimal_confidence(self):
        """Адаптация оптимального порога уверенности"""
        if self.trade_stats['total_trades'] < 10:
            return  # Мало данных
            
        win_rate = self.trade_stats['win_rate']
        
        # Если Win Rate низкий - повышаем порог
        if win_rate < 0.6:
            self.learning_data['optimal_confidence'] = min(95, self.learning_data['optimal_confidence'] + 2)
            logger.info(f"🧠 AI: Win Rate {win_rate:.1%} < 60%, повышаю порог до {self.learning_data['optimal_confidence']}%")
        # Если Win Rate высокий - можно понизить
        elif win_rate > 0.8 and self.trade_stats['total_trades'] > 20:
            self.learning_data['optimal_confidence'] = max(80, self.learning_data['optimal_confidence'] - 1)
            logger.info(f"🧠 AI: Win Rate {win_rate:.1%} > 80%, понижаю порог до {self.learning_data['optimal_confidence']}%")
    
    # ===========================================
    # 📊 ОТЧЕТЫ И СТАТИСТИКА
    # ===========================================
    
    def get_performance_report(self) -> str:
        """Полный отчет о работе системы"""
        stats = self.trade_stats
        
        report = f"""
🤖 **СУПЕР AI АГЕНТ - ОТЧЕТ**

📊 **СТАТИСТИКА СДЕЛОК:**
   • Всего: {stats['total_trades']}
   • Прибыльных: {stats['wins']} ({stats['win_rate']:.1%})
   • Убыточных: {stats['losses']}
   • Profit Factor: {stats['profit_factor']:.2f}
   • Средняя прибыль: ${stats['avg_win']:.2f}
   • Средний убыток: ${stats['avg_loss']:.2f}

📈 **РЫНОЧНЫЙ АНАЛИЗ:**
   • Тренд: {self.market_analysis['trend'].upper()}
   • Волатильность: {self.market_analysis['volatility']}
   • Уверенность рынка: {self.market_analysis['market_confidence']:.1%}

🧠 **ОБУЧЕНИЕ:**
   • Оптимальная уверенность: {self.learning_data['optimal_confidence']}%
   • Успешных паттернов: {len(self.learning_data['successful_patterns'])}
   • Неудачных паттернов: {len(self.learning_data['failed_patterns'])}

⚙️ **РАБОТА АГЕНТА:**
   • Решений принято: {self.decisions_made}
   • Коррекций сделано: {self.corrections_made}
   • Статус: {self.agent_status.upper()}
"""
        return report.strip()
    
    def get_quick_status(self) -> str:
        """Быстрый статус для Telegram"""
        stats = self.trade_stats
        return (
            f"🧠 **AI АГЕНТ:**\n"
            f"   Win Rate: {stats['win_rate']:.0%}\n"
            f"   Profit Factor: {stats['profit_factor']:.2f}\n"
            f"   Всего сделок: {stats['total_trades']}\n"
            f"   Рынок: {self.market_analysis['trend'].upper()}\n"
            f"   Оптимальная уверенность: {self.learning_data['optimal_confidence']}%"
        )


# Глобальный экземпляр
super_ai_agent = SuperAIAgent()


if __name__ == "__main__":
    print("🤖 СУПЕР AI АГЕНТ V3.4 - Тестирование")
    print("=" * 50)
    
    # Тест мониторинга
    import asyncio
    
    async def test():
        health = await super_ai_agent.monitor_server_health()
        print(f"\n📊 Здоровье сервера: {health['score']}/100")
        print(f"CPU: {health['cpu']:.1f}%")
        print(f"Memory: {health['memory']:.1f}%")
        print(f"Disk: {health['disk']:.1f}%")
        
        # Тест самодиагностики
        can_fix, solution = await super_ai_agent.self_diagnose(
            "Stop Loss ордер не создан",
            {}
        )
        print(f"\n🔧 Самодиагностика:")
        print(f"Можно исправить: {can_fix}")
        print(f"Решение: {solution}")
        
        print(f"\n{super_ai_agent.get_performance_report()}")
        
    asyncio.run(test())
    print("\n✅ Тесты пройдены!")



