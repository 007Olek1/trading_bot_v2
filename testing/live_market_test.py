"""
2️⃣ LIVE MARKET TESTING ANALYSIS
Анализ рынка в реальном времени БЕЗ реального исполнения сделок
Безопасное тестирование стратегий на живых данных
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
import json
from pathlib import Path

from pybit.unified_trading import HTTP
import pandas as pd

import config
from indicators import MarketIndicators
from strategies import TrendVolumeStrategy, ManipulationDetector, GlobalTrendAnalyzer
from utils import setup_logging


class LiveMarketTester:
    """Тестирование на реальном рынке без исполнения сделок"""
    
    def __init__(self, test_mode: bool = True):
        """
        Args:
            test_mode: True = только анализ, False = реальные сделки
        """
        self.test_mode = test_mode
        self.logger = setup_logging(
            Path(__file__).parent / "logs" / "live_market_test.log",
            "INFO"
        )
        
        # Создаём директорию для логов
        (Path(__file__).parent / "logs").mkdir(exist_ok=True)
        
        self.logger.info("="*80)
        self.logger.info("2️⃣ LIVE MARKET TESTING - ЗАПУСК")
        self.logger.info(f"Режим: {'ТЕСТОВЫЙ (без реальных сделок)' if test_mode else 'РЕАЛЬНЫЙ'}")
        self.logger.info("="*80)
        
        # API клиент
        self.client = HTTP(
            testnet=config.USE_TESTNET,
            api_key=config.BYBIT_API_KEY,
            api_secret=config.BYBIT_API_SECRET,
        )
        
        # Индикаторы и стратегии
        self.indicators_calc = MarketIndicators(config.INDICATOR_PARAMS)
        self.trend_volume = TrendVolumeStrategy(config.INDICATOR_PARAMS)
        self.manipulation = ManipulationDetector()
        self.global_trend = GlobalTrendAnalyzer()
        
        # Виртуальные позиции (для тестирования)
        self.virtual_positions: Dict[str, Dict] = {}
        
        # Статистика
        self.signals_found = 0
        self.signals_tested = 0
        self.test_results = []
        
        self.active = True
    
    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Получение свечей с биржи"""
        try:
            response = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            if response['retCode'] != 0:
                return None
            
            klines = response['result']['list']
            if not klines:
                return None
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка получения данных {symbol} {interval}: {e}")
            return None
    
    def get_mtf_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Получение MTF данных"""
        mtf_data = {}
        
        for tf_name, tf_value in config.TIMEFRAMES.items():
            df = self.get_klines(symbol, tf_value, limit=200)
            if df is not None and len(df) > 0:
                mtf_data[tf_name] = df
        
        return mtf_data
    
    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Анализ одного символа"""
        try:
            # Получаем MTF данные
            mtf_data = self.get_mtf_data(symbol)
            
            if not mtf_data or len(mtf_data) < config.MIN_TIMEFRAME_ALIGNMENT:
                return None
            
            # Анализируем каждый таймфрейм
            signals = {}
            
            for tf_name, df in mtf_data.items():
                if len(df) < 50:
                    continue
                
                # Рассчитываем индикаторы
                indicators = self.indicators_calc.calculate_all(df)
                
                # Применяем стратегии
                strategy_signals = {}
                
                if config.STRATEGIES['trend_volume_bb']['enabled']:
                    trend_signal = self.trend_volume.analyze(df, indicators)
                    strategy_signals['trend_volume_bb'] = trend_signal
                
                if config.STRATEGIES['manipulation_detector']['enabled']:
                    manip_signal = self.manipulation.analyze(df, indicators)
                    strategy_signals['manipulation_detector'] = manip_signal
                
                if config.STRATEGIES['global_trend']['enabled'] and tf_name in ['4h', '1d']:
                    global_signal = self.global_trend.analyze(df, indicators)
                    strategy_signals['global_trend'] = global_signal
                
                # Объединяем сигналы стратегий
                tf_signal = self._combine_strategy_signals(strategy_signals)
                signals[tf_name] = tf_signal
            
            # Объединяем MTF сигналы
            final_signal = self._combine_mtf_signals(signals)
            
            if final_signal and final_signal['confidence'] >= config.SIGNAL_THRESHOLDS['min_confidence']:
                final_signal['symbol'] = symbol
                final_signal['price'] = mtf_data[config.PRIMARY_TIMEFRAME]['close'].iloc[-1]
                final_signal['timestamp'] = datetime.now(timezone.utc)
                return final_signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа {symbol}: {e}")
            return None
    
    def _combine_strategy_signals(self, strategy_signals: Dict) -> Dict:
        """Объединение сигналов стратегий"""
        if not strategy_signals:
            return {'direction': None, 'confidence': 0.0}
        
        long_weight = 0.0
        short_weight = 0.0
        total_weight = 0.0
        
        for strategy_name, signal in strategy_signals.items():
            if signal is None or signal['direction'] is None:
                continue
            
            weight = config.STRATEGIES.get(strategy_name, {}).get('weight', 0.0)
            confidence = signal.get('confidence', 0.0)
            
            if signal['direction'] == 'LONG':
                long_weight += weight * confidence
            elif signal['direction'] == 'SHORT':
                short_weight += weight * confidence
            
            total_weight += weight
        
        if total_weight == 0:
            return {'direction': None, 'confidence': 0.0}
        
        if long_weight > short_weight:
            direction = 'LONG'
            confidence = long_weight / total_weight
        elif short_weight > long_weight:
            direction = 'SHORT'
            confidence = short_weight / total_weight
        else:
            direction = None
            confidence = 0.0
        
        return {
            'direction': direction,
            'confidence': confidence,
            'long_weight': long_weight,
            'short_weight': short_weight
        }
    
    def _combine_mtf_signals(self, signals: Dict[str, Dict]) -> Optional[Dict]:
        """Объединение MTF сигналов"""
        if not signals:
            return None
        
        long_votes = 0
        short_votes = 0
        total_confidence = 0.0
        
        for tf_name, signal in signals.items():
            if signal['direction'] == 'LONG':
                long_votes += 1
                total_confidence += signal['confidence']
            elif signal['direction'] == 'SHORT':
                short_votes += 1
                total_confidence += signal['confidence']
        
        total_votes = long_votes + short_votes
        
        if total_votes < config.MIN_TIMEFRAME_ALIGNMENT:
            return None
        
        if long_votes > short_votes:
            direction = 'LONG'
            alignment = long_votes / len(signals)
        elif short_votes > long_votes:
            direction = 'SHORT'
            alignment = short_votes / len(signals)
        else:
            return None
        
        avg_confidence = total_confidence / total_votes if total_votes > 0 else 0.0
        final_confidence = avg_confidence * alignment
        
        return {
            'direction': direction,
            'confidence': final_confidence,
            'timeframes_aligned': total_votes,
            'total_timeframes': len(signals),
            'alignment_ratio': alignment,
            'details': signals
        }
    
    def create_virtual_position(self, signal: Dict):
        """Создание виртуальной позиции (для тестирования)"""
        symbol = signal['symbol']
        
        if symbol in self.virtual_positions:
            self.logger.warning(f"⚠️ Виртуальная позиция для {symbol} уже существует")
            return
        
        self.virtual_positions[symbol] = {
            'direction': signal['direction'],
            'entry_price': signal['price'],
            'entry_time': signal['timestamp'],
            'confidence': signal['confidence'],
            'timeframes_aligned': signal['timeframes_aligned'],
            'status': 'OPEN'
        }
        
        self.logger.info(
            f"🎯 ВИРТУАЛЬНАЯ ПОЗИЦИЯ ОТКРЫТА: {symbol} {signal['direction']} "
            f"@ ${signal['price']:.6f} (уверенность: {signal['confidence']:.1%})"
        )
        
        self.signals_tested += 1
    
    def check_virtual_positions(self):
        """Проверка виртуальных позиций"""
        for symbol in list(self.virtual_positions.keys()):
            pos = self.virtual_positions[symbol]
            
            if pos['status'] != 'OPEN':
                continue
            
            # Получаем текущую цену
            mtf_data = self.get_mtf_data(symbol)
            if not mtf_data or config.PRIMARY_TIMEFRAME not in mtf_data:
                continue
            
            current_price = mtf_data[config.PRIMARY_TIMEFRAME]['close'].iloc[-1]
            
            # Рассчитываем PnL
            if pos['direction'] == 'LONG':
                pnl_percent = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
            else:  # SHORT
                pnl_percent = ((pos['entry_price'] - current_price) / pos['entry_price']) * 100
            
            # Проверяем TP/SL
            tp_hit = False
            sl_hit = False
            
            for tp_level in config.TAKE_PROFIT_LEVELS:
                if pnl_percent >= tp_level['percent']:
                    tp_hit = True
                    break
            
            max_loss_percent = (config.STOP_LOSS_MAX_USD / (config.POSITION_SIZE_USD * config.LEVERAGE)) * 100
            if pnl_percent <= -max_loss_percent:
                sl_hit = True
            
            # Закрываем позицию если достигнут TP или SL
            if tp_hit or sl_hit:
                pos['status'] = 'CLOSED'
                pos['exit_price'] = current_price
                pos['exit_time'] = datetime.now(timezone.utc)
                pos['pnl_percent'] = pnl_percent
                pos['result'] = 'TP' if tp_hit else 'SL'
                
                duration = (pos['exit_time'] - pos['entry_time']).total_seconds() / 60
                
                self.logger.info(
                    f"{'💚' if tp_hit else '❤️'} ВИРТУАЛЬНАЯ ПОЗИЦИЯ ЗАКРЫТА: {symbol} "
                    f"PnL: {pnl_percent:+.2f}% ({pos['result']}) "
                    f"Время: {duration:.0f}м"
                )
                
                # Сохраняем результат
                self.test_results.append(pos.copy())
    
    async def scan_market(self):
        """Сканирование рынка"""
        self.logger.info("🔍 Начинаю сканирование рынка...")
        
        signals_found = []
        
        for symbol in config.WATCHLIST[:10]:  # Ограничиваем для теста
            # Пропускаем если уже есть виртуальная позиция
            if symbol in self.virtual_positions and self.virtual_positions[symbol]['status'] == 'OPEN':
                continue
            
            signal = self.analyze_symbol(symbol)
            
            if signal:
                signals_found.append(signal)
                self.signals_found += 1
                
                self.logger.info(
                    f"📊 СИГНАЛ: {symbol} {signal['direction']} "
                    f"(уверенность: {signal['confidence']:.1%}, "
                    f"таймфреймов: {signal['timeframes_aligned']}/{signal['total_timeframes']})"
                )
        
        # Создаём виртуальные позиции для найденных сигналов
        for signal in signals_found:
            if len([p for p in self.virtual_positions.values() if p['status'] == 'OPEN']) < config.MAX_CONCURRENT_POSITIONS:
                self.create_virtual_position(signal)
        
        if not signals_found:
            self.logger.info("📭 Сигналов не найдено")
    
    async def run(self, duration_minutes: int = 60):
        """
        Запуск тестирования
        
        Args:
            duration_minutes: Длительность теста в минутах
        """
        self.logger.info(f"🚀 Запуск тестирования на {duration_minutes} минут")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        scan_interval = 300  # Сканирование каждые 5 минут
        check_interval = 60  # Проверка позиций каждую минуту
        
        last_scan = 0
        last_check = 0
        
        while self.active and time.time() < end_time:
            try:
                current_time = time.time()
                
                # Сканирование рынка
                if current_time - last_scan >= scan_interval:
                    await self.scan_market()
                    last_scan = current_time
                
                # Проверка виртуальных позиций
                if current_time - last_check >= check_interval:
                    self.check_virtual_positions()
                    last_check = current_time
                
                # Пауза
                await asyncio.sleep(10)
                
            except KeyboardInterrupt:
                self.logger.info("⏹️ Тестирование остановлено пользователем")
                break
            except Exception as e:
                self.logger.error(f"❌ Ошибка: {e}")
                await asyncio.sleep(60)
        
        # Итоговый отчёт
        self.generate_report()
    
    def generate_report(self):
        """Генерация итогового отчёта"""
        self.logger.info("\n" + "="*80)
        self.logger.info("📊 ИТОГОВЫЙ ОТЧЁТ LIVE MARKET TESTING")
        self.logger.info("="*80)
        
        self.logger.info(f"🔍 Сигналов найдено: {self.signals_found}")
        self.logger.info(f"🎯 Сигналов протестировано: {self.signals_tested}")
        
        if self.test_results:
            tp_count = sum(1 for r in self.test_results if r['result'] == 'TP')
            sl_count = sum(1 for r in self.test_results if r['result'] == 'SL')
            
            avg_pnl = sum(r['pnl_percent'] for r in self.test_results) / len(self.test_results)
            
            self.logger.info(f"\n📈 Закрытые позиции: {len(self.test_results)}")
            self.logger.info(f"💚 TP: {tp_count} ({tp_count/len(self.test_results)*100:.1f}%)")
            self.logger.info(f"❤️ SL: {sl_count} ({sl_count/len(self.test_results)*100:.1f}%)")
            self.logger.info(f"📊 Средний PnL: {avg_pnl:+.2f}%")
            
            # Сохраняем результаты в файл
            results_file = Path(__file__).parent / "logs" / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            
            self.logger.info(f"\n💾 Результаты сохранены: {results_file}")
        else:
            self.logger.info("\n⚠️ Нет закрытых позиций для анализа")
        
        # Открытые позиции
        open_positions = [p for p in self.virtual_positions.values() if p['status'] == 'OPEN']
        if open_positions:
            self.logger.info(f"\n📌 Открытых позиций: {len(open_positions)}")
        
        self.logger.info("="*80 + "\n")


async def main():
    """Точка входа"""
    print("\n" + "="*80)
    print("2️⃣ LIVE MARKET TESTING ANALYSIS")
    print("Тестирование на реальном рынке БЕЗ исполнения сделок")
    print("="*80 + "\n")
    
    # Запрашиваем длительность теста
    try:
        duration = int(input("Введите длительность теста в минутах (по умолчанию 60): ") or "60")
    except ValueError:
        duration = 60
    
    tester = LiveMarketTester(test_mode=True)
    await tester.run(duration_minutes=duration)


if __name__ == "__main__":
    asyncio.run(main())
