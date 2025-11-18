"""
🧪 BACKTESTING СИСТЕМА
Тестирование стратегий на исторических данных
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from pybit.unified_trading import HTTP

import config
from indicators import MarketIndicators
from strategies import TrendVolumeStrategy, ManipulationDetector, GlobalTrendAnalyzer


class Backtester:
    """Система бэктестинга торговых стратегий"""
    
    def __init__(self, client: HTTP, logger: logging.Logger = None):
        self.client = client
        self.logger = logger or logging.getLogger(__name__)
        
        self.indicators = MarketIndicators(config.INDICATOR_PARAMS)
        self.trend_volume = TrendVolumeStrategy(config.INDICATOR_PARAMS)
        self.manip_detector = ManipulationDetector()
        self.global_trend = GlobalTrendAnalyzer()
        
        self.trades = []
        self.equity_curve = []
    
    def get_historical_data(self, symbol: str, interval: str, days: int = 30) -> pd.DataFrame:
        """Получение исторических данных"""
        try:
            # Рассчитываем количество свечей
            intervals_per_day = {
                '5': 288, '15': 96, '30': 48, '60': 24, '240': 6, 'D': 1
            }
            limit = min(intervals_per_day.get(interval, 96) * days, 1000)
            
            response = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            if response['retCode'] != 0:
                return None
            
            klines = response['result']['list']
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
        
        except Exception as e:
            self.logger.error(f"Ошибка получения данных {symbol}: {e}")
            return None
    
    def analyze_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """Анализ сигнала (упрощенная версия)"""
        if len(df) < 50:
            return None
        
        indicators = self.indicators.calculate_all(df)
        
        # Применяем стратегии
        trend_signal = self.trend_volume.analyze(df, indicators)
        manip_signal = self.manip_detector.analyze(df, indicators)
        
        if not trend_signal or not manip_signal:
            return None
        
        # Объединяем сигналы
        if trend_signal['direction'] == manip_signal['direction']:
            confidence = (trend_signal['confidence'] + manip_signal['confidence']) / 2
            return {
                'direction': trend_signal['direction'],
                'confidence': confidence
            }
        
        return None
    
    def simulate_trade(self, entry_price: float, direction: str, 
                      df_future: pd.DataFrame, leverage: int = 10) -> Dict:
        """Симуляция сделки"""
        # TP и SL
        if direction == 'LONG':
            tp_price = entry_price * 1.08  # +8%
            sl_price = entry_price * 0.99  # -1%
        else:  # SHORT
            tp_price = entry_price * 0.92  # -8%
            sl_price = entry_price * 1.01  # +1%
        
        # Проверяем каждую свечу
        for i, row in df_future.iterrows():
            high = row['high']
            low = row['low']
            
            # Проверка TP/SL
            if direction == 'LONG':
                if high >= tp_price:
                    pnl_percent = 8.0
                    roe = pnl_percent * leverage
                    return {
                        'exit_price': tp_price,
                        'pnl_percent': pnl_percent,
                        'roe': roe,
                        'result': 'WIN',
                        'reason': 'TP'
                    }
                elif low <= sl_price:
                    pnl_percent = -1.0
                    roe = pnl_percent * leverage
                    return {
                        'exit_price': sl_price,
                        'pnl_percent': pnl_percent,
                        'roe': roe,
                        'result': 'LOSS',
                        'reason': 'SL'
                    }
            else:  # SHORT
                if low <= tp_price:
                    pnl_percent = 8.0
                    roe = pnl_percent * leverage
                    return {
                        'exit_price': tp_price,
                        'pnl_percent': pnl_percent,
                        'roe': roe,
                        'result': 'WIN',
                        'reason': 'TP'
                    }
                elif high >= sl_price:
                    pnl_percent = -1.0
                    roe = pnl_percent * leverage
                    return {
                        'exit_price': sl_price,
                        'pnl_percent': pnl_percent,
                        'roe': roe,
                        'result': 'LOSS',
                        'reason': 'SL'
                    }
        
        # Не достигли ни TP ни SL
        return None
    
    def run_backtest(self, symbol: str, days: int = 30, 
                    interval: str = '15', leverage: int = 10) -> Dict:
        """Запуск бэктеста"""
        self.logger.info(f"🧪 Backtesting {symbol} за последние {days} дней...")
        
        # Получаем данные
        df = self.get_historical_data(symbol, interval, days)
        if df is None or len(df) < 100:
            return None
        
        self.trades = []
        initial_balance = 100.0
        balance = initial_balance
        
        # Проходим по данным
        for i in range(50, len(df) - 50):  # Оставляем запас для будущих данных
            # Анализируем сигнал
            df_past = df.iloc[:i+1]
            signal = self.analyze_signal(df_past)
            
            if signal and signal['confidence'] >= config.SIGNAL_THRESHOLDS['min_confidence']:
                entry_price = df.iloc[i]['close']
                direction = signal['direction']
                
                # Симулируем сделку
                df_future = df.iloc[i+1:i+51]  # Следующие 50 свечей
                result = self.simulate_trade(entry_price, direction, df_future, leverage)
                
                if result:
                    # Рассчитываем P&L
                    position_size = 1.0  # $1 позиция
                    pnl = position_size * (result['roe'] / 100)
                    balance += pnl
                    
                    trade = {
                        'timestamp': df.iloc[i]['timestamp'],
                        'symbol': symbol,
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': result['exit_price'],
                        'confidence': signal['confidence'],
                        'pnl_percent': result['pnl_percent'],
                        'roe': result['roe'],
                        'pnl_usd': pnl,
                        'balance': balance,
                        'result': result['result'],
                        'reason': result['reason']
                    }
                    
                    self.trades.append(trade)
                    self.equity_curve.append(balance)
        
        # Рассчитываем метрики
        if not self.trades:
            return {
                'symbol': symbol,
                'period_days': days,
                'total_trades': 0,
                'message': 'Нет сделок за период'
            }
        
        wins = [t for t in self.trades if t['result'] == 'WIN']
        losses = [t for t in self.trades if t['result'] == 'LOSS']
        
        total_pnl = sum(t['pnl_usd'] for t in self.trades)
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        
        # Максимальная просадка
        peak = initial_balance
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        results = {
            'symbol': symbol,
            'period_days': days,
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl_usd': total_pnl,
            'total_pnl_percent': (balance - initial_balance) / initial_balance * 100,
            'avg_win': sum(t['pnl_usd'] for t in wins) / len(wins) if wins else 0,
            'avg_loss': sum(t['pnl_usd'] for t in losses) / len(losses) if losses else 0,
            'best_trade': max((t['pnl_usd'] for t in self.trades), default=0),
            'worst_trade': min((t['pnl_usd'] for t in self.trades), default=0),
            'max_drawdown': max_dd,
            'final_balance': balance,
            'profit_factor': abs(sum(t['pnl_usd'] for t in wins) / sum(t['pnl_usd'] for t in losses)) if losses else 0,
        }
        
        self.logger.info(f"✅ Backtest завершен: {len(self.trades)} сделок, Win Rate: {win_rate:.1f}%")
        
        return results
    
    def print_results(self, results: Dict):
        """Красивый вывод результатов"""
        print("\n" + "="*80)
        print(f"📊 РЕЗУЛЬТАТЫ БЭКТЕСТА: {results['symbol']}")
        print("="*80)
        print(f"📅 Период: {results['period_days']} дней")
        print(f"📈 Всего сделок: {results['total_trades']}")
        print(f"✅ Прибыльных: {results['wins']}")
        print(f"❌ Убыточных: {results['losses']}")
        print(f"📊 Win Rate: {results['win_rate']:.1f}%")
        print()
        print(f"💰 Общий P&L: ${results['total_pnl_usd']:.2f} ({results['total_pnl_percent']:+.1f}%)")
        print(f"📊 Средняя прибыль: ${results['avg_win']:.2f}")
        print(f"📊 Средний убыток: ${results['avg_loss']:.2f}")
        print(f"🏆 Лучшая сделка: ${results['best_trade']:.2f}")
        print(f"💔 Худшая сделка: ${results['worst_trade']:.2f}")
        print(f"📉 Макс. просадка: {results['max_drawdown']:.1f}%")
        print(f"📊 Profit Factor: {results['profit_factor']:.2f}")
        print(f"💵 Финальный баланс: ${results['final_balance']:.2f}")
        print("="*80)
    
    def export_trades(self, filename: str = "backtest_trades.csv"):
        """Экспорт сделок в CSV"""
        if not self.trades:
            return
        
        df = pd.DataFrame(self.trades)
        df.to_csv(filename, index=False)
        self.logger.info(f"📁 Сделки экспортированы в {filename}")


if __name__ == "__main__":
    # Пример использования
    import sys
    sys.path.insert(0, '.')
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    client = HTTP(
        testnet=config.USE_TESTNET,
        api_key=config.BYBIT_API_KEY,
        api_secret=config.BYBIT_API_SECRET
    )
    
    backtester = Backtester(client, logger)
    
    # Тест на BTC
    results = backtester.run_backtest('BTCUSDT', days=30, leverage=10)
    
    if results:
        backtester.print_results(results)
        backtester.export_trades()
