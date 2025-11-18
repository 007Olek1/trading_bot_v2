"""
⚡ SCALPING ENGINE
Быстрые сделки: 5m-15m-30m-1h-4h | 20x leverage | +1-5% = +20-100% ROE
Проверка каждые 10 секунд
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
import logging
import pandas as pd

import config
from indicators import MarketIndicators
from strategies import TrendVolumeStrategy, ManipulationDetector
from utils import calculate_position_size, round_price, round_quantity
from trade_logger import TradeLogger


class ScalpingEngine:
    """Движок скальпинг-торговли"""
    
    def __init__(self, client, logger: logging.Logger):
        self.client = client
        self.logger = logger
        
        self.indicators = MarketIndicators(config.INDICATOR_PARAMS)
        self.trend_volume = TrendVolumeStrategy(config.INDICATOR_PARAMS)
        self.manip_detector = ManipulationDetector()
        
        # Trade logger
        self.trade_logger = TradeLogger(logs_dir="logs")
        
        self.scalping_positions: Dict[str, Dict] = {}
        self.active = True
    
    def get_klines(self, symbol: str, interval: str, limit: int = 200):
        """Получение свечей"""
        try:
            response = self.client.get_kline(
                category="linear", symbol=symbol, interval=interval, limit=limit
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
            
            return df.sort_values('timestamp').reset_index(drop=True)
        except:
            return None
    
    def get_mtf_data(self, symbol: str) -> Dict:
        """MTF данные для скальпинга"""
        mtf_data = {}
        for tf_name, tf_value in config.SCALPING_TIMEFRAMES.items():
            df = self.get_klines(symbol, tf_value, 200)
            if df is not None and len(df) > 0:
                mtf_data[tf_name] = df
        return mtf_data
    
    def analyze_signal(self, symbol: str, mtf_data: Dict) -> Optional[Dict]:
        """Анализ скальпинг-сигнала"""
        if not mtf_data or len(mtf_data) < config.SCALPING_MIN_TIMEFRAME_ALIGNMENT:
            return None
        
        signals = {}
        
        for tf_name, df in mtf_data.items():
            if len(df) < 50:
                continue
            
            indicators = self.indicators.calculate_all(df)
            
            # Стратегии
            trend_sig = self.trend_volume.analyze(df, indicators)
            manip_sig = self.manip_detector.analyze(df, indicators)
            
            # Объединяем
            tf_signal = self._combine_signals([trend_sig, manip_sig])
            signals[tf_name] = tf_signal
        
        # MTF объединение
        final = self._combine_mtf(signals)
        
        if final and final['confidence'] >= config.SCALPING_SIGNAL_THRESHOLDS['min_confidence']:
            if final['timeframes_aligned'] >= config.SCALPING_MIN_TIMEFRAME_ALIGNMENT:
                self.logger.info(
                    f"⚡ SCALPING: {symbol} {final['direction']} | "
                    f"Conf: {final['confidence']:.0%} | TF: {final['timeframes_aligned']}/{len(signals)}"
                )
                return final
        
        return None
    
    def _combine_signals(self, signals: list) -> Dict:
        """Объединение сигналов стратегий"""
        long_w, short_w = 0.0, 0.0
        
        for sig in signals:
            if sig and sig['direction']:
                conf = sig.get('confidence', 0.0)
                if sig['direction'] == 'LONG':
                    long_w += conf
                elif sig['direction'] == 'SHORT':
                    short_w += conf
        
        if long_w > short_w and long_w > 0:
            return {'direction': 'LONG', 'confidence': long_w / len(signals)}
        elif short_w > long_w and short_w > 0:
            return {'direction': 'SHORT', 'confidence': short_w / len(signals)}
        
        return {'direction': None, 'confidence': 0.0}
    
    def _combine_mtf(self, signals: Dict) -> Optional[Dict]:
        """MTF объединение"""
        if not signals:
            return None
        
        long_votes, short_votes, total_conf = 0, 0, 0.0
        
        for sig in signals.values():
            if sig['direction'] == 'LONG':
                long_votes += 1
                total_conf += sig['confidence']
            elif sig['direction'] == 'SHORT':
                short_votes += 1
                total_conf += sig['confidence']
        
        total_votes = long_votes + short_votes
        
        if total_votes < config.SCALPING_MIN_TIMEFRAME_ALIGNMENT:
            return None
        
        if long_votes > short_votes:
            direction = 'LONG'
            alignment = long_votes / len(signals)
        elif short_votes > long_votes:
            direction = 'SHORT'
            alignment = short_votes / len(signals)
        else:
            return None
        
        avg_conf = total_conf / total_votes if total_votes > 0 else 0.0
        
        return {
            'direction': direction,
            'confidence': avg_conf * alignment,
            'timeframes_aligned': total_votes,
            'total_timeframes': len(signals)
        }
    
    async def open_position(self, symbol: str, direction: str, confidence: float, price: float):
        """Открыть скальпинг-позицию"""
        try:
            self.logger.info(f"⚡ Открываю SCALPING: {symbol} {direction} @ ${price:.4f}")
            
            qty = calculate_position_size(
                self.client, symbol, 
                config.SCALPING_POSITION_SIZE_USD,
                config.SCALPING_LEVERAGE, price
            )
            
            if not qty or qty <= 0:
                return False
            
            # Плечо 20x
            try:
                self.client.set_leverage(
                    category="linear", symbol=symbol,
                    buyLeverage=str(config.SCALPING_LEVERAGE),
                    sellLeverage=str(config.SCALPING_LEVERAGE)
                )
            except Exception as e:
                self.logger.warning(f"⚠️ Плечо 20x: {e}")
            
            # Открываем
            side = "Buy" if direction == "LONG" else "Sell"
            order = self.client.place_order(
                category="linear", symbol=symbol, side=side,
                orderType="Market", qty=str(qty),
                timeInForce="GTC", positionIdx=0
            )
            
            if order['retCode'] != 0:
                self.logger.error(f"❌ Ошибка: {order.get('retMsg')}")
                return False
            
            # Логируем сделку
            trade_id = self.trade_logger.log_trade_open(
                symbol=symbol,
                direction=direction,
                entry_price=price,
                quantity=qty,
                leverage=config.SCALPING_LEVERAGE,
                confidence=confidence,
                strategy="Scalping_MTF",
                mode="scalping"
            )
            
            # Сохраняем
            self.scalping_positions[symbol] = {
                'trade_id': trade_id,
                'direction': direction,
                'entry_price': price,
                'quantity': qty,
                'initial_quantity': qty,
                'confidence': confidence,
                'open_time': datetime.now(timezone.utc),
                'leverage': config.SCALPING_LEVERAGE,
                'tp_hits': [],
                'max_profit_percent': 0.0,
                'trailing_sl_active': False,
                'order_id': order['result']['orderId']
            }
            
            # SL
            await self.set_stop_loss(symbol)
            
            self.logger.info(f"✅ SCALPING открыт: {symbol} {direction} | Qty: {qty}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка открытия {symbol}: {e}")
            return False
    
    async def set_stop_loss(self, symbol: str):
        """Установить SL"""
        pos = self.scalping_positions.get(symbol)
        if not pos:
            return
        
        if pos['direction'] == 'LONG':
            sl = pos['entry_price'] * (1 - config.SCALPING_STOP_LOSS_PERCENT / 100)
        else:
            sl = pos['entry_price'] * (1 + config.SCALPING_STOP_LOSS_PERCENT / 100)
        
        try:
            self.client.set_trading_stop(
                category="linear", symbol=symbol,
                stopLoss=str(round_price(symbol, sl)),
                positionIdx=0
            )
            pos['sl_price'] = sl
            self.logger.info(f"🛑 SL: {symbol} @ ${sl:.4f} (-{config.SCALPING_STOP_LOSS_PERCENT}%)")
        except Exception as e:
            self.logger.error(f"❌ SL ошибка {symbol}: {e}")
    
    async def monitor_positions(self):
        """Мониторинг каждые 10 сек"""
        while self.active:
            if self.scalping_positions:
                for symbol in list(self.scalping_positions.keys()):
                    await self.check_position(symbol)
            await asyncio.sleep(config.SCALPING_MONITORING_INTERVAL_SECONDS)
    
    async def check_position(self, symbol: str):
        """Проверка позиции"""
        pos = self.scalping_positions.get(symbol)
        if not pos:
            return
        
        # Текущая цена
        df = self.get_klines(symbol, config.SCALPING_TIMEFRAMES[config.SCALPING_PRIMARY_TIMEFRAME], 10)
        if df is None or len(df) == 0:
            return
        
        price = df['close'].iloc[-1]
        
        # Проверки
        await self.check_tp_levels(symbol, price)
        await self.check_trailing_sl(symbol, price)
        await self.check_duration(symbol)
    
    async def check_tp_levels(self, symbol: str, price: float):
        """Проверка TP и частичное закрытие"""
        pos = self.scalping_positions[symbol]
        
        # PnL%
        if pos['direction'] == 'LONG':
            pnl_pct = ((price - pos['entry_price']) / pos['entry_price']) * 100
        else:
            pnl_pct = ((pos['entry_price'] - price) / pos['entry_price']) * 100
        
        if pnl_pct > pos['max_profit_percent']:
            pos['max_profit_percent'] = pnl_pct
        
        # Проверяем TP уровни
        for tp in config.SCALPING_TP_LEVELS:
            tp_key = f"tp_{tp['price_move_percent']}"
            
            if tp_key in pos['tp_hits']:
                continue
            
            if pnl_pct >= tp['price_move_percent']:
                await self.partial_close(symbol, tp, price, pnl_pct)
    
    async def partial_close(self, symbol: str, tp: Dict, price: float, pnl_pct: float):
        """Частичное закрытие"""
        pos = self.scalping_positions[symbol]
        
        close_pct = tp['close_percent']
        close_qty = pos['quantity'] * (close_pct / 100)
        close_qty = round_quantity(symbol, close_qty)
        
        if close_qty <= 0:
            return
        
        side = "Sell" if pos['direction'] == "LONG" else "Buy"
        
        try:
            order = self.client.place_order(
                category="linear", symbol=symbol, side=side,
                orderType="Market", qty=str(close_qty),
                timeInForce="GTC", positionIdx=0, reduceOnly=True
            )
            
            if order['retCode'] == 0:
                tp_key = f"tp_{tp['price_move_percent']}"
                pos['tp_hits'].append(tp_key)
                pos['quantity'] -= close_qty
                
                roe = pnl_pct * config.SCALPING_LEVERAGE
                pnl_usd = (config.SCALPING_POSITION_SIZE_USD * (close_pct / 100) * (roe / 100))
                
                self.logger.info(
                    f"🎯 TP{len(pos['tp_hits'])}: {symbol} | "
                    f"${price:.4f} | +{pnl_pct:.2f}% | ROE: +{roe:.0f}% | "
                    f"Closed: {close_pct}% | P&L: +${pnl_usd:.2f}"
                )
                
                # Закрыли всё?
                if pos['quantity'] <= 0 or len(pos['tp_hits']) >= len(config.SCALPING_TP_LEVELS):
                    await self.close_position(symbol, "all_tp_hit")
        
        except Exception as e:
            self.logger.error(f"❌ Частичное закрытие {symbol}: {e}")
    
    async def check_trailing_sl(self, symbol: str, price: float):
        """Trailing SL"""
        if not config.SCALPING_USE_TRAILING_SL:
            return
        
        pos = self.scalping_positions[symbol]
        
        if pos['direction'] == 'LONG':
            pnl_pct = ((price - pos['entry_price']) / pos['entry_price']) * 100
        else:
            pnl_pct = ((pos['entry_price'] - price) / pos['entry_price']) * 100
        
        if pnl_pct >= config.SCALPING_TRAILING_SL_ACTIVATION_PERCENT:
            if not pos.get('trailing_sl_active'):
                pos['trailing_sl_active'] = True
                self.logger.info(f"🔄 Trailing SL: {symbol}")
            
            callback = config.SCALPING_TRAILING_SL_CALLBACK_PERCENT / 100
            
            if pos['direction'] == 'LONG':
                new_sl = price * (1 - callback)
            else:
                new_sl = price * (1 + callback)
            
            should_update = False
            if pos['direction'] == 'LONG' and new_sl > pos.get('sl_price', 0):
                should_update = True
            elif pos['direction'] == 'SHORT' and new_sl < pos.get('sl_price', float('inf')):
                should_update = True
            
            if should_update:
                try:
                    self.client.set_trading_stop(
                        category="linear", symbol=symbol,
                        stopLoss=str(round_price(symbol, new_sl)),
                        positionIdx=0
                    )
                    pos['sl_price'] = new_sl
                    self.logger.info(f"🔄 Trailing SL: {symbol} @ ${new_sl:.4f}")
                except Exception as e:
                    self.logger.error(f"❌ Trailing SL {symbol}: {e}")
    
    async def check_duration(self, symbol: str):
        """Проверка времени"""
        pos = self.scalping_positions[symbol]
        duration = datetime.now(timezone.utc) - pos['open_time']
        max_dur = timedelta(minutes=config.SCALPING_MAX_POSITION_DURATION_MINUTES)
        
        if duration > max_dur:
            self.logger.warning(f"⏰ {symbol} слишком долго, закрываю...")
            await self.close_position(symbol, "max_duration")
    
    async def close_position(self, symbol: str, reason: str = "manual"):
        """Закрыть позицию"""
        pos = self.scalping_positions.get(symbol)
        if not pos:
            return
        
        try:
            if pos['quantity'] <= 0:
                del self.scalping_positions[symbol]
                return
            
            side = "Sell" if pos['direction'] == "LONG" else "Buy"
            
            order = self.client.place_order(
                category="linear", symbol=symbol, side=side,
                orderType="Market", qty=str(pos['quantity']),
                timeInForce="GTC", positionIdx=0, reduceOnly=True
            )
            
            if order['retCode'] == 0:
                # Получаем текущую цену
                df = self.get_klines(symbol, config.SCALPING_TIMEFRAMES[config.SCALPING_PRIMARY_TIMEFRAME], 10)
                exit_price = df['close'].iloc[-1] if df is not None and len(df) > 0 else pos['entry_price']
                
                # Логируем закрытие
                if 'trade_id' in pos:
                    self.trade_logger.log_trade_close(
                        trade_id=pos['trade_id'],
                        exit_price=exit_price,
                        close_reason=reason,
                        tp_hits=pos.get('tp_hits', [])
                    )
                
                self.logger.info(f"✅ Закрыт: {symbol} | Reason: {reason}")
                del self.scalping_positions[symbol]
        
        except Exception as e:
            self.logger.error(f"❌ Закрытие {symbol}: {e}")
