"""
🤖 DISCO57 BOT - ГЛАВНЫЙ МОДУЛЬ
Автоматизированный торговый бот для Bybit фьючерсов
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import logging

from pybit.unified_trading import HTTP
import pandas as pd

import config
from utils import (
    setup_logging,
    save_trade_log,
    calculate_position_size,
    calculate_sl_tp_prices,
    format_telegram_message,
    round_price,
    round_quantity,
)
from indicators import MarketIndicators, detect_market_mode


class Disco57Bot:
    """Главный класс торгового бота Disco57"""
    
    def __init__(self):
        """Инициализация бота"""
        self.logger = setup_logging(config.LOG_FILE, config.LOG_LEVEL)
        self.logger.info("="*70)
        self.logger.info("🤖 DISCO57 BOT - ИНИЦИАЛИЗАЦИЯ")
        self.logger.info("="*70)
        
        # Bybit API клиент
        self.client = HTTP(
            testnet=config.USE_TESTNET,
            api_key=config.BYBIT_API_KEY,
            api_secret=config.BYBIT_API_SECRET,
        )
        
        # Индикаторы
        self.indicators_calculator = MarketIndicators(config.INDICATOR_PARAMS)
        
        # Состояние
        self.active = True
        self.open_positions: Dict[str, Dict] = {}  # {symbol: position_info}
        self.last_analysis_time = None
        self.cycle_count = 0
        
        self.logger.info(f"✅ Watchlist: {len(config.WATCHLIST)} монет")
        self.logger.info(f"✅ Таймфреймы: {list(config.TIMEFRAMES.values())}")
        self.logger.info(f"✅ Макс. позиций: {config.MAX_CONCURRENT_POSITIONS}")
        self.logger.info(f"✅ Размер позиции: ${config.POSITION_SIZE_USD} × {config.LEVERAGE}x = ${config.POSITION_SIZE_USD * config.LEVERAGE}")
    
    # ═══════════════════════════════════════════════════════════════════
    # ПОЛУЧЕНИЕ ДАННЫХ
    # ═══════════════════════════════════════════════════════════════════
    
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
                self.logger.warning(f"Ошибка получения данных для {symbol} {interval}: {response.get('retMsg')}")
                return None
            
            klines = response['result']['list']
            
            if not klines:
                return None
            
            # Конвертация в DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            
            # Преобразование типов
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Сортировка по времени (от старых к новым)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка get_klines для {symbol} {interval}: {e}")
            return None
    
    def get_multitimeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Получение данных по всем таймфреймам"""
        data = {}
        
        for tf_name, tf_value in config.TIMEFRAMES.items():
            df = self.get_klines(symbol, tf_value)
            if df is not None and len(df) >= 200:
                data[tf_name] = df
        
        return data
    
    # ═══════════════════════════════════════════════════════════════════
    # РАБОТА С ПОЗИЦИЯМИ
    # ═══════════════════════════════════════════════════════════════════
    
    def get_active_positions(self) -> List[Dict]:
        """Получение всех активных позиций"""
        try:
            response = self.client.get_positions(
                category="linear",
                settleCoin="USDT"
            )
            
            if response['retCode'] != 0:
                self.logger.error(f"Ошибка получения позиций: {response.get('retMsg')}")
                return []
            
            positions = []
            for pos in response['result']['list']:
                size = float(pos.get('size', 0))
                if size > 0:
                    positions.append({
                        'symbol': pos['symbol'],
                        'side': pos['side'],
                        'size': size,
                        'entry_price': float(pos.get('avgPrice', 0)),
                        'mark_price': float(pos.get('markPrice', 0)),
                        'unrealized_pnl': float(pos.get('unrealisedPnl', 0)),
                        'leverage': float(pos.get('leverage', 0)),
                        'take_profit': pos.get('takeProfit', ''),
                        'stop_loss': pos.get('stopLoss', ''),
                        'created_time': pos.get('createdTime', ''),
                    })
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Ошибка get_active_positions: {e}")
            return []
    
    def can_open_position(self, symbol: str) -> bool:
        """Проверка возможности открытия позиции"""
        active_positions = self.get_active_positions()
        
        # Проверка максимального количества позиций
        if len(active_positions) >= config.MAX_CONCURRENT_POSITIONS:
            return False
        
        # Проверка что по этому символу нет открытой позиции
        for pos in active_positions:
            if pos['symbol'] == symbol:
                return False
        
        return True
    
    def get_balance(self) -> float:
        """Получение доступного баланса"""
        try:
            response = self.client.get_wallet_balance(accountType="UNIFIED")
            
            if response['retCode'] != 0:
                self.logger.error(f"Ошибка получения баланса: {response.get('retMsg')}")
                return 0.0
            
            balance_list = response['result']['list']
            if balance_list:
                for coin in balance_list[0].get('coin', []):
                    if coin.get('coin') == 'USDT':
                        return float(coin.get('availableToWithdraw', 0))
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Ошибка get_balance: {e}")
            return 0.0
    
    # ═══════════════════════════════════════════════════════════════════
    # ОТКРЫТИЕ/ЗАКРЫТИЕ ПОЗИЦИЙ
    # ═══════════════════════════════════════════════════════════════════
    
    def open_position(self, symbol: str, side: str, price: float, confidence: float, timeframes_aligned: int) -> bool:
        """
        Открытие позиции с автоматическим SL/TP
        
        Args:
            symbol: Символ монеты (BTCUSDT)
            side: "Buy" или "Sell"
            price: Текущая цена
            confidence: Уверенность в сигнале (%)
            timeframes_aligned: Количество подтверждающих таймфреймов
        
        Returns:
            True если позиция успешно открыта
        """
        try:
            # Проверка баланса
            balance = self.get_balance()
            
            # Расчет размера позиции
            qty = calculate_position_size(balance, config.POSITION_SIZE_USD, config.LEVERAGE, price)
            
            if qty <= 0:
                self.logger.warning(f"Недостаточно баланса для открытия позиции {symbol}")
                return False
            
            # Округление количества
            qty = round_quantity(qty, 0.001)
            
            self.logger.info(f"🔄 Открытие позиции {symbol} {side} | Qty: {qty} | Price: ${price:.6f}")
            
            # Размещение ордера
            response = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(qty),
                leverage=str(config.LEVERAGE),
                positionIdx=0,  # One-way mode
            )
            
            if response['retCode'] != 0:
                self.logger.error(f"Ошибка открытия позиции: {response.get('retMsg')}")
                return False
            
            order_id = response['result'].get('orderId', '')
            self.logger.info(f"✅ Ордер размещен: {order_id}")
            
            # Ждем пока позиция откроется
            time.sleep(1)
            
            # Получаем актуальную цену входа
            positions = self.get_active_positions()
            entry_price = price
            for pos in positions:
                if pos['symbol'] == symbol:
                    entry_price = pos['entry_price']
                    break
            
            # Рассчитываем SL/TP
            sl_tp = calculate_sl_tp_prices(
                entry_price,
                side,
                config.STOP_LOSS_PERCENT,
                config.TAKE_PROFIT_PERCENT
            )
            
            # Устанавливаем SL/TP
            self.logger.info(f"🛡️ Установка SL/TP: SL=${sl_tp['stop_loss']:.6f} | TP=${sl_tp['take_profit']:.6f}")
            
            sl_tp_response = self.client.set_trading_stop(
                category="linear",
                symbol=symbol,
                stopLoss=str(sl_tp['stop_loss']),
                takeProfit=str(sl_tp['take_profit']),
                positionIdx=0,
            )
            
            if sl_tp_response['retCode'] != 0:
                self.logger.warning(f"⚠️ Не удалось установить SL/TP: {sl_tp_response.get('retMsg')}")
            else:
                self.logger.info(f"✅ SL/TP установлены успешно")
            
            # Логирование сделки
            trade_data = {
                "type": "trade_open",
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "size": qty,
                "stop_loss": sl_tp['stop_loss'],
                "take_profit": sl_tp['take_profit'],
                "sl_percent": config.STOP_LOSS_PERCENT,
                "tp_percent": config.TAKE_PROFIT_PERCENT,
                "confidence": confidence,
                "timeframes_aligned": timeframes_aligned,
                "order_id": order_id,
            }
            
            save_trade_log(trade_data, config.TRADES_LOG_FILE)
            
            # Отправка в Telegram
            if config.TELEGRAM_TOKEN and config.TELEGRAM_CHAT_ID:
                self.send_telegram(format_telegram_message(trade_data))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка open_position для {symbol}: {e}", exc_info=True)
            return False
    
    def close_position(self, symbol: str, reason: str = "manual") -> bool:
        """Закрытие позиции"""
        try:
            # Получаем информацию о позиции
            positions = self.get_active_positions()
            position = None
            for pos in positions:
                if pos['symbol'] == symbol:
                    position = pos
                    break
            
            if not position:
                self.logger.warning(f"Позиция {symbol} не найдена")
                return False
            
            # Закрываем позицию market ордером
            close_side = "Sell" if position['side'] == "Buy" else "Buy"
            
            response = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=close_side,
                orderType="Market",
                qty=str(position['size']),
                reduceOnly=True,
                positionIdx=0,
            )
            
            if response['retCode'] != 0:
                self.logger.error(f"Ошибка закрытия позиции {symbol}: {response.get('retMsg')}")
                return False
            
            # Расчет PnL
            pnl = position['unrealized_pnl']
            entry = position['entry_price']
            exit_price = position['mark_price']
            pnl_percent = (pnl / (config.POSITION_SIZE_USD * config.LEVERAGE)) * 100
            
            self.logger.info(f"✅ Позиция {symbol} закрыта | PnL: ${pnl:.2f} ({pnl_percent:.2f}%)")
            
            # Логирование
            trade_data = {
                "type": "trade_close",
                "symbol": symbol,
                "side": position['side'],
                "entry_price": entry,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
                "reason": reason,
            }
            
            save_trade_log(trade_data, config.TRADES_LOG_FILE)
            
            # Отправка в Telegram
            if config.TELEGRAM_TOKEN and config.TELEGRAM_CHAT_ID:
                self.send_telegram(format_telegram_message(trade_data))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка close_position для {symbol}: {e}", exc_info=True)
            return False
    
    # ═══════════════════════════════════════════════════════════════════
    # АНАЛИЗ РЫНКА
    # ═══════════════════════════════════════════════════════════════════
    
    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Анализ одного символа по всем таймфреймам
        
        Returns:
            None если нет данных или dict с результатами анализа
        """
        try:
            # Получаем данные по всем таймфреймам
            mtf_data = self.get_multitimeframe_data(symbol)
            
            if len(mtf_data) < 2:  # Минимум 2 таймфрейма
                return None
            
            # Расчет индикаторов для каждого таймфрейма
            mtf_indicators = {}
            for tf, df in mtf_data.items():
                indicators = self.indicators_calculator.calculate_all(df)
                if indicators:
                    mtf_indicators[tf] = indicators
            
            if not mtf_indicators:
                return None
            
            # Генерация сигнала
            signal, confidence, aligned_tf = self.indicators_calculator.generate_signal(mtf_indicators)
            
            # Определение режима рынка
            primary_indicators = mtf_indicators.get(config.PRIMARY_TIMEFRAME)
            market_mode = detect_market_mode(primary_indicators, config.MARKET_MODES)
            
            # Текущая цена
            current_price = primary_indicators.get('price', 0) if primary_indicators else 0
            
            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'aligned_timeframes': aligned_tf,
                'market_mode': market_mode,
                'price': current_price,
                'indicators': primary_indicators,
                'mtf_indicators': mtf_indicators,
            }
            
        except Exception as e:
            self.logger.debug(f"Ошибка analyze_symbol для {symbol}: {e}")
            return None
    
    def scan_all_symbols(self) -> List[Dict]:
        """Сканирование всех монет из watchlist"""
        opportunities = []
        
        for symbol in config.WATCHLIST:
            analysis = self.analyze_symbol(symbol)
            
            if analysis and analysis['signal'] in ["BUY", "SELL"]:
                # Фильтруем по уверенности
                if analysis['confidence'] >= config.SIGNAL_THRESHOLDS['min_confidence']:
                    # Проверяем режим рынка
                    market_mode = analysis['market_mode']
                    if market_mode == "ranging":
                        continue  # Не торговать во флэте
                    
                    opportunities.append(analysis)
        
        # Сортируем по уверенности
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        
        return opportunities
    
    # ═══════════════════════════════════════════════════════════════════
    # ТОРГОВЫЙ ЦИКЛ
    # ═══════════════════════════════════════════════════════════════════
    
    async def trading_cycle(self):
        """Основной торговый цикл"""
        self.cycle_count += 1
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"🔄 ЦИКЛ #{self.cycle_count} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"{'='*70}")
        
        try:
            # Проверка активных позиций
            active_positions = self.get_active_positions()
            self.logger.info(f"📊 Открыто позиций: {len(active_positions)}/{config.MAX_CONCURRENT_POSITIONS}")
            
            for pos in active_positions:
                self.logger.info(f"  • {pos['symbol']} {pos['side']} | PnL: ${pos['unrealized_pnl']:.2f}")
            
            # Проверка на старые позиции (> 24 часа)
            for pos in active_positions:
                if pos.get('created_time'):
                    created_ms = int(pos['created_time'])
                    age_hours = (time.time() * 1000 - created_ms) / (1000 * 3600)
                    if age_hours > config.MAX_POSITION_DURATION_HOURS:
                        self.logger.warning(f"⏰ Позиция {pos['symbol']} открыта {age_hours:.1f}ч - закрываем")
                        self.close_position(pos['symbol'], reason="timeout")
            
            # Если есть свободные слоты - сканируем рынок
            if len(active_positions) < config.MAX_CONCURRENT_POSITIONS:
                self.logger.info("🔍 Сканирование рынка...")
                
                opportunities = self.scan_all_symbols()
                
                self.logger.info(f"✅ Найдено возможностей: {len(opportunities)}")
                
                # Выводим топ-5
                for i, opp in enumerate(opportunities[:5], 1):
                    self.logger.info(
                        f"  {i}. {opp['symbol']}: {opp['signal']} | "
                        f"Confidence: {opp['confidence']:.1f}% | "
                        f"Aligned: {opp['aligned_timeframes']}/4 | "
                        f"Mode: {opp['market_mode']}"
                    )
                
                # Открываем позиции для лучших возможностей
                for opp in opportunities:
                    if len(self.get_active_positions()) >= config.MAX_CONCURRENT_POSITIONS:
                        break
                    
                    if not self.can_open_position(opp['symbol']):
                        continue
                    
                    # Проверка уверенности и выравнивания
                    if opp['confidence'] >= config.SIGNAL_THRESHOLDS['min_confidence']:
                        if opp['aligned_timeframes'] >= config.MIN_TIMEFRAME_ALIGNMENT:
                            self.logger.info(f"🎯 Открываем позицию: {opp['symbol']} {opp['signal']}")
                            self.open_position(
                                opp['symbol'],
                                opp['signal'].title(),  # "BUY" -> "Buy"
                                opp['price'],
                                opp['confidence'],
                                opp['aligned_timeframes']
                            )
                            await asyncio.sleep(2)  # Пауза между открытиями
            
        except Exception as e:
            self.logger.error(f"Ошибка в trading_cycle: {e}", exc_info=True)
    
    async def monitoring_cycle(self):
        """Мониторинг открытых позиций (Trailing TP и т.д.)"""
        if not config.USE_TRAILING_TP:
            return
        
        try:
            positions = self.get_active_positions()
            
            for pos in positions:
                # Расчет профита в процентах
                entry = pos['entry_price']
                mark = pos['mark_price']
                side = pos['side']
                
                if side == "Buy":
                    profit_percent = ((mark - entry) / entry) * 100
                else:  # Sell
                    profit_percent = ((entry - mark) / entry) * 100
                
                # Активация trailing если профит > 5%
                if profit_percent >= config.TRAILING_TP_ACTIVATION_PERCENT:
                    # Рассчитываем новый TP (trailing)
                    new_tp_distance = config.TRAILING_TP_CALLBACK_PERCENT
                    
                    if side == "Buy":
                        new_tp = mark * (1 - new_tp_distance / 100)
                    else:
                        new_tp = mark * (1 + new_tp_distance / 100)
                    
                    # Обновляем TP
                    current_tp = float(pos.get('take_profit', 0)) if pos.get('take_profit') else 0
                    
                    # Обновляем только если новый TP лучше текущего
                    should_update = False
                    if side == "Buy" and new_tp > current_tp:
                        should_update = True
                    elif side == "Sell" and new_tp < current_tp:
                        should_update = True
                    
                    if should_update:
                        self.client.set_trading_stop(
                            category="linear",
                            symbol=pos['symbol'],
                            takeProfit=str(round_price(new_tp)),
                            positionIdx=0,
                        )
                        self.logger.info(f"📈 Trailing TP обновлен для {pos['symbol']}: ${new_tp:.6f}")
        
        except Exception as e:
            self.logger.error(f"Ошибка monitoring_cycle: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # TELEGRAM
    # ═══════════════════════════════════════════════════════════════════
    
    def send_telegram(self, message: str):
        """Отправка сообщения в Telegram"""
        try:
            import requests
            
            url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
            data = {
                "chat_id": config.TELEGRAM_CHAT_ID,
                "text": message,
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code != 200:
                self.logger.warning(f"Ошибка отправки в Telegram: {response.text}")
        
        except Exception as e:
            self.logger.warning(f"Не удалось отправить сообщение в Telegram: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # ГЛАВНЫЙ ЦИКЛ
    # ═══════════════════════════════════════════════════════════════════
    
    async def run(self):
        """Запуск бота"""
        self.logger.info("\n" + "="*70)
        self.logger.info("🚀 DISCO57 BOT ЗАПУЩЕН")
        self.logger.info("="*70)
        
        # Отправляем стартовое уведомление
        balance = self.get_balance()
        if config.TELEGRAM_TOKEN and config.TELEGRAM_CHAT_ID:
            startup_msg = f"""
🚀 DISCO57 BOT ЗАПУЩЕН

💰 Баланс: ${balance:.2f}
📊 Монет: {len(config.WATCHLIST)}
🎯 Макс. позиций: {config.MAX_CONCURRENT_POSITIONS}
⚙️ Размер: ${config.POSITION_SIZE_USD} × {config.LEVERAGE}x

📈 SL: -{config.STOP_LOSS_PERCENT}%
💎 TP: +{config.TAKE_PROFIT_PERCENT}%

Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            self.send_telegram(startup_msg)
        
        # Счетчики для циклов
        analysis_counter = 0
        monitoring_counter = 0
        
        try:
            while self.active:
                # Торговый анализ (каждые 60 секунд)
                if analysis_counter % config.ANALYSIS_INTERVAL_SECONDS == 0:
                    await self.trading_cycle()
                
                # Мониторинг позиций (каждые 10 секунд)
                if monitoring_counter % config.MONITORING_INTERVAL_SECONDS == 0:
                    await self.monitoring_cycle()
                
                analysis_counter += 1
                monitoring_counter += 1
                
                await asyncio.sleep(1)  # Базовый интервал 1 секунда
                
        except KeyboardInterrupt:
            self.logger.info("\n⚠️ Получен сигнал остановки")
        except Exception as e:
            self.logger.error(f"Критическая ошибка в main loop: {e}", exc_info=True)
        finally:
            self.logger.info("🛑 Бот остановлен")
            if config.TELEGRAM_TOKEN and config.TELEGRAM_CHAT_ID:
                self.send_telegram("🛑 DISCO57 BOT ОСТАНОВЛЕН")


# ═══════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    🤖 DISCO57 TRADING BOT                    ║
    ║                     Bybit Futures Bot                        ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    bot = Disco57Bot()
    asyncio.run(bot.run())

