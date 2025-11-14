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
        self.logger.info(f"✅ Trailing TP: {'Включен' if config.USE_TRAILING_TP else 'Выключен'}")
    
    # ═══════════════════════════════════════════════════════════════════
    # ПОЛУЧЕНИЕ ДАННЫХ
    # ═══════════════════════════════════════════════════════════════════
    
    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Получение свечей с биржи"""
        try:
            # Конвертируем интервал в формат Bybit API
            # 30m -> 30, 1h -> 60, 4h -> 240, 1d -> D
            interval_map = {
                '30m': '30',
                '1h': '60',
                '4h': '240',
                '1d': 'D'
            }
            api_interval = interval_map.get(interval, interval)
            
            self.logger.debug(f"Запрос данных: {symbol} {interval} (API: {api_interval})")
            
            response = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=api_interval,
                limit=limit
            )
            
            if response['retCode'] != 0:
                error_msg = response.get('retMsg', 'Unknown error')
                # Если символ не существует, не логируем как ошибку
                if 'Symbol Is Invalid' in error_msg or '10001' in str(response.get('retCode', '')):
                    self.logger.debug(f"⚠️ {symbol} {interval}: символ не существует на Bybit фьючерсах")
                else:
                    self.logger.warning(f"Ошибка получения данных для {symbol} {interval}: {error_msg}")
                return None
            
            klines = response['result']['list']
            
            if not klines:
                self.logger.warning(f"Пустой список свечей для {symbol} {interval}")
                return None
            
            # Конвертация в DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            
            # Преобразование типов
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Сортировка по времени (от старых к новым)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.debug(f"✅ {symbol} {interval}: получено {len(df)} свечей")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка get_klines для {symbol} {interval}: {e}", exc_info=True)
            return None
    
    def get_multitimeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Получение данных по всем таймфреймам"""
        data = {}
        
        for tf_name, tf_value in config.TIMEFRAMES.items():
            df = self.get_klines(symbol, tf_value)
            if df is not None:
                if len(df) >= 100:  # Уменьшил требование до 100 свечей
                    data[tf_name] = df
                    self.logger.debug(f"✅ {symbol} {tf_name}: получено {len(df)} свечей")
                else:
                    self.logger.warning(f"⚠️ {symbol} {tf_name}: недостаточно свечей ({len(df)}/100)")
            else:
                self.logger.warning(f"⚠️ {symbol} {tf_name}: не удалось получить данные")
            # Задержка между запросами разных таймфреймов (0.5 сек)
            time.sleep(0.5)
        
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
            
            # Безопасный парсер чисел (строки/пустые значения)
            def _to_float(value) -> float:
                try:
                    if value is None:
                        return 0.0
                    s = str(value).strip()
                    if s == "" or s.lower() == "null":
                        return 0.0
                    return float(s)
                except Exception:
                    return 0.0
            
            result = response.get("result") or {}
            balance_list = result.get("list") or []
            if balance_list:
                account = balance_list[0] or {}
                
                # 1) Сначала пробуем availableBalance (что реально доступно)
                available_balance = _to_float(account.get("availableBalance"))
                if available_balance > 0:
                    self.logger.debug(f"Доступный баланс (availableBalance): ${available_balance:.2f}")
                    return available_balance
                
                # 2) Затем totalEquity (общая эквити)
                total_equity = _to_float(account.get("totalEquity"))
                if total_equity > 0:
                    self.logger.debug(f"Общая эквити (totalEquity): ${total_equity:.2f}")
                    return total_equity
                
                # 3) Альтернатива: по монете USDT в списке coin
                coins = account.get("coin") or []
                for coin in coins:
                    if coin.get("coin") == "USDT":
                        wallet_balance = _to_float(coin.get("walletBalance"))
                        available_to_withdraw = _to_float(coin.get("availableToWithdraw"))
                        candidate = max(wallet_balance, available_to_withdraw)
                        if candidate > 0:
                            self.logger.debug(f"USDT баланс (wallet/availableToWithdraw): ${candidate:.2f}")
                            return candidate
            
            self.logger.warning("Не удалось определить баланс (0). Проверьте права API/аккаунт.")
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
            
            # Округление количества до правильного шага для символа
            # Получаем информацию о символе для правильного округления
            qty_step = 0.001  # Дефолтное значение
            try:
                symbol_info = self.client.get_instruments_info(category="linear", symbol=symbol)
                if symbol_info['retCode'] == 0 and symbol_info.get('result'):
                    result_list = symbol_info['result'].get('list', [])
                    for item in result_list:
                        if item.get('symbol') == symbol:
                            # lotSizeFilter может быть словарем или списком
                            lot_size_filter = item.get('lotSizeFilter', {})
                            
                            # Если это словарь, берем напрямую
                            if isinstance(lot_size_filter, dict):
                                qty_step_str = lot_size_filter.get('qtyStep', '0.001')
                                qty_step = float(qty_step_str)
                            # Если это список, берем первый элемент
                            elif isinstance(lot_size_filter, list) and len(lot_size_filter) > 0:
                                first_filter = lot_size_filter[0]
                                if isinstance(first_filter, dict):
                                    qty_step_str = first_filter.get('qtyStep', '0.001')
                                    qty_step = float(qty_step_str)
                            
                            break
                    
                    # Округляем количество
                    qty = round_quantity(qty, qty_step)
                    self.logger.debug(f"Округление {symbol}: шаг={qty_step}, qty={qty}")
                else:
                    qty = round_quantity(qty, qty_step)
                    self.logger.debug(f"Используем дефолтный шаг {qty_step} для {symbol}")
            except Exception as e:
                self.logger.warning(f"Не удалось получить info для {symbol}, используем дефолт: {e}")
                qty = round_quantity(qty, qty_step)
            
            # Убеждаемся что qty не имеет лишних знаков
            # Преобразуем в строку и обратно для проверки
            qty_str = f"{qty:.10f}".rstrip('0').rstrip('.')
            qty = float(qty_str) if qty_str else qty
            
            # Проверка минимального количества
            if qty < qty_step:
                self.logger.warning(f"Количество {qty} меньше минимального шага {qty_step} для {symbol}")
                return False
            
            # Форматируем qty как строку без лишних нулей
            qty_formatted = f"{qty:.10f}".rstrip('0').rstrip('.')
            if '.' not in qty_formatted:
                qty_formatted = str(int(qty))
            
            self.logger.info(f"🔄 Открытие позиции {symbol} {side} | Qty: {qty_formatted} | Price: ${price:.6f}")
            
            # Размещение ордера
            response = self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=qty_formatted,
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
    
    def analyze_symbol(self, symbol: str, detailed_log: bool = False) -> Optional[Dict]:
        """
        Анализ одного символа по всем таймфреймам
        
        Args:
            symbol: Символ для анализа
            detailed_log: Включить детальное логирование
        
        Returns:
            None если нет данных или dict с результатами анализа
        """
        try:
            if detailed_log:
                self.logger.info(f"\n{'='*70}")
                self.logger.info(f"📊 ДЕТАЛЬНЫЙ АНАЛИЗ: {symbol}")
                self.logger.info(f"{'='*70}")
            
            # Получаем данные по всем таймфреймам
            if detailed_log:
                self.logger.info(f"1️⃣ Получение данных по таймфреймам...")
            mtf_data = self.get_multitimeframe_data(symbol)
            
            if len(mtf_data) < 2:  # Минимум 2 таймфрейма
                if detailed_log:
                    self.logger.warning(f"❌ {symbol}: недостаточно таймфреймов ({len(mtf_data)}/4)")
                return None
            
            if detailed_log:
                self.logger.info(f"✅ Получено таймфреймов: {len(mtf_data)}/4")
                for tf_name in mtf_data.keys():
                    self.logger.info(f"   • {tf_name}: {len(mtf_data[tf_name])} свечей")
            
            # Расчет индикаторов для каждого таймфрейма
            if detailed_log:
                self.logger.info(f"\n2️⃣ Расчет технических индикаторов...")
            mtf_indicators = {}
            for tf, df in mtf_data.items():
                try:
                    if detailed_log:
                        self.logger.info(f"   📈 {tf}: расчет индикаторов...")
                    indicators = self.indicators_calculator.calculate_all(df)
                    if indicators:
                        mtf_indicators[tf] = indicators
                        if detailed_log:
                            # Показываем ключевые индикаторы
                            try:
                                rsi = indicators.get('rsi', 0)
                                macd = indicators.get('macd', 0)
                                adx = indicators.get('adx', 0)
                                ema_20 = indicators.get('ema_20', 0)
                                ema_50 = indicators.get('ema_50', 0)
                                price = indicators.get('price', 0)
                                
                                # Безопасное форматирование
                                rsi_str = f"{float(rsi):.2f}" if isinstance(rsi, (int, float)) else str(rsi)
                                macd_str = f"{float(macd):.4f}" if isinstance(macd, (int, float)) else str(macd)
                                adx_str = f"{float(adx):.2f}" if isinstance(adx, (int, float)) else str(adx)
                                ema20_str = f"{float(ema_20):.2f}" if isinstance(ema_20, (int, float)) else str(ema_20)
                                ema50_str = f"{float(ema_50):.2f}" if isinstance(ema_50, (int, float)) else str(ema_50)
                                price_str = f"{float(price):.2f}" if isinstance(price, (int, float)) else str(price)
                                
                                self.logger.info(f"      RSI: {rsi_str} | MACD: {macd_str} | ADX: {adx_str}")
                                self.logger.info(f"      EMA20: {ema20_str} | EMA50: {ema50_str} | Цена: {price_str}")
                            except Exception as e:
                                self.logger.debug(f"      Ошибка форматирования индикаторов: {e}")
                    else:
                        if detailed_log:
                            self.logger.warning(f"      ❌ Индикаторы не рассчитаны")
                except Exception as e:
                    if detailed_log:
                        self.logger.error(f"      ❌ Ошибка: {e}")
            
            if not mtf_indicators:
                if detailed_log:
                    self.logger.warning(f"❌ {symbol}: нет индикаторов для анализа")
                return None
            
            # Генерация сигнала
            if detailed_log:
                self.logger.info(f"\n3️⃣ Генерация торгового сигнала...")
            try:
                signal, confidence, aligned_tf = self.indicators_calculator.generate_signal(mtf_indicators)
                if detailed_log:
                    self.logger.info(f"   Сигнал: {signal}")
                    self.logger.info(f"   Уверенность: {confidence:.1f}%")
                    self.logger.info(f"   Выровнено таймфреймов: {aligned_tf}/4")
            except Exception as e:
                if detailed_log:
                    self.logger.error(f"   ❌ Ошибка генерации сигнала: {e}")
                return None
            
            # Определение режима рынка
            primary_indicators = mtf_indicators.get(config.PRIMARY_TIMEFRAME)
            if not primary_indicators:
                if detailed_log:
                    self.logger.warning(f"❌ {symbol}: нет primary индикаторов")
                return None
                
            market_mode = detect_market_mode(primary_indicators, config.MARKET_MODES)
            if detailed_log:
                self.logger.info(f"\n4️⃣ Режим рынка: {market_mode}")
            
            # Текущая цена
            current_price = primary_indicators.get('price', 0) if primary_indicators else 0
            
            if detailed_log:
                self.logger.info(f"\n✅ Анализ завершен: {symbol} | {signal} | {confidence:.1f}% | {market_mode}")
                self.logger.info(f"{'='*70}\n")
            
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
            if detailed_log:
                self.logger.error(f"❌ Ошибка analyze_symbol для {symbol}: {e}", exc_info=True)
            return None
    
    def scan_all_symbols(self) -> List[Dict]:
        """Сканирование всех монет из watchlist"""
        opportunities = []
        all_analyses = []  # Для детального логирования
        
        for i, symbol in enumerate(config.WATCHLIST):
            analysis = self.analyze_symbol(symbol)
            
            if analysis:
                all_analyses.append(analysis)
                
                if analysis['signal'] in ["BUY", "SELL"]:
                    # Фильтруем по уверенности
                    if analysis['confidence'] >= config.SIGNAL_THRESHOLDS['min_confidence']:
                        # Проверяем режим рынка
                        market_mode = analysis['market_mode']
                        if market_mode == "ranging":
                            continue  # Не торговать во флэте
                        
                        opportunities.append(analysis)
            
            # Задержка между монетами для соблюдения rate limit (2 сек)
            if i < len(config.WATCHLIST) - 1:
                time.sleep(2)
        
        # Сортируем все анализы по уверенности для логирования
        all_analyses.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ ТОП-10
        if all_analyses:
            self.logger.info("\n" + "="*70)
            self.logger.info("📊 ТОП-10 МОНЕТ ПО УВЕРЕННОСТИ:")
            self.logger.info("="*70)
            for i, analysis in enumerate(all_analyses[:10], 1):
                signal = analysis.get('signal', 'N/A')
                confidence = analysis.get('confidence', 0)
                aligned = analysis.get('aligned_timeframes', 0)
                mode = analysis.get('market_mode', 'N/A')
                symbol = analysis.get('symbol', 'N/A')
                
                # Причина отклонения
                reason = ""
                if signal == "HOLD":
                    reason = "❌ HOLD сигнал"
                elif confidence < config.SIGNAL_THRESHOLDS['min_confidence']:
                    reason = f"❌ Низкая уверенность (<{config.SIGNAL_THRESHOLDS['min_confidence']*100:.0f}%)"
                elif aligned < config.MIN_TIMEFRAME_ALIGNMENT:
                    reason = f"❌ Мало таймфреймов (<{config.MIN_TIMEFRAME_ALIGNMENT})"
                elif mode == "ranging":
                    reason = "❌ Флэт"
                else:
                    reason = "✅ ПОДХОДИТ"
                
                self.logger.info(
                    f"  {i}. {symbol}: {signal} | "
                    f"Уверенность: {confidence:.1f}% | "
                    f"Выровнено: {aligned}/4 | "
                    f"Режим: {mode} | "
                    f"{reason}"
                )
            self.logger.info("="*70)
        else:
            self.logger.warning("⚠️ Не удалось проанализировать ни одной монеты")
        
        # Сортируем возможности по уверенности
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
                    confidence_percent = opp['confidence'] / 100.0  # Конвертируем проценты в долю
                    if confidence_percent >= config.SIGNAL_THRESHOLDS['min_confidence']:
                        if opp['aligned_timeframes'] >= config.MIN_TIMEFRAME_ALIGNMENT:
                            self.logger.info(f"🎯 Открываем позицию: {opp['symbol']} {opp['signal']} | Уверенность: {opp['confidence']:.1f}%")
                            result = self.open_position(
                                opp['symbol'],
                                opp['signal'].title(),  # "BUY" -> "Buy"
                                opp['price'],
                                opp['confidence'],
                                opp['aligned_timeframes']
                            )
                            if result:
                                self.logger.info(f"✅ Позиция {opp['symbol']} успешно открыта")
                            else:
                                self.logger.warning(f"❌ Не удалось открыть позицию {opp['symbol']}")
                            await asyncio.sleep(2)  # Пауза между открытиями
                        else:
                            self.logger.debug(f"⏭️ {opp['symbol']}: недостаточно выровненных таймфреймов ({opp['aligned_timeframes']}/{config.MIN_TIMEFRAME_ALIGNMENT})")
                    else:
                        self.logger.debug(f"⏭️ {opp['symbol']}: низкая уверенность ({opp['confidence']:.1f}% < {config.SIGNAL_THRESHOLDS['min_confidence']*100:.0f}%)")
            
        except Exception as e:
            self.logger.error(f"Ошибка в trading_cycle: {e}", exc_info=True)
    
    def generate_diagnostic_report(self) -> Dict:
        """
        Генеральная диагностика бота и рынка
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("🔍 ГЕНЕРАЛЬНАЯ ДИАГНОСТИКА БОТА И РЫНКА")
        self.logger.info("="*70)
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'bot_status': {},
            'market_analysis': {},
            'indicators_status': {},
            'positions': {},
            'errors': []
        }
        
        try:
            # 1. СТАТУС БОТА
            self.logger.info("\n1️⃣ СТАТУС БОТА:")
            balance = self.get_balance()
            active_positions = self.get_active_positions()
            
            report['bot_status'] = {
                'balance': balance,
                'active_positions': len(active_positions),
                'max_positions': config.MAX_CONCURRENT_POSITIONS,
                'watchlist_size': len(config.WATCHLIST),
                'cycle_count': self.cycle_count,
                'active': self.active
            }
            
            self.logger.info(f"   💰 Баланс: ${balance:.2f}")
            self.logger.info(f"   📊 Позиций: {len(active_positions)}/{config.MAX_CONCURRENT_POSITIONS}")
            self.logger.info(f"   📋 Watchlist: {len(config.WATCHLIST)} монет")
            self.logger.info(f"   🔄 Циклов выполнено: {self.cycle_count}")
            self.logger.info(f"   ✅ Статус: {'Активен' if self.active else 'Остановлен'}")
            
            # 2. АНАЛИЗ РЫНКА (на примере BTC)
            self.logger.info("\n2️⃣ АНАЛИЗ РЫНКА (BTCUSDT как индикатор):")
            btc_analysis = self.analyze_symbol("BTCUSDT", detailed_log=True)
            
            if btc_analysis:
                report['market_analysis'] = {
                    'btc_signal': btc_analysis['signal'],
                    'btc_confidence': btc_analysis['confidence'],
                    'btc_market_mode': btc_analysis['market_mode'],
                    'btc_price': btc_analysis['price'],
                    'btc_aligned_tf': btc_analysis['aligned_timeframes']
                }
                
                # Общее направление рынка
                market_direction = "НЕОПРЕДЕЛЕННО"
                if btc_analysis['signal'] == "BUY":
                    market_direction = "🟢 БЫЧИЙ (восходящий тренд)"
                elif btc_analysis['signal'] == "SELL":
                    market_direction = "🔴 МЕДВЕЖИЙ (нисходящий тренд)"
                else:
                    market_direction = "⚪ БОКОВИК (флэт)"
                
                self.logger.info(f"   📈 Направление рынка: {market_direction}")
                self.logger.info(f"   💹 Режим: {btc_analysis['market_mode']}")
                self.logger.info(f"   💰 Цена BTC: ${btc_analysis['price']:.2f}")
            else:
                report['errors'].append("Не удалось проанализировать BTCUSDT")
                self.logger.warning("   ❌ Не удалось получить анализ BTC")
            
            # 3. ПРОВЕРКА ИНДИКАТОРОВ
            self.logger.info("\n3️⃣ ПРОВЕРКА РАБОТЫ ИНДИКАТОРОВ:")
            test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"][:2]  # Тестируем 2 монеты
            indicators_ok = 0
            indicators_failed = 0
            
            for symbol in test_symbols:
                try:
                    mtf_data = self.get_multitimeframe_data(symbol)
                    if len(mtf_data) >= 2:
                        for tf, df in list(mtf_data.items())[:1]:  # Проверяем только первый таймфрейм
                            indicators = self.indicators_calculator.calculate_all(df)
                            if indicators and len(indicators) > 10:
                                indicators_ok += 1
                                self.logger.info(f"   ✅ {symbol} {tf}: {len(indicators)} индикаторов рассчитано")
                            else:
                                indicators_failed += 1
                                self.logger.warning(f"   ❌ {symbol} {tf}: индикаторы не рассчитаны")
                except Exception as e:
                    indicators_failed += 1
                    self.logger.error(f"   ❌ {symbol}: ошибка {e}")
            
            report['indicators_status'] = {
                'ok': indicators_ok,
                'failed': indicators_failed,
                'success_rate': (indicators_ok / (indicators_ok + indicators_failed) * 100) if (indicators_ok + indicators_failed) > 0 else 0
            }
            
            # 4. АКТИВНЫЕ ПОЗИЦИИ
            self.logger.info("\n4️⃣ АКТИВНЫЕ ПОЗИЦИИ:")
            if active_positions:
                total_pnl = 0
                for pos in active_positions:
                    pnl = pos['unrealized_pnl']
                    total_pnl += pnl
                    self.logger.info(f"   • {pos['symbol']} {pos['side']}: PnL ${pnl:.2f}")
                
                report['positions'] = {
                    'count': len(active_positions),
                    'total_pnl': total_pnl,
                    'positions': active_positions
                }
                self.logger.info(f"   💵 Общий PnL: ${total_pnl:.2f}")
            else:
                self.logger.info("   📭 Нет открытых позиций")
                report['positions'] = {'count': 0, 'total_pnl': 0}
            
            # 5. СТАТИСТИКА СИГНАЛОВ
            self.logger.info("\n5️⃣ СТАТИСТИКА СИГНАЛОВ (последний цикл):")
            self.logger.info("   ⏳ Сканирование топ-10 монет для статистики...")
            
            top_symbols = config.WATCHLIST[:10]
            signals_stats = {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'total': 0}
            confidence_sum = 0
            
            for symbol in top_symbols:
                analysis = self.analyze_symbol(symbol)
                if analysis:
                    signals_stats[analysis['signal']] += 1
                    signals_stats['total'] += 1
                    confidence_sum += analysis['confidence']
            
            avg_confidence = confidence_sum / signals_stats['total'] if signals_stats['total'] > 0 else 0
            
            report['signals_stats'] = {
                'buy': signals_stats['BUY'],
                'sell': signals_stats['SELL'],
                'hold': signals_stats['HOLD'],
                'total': signals_stats['total'],
                'avg_confidence': avg_confidence
            }
            
            self.logger.info(f"   📊 BUY: {signals_stats['BUY']} | SELL: {signals_stats['SELL']} | HOLD: {signals_stats['HOLD']}")
            self.logger.info(f"   📈 Средняя уверенность: {avg_confidence:.1f}%")
            
            # 6. ИТОГОВЫЙ ВЕРДИКТ
            self.logger.info("\n" + "="*70)
            self.logger.info("✅ ДИАГНОСТИКА ЗАВЕРШЕНА")
            self.logger.info("="*70)
            
            # Проверка здоровья системы
            health_issues = []
            if balance <= 0:
                health_issues.append("⚠️ Баланс = 0")
            if indicators_failed > indicators_ok:
                health_issues.append("⚠️ Много ошибок индикаторов")
            if not btc_analysis:
                health_issues.append("⚠️ Не удается анализировать BTC")
            
            if health_issues:
                self.logger.warning("⚠️ ОБНАРУЖЕНЫ ПРОБЛЕМЫ:")
                for issue in health_issues:
                    self.logger.warning(f"   {issue}")
            else:
                self.logger.info("✅ Все системы работают нормально")
            
            report['health_status'] = 'OK' if not health_issues else 'ISSUES'
            report['health_issues'] = health_issues
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка генерации диагностики: {e}", exc_info=True)
            report['errors'].append(str(e))
        
        return report
    
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
                    
                    # Автоматическая диагностика каждые 10 циклов
                    if self.cycle_count % 10 == 0:
                        self.logger.info("\n🔍 АВТОМАТИЧЕСКАЯ ДИАГНОСТИКА (каждые 10 циклов)")
                        self.generate_diagnostic_report()
                
                # Мониторинг позиций (каждые 10 секунд)
                if monitoring_counter % config.MONITORING_INTERVAL_SECONDS == 0:
                    await self.monitoring_cycle()
                
                analysis_counter += 1
                monitoring_counter += 1
                
                await asyncio.sleep(1)  # Базовый интервал 1 секунда
                
        except KeyboardInterrupt:
            self.logger.info("\n⚠️ Получен сигнал остановки")
            # Генерируем финальный отчет перед остановкой
            self.generate_diagnostic_report()
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

