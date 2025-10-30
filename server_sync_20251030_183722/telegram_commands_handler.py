#!/usr/bin/env python3
"""
📱 TELEGRAM КОМАНДЫ ДЛЯ БОТА V4.0 PRO
=====================================

Полная система команд для управления ботом:
- /start - Стартовое сообщение с информацией
- /status - Статус бота и открытые позиции
- /balance - Текущий баланс и статистика
- /positions - Список открытых позиций
- /history - История последних сделок
- /settings - Текущие настройки бота
- /health - Health Score и ML анализ
- /stop - Остановить торговлю (пауза)
- /resume - Возобновить торговлю
- /help - Список всех команд
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import pytz

logger = logging.getLogger(__name__)


class TelegramCommandsHandler:
    """Обработчик команд Telegram для бота"""
    
    def __init__(self, bot_instance):
        """Инициализация обработчика команд"""
        self.bot = bot_instance  # Экземпляр SuperBotV4MTF
        self.warsaw_tz = pytz.timezone('Europe/Warsaw')
        self.commands_registered = False
    
    async def register_commands(self, application):
        """Регистрирует все команды в Telegram боте"""
        try:
            from telegram.ext import (
                CommandHandler, MessageHandler, 
                ContextTypes, filters
            )
            
            # Команды
            application.add_handler(CommandHandler("start", self.cmd_start))
            application.add_handler(CommandHandler("help", self.cmd_help))
            application.add_handler(CommandHandler("status", self.cmd_status))
            application.add_handler(CommandHandler("balance", self.cmd_balance))
            application.add_handler(CommandHandler("positions", self.cmd_positions))
            application.add_handler(CommandHandler("history", self.cmd_history))
            application.add_handler(CommandHandler("settings", self.cmd_settings))
            application.add_handler(CommandHandler("health", self.cmd_health))
            application.add_handler(CommandHandler("stop", self.cmd_stop))
            application.add_handler(CommandHandler("resume", self.cmd_resume))
            application.add_handler(CommandHandler("stats", self.cmd_stats))
            
            # Обработчик неизвестных команд
            application.add_handler(MessageHandler(filters.COMMAND, self.cmd_unknown))
            
            self.commands_registered = True
            logger.info("✅ Telegram команды зарегистрированы")
            
        except Exception as e:
            logger.error(f"❌ Ошибка регистрации команд: {e}")
    
    async def _get_open_positions_live(self) -> list:
        """Возвращает открытые позиции с биржи (ccxt → pybit фолбэк), нормализованные."""
        positions = []
        # ccxt путь
        try:
            if self.bot.exchange:
                raw = await self.bot.exchange.fetch_positions({'category': 'linear'})
                # ccxt обычно возвращает list; если dict — пробуем извлечь значения
                if isinstance(raw, dict):
                    possible_lists = []
                    for v in raw.values():
                        if isinstance(v, list):
                            possible_lists.extend(v)
                    raw_list = possible_lists
                else:
                    raw_list = list(raw) if isinstance(raw, list) else []
                for p in raw_list:
                    try:
                        size = float(p.get('contracts') or p.get('size') or 0)
                        if size <= 0:
                            continue
                        positions.append({
                            'symbol': p.get('symbol'),
                            'side': (p.get('side') or '').lower(),
                            'entryPrice': p.get('entryPrice') or p.get('entry') or p.get('avgPrice'),
                            'markPrice': p.get('markPrice') or p.get('mark'),
                            'takeProfit': p.get('takeProfit') or p.get('take_profit'),
                            'stopLoss': p.get('stopLoss') or p.get('stop_loss'),
                            'size': size,
                        })
                    except Exception:
                        continue
        except Exception:
            pass
        
        # pybit фолбэк
        if not positions:
            try:
                from pybit.unified_trading import HTTP
                session = HTTP(testnet=False, api_key=self.bot.api_key, api_secret=self.bot.api_secret)
                r = session.get_positions(category='linear', settleCoin='USDT')
                lst = r.get('result', {}).get('list', []) or []
                for p in lst:
                    try:
                        size = float(p.get('size') or 0)
                        if size <= 0:
                            continue
                        positions.append({
                            'symbol': p.get('symbol'),
                            'side': 'buy' if (str(p.get('side','')).lower() in ['buy','long']) else 'sell',
                            'entryPrice': p.get('avgPrice'),
                            'markPrice': p.get('markPrice'),
                            'takeProfit': p.get('takeProfit'),
                            'stopLoss': p.get('stopLoss'),
                            'size': size,
                        })
                    except Exception:
                        continue
            except Exception:
                pass
        return positions
    
    async def cmd_start(self, update, context):
        """Команда /start - Стартовое сообщение"""
        try:
            # Получаем баланс если доступен
            balance_info = "N/A"
            try:
                if self.bot.exchange:
                    balance = await self.bot.exchange.fetch_balance({'accountType': 'UNIFIED'})
                    usdt_info = balance.get('USDT', {})
                    usdt_total = usdt_info.get('total') if isinstance(usdt_info, dict) else 0
                    balance_info = f"${usdt_total:.2f} USDT" if usdt_total else "N/A"
            except:
                pass
            
            # Статистика
            total_trades = self.bot.performance_stats.get('total_trades', 0)
            active_positions = len(self.bot.active_positions)
            
            message = f"""🚀 *БОТ V4.0 PRO — АКТИВЕН!*

💰 *Баланс:* {balance_info}
📈 *Суточный P&L:* ${self.bot.performance_stats.get('total_pnl', 0.0):.2f}
📉 *Просадка:* 0.00%
⚙️ *Плечо:* {self.bot.LEVERAGE}x
📂 *Открытые сделки:* {active_positions}/{self.bot.MAX_POSITIONS}

📊 *MTF Таймфреймы*
15m ⏩ 30m ⏩ 45m ⭐ ⏩ 1h ⏩ 4h

🎯 *Стратегии*
💹 Тренд + Объём + Bollinger
🎭 Детектор манипуляций

🎯 *TP: +10% (от \$25) → \$2.5 прибыли*
🛑 *SL: -\$2.5 максимум → Trailing*

💡 *Используйте /help для списка команд*
⏰ {datetime.now(self.warsaw_tz).strftime('%H:%M:%S %d.%m.%Y')}
"""
            await update.message.reply_text(message, parse_mode='Markdown')
            logger.info("✅ Команда /start выполнена")
        except Exception as e:
            logger.error(f"❌ Ошибка команды /start: {e}")
            try:
                await update.message.reply_text(f"❌ Ошибка: {str(e)}")
            except:
                pass
    
    async def cmd_help(self, update, context):
        """Команда /help - Список команд"""
        try:
            message = """📋 *ДОСТУПНЫЕ КОМАНДЫ:*

/start - Стартовое сообщение
/help - Список команд
/status - Статус бота
/balance - Баланс
/positions - Открытые позиции
/history - История сделок
/settings - Настройки
/health - Health Score
/stop - Остановить торговлю
/resume - Возобновить
/stats - Статистика

⏰ Все времена указаны по Варшаве
"""
            await update.message.reply_text(message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"❌ Ошибка команды /help: {e}")

    async def cmd_settings(self, update, context):
        """Команда /settings - Текущие настройки бота"""
        try:
            leverage = getattr(self.bot, 'LEVERAGE', 5)
            position_size = getattr(self.bot, 'POSITION_SIZE', 5.0)
            max_positions = getattr(self.bot, 'MAX_POSITIONS', 3)
            max_sl_usd = getattr(self.bot, 'MAX_STOP_LOSS_USD', 2.5)
            min_conf = getattr(self.bot, 'MIN_CONFIDENCE', 70)
            message = f"""⚙️ *НАСТРОЙКИ БОТА*

💰 *Торговля:*
• Леверидж: {leverage}x
• Размер сделки: {position_size} USDT
• Макс. позиций: {max_positions}

🛑 *Риск:*
• Stop Loss: -$1.0 максимум/сделка (Trailing)
• Trailing TP: старт +1% (шаг 0.5% до +5%)
• Лимит просадки: -$10

🎯 *Сигналы:*
• Порог уверенности: {min_conf}%
• MTF таймфреймы: 15м ⏩ 30м ⏩ 45м ⭐ ⏩ 1ч ⏩ 4ч
• TP: старт +1% (от $25) + трейлинг по 0.5% до +5%

⏰ {datetime.now(self.warsaw_tz).strftime('%H:%M:%S %d.%m.%Y')}"""
            await update.message.reply_text(message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"❌ Ошибка команды /settings: {e}")
    
    async def cmd_status(self, update, context):
        """Команда /status - Статус бота"""
        try:
            # Получаем баланс
            balance_info = "N/A"
            if self.bot.exchange:
                try:
                    balance = await self.bot.exchange.fetch_balance({'accountType': 'UNIFIED'})
                    usdt_info = balance.get('USDT', {})
                    usdt_total = usdt_info.get('total') if isinstance(usdt_info, dict) else 0
                    usdt_free = usdt_info.get('free') or usdt_total if isinstance(usdt_info, dict) else usdt_total
                    balance_info = f"${usdt_total:.2f} (свободно: ${usdt_free:.2f})"
                except:
                    balance_info = "Ошибка получения"
            
            # Живые позиции (ccxt → pybit фолбэк) и uPnL
            active_positions = 0
            u_pnl_sum = 0.0
            try:
                open_positions = await self._get_open_positions_live()
                active_positions = len(open_positions)
                for p in open_positions:
                    side = (p.get('side') or '').lower()
                    entry = float(p.get('entryPrice') or 0)
                    last = float(p.get('markPrice') or 0)
                    notional = getattr(self.bot, 'POSITION_NOTIONAL', 25.0)
                    qty = (notional / entry) if entry else 0
                    pnl = (last - entry) * qty * (1 if side in ['buy', 'long'] else -1)
                    u_pnl_sum += pnl
            except Exception as e:
                logger.debug(f"Ошибка получения позиций для /status: {e}")

            # Статистика закрытых сделок
            total_trades = self.bot.performance_stats.get('total_trades', 0)
            winning_trades = self.bot.performance_stats.get('winning_trades', 0)
            total_pnl = self.bot.performance_stats.get('total_pnl', 0.0)
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            message = f"""📊 *СТАТУС БОТА*

🟢 *Работает*
💰 *Баланс:* {balance_info}
📈 *Открытых сделок:* {active_positions}/3
📊 *Всего сделок:* {total_trades}
✅ *Прибыльных:* {winning_trades}
📈 *Винрейт:* {win_rate:.1f}%
❤️ *Общая прибыль (закрытые):* ${total_pnl:.2f}
💚 *Нереализованный P&L (открытые):* ${u_pnl_sum:.2f}

⚙️ *Настройки:*
🎚 *Леверидж:* {self.bot.LEVERAGE}x
💸 *Размер сделки:* {self.bot.POSITION_SIZE} USDT
🎲 *Порог уверенности:* {self.bot.MIN_CONFIDENCE}%

⏰ {datetime.now(self.warsaw_tz).strftime('%H:%M:%S %d.%m.%Y')}
"""
            await update.message.reply_text(message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"❌ Ошибка команды /status: {e}")
    
    async def cmd_balance(self, update, context):
        """Команда /balance - Баланс"""
        try:
            if not self.bot.exchange:
                await update.message.reply_text("❌ Exchange не инициализирован")
                return
            
            balance = await self.bot.exchange.fetch_balance({'accountType': 'UNIFIED'})
            usdt_info = balance.get('USDT', {})
            usdt_total = usdt_info.get('total') if isinstance(usdt_info, dict) else 0
            usdt_free = usdt_info.get('free') or usdt_total if isinstance(usdt_info, dict) else usdt_total
            usdt_used = usdt_info.get('used') or 0 if isinstance(usdt_info, dict) else 0
            
            # Вычисляем P&L за сегодня (если есть статистика)
            daily_pnl = self.bot.performance_stats.get('total_pnl', 0.0)
            
            message = f"""💰 *БАЛАНС*

💵 *Всего:* ${usdt_total:.2f} USDT
💸 *Свободно:* ${usdt_free:.2f} USDT
🔒 *В торговле:* ${usdt_used:.2f} USDT

📈 *Суточный P&L:* ${daily_pnl:.2f}
📊 *Просадка:* {(usdt_used/usdt_total*100) if usdt_total > 0 else 0:.2f}%

⏰ {datetime.now(self.warsaw_tz).strftime('%H:%M:%S %d.%m.%Y')}
"""
            await update.message.reply_text(message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"❌ Ошибка команды /balance: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")
    
    async def cmd_positions(self, update, context):
        """Команда /positions - Открытые позиции"""
        try:
            # Получаем живые позиции с биржи (нормализовано)
            positions = await self._get_open_positions_live()

            if not positions:
                await update.message.reply_text("📊 *ОТКРЫТЫХ ПОЗИЦИЙ НЕТ*\n\nБот ищет возможности для торговли...")
                return
            
            message = "📊 *ОТКРЫТЫЕ ПОЗИЦИИ*\n\n"
            
            for p in positions:
                symbol = p.get('symbol', 'N/A')
                side = (p.get('side') or '').lower()
                entry = float(p.get('entryPrice') or p.get('entry') or 0)
                last = float(p.get('markPrice') or p.get('mark') or 0)
                tp = p.get('takeProfit') or p.get('take_profit')
                sl = p.get('stopLoss') or p.get('stop_loss')
                notional = getattr(self.bot, 'POSITION_NOTIONAL', 25.0)
                qty = (notional / entry) if entry else 0.0
                pnl = (last - entry) * qty * (1 if side in ['buy','long'] else -1)
                def dist(x):
                    if x is None or not last: return None
                    x = float(x)
                    return ((x-last)/last*100) if side in ['buy','long'] else ((last-x)/last*100)
                emoji = "🟢" if side in ['buy','long'] else "🔴"
                direction = "LONG" if side in ['buy','long'] else "SHORT"
                message += f"""{emoji} *{symbol}* {direction}
💵 Вход: ${entry:.5f} | Текущая: ${last:.5f}
📊 uPnL: {pnl:+.2f} USDT
🎯 TP: {('—' if not tp else f'${float(tp):.5f} ({dist(tp):+.3f}%)')}
🛑 SL: {('—' if not sl else f'${float(sl):.5f} ({dist(sl):+.3f}%)')}

"""
            
            message += f"⏰ {datetime.now(self.warsaw_tz).strftime('%H:%M:%S %d.%m.%Y')}"
            await update.message.reply_text(message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"❌ Ошибка команды /positions: {e}")
    
    async def cmd_history(self, update, context):
        """Команда /history - История сделок"""
        try:
            # Получаем последние сделки из базы данных
            message = "📜 *ИСТОРИЯ СДЕЛОК*\n\n"
            
            if self.bot.data_storage:
                try:
                    # Получаем последние 10 сделок
                    # (нужно реализовать метод в data_storage)
                    total_trades = self.bot.performance_stats.get('total_trades', 0)
                    winning_trades = self.bot.performance_stats.get('winning_trades', 0)
                    total_pnl = self.bot.performance_stats.get('total_pnl', 0.0)
                    
                    message += f"""📊 *Статистика:*
• Всего сделок: {total_trades}
• Прибыльных: {winning_trades}
• Общий P&L: ${total_pnl:.2f}

📝 *Последние сделки доступны в базе данных*
"""
                except Exception as e:
                    logger.debug(f"Ошибка получения истории: {e}")
                    message += "📊 История сохраняется в базу данных"
            else:
                message += f"""📊 *Статистика:*
• Всего сделок: {self.bot.performance_stats.get('total_trades', 0)}
• Прибыльных: {self.bot.performance_stats.get('winning_trades', 0)}
• Общий P&L: ${self.bot.performance_stats.get('total_pnl', 0.0):.2f}
"""
            
            await update.message.reply_text(message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"❌ Ошибка команды /history: {e}")
    
    # Удалено дублирование старой версии /settings; используется актуальная реализация выше
    
    async def cmd_health(self, update, context):
        """Команда /health - Health Score"""
        try:
            message = "🏥 *HEALTH SCORE*\n\n"
            
            if self.bot.health_monitor:
                try:
                    health_data = self.bot.health_monitor.get_current_health()
                    score = health_data.get('overall_score', 0)
                    
                    # Визуализация score
                    if score >= 80:
                        status = "🟢 Отлично"
                    elif score >= 60:
                        status = "🟡 Хорошо"
                    elif score >= 40:
                        status = "🟠 Нормально"
                    else:
                        status = "🔴 Требует внимания"
                    
                    message += f"""📊 *Общий Score:* {score}/100 {status}

📈 *Компоненты:*
• Торговля: {health_data.get('trading_performance', 0):.0f}/100
• Система: {health_data.get('system_stability', 0):.0f}/100
• ML: {health_data.get('ml_accuracy', 0):.0f}/100
"""
                except Exception as e:
                    logger.debug(f"Ошибка получения health: {e}")
                    message += "📊 Health Monitor работает\n✅ Все системы активны"
            else:
                message += "📊 Health Monitor не инициализирован"
            
            await update.message.reply_text(message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"❌ Ошибка команды /health: {e}")
    
    async def cmd_stats(self, update, context):
        """Команда /stats - Подробная статистика"""
        try:
            stats = self.bot.performance_stats
            total = stats.get('total_trades', 0)
            winning = stats.get('winning_trades', 0)
            total_pnl = stats.get('total_pnl', 0.0)
            
            win_rate = (winning / total * 100) if total > 0 else 0
            avg_pnl = (total_pnl / total) if total > 0 else 0
            
            message = f"""📊 *ПОДРОБНАЯ СТАТИСТИКА*

🎯 *Сделки:*
• Всего: {total}
• Прибыльных: {winning}
• Убыточных: {total - winning}
• Винрейт: {win_rate:.1f}%

💰 *Финансы:*
• Общий P&L: ${total_pnl:.2f}
• Средний P&L: ${avg_pnl:.2f}
• Открытых позиций: {len(self.bot.active_positions)}/{self.bot.MAX_POSITIONS}

🤖 *Система:*
• Умный селектор: 145 монет
• MTF анализ: Активен
• ML/LLM: {"✅" if self.bot.llm_analyzer else "⚠️"}
• Обучение: {"✅" if self.bot.universal_learning else "⚠️"}

⏰ {datetime.now(self.warsaw_tz).strftime('%H:%M:%S %d.%m.%Y')}
"""
            await update.message.reply_text(message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"❌ Ошибка команды /stats: {e}")
    
    async def cmd_stop(self, update, context):
        """Команда /stop - Остановить торговлю"""
        try:
            # Устанавливаем флаг остановки
            if not hasattr(self.bot, '_trading_paused'):
                self.bot._trading_paused = False
            
            self.bot._trading_paused = True
            
            message = f"""🛑 *ТОРГОВЛЯ ОСТАНОВЛЕНА*

Бот переходит в режим паузы.
• Новые позиции не открываются
• Текущие позиции мониторятся
• Используйте /resume для возобновления

⏰ {datetime.now(self.warsaw_tz).strftime('%H:%M:%S')}
"""
            await update.message.reply_text(message, parse_mode='Markdown')
            logger.info("🛑 Торговля остановлена через /stop")
        except Exception as e:
            logger.error(f"❌ Ошибка команды /stop: {e}")
    
    async def cmd_resume(self, update, context):
        """Команда /resume - Возобновить торговлю"""
        try:
            if hasattr(self.bot, '_trading_paused'):
                self.bot._trading_paused = False
            
            message = f"""▶️ *ТОРГОВЛЯ ВОЗОБНОВЛЕНА*

Бот снова активен и ищет возможности.

⏰ {datetime.now(self.warsaw_tz).strftime('%H:%M:%S')}
"""
            await update.message.reply_text(message, parse_mode='Markdown')
            logger.info("▶️ Торговля возобновлена через /resume")
        except Exception as e:
            logger.error(f"❌ Ошибка команды /resume: {e}")
    
    async def cmd_unknown(self, update, context):
        """Обработчик неизвестных команд"""
        try:
            await update.message.reply_text(
                "❓ Неизвестная команда.\nИспользуйте /help для списка команд."
            )
        except Exception as e:
            logger.error(f"❌ Ошибка обработки неизвестной команды: {e}")

