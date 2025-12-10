#!/usr/bin/env python3
"""
Telegram Notifier для GoldTrigger Bot
Отправка уведомлений о сделках
"""

import logging
import os
import asyncio
from datetime import datetime
import pytz
from typing import Optional
import aiohttp
from dotenv import load_dotenv

load_dotenv()

# Часовой пояс Варшавы
WARSAW_TZ = pytz.timezone('Europe/Warsaw')

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Отправка уведомлений в Telegram"""
    
    def __init__(self):
        self.token = os.getenv('TELEGRAM_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.token and self.chat_id)
        
        if not self.enabled:
            logger.warning("Telegram уведомления отключены (не указаны TOKEN или CHAT_ID)")
        else:
            logger.info("Telegram уведомления включены")
        
        self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        self.retry_count = 3
    
    async def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """
        Отправить сообщение в Telegram
        
        Args:
            message: Текст сообщения
            parse_mode: Режим парсинга ('HTML' или 'Markdown')
        
        Returns:
            True если успешно отправлено
        """
        if not self.enabled:
            return False
        
        for attempt in range(self.retry_count):
            try:
                async with aiohttp.ClientSession() as session:
                    payload = {
                        'chat_id': self.chat_id,
                        'text': message,
                        'parse_mode': parse_mode
                    }
                    
                    async with session.post(self.api_url, json=payload) as response:
                        if response.status == 200:
                            logger.debug("Telegram сообщение отправлено")
                            return True
                        else:
                            error_text = await response.text()
                            logger.warning(f"Ошибка Telegram API: {response.status} - {error_text}")
                            
            except Exception as e:
                if attempt < self.retry_count - 1:
                    logger.warning(f"Ошибка отправки в Telegram (попытка {attempt + 1}): {e}")
                    await asyncio.sleep(1)
                else:
                    logger.error(f"Не удалось отправить в Telegram после {self.retry_count} попыток: {e}")
        
        return False
    
    async def send_startup_message(self):
        """Отправить сообщение о запуске бота"""
        message = (
            "⚡️🌟 <b>GoldTrigger Swing Trader</b>\n\n"
            "🤖 <b>Система</b>\n"
            "• 145 пар: 1h скан + 30m подтверждение\n"
            "• EMA50/200 тренд + EMA9/21 cross + RSI + объём\n"
            "• Disco57-Swing PPO фильтр, TradeHistoryDB, Telegram-команды\n\n"
            "📈 <b>Стратегия</b>\n"
            "• Риск 0.5–0.8% баланса, ATR-основанный SL (~1–2%)\n"
            "• Leverage 20x, цель R:R ≥ 1:6, автотрейлинг после 3R\n"
            "• Макс. позиций: 3, кулдаун 90 мин, retention 72ч\n\n"
            "🎯 Swing режим активен — ждём сильные трендовые сигналы!"
        )
        await self.send_message(message)
    
    async def send_trade_opened(self, symbol: str, side: str, entry_price: float,
                               sl_usd: float, tp_usd: float, sl_price: float = 0,
                               signal_strength: int = 0, disco_confidence: float = 0):
        """Уведомление об открытии сделки - ТРЕНДОВАЯ СТРАТЕГИЯ"""
        side_emoji = "🟢" if side == "long" else "🔴"
        side_text = "LONG" if side == "long" else "SHORT"
        warsaw_time = datetime.now(WARSAW_TZ).strftime('%H:%M:%S')
        short_symbol = symbol.replace('/USDT:USDT', '').replace('USDT', '')
        if sl_price == 0:
            sl_price = entry_price * (0.99 if side == "long" else 1.01)
        sl_dist_pct = abs(entry_price - sl_price) / entry_price * 100
        rr_target = tp_usd / sl_usd if sl_usd else 6.0
        
        message = (
            f"{side_emoji} <b>SWING ENTRY</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🪙 <b>{short_symbol}</b> {side_text} 20x\n"
            f"⏰ {warsaw_time} (Варшава)\n\n"
            f"📍 <b>Вход:</b> ${entry_price:.6f}\n"
            f"🛡 <b>SL:</b> ${sl_price:.6f} (≈{sl_dist_pct:.2f}%)\n"
            f"💵 <b>Риск:</b> −${sl_usd:.2f}\n"
            f"🎯 <b>Цель:</b> +${tp_usd:.2f} (R:R ≈ {rr_target:.1f})\n\n"
            f"📚 Swing-условия: 1h EMA50/200 тренд, EMA9×EMA21, RSI/объём + pullback 30m\n"
        )
        
        if signal_strength > 0 or disco_confidence > 0:
            message += f"🤖 Сигнал: {signal_strength}/5 • Disco57: {disco_confidence:.0f}%\n\n"
        
        message += "💎 Поддерживаем позицию пока тренд и фильтры согласованы."
        
        await self.send_message(message)
    
    async def send_trailing_activated(self, symbol: str, side: str, profit_usd: float,
                                      current_price: float = 0, entry_price: float = 0):
        """Уведомление об активации трейлинга"""
        warsaw_time = datetime.now(WARSAW_TZ).strftime('%H:%M:%S')
        short_symbol = symbol.replace('/USDT:USDT', '').replace('USDT', '')
        
        # Расчет ROI
        roi_pct = (profit_usd / 25) * 100  # $25 exposure
        
        message = (
            f"🔄 <b>TRAILING АКТИВИРОВАН</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🪙 <b>{short_symbol}</b> {side.upper()} 25x\n"
            f"⏰ {warsaw_time} (Варшава)\n\n"
            f"💰 <b>Прибыль:</b> +${profit_usd:.2f} (+{roi_pct:.1f}%)\n"
        )
        
        if entry_price > 0 and current_price > 0:
            message += f"📍 Вход: ${entry_price:.6f}\n"
            message += f"📈 Сейчас: ${current_price:.6f}\n\n"
        
        message += "🎯 SL удерживает +30% ROI и двигается шагом ≈+10% ROI (0.40%)"
        
        await self.send_message(message)
    
    async def send_trade_closed(self, symbol: str, side: str, entry_price: float,
                               exit_price: float, pnl_usd: float, reason: str,
                               daily_pnl: float, duration_min: int = 0):
        """Уведомление о закрытии сделки"""
        warsaw_time = datetime.now(WARSAW_TZ).strftime('%H:%M:%S')
        short_symbol = symbol.replace('/USDT:USDT', '').replace('USDT', '')
        
        status_emoji = "✅" if pnl_usd > 0 else "❌"
        pnl_sign = "+" if pnl_usd > 0 else ""
        
        # Расчет ROI
        roi_pct = (pnl_usd / 25) * 100
        
        # Расчет изменения цены
        if side == "long":
            price_change_pct = (exit_price - entry_price) / entry_price * 100
        else:
            price_change_pct = (entry_price - exit_price) / entry_price * 100
        
        message = (
            f"{status_emoji} <b>CLOSED ({reason})</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🪙 <b>{short_symbol}</b> {side.upper()} 25x\n"
            f"⏰ {warsaw_time} (Варшава)\n\n"
            f"📍 <b>Вход:</b> ${entry_price:.6f}\n"
            f"📍 <b>Выход:</b> ${exit_price:.6f}\n"
            f"📊 <b>Изменение:</b> {price_change_pct:+.2f}%\n\n"
            f"💰 <b>PnL:</b> {pnl_sign}${pnl_usd:.2f} ({roi_pct:+.1f}% ROI)\n"
        )
        
        if duration_min > 0:
            message += f"⏱ <b>Время:</b> {duration_min} мин\n"
        
        message += f"\n📊 <b>Дневной PnL:</b> {'+' if daily_pnl > 0 else ''}${daily_pnl:.2f}"
        
        await self.send_message(message)
    
    async def send_daily_summary(self, trades_count: int, pnl: float, 
                                win_rate: float):
        """Дневная сводка"""
        pnl_emoji = "📈" if pnl > 0 else "📉"
        
        message = (
            f"{pnl_emoji} <b>Дневная сводка</b>\n\n"
            f"Сделок: {trades_count}\n"
            f"PnL: {'+' if pnl > 0 else ''}${pnl:.2f}\n"
            f"Win Rate: {win_rate:.1f}%"
        )
        await self.send_message(message)
    
    async def send_error(self, error_message: str):
        """Уведомление об ошибке"""
        message = f"⚠️ <b>ОШИБКА</b>\n\n{error_message}"
        await self.send_message(message)
    
    async def send_daily_limit_reached(self, loss: float):
        """Уведомление о достижении дневного лимита"""
        message = (
            f"🛑 <b>ДНЕВНОЙ ЛИМИТ УБЫТКА</b>\n\n"
            f"Убыток: -${abs(loss):.2f}\n"
            f"Лимит: -$5.00\n\n"
            f"Торговля приостановлена на 24 часа"
        )
        await self.send_message(message)


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

async def test_telegram():
    """Тест Telegram уведомлений"""
    logging.basicConfig(level=logging.INFO)
    
    notifier = TelegramNotifier()
    
    if not notifier.enabled:
        print("Telegram не настроен. Установите TELEGRAM_TOKEN и TELEGRAM_CHAT_ID в .env")
        return
    
    print("Отправка тестовых сообщений...")
    
    # Тест запуска
    await notifier.send_startup_message()
    await asyncio.sleep(1)
    
    # Тест открытия
    await notifier.send_trade_opened(
        symbol='BTC/USDT:USDT',
        side='long',
        entry_price=50000.0,
        sl_usd=0.15,
        tp_usd=0.50
    )
    await asyncio.sleep(1)
    
    # Тест трейлинга
    await notifier.send_trailing_activated(
        symbol='BTC/USDT:USDT',
        side='long',
        profit_usd=0.35
    )
    await asyncio.sleep(1)
    
    # Тест закрытия
    await notifier.send_trade_closed(
        symbol='BTC/USDT:USDT',
        side='long',
        entry_price=50000.0,
        exit_price=50250.0,
        pnl_usd=0.52,
        reason='TP',
        daily_pnl=0.52
    )
    
    print("Тестовые сообщения отправлены!")


if __name__ == '__main__':
    asyncio.run(test_telegram())
