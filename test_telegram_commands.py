#!/usr/bin/env python3
"""
Тест Telegram команд
"""
import asyncio
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update
import os
from dotenv import load_dotenv

load_dotenv()

# Глобальный флаг для остановки
bot_running = True


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start"""
    await update.message.reply_text(
        "🤖 *БОТ V2.0* - Управление\n\n"
        "*📊 Информация:*\n"
        "/status - краткий статус\n"
        "/positions - открытые позиции\n"
        "/history - история сделок\n\n"
        "*⚙️ Управление:*\n"
        "/pause - пауза торговли\n"
        "/resume - возобновить торговлю\n"
        "/close_all - закрыть все позиции\n"
        "/stop - остановить бота",
        parse_mode="Markdown"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /status"""
    await update.message.reply_text("📊 *СТАТУС:* ✅ Работает\n\n🤖 Бот активен", parse_mode="Markdown")


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /stop"""
    global bot_running
    await update.message.reply_text("🛑 Останавливаю бота...")
    bot_running = False
    print("✅ Команда /stop получена! bot_running = False")


async def main():
    """Тест"""
    print("="*70)
    print("🧪 ТЕСТ TELEGRAM КОМАНД")
    print("="*70)
    
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN не найден!")
        return
    
    print(f"✅ Token: {token[:10]}...")
    
    # Создаём приложение
    app = Application.builder().token(token).build()
    
    # Добавляем обработчики
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("stop", cmd_stop))
    
    print("\n📱 Инициализация Telegram бота...")
    await app.initialize()
    await app.start()
    
    print("📱 Запуск polling...")
    await app.updater.start_polling(drop_pending_updates=True)
    
    print("\n" + "="*70)
    print("✅ БОТА ЗАПУЩЕН! Отправьте команды в Telegram:")
    print("   /start - список команд")
    print("   /status - статус")
    print("   /stop - остановка")
    print("="*70)
    print("\n⏳ Жду команд (макс 60 сек)...\n")
    
    # Ждём команды
    for i in range(60):
        if not bot_running:
            print(f"\n✅ БОТ ОСТАНОВЛЕН через /stop команду!")
            break
        await asyncio.sleep(1)
        if i % 10 == 0 and i > 0:
            print(f"⏳ {60-i} сек до таймаута...")
    else:
        print("\n⏱️ Таймаут! Команды не получены за 60 сек.")
    
    # Остановка
    print("\n🛑 Остановка polling...")
    await app.updater.stop()
    await app.stop()
    await app.shutdown()
    
    print("\n" + "="*70)
    if not bot_running:
        print("✅ ТЕСТ ПРОЙДЕН! Команда /stop работает!")
    else:
        print("⚠️ ТЕСТ НЕ ЗАВЕРШЁН: Команда /stop не получена")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())

