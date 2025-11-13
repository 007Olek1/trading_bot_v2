#!/usr/bin/env python3
"""
Тест всех Telegram команд бота
Проверяет что все команды зарегистрированы и могут быть вызваны
"""
import sys
import os
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path("/opt/bot")))

import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

async def test_telegram_commands():
    """Проверка всех Telegram команд"""
    logger.info("\n" + "="*70)
    logger.info("📱 ПРОВЕРКА TELEGRAM КОМАНД")
    logger.info("="*70)
    
    try:
        from telegram_commands_handler import TelegramCommandsHandler
        from super_bot_v4_mtf import SuperBotV4MTF
        from dotenv import load_dotenv
        
        # Загружаем переменные окружения
        env_file = Path("/opt/bot/.env")
        if env_file.exists():
            load_dotenv(env_file, override=True)
        else:
            logger.error("❌ .env файл не найден")
            return False
        
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not telegram_token:
            logger.error("❌ TELEGRAM_BOT_TOKEN не найден")
            return False
        
        logger.info("✅ Telegram токен найден")
        
        # Инициализируем бот (минимально, только для проверки команд)
        bot = SuperBotV4MTF()
        
        # Проверяем наличие обработчика команд
        commands_handler = TelegramCommandsHandler(bot)
        logger.info("✅ TelegramCommandsHandler создан")
        
        # Проверяем наличие всех методов команд
        required_commands = {
            '/start': 'cmd_start',
            '/help': 'cmd_help',
            '/status': 'cmd_status',
            '/balance': 'cmd_balance',
            '/positions': 'cmd_positions',
            '/history': 'cmd_history',
            '/settings': 'cmd_settings',
            '/health': 'cmd_health',
            '/stop': 'cmd_stop',
            '/resume': 'cmd_resume',
            '/stats': 'cmd_stats',
        }
        
        logger.info("\n📋 Проверка методов команд:")
        all_ok = True
        for command, method_name in required_commands.items():
            if hasattr(commands_handler, method_name):
                method = getattr(commands_handler, method_name)
                if callable(method):
                    logger.info(f"   ✅ {command} -> {method_name}()")
                else:
                    logger.error(f"   ❌ {command} -> {method_name} не вызываемый")
                    all_ok = False
            else:
                logger.error(f"   ❌ {command} -> {method_name} не найден")
                all_ok = False
        
        # Проверяем метод регистрации
        if hasattr(commands_handler, 'register_commands'):
            logger.info(f"\n   ✅ register_commands() доступен")
        else:
            logger.error(f"\n   ❌ register_commands() не найден")
            all_ok = False
        
        # Проверяем вспомогательные методы
        helper_methods = ['_get_open_positions_live', '_format_price', '_format_time']
        logger.info("\n🔧 Проверка вспомогательных методов:")
        for method_name in helper_methods:
            if hasattr(commands_handler, method_name):
                logger.info(f"   ✅ {method_name}() доступен")
            else:
                logger.warning(f"   ⚠️ {method_name}() не найден (может быть опциональным)")
        
        logger.info("\n" + "="*70)
        if all_ok:
            logger.info("✅ ВСЕ КОМАНДЫ ПРОВЕРЕНЫ И ДОСТУПНЫ!")
            logger.info("="*70)
            logger.info("\n📱 Команды готовы к использованию:")
            for cmd in required_commands.keys():
                logger.info(f"   • {cmd}")
            return True
        else:
            logger.error("❌ НЕКОТОРЫЕ КОМАНДЫ НЕ РАБОТАЮТ!")
            logger.info("="*70)
            return False
            
    except Exception as e:
        logger.error(f"❌ Ошибка проверки команд: {e}", exc_info=True)
        return False

async def test_bot_telegram_integration():
    """Проверка интеграции Telegram в боте"""
    logger.info("\n" + "="*70)
    logger.info("🔗 ПРОВЕРКА ИНТЕГРАЦИИ TELEGRAM В БОТЕ")
    logger.info("="*70)
    
    try:
        from super_bot_v4_mtf import SuperBotV4MTF
        
        # Проверяем что бот имеет необходимые атрибуты для Telegram
        bot = SuperBotV4MTF()
        
        required_attrs = {
            'telegram_token': 'Токен Telegram',
            'telegram_chat_id': 'Chat ID',
            'application': 'Application объект',
            'commands_handler': 'Обработчик команд',
        }
        
        all_ok = True
        for attr, desc in required_attrs.items():
            if hasattr(bot, attr):
                logger.info(f"   ✅ {attr} ({desc}) - доступен")
            else:
                logger.warning(f"   ⚠️ {attr} ({desc}) - не найден (инициализируется позже)")
        
        # Проверяем методы работы с Telegram
        telegram_methods = [
            'send_telegram_v4',
            'send_startup_message_v4',
            'send_enhanced_signal_v4',
            'send_position_closed_v4',
        ]
        
        logger.info("\n📨 Проверка методов отправки сообщений:")
        for method_name in telegram_methods:
            if hasattr(bot, method_name):
                logger.info(f"   ✅ {method_name}() доступен")
            else:
                logger.error(f"   ❌ {method_name}() не найден")
                all_ok = False
        
        logger.info("\n" + "="*70)
        if all_ok:
            logger.info("✅ ИНТЕГРАЦИЯ TELEGRAM РАБОТАЕТ!")
            logger.info("="*70)
            return True
        else:
            logger.error("❌ ЕСТЬ ПРОБЛЕМЫ С ИНТЕГРАЦИЕЙ!")
            logger.info("="*70)
            return False
            
    except Exception as e:
        logger.error(f"❌ Ошибка проверки интеграции: {e}", exc_info=True)
        return False

async def main():
    """Главная функция"""
    logger.info("\n" + "="*70)
    logger.info("🧪 ТЕСТИРОВАНИЕ TELEGRAM КОМАНД")
    logger.info("="*70)
    
    test1 = await test_telegram_commands()
    test2 = await test_bot_telegram_integration()
    
    logger.info("\n" + "="*70)
    logger.info("📊 ИТОГОВЫЙ ОТЧЕТ")
    logger.info("="*70)
    logger.info(f"   Команды: {'✅ РАБОТАЮТ' if test1 else '❌ ОШИБКИ'}")
    logger.info(f"   Интеграция: {'✅ РАБОТАЕТ' if test2 else '❌ ОШИБКИ'}")
    
    if test1 and test2:
        logger.info("\n🎉 ВСЕ TELEGRAM КОМАНДЫ ГОТОВЫ К ИСПОЛЬЗОВАНИЮ!")
        logger.info("="*70)
        logger.info("\n📱 Доступные команды:")
        logger.info("   /start - Стартовое сообщение")
        logger.info("   /help - Список команд")
        logger.info("   /status - Статус бота")
        logger.info("   /balance - Баланс")
        logger.info("   /positions - Открытые позиции")
        logger.info("   /history - История сделок")
        logger.info("   /settings - Настройки")
        logger.info("   /health - Health Score")
        logger.info("   /stop - Остановить торговлю")
        logger.info("   /resume - Возобновить")
        logger.info("   /stats - Статистика")
    else:
        logger.error("\n⚠️ НЕКОТОРЫЕ КОМПОНЕНТЫ НЕ РАБОТАЮТ!")
        logger.info("="*70)
    
    return test1 and test2

if __name__ == "__main__":
    asyncio.run(main())
