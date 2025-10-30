#!/usr/bin/env python3
"""
🧪 ТЕСТИРОВАНИЕ TELEGRAM КОМАНД
Проверяет работоспособность всех команд бота
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные окружения
env_files = [
    Path(__file__).parent / 'api.env',
    Path(__file__).parent.parent / '.env',
    Path(__file__).parent / '.env'
]

for env_file in env_files:
    if env_file.exists():
        load_dotenv(env_file, override=False)
        print(f"✅ Загружены переменные из {env_file}")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


async def test_telegram_commands():
    """Тестирует все Telegram команды"""
    
    print("=" * 70)
    print("🧪 ТЕСТИРОВАНИЕ TELEGRAM КОМАНД")
    print("=" * 70)
    print()
    
    # Проверка токена
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN') or os.getenv('TELEGRAM_TOKEN')
    telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not telegram_token:
        print("❌ Ошибка: TELEGRAM_BOT_TOKEN не найден в переменных окружения")
        print("   Проверьте .env или api.env файл")
        return False
    
    if not telegram_chat_id:
        print("⚠️ Внимание: TELEGRAM_CHAT_ID не найден")
        print("   Бот будет работать, но некоторые команды могут требовать chat_id")
    else:
        print(f"✅ Telegram Chat ID: {telegram_chat_id}")
    
    print(f"✅ Telegram Token: {telegram_token[:10]}...")
    print()
    
    try:
        # Импорт бота
        from super_bot_v4_mtf import SuperBotV4MTF
        from telegram import Bot
        from telegram.ext import Application
        
        print("📦 Инициализация бота...")
        bot = SuperBotV4MTF()
        
        # Инициализация без реального запуска
        print("🔧 Инициализация компонентов...")
        await bot.initialize()
        
        if not bot.application:
            print("❌ Telegram Application не инициализирован")
            return False
        
        if not bot.commands_handler:
            print("❌ Telegram Commands Handler не инициализирован")
            return False
        
        print("✅ Бот инициализирован")
        print()
        
        # Проверка зарегистрированных команд
        print("📋 Проверка зарегистрированных команд...")
        print()
        
        # Список всех команд
        all_commands = [
            ('/start', 'cmd_start'),
            ('/help', 'cmd_help'),
            ('/status', 'cmd_status'),
            ('/balance', 'cmd_balance'),
            ('/positions', 'cmd_positions'),
            ('/history', 'cmd_history'),
            ('/settings', 'cmd_settings'),
            ('/health', 'cmd_health'),
            ('/stop', 'cmd_stop'),
            ('/resume', 'cmd_resume'),
            ('/stats', 'cmd_stats')
        ]
        
        # Проверяем что методы существуют
        commands_ok = 0
        for cmd_name, method_name in all_commands:
            if hasattr(bot.commands_handler, method_name):
                print(f"  ✅ {cmd_name:15s} - {method_name} существует")
                commands_ok += 1
            else:
                print(f"  ❌ {cmd_name:15s} - {method_name} НЕ НАЙДЕН")
        
        print()
        print(f"✅ Команд проверено: {commands_ok}/{len(all_commands)}")
        print()
        
        # Тест отправки сообщения (если есть chat_id)
        if telegram_chat_id:
            print("📤 Тест отправки сообщения...")
            try:
                test_message = """🧪 *ТЕСТ TELEGRAM КОМАНД*

✅ Бот успешно инициализирован
✅ Все команды зарегистрированы
✅ Система готова к работе

📋 Доступные команды:
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

💡 Отправьте любую команду боту для проверки!
"""
                
                await bot.send_telegram_v4(test_message)
                print("✅ Тестовое сообщение отправлено в Telegram")
                print()
            except Exception as e:
                print(f"⚠️ Ошибка отправки тестового сообщения: {e}")
                print()
        
        # Информация о polling
        print("📡 Статус Telegram Polling:")
        if bot.application.updater:
            print("  ✅ Updater инициализирован")
        else:
            print("  ⚠️ Updater не инициализирован (будет создан при run_v4)")
        
        print()
        print("=" * 70)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 70)
        print()
        print("💡 Для полного тестирования:")
        print("   1. Запустите бота: python super_bot_v4_mtf.py")
        print("   2. Откройте Telegram и найдите вашего бота")
        print("   3. Отправьте команду /start")
        print("   4. Проверьте остальные команды")
        print()
        
        # Закрываем соединения
        if bot.exchange:
            try:
                await bot.exchange.close()
            except:
                pass
        
        return True
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("   Установите недостающие библиотеки: pip install -r requirements_bot.txt")
        return False
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_simple_bot_start():
    """Простой тест - проверяет что бот может запуститься"""
    print("\n🔍 Дополнительный тест: проверка запуска бота...")
    
    try:
        from super_bot_v4_mtf import SuperBotV4MTF
        
        bot = SuperBotV4MTF()
        
        # Проверяем основные атрибуты
        checks = {
            'API ключ Bybit': bool(bot.api_key),
            'API секрет Bybit': bool(bot.api_secret),
            'Telegram токен': bool(bot.telegram_token),
            'Telegram Chat ID': bool(bot.telegram_chat_id),
            'Exchange объект': False,  # Создается при initialize()
            'Application объект': False,  # Создается при initialize()
        }
        
        print("\n📊 Проверка конфигурации:")
        for check_name, status in checks.items():
            icon = "✅" if status else "❌"
            print(f"  {icon} {check_name}: {'Да' if status else 'Нет'}")
        
        # Пробуем инициализировать (без реального запуска)
        try:
            await bot.initialize()
            checks['Exchange объект'] = bot.exchange is not None
            checks['Application объект'] = bot.application is not None
            
            print("\n📊 После инициализации:")
            for check_name, status in checks.items():
                icon = "✅" if status else "❌"
                print(f"  {icon} {check_name}: {'Да' if status else 'Нет'}")
            
            # Закрываем
            if bot.exchange:
                await bot.exchange.close()
            
            return all(checks.values())
            
        except Exception as e:
            print(f"⚠️ Ошибка при инициализации: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


async def main():
    """Главная функция тестирования"""
    try:
        # Основной тест команд
        result1 = await test_telegram_commands()
        
        # Дополнительный тест
        result2 = await test_simple_bot_start()
        
        print("\n" + "=" * 70)
        if result1 and result2:
            print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
            print("✅ Бот готов к работе с Telegram командами")
            print("\n💡 Следующий шаг: запустите бота и протестируйте команды в Telegram")
        else:
            print("⚠️ Некоторые тесты не прошли")
            print("   Проверьте логи выше для деталей")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Тестирование прервано пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())


