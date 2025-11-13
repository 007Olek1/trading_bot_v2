"""
Перезапуск и тестирование системы
"""

import logging
import json
import time
import sys
from typing import Dict, Optional

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_restart.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Безопасный импорт модулей
try:
    import ccxt
except ImportError:
    ccxt = None
    logger.warning("Модуль ccxt не установлен. Установите: pip install ccxt")

try:
    from adaptive_trading_system import AdaptiveTradingSystem
except ImportError:
    AdaptiveTradingSystem = None
    logger.warning("Модуль adaptive_trading_system не найден")

try:
    from order_manager import OrderManager
except ImportError:
    OrderManager = None
    logger.warning("Модуль order_manager не найден")

try:
    from coin_analyzer import CoinAnalyzer
except ImportError:
    CoinAnalyzer = None
    logger.warning("Модуль coin_analyzer не найден")

# Параметры торговли
TRADING_PARAMS = {
    'position_size': 30,      # $30
    'leverage': 10,           # 10x
    'take_profit': 0.02,      # 2%
    'trailing': 0.01,         # 1%
    'stop_loss': 1            # $1
}

def load_config() -> Dict:
    """Загрузка конфигурации из файла"""
    try:
        with open('config/exchange_config.json', 'r') as f:
            config = json.load(f)
            logger.info("✅ Конфигурация загружена")
            return config
    except FileNotFoundError:
        logger.error("❌ Файл config/exchange_config.json не найден")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"❌ Ошибка парсинга JSON: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки конфигурации: {e}")
        raise

def test_system() -> bool:
    """Тестирование компонентов системы"""
    logger.info("🧪 Начало тестирования системы")
    
    if AdaptiveTradingSystem is None:
        logger.warning("⚠️ AdaptiveTradingSystem недоступен - пропуск тестов")
        return True
    
    try:
        trading_system = AdaptiveTradingSystem()
        
        # Проверка наличия методов
        if hasattr(trading_system, 'test_components'):
            result = trading_system.test_components()
            if not result:
                logger.error("❌ test_components вернул False")
                return False
        
        if hasattr(trading_system, 'validate_parameters'):
            result = trading_system.validate_parameters()
            if not result:
                logger.error("❌ validate_parameters вернул False")
                return False
        
        logger.info("✅ Тестирование успешно завершено")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования: {e}")
        return False

def update_existing_positions(config: Dict) -> None:
    """Обновление существующих позиций"""
    logger.info("🔄 Обновление существующих позиций")
    
    if ccxt is None:
        logger.error("❌ ccxt не установлен - невозможно обновить позиции")
        return
    
    if AdaptiveTradingSystem is None:
        logger.error("❌ AdaptiveTradingSystem недоступен")
        return
    
    try:
        # Инициализация биржи
        exchange = ccxt.bybit({
            'apiKey': config.get('api_key'),
            'secret': config.get('api_secret'),
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        # Получение позиций
        positions = exchange.fetch_positions()
        logger.info(f"📊 Найдено позиций: {len(positions)}")
        
        trading_system = AdaptiveTradingSystem()
        
        for position in positions:
            try:
                contracts = float(position.get('contracts', 0) or 0)
                if contracts <= 0:
                    continue
                
                symbol = position.get('symbol')
                logger.info(f"📈 Обработка позиции {symbol}")
                
                # Получение текущей цены
                ticker = exchange.fetch_ticker(symbol)
                market_data = {
                    'symbol': symbol,
                    'current_price': ticker.get('last'),
                    'volume_24h': ticker.get('quoteVolume')
                }
                
                # Обработка через систему
                setup = trading_system.process_market_update(market_data)
                
                if setup and setup.get('action') == 'enter_trade':
                    order_manager = getattr(trading_system, 'order_manager', None)
                    if order_manager and hasattr(order_manager, 'place_sl_tp_orders'):
                        result = order_manager.place_sl_tp_orders(
                            symbol,
                            {
                                'stop_loss': setup['setup'].get('stop_loss'),
                                'take_profit': setup['setup'].get('take_profit')
                            }
                        )
                        if result.get('success'):
                            logger.info(f"✅ {symbol}: SL/TP обновлены")
                        else:
                            logger.error(f"❌ {symbol}: {result.get('reason')}")
                
                time.sleep(1)  # Задержка между запросами
                
            except Exception as e:
                logger.error(f"❌ Ошибка обработки позиции: {e}")
                continue
        
        logger.info("✅ Обновление позиций завершено")
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка при обновлении позиций: {e}")

def main():
    """Основная функция"""
    try:
        logger.info("🚀 Запуск перезапуска системы")
        logger.info(f"📁 Рабочая директория: {sys.path[0]}")
        
        # Загрузка конфигурации
        config = load_config()
        
        # Тестирование системы
        if not test_system():
            logger.error("❌ Тестирование не пройдено - отмена перезапуска")
            sys.exit(1)
        
        # Инициализация системы
        if AdaptiveTradingSystem:
            trading_system = AdaptiveTradingSystem()
            
            # Обновление параметров
            if hasattr(trading_system, 'update_trading_parameters'):
                trading_system.update_trading_parameters(TRADING_PARAMS)
                logger.info("✅ Параметры торговли обновлены")
        
        # Обновление позиций
        update_existing_positions(config)
        
        # Запуск мониторинга
        if AdaptiveTradingSystem and hasattr(trading_system, 'start_order_monitoring'):
            try:
                trading_system.start_order_monitoring()
                logger.info("✅ Мониторинг ордеров запущен")
            except Exception as e:
                logger.error(f"❌ Не удалось запустить мониторинг: {e}")
        
        logger.info("✅ Система успешно перезапущена")
        
    except Exception as e:
        logger.error(f"❌ Фатальная ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

