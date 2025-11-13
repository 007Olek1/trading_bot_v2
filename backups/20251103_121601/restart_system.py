)

logger = logging.getLogger(__name__)

def load_config() -> Dict:
    """Загрузка конфигурации"""
    try:
        with open('config/exchange_config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {str(e)}")
        raise

def test_system() -> bool:
    """Тестирование системы"""
    try:
        logger.info("🧪 Начало тестирования системы")

        # Инициализация компонентов
        trading_system = AdaptiveTradingSystem()

        # Тест анализатора монет
        test_market_data = {
            'volume_24h': 2000000,
            'market_cap': 10000000,
            'price_history': [100, 101, 102, 103, 102, 103, 104],
            'volume_history': [1000000, 1100000, 900000, 1200000, 1100000]
        }

        coin_analysis = trading_system.coin_analyzer.analyze_coin('TEST/USDT', test_market_data)
        if not coin_analysis:
            logger.error("❌ Ошибка анализатора монет")
            return False

        # Тест параметров торговли
        trade_params = trading_system.parameter_system.get_trading_parameters()
        if not all(k in trade_params for k in ['position_size', 'take_profit_percent', 'trailing_percent']):
            logger.error("❌ Ошибка параметров торговли")
            return False

        # Тест расчета сделки
        market_data = {
            'symbol': 'TEST/USDT',
            'current_price': 100,
            'volume_24h': 2000000,
            'rsi': 40,
            'macd': 0.002,
            'bb_position': 0.3
        }

        trade_setup = trading_system.process_market_update(market_data)
        if trade_setup['action'] not in ['enter_trade', 'wait', 'error']:
            logger.error("❌ Ошибка расчета сделки")
            return False

        logger.info("✅ Тестирование успешно завершено")
        return True

    except Exception as e:
        logger.error(f"❌ Ошибка тестирования: {str(e)}")
        return False

def update_existing_positions(exchange_config: Dict) -> None:
    """Обновление параметров существующих позиций"""
    try:
        logger.info("🔄 Обновление существующих позиций")

        # Инициализация биржи
        exchange = ccxt.bybit({
            'apiKey': exchange_config['api_key'],
            'secret': exchange_config['api_secret'],
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

        # Получаем открытые позиции
        positions = exchange.fetch_positions()

        # Инициализация торговой системы
        trading_system = AdaptiveTradingSystem()

        for position in positions:
            if position['contracts'] > 0:
                symbol = position['symbol']
                entry_price = float(position['entryPrice'])
                position_size = abs(float(position['contracts']) * entry_price)

                logger.info(f"📊 Обновление позиции {symbol}")

                # Получаем текущие рыночные данные
                ticker = exchange.fetch_ticker(symbol)
                market_data = {
                    'symbol': symbol,
                    'current_price': ticker['last'],
                    'volume_24h': ticker['quoteVolume']
                }

                # Рассчитываем новые уровни
                setup = trading_system.process_market_update(market_data)

                if setup['action'] == 'enter_trade':
                    # Обновляем SL/TP
                    order_result = trading_system.order_manager.place_sl_tp_orders(
                        symbol,
                        {
                            'stop_loss': setup['setup']['stop_loss'],
                            'take_profit': setup['setup']['take_profit']
                        }
                    )

                    if order_result['success']:
                        logger.info(f"✅ {symbol}: SL/TP обновлены")
                    else:
                        logger.error(f"❌ {symbol}: Ошибка обновления SL/TP: {order_result['reason']}")

                # Пауза между обновлениями
                time.sleep(1)

        logger.info("✅ Обновление позиций завершено")

    except Exception as e:
        logger.error(f"❌ Ошибка обновления позиций: {str(e)}")

def main():
    try:
        logger.info("🚀 Начало перезапуска системы")

        # Загрузка конфигурации
        config = load_config()

        # Тестирование системы
        if not test_system():
            logger.error("❌ Тестирование не пройдено, отмена перезапуска")
            return

        # Обновление существующих позиций
        update_existing_positions(config)

        # Запуск мониторинга ордеров
        trading_system = AdaptiveTradingSystem()
        trading_system.start_order_monitoring()

        logger.info("✅ Система успешно перезапущена")

    except Exception as e:
        logger.error(f"❌ Ошибка перезапуска системы: {str(e)}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
🔄 Перезапуск и тестирование системы
==================================
"""

import logging
from typing import Dict, List
import json
import time
from datetime import datetime
import ccxt
from adaptive_trading_system import AdaptiveTradingSystem
from order_manager import OrderManager
from coin_analyzer import CoinAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_restart.log'),
        logging.StreamHandler()
    ]
