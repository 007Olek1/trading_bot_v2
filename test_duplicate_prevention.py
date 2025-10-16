#!/usr/bin/env python3
"""
🧪 ТЕСТ СИСТЕМЫ ПРЕДОТВРАЩЕНИЯ ДУБЛИРОВАНИЯ СДЕЛОК
Проверяет что бот не открывает повторные сделки по одним и тем же монетам
"""

import asyncio
import sys
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

# Импортируем наши боты
sys.path.append('.')
from trading_bot_v3_main import TradingBotV2 as TradingBotV3
from trading_bot_v2_main import TradingBotV2


class DuplicatePreventionTester:
    """Тестер системы предотвращения дублирования"""
    
    def __init__(self):
        self.test_results = []
    
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Логирование результата теста"""
        status = "✅ PASSED" if passed else "❌ FAILED"
        self.test_results.append({
            'name': test_name,
            'passed': passed,
            'details': details
        })
        print(f"{status}: {test_name}")
        if details:
            print(f"   {details}")
    
    def test_v3_cooldown_system(self):
        """Тест системы cooldown для V3 бота"""
        print("\n🧪 Тестирование V3 Bot Cooldown System...")
        
        bot = TradingBotV3()
        
        # Тест 1: Символ не на cooldown изначально
        is_cooldown = bot._is_symbol_on_cooldown("BTCUSDT")
        self.log_test(
            "V3: Новый символ не на cooldown",
            not is_cooldown,
            f"BTCUSDT cooldown: {is_cooldown}"
        )
        
        # Тест 2: Добавление в cooldown
        bot._add_symbol_to_cooldown("BTCUSDT", "buy")
        is_cooldown = bot._is_symbol_on_cooldown("BTCUSDT")
        self.log_test(
            "V3: Символ добавлен в cooldown",
            is_cooldown,
            f"BTCUSDT после добавления: {is_cooldown}"
        )
        
        # Тест 3: Время cooldown корректно
        remaining = bot._get_cooldown_remaining_hours("BTCUSDT")
        self.log_test(
            "V3: Время cooldown корректно",
            5.9 <= remaining <= 6.0,
            f"Осталось часов: {remaining:.2f}"
        )
        
        # Тест 4: Искусственно истекший cooldown
        past_time = datetime.now() - timedelta(hours=7)
        bot.symbol_trade_history["ETHUSDT"] = {
            'last_trade_time': past_time,
            'last_side': 'sell'
        }
        is_cooldown = bot._is_symbol_on_cooldown("ETHUSDT")
        self.log_test(
            "V3: Истекший cooldown не активен",
            not is_cooldown,
            f"ETHUSDT (7ч назад): {is_cooldown}"
        )
        
        # Тест 5: Очистка истекших cooldown
        initial_count = len(bot.symbol_trade_history)
        bot._cleanup_expired_cooldowns()
        final_count = len(bot.symbol_trade_history)
        self.log_test(
            "V3: Очистка истекших cooldown",
            final_count < initial_count,
            f"Было: {initial_count}, стало: {final_count}"
        )
    
    def test_v2_cooldown_system(self):
        """Тест системы cooldown для V2 бота"""
        print("\n🧪 Тестирование V2 Bot Cooldown System...")
        
        bot = TradingBotV2()
        
        # Тест 1: Символ не на cooldown изначально
        is_cooldown, _ = bot.is_symbol_on_cooldown("BTCUSDT")
        self.log_test(
            "V2: Новый символ не на cooldown",
            not is_cooldown,
            f"BTCUSDT cooldown: {is_cooldown}"
        )
        
        # Тест 2: Добавление в cooldown
        bot.add_symbol_to_cooldown("BTCUSDT", "buy")
        is_cooldown, remaining = bot.is_symbol_on_cooldown("BTCUSDT")
        self.log_test(
            "V2: Символ добавлен в cooldown",
            is_cooldown,
            f"BTCUSDT после добавления: {is_cooldown}, осталось: {remaining:.2f}ч"
        )
        
        # Тест 3: Искусственно истекший cooldown
        past_time = datetime.now() - timedelta(hours=7)
        bot.symbol_cooldown["ETHUSDT"] = past_time
        bot.symbol_last_side["ETHUSDT"] = "sell"
        is_cooldown, _ = bot.is_symbol_on_cooldown("ETHUSDT")
        self.log_test(
            "V2: Истекший cooldown не активен",
            not is_cooldown,
            f"ETHUSDT (7ч назад): {is_cooldown}"
        )
    
    async def test_duplicate_prevention_in_open_position(self):
        """Тест предотвращения дублирования в методе open_position"""
        print("\n🧪 Тестирование предотвращения дублирования в open_position...")
        
        # Мокаем exchange_manager
        from unittest.mock import patch
        
        with patch('trading_bot_v3_main.exchange_manager') as mock_exchange:
            # Настраиваем мок
            mock_exchange.fetch_positions = AsyncMock(return_value=[
                {'symbol': 'BTCUSDT', 'contracts': 0.001, 'side': 'buy'}
            ])
            
            bot = TradingBotV3()
            
            # Добавляем позицию в список
            bot.open_positions.append({
                'symbol': 'BTCUSDT',
                'side': 'buy',
                'amount': 0.001
            })
            
            # Пытаемся открыть дублирующую позицию
            signal_data = {
                'signal': 'buy',
                'confidence': 90,
                'reason': 'Test signal'
            }
            
            result = await bot.open_position('BTCUSDT', 'buy', signal_data)
            
            self.log_test(
                "V3: Предотвращение дублирования по позициям",
                result is None,
                f"Результат: {result}, статистика: {bot.duplicate_prevention_stats}"
            )
    
    def test_position_sync_cooldown_preservation(self):
        """Тест сохранения cooldown при синхронизации позиций"""
        print("\n🧪 Тестирование сохранения cooldown при синхронизации...")
        
        bot = TradingBotV3()
        
        # Добавляем позицию
        bot.open_positions.append({
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'amount': 0.001
        })
        
        # Добавляем в cooldown
        bot._add_symbol_to_cooldown('ETHUSDT', 'sell')
        
        # Симулируем синхронизацию где BTCUSDT больше нет
        old_positions = bot.open_positions.copy()
        bot.open_positions = []  # Имитируем что позиция закрылась
        
        # Симулируем логику из sync_positions_from_exchange
        current_position_symbols = {p['symbol'] for p in old_positions}
        new_position_symbols = {p['symbol'] for p in bot.open_positions}
        closed_symbols = current_position_symbols - new_position_symbols
        
        for symbol in closed_symbols:
            if symbol not in bot.symbol_trade_history:
                bot._add_symbol_to_cooldown(symbol, "unknown")
        
        # Проверяем что BTCUSDT добавлен в cooldown
        btc_cooldown = bot._is_symbol_on_cooldown('BTCUSDT')
        eth_cooldown = bot._is_symbol_on_cooldown('ETHUSDT')
        
        self.log_test(
            "V3: Закрытая позиция добавлена в cooldown",
            btc_cooldown,
            f"BTCUSDT cooldown: {btc_cooldown}"
        )
        
        self.log_test(
            "V3: Существующий cooldown сохранен",
            eth_cooldown,
            f"ETHUSDT cooldown: {eth_cooldown}"
        )
    
    def run_all_tests(self):
        """Запуск всех тестов"""
        print("🚀 Запуск тестов системы предотвращения дублирования сделок...")
        print("=" * 70)
        
        # Синхронные тесты
        self.test_v3_cooldown_system()
        self.test_v2_cooldown_system()
        self.test_position_sync_cooldown_preservation()
        
        # Асинхронные тесты
        asyncio.run(self.test_duplicate_prevention_in_open_position())
        
        # Результаты
        print("\n" + "=" * 70)
        print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
        print("=" * 70)
        
        passed_count = sum(1 for test in self.test_results if test['passed'])
        total_count = len(self.test_results)
        
        for test in self.test_results:
            status = "✅" if test['passed'] else "❌"
            print(f"{status} {test['name']}")
            if test['details'] and not test['passed']:
                print(f"   ⚠️ {test['details']}")
        
        print(f"\n📈 ИТОГО: {passed_count}/{total_count} тестов пройдено")
        
        if passed_count == total_count:
            print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Система предотвращения дублирования работает корректно.")
        else:
            print("⚠️ ЕСТЬ ПРОБЛЕМЫ! Требуется доработка системы.")
        
        return passed_count == total_count


if __name__ == "__main__":
    tester = DuplicatePreventionTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ Система готова к использованию!")
        sys.exit(0)
    else:
        print("\n❌ Требуется исправление ошибок!")
        sys.exit(1)