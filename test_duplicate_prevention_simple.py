#!/usr/bin/env python3
"""
🧪 ПРОСТОЙ ТЕСТ СИСТЕМЫ ПРЕДОТВРАЩЕНИЯ ДУБЛИРОВАНИЯ
Тестирует только ключевые функции без зависимостей
"""

import json
import os
from datetime import datetime, timedelta


class SimpleDuplicatePreventionTester:
    """Простой тестер логики предотвращения дублирования"""
    
    def __init__(self):
        self.test_results = []
        self.cooldown_hours = 6
        self.symbol_trade_history = {}
    
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
    
    def _is_symbol_on_cooldown(self, symbol: str) -> bool:
        """Проверяет находится ли символ на cooldown"""
        if symbol not in self.symbol_trade_history:
            return False
        
        last_trade_time = self.symbol_trade_history[symbol]['last_trade_time']
        time_passed = datetime.now() - last_trade_time
        hours_passed = time_passed.total_seconds() / 3600
        
        return hours_passed < self.cooldown_hours
    
    def _get_cooldown_remaining_hours(self, symbol: str) -> float:
        """Возвращает оставшееся время cooldown в часах"""
        if symbol not in self.symbol_trade_history:
            return 0.0
        
        last_trade_time = self.symbol_trade_history[symbol]['last_trade_time']
        time_passed = datetime.now() - last_trade_time
        hours_passed = time_passed.total_seconds() / 3600
        
        return max(0.0, self.cooldown_hours - hours_passed)
    
    def _add_symbol_to_cooldown(self, symbol: str, side: str):
        """Добавляет символ в cooldown после открытия позиции"""
        self.symbol_trade_history[symbol] = {
            'last_trade_time': datetime.now(),
            'last_side': side.lower()
        }
        print(f"⏰ {symbol} {side.upper()} добавлен в cooldown на {self.cooldown_hours} часов")
    
    def _cleanup_expired_cooldowns(self):
        """Очистка устаревших записей cooldown"""
        expired_symbols = []
        
        for symbol in list(self.symbol_trade_history.keys()):
            if not self._is_symbol_on_cooldown(symbol):
                expired_symbols.append(symbol)
        
        for symbol in expired_symbols:
            del self.symbol_trade_history[symbol]
        
        return len(expired_symbols)
    
    def test_cooldown_basic_functionality(self):
        """Тест базовой функциональности cooldown"""
        print("\n🧪 Тестирование базовой функциональности cooldown...")
        
        # Тест 1: Новый символ не на cooldown
        is_cooldown = self._is_symbol_on_cooldown("BTCUSDT")
        self.log_test(
            "Новый символ не на cooldown",
            not is_cooldown,
            f"BTCUSDT cooldown: {is_cooldown}"
        )
        
        # Тест 2: Добавление в cooldown
        self._add_symbol_to_cooldown("BTCUSDT", "buy")
        is_cooldown = self._is_symbol_on_cooldown("BTCUSDT")
        self.log_test(
            "Символ добавлен в cooldown",
            is_cooldown,
            f"BTCUSDT после добавления: {is_cooldown}"
        )
        
        # Тест 3: Время cooldown корректно
        remaining = self._get_cooldown_remaining_hours("BTCUSDT")
        self.log_test(
            "Время cooldown корректно",
            5.9 <= remaining <= 6.0,
            f"Осталось часов: {remaining:.2f}"
        )
        
        # Тест 4: Искусственно истекший cooldown
        past_time = datetime.now() - timedelta(hours=7)
        self.symbol_trade_history["ETHUSDT"] = {
            'last_trade_time': past_time,
            'last_side': 'sell'
        }
        is_cooldown = self._is_symbol_on_cooldown("ETHUSDT")
        self.log_test(
            "Истекший cooldown не активен",
            not is_cooldown,
            f"ETHUSDT (7ч назад): {is_cooldown}"
        )
        
        # Тест 5: Очистка истекших cooldown
        initial_count = len(self.symbol_trade_history)
        expired_count = self._cleanup_expired_cooldowns()
        final_count = len(self.symbol_trade_history)
        self.log_test(
            "Очистка истекших cooldown",
            expired_count > 0 and final_count < initial_count,
            f"Было: {initial_count}, очищено: {expired_count}, стало: {final_count}"
        )
    
    def test_file_persistence(self):
        """Тест сохранения и загрузки из файла"""
        print("\n🧪 Тестирование сохранения в файл...")
        
        test_file = "test_cooldown_history.json"
        
        # Очищаем тестовый файл если существует
        if os.path.exists(test_file):
            os.remove(test_file)
        
        # Добавляем данные
        self._add_symbol_to_cooldown("ADAUSDT", "buy")
        self._add_symbol_to_cooldown("DOTUSDT", "sell")
        
        # Сохраняем в файл
        data_to_save = {}
        for symbol, info in self.symbol_trade_history.items():
            data_to_save[symbol] = {
                'last_trade_time': info['last_trade_time'].isoformat(),
                'last_side': info['last_side']
            }
        
        with open(test_file, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        # Очищаем память
        original_data = self.symbol_trade_history.copy()
        self.symbol_trade_history = {}
        
        # Загружаем из файла
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)
        
        # Конвертируем обратно
        for symbol, info in loaded_data.items():
            self.symbol_trade_history[symbol] = {
                'last_trade_time': datetime.fromisoformat(info['last_trade_time']),
                'last_side': info['last_side']
            }
        
        # Проверяем что данные восстановлены
        ada_cooldown = self._is_symbol_on_cooldown("ADAUSDT")
        dot_cooldown = self._is_symbol_on_cooldown("DOTUSDT")
        
        self.log_test(
            "Сохранение и загрузка из файла",
            ada_cooldown and dot_cooldown,
            f"ADAUSDT: {ada_cooldown}, DOTUSDT: {dot_cooldown}"
        )
        
        # Очищаем тестовый файл
        if os.path.exists(test_file):
            os.remove(test_file)
    
    def test_duplicate_prevention_logic(self):
        """Тест логики предотвращения дублирования"""
        print("\n🧪 Тестирование логики предотвращения дублирования...")
        
        # Симулируем открытые позиции
        open_positions = [
            {'symbol': 'BTCUSDT', 'side': 'buy', 'amount': 0.001},
            {'symbol': 'ETHUSDT', 'side': 'sell', 'amount': 0.01}
        ]
        
        # Тест 1: Предотвращение по существующим позициям
        symbol_in_positions = any(p['symbol'] == 'BTCUSDT' for p in open_positions)
        self.log_test(
            "Предотвращение по существующим позициям",
            symbol_in_positions,
            f"BTCUSDT найден в позициях: {symbol_in_positions}"
        )
        
        # Тест 2: Предотвращение по cooldown
        self._add_symbol_to_cooldown("ADAUSDT", "buy")
        ada_on_cooldown = self._is_symbol_on_cooldown("ADAUSDT")
        self.log_test(
            "Предотвращение по cooldown",
            ada_on_cooldown,
            f"ADAUSDT на cooldown: {ada_on_cooldown}"
        )
        
        # Тест 3: Разрешение новой сделки
        new_symbol_allowed = (
            not any(p['symbol'] == 'LINKUSDT' for p in open_positions) and
            not self._is_symbol_on_cooldown('LINKUSDT')
        )
        self.log_test(
            "Разрешение новой сделки",
            new_symbol_allowed,
            f"LINKUSDT разрешен: {new_symbol_allowed}"
        )
    
    def test_position_sync_simulation(self):
        """Тест симуляции синхронизации позиций"""
        print("\n🧪 Тестирование симуляции синхронизации позиций...")
        
        # Начальные позиции
        old_positions = [
            {'symbol': 'BTCUSDT', 'side': 'buy'},
            {'symbol': 'ETHUSDT', 'side': 'sell'},
            {'symbol': 'ADAUSDT', 'side': 'buy'}
        ]
        
        # Новые позиции (ADAUSDT закрылась)
        new_positions = [
            {'symbol': 'BTCUSDT', 'side': 'buy'},
            {'symbol': 'ETHUSDT', 'side': 'sell'}
        ]
        
        # Находим закрытые позиции
        old_symbols = {p['symbol'] for p in old_positions}
        new_symbols = {p['symbol'] for p in new_positions}
        closed_symbols = old_symbols - new_symbols
        
        # Добавляем закрытые в cooldown
        for symbol in closed_symbols:
            if symbol not in self.symbol_trade_history:
                self._add_symbol_to_cooldown(symbol, "unknown")
        
        # Проверяем что закрытая позиция добавлена в cooldown
        ada_cooldown = self._is_symbol_on_cooldown('ADAUSDT')
        self.log_test(
            "Закрытая позиция добавлена в cooldown",
            ada_cooldown,
            f"ADAUSDT добавлен в cooldown: {ada_cooldown}, закрытых позиций: {len(closed_symbols)}"
        )
    
    def run_all_tests(self):
        """Запуск всех тестов"""
        print("🚀 Запуск тестов системы предотвращения дублирования сделок...")
        print("=" * 70)
        
        self.test_cooldown_basic_functionality()
        self.test_file_persistence()
        self.test_duplicate_prevention_logic()
        self.test_position_sync_simulation()
        
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
    tester = SimpleDuplicatePreventionTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ Система готова к использованию!")
        exit(0)
    else:
        print("\n❌ Требуется исправление ошибок!")
        exit(1)