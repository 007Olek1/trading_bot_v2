"""
🧪 ПОЛНАЯ СИСТЕМА ТЕСТИРОВАНИЯ
Объединяет все 4 компонента тестирования
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from pathlib import Path
from datetime import datetime

# Импорт всех тестовых модулей
from test_indicators import IndicatorValidator, StrategyValidator
from live_market_test import LiveMarketTester
from performance_optimizer import PerformanceOptimizer
from trade_analyzer import TradeAnalyzer


class FullTestingSuite:
    """Полная система тестирования"""
    
    def __init__(self):
        self.logs_dir = Path(__file__).parent / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*80)
        print("🧪 TRADING BOT V4.0 MTF - ПОЛНАЯ СИСТЕМА ТЕСТИРОВАНИЯ")
        print("="*80)
        print("\nСистема включает:")
        print("1️⃣ Comprehensive Bot Testing - валидация индикаторов и стратегий")
        print("2️⃣ Live Market Testing - тестирование на реальных данных БЕЗ сделок")
        print("3️⃣ Performance Optimization - анализ и оптимизация параметров")
        print("4️⃣ Trade Analysis - детальный анализ результатов")
        print("="*80 + "\n")
    
    async def run_phase_1(self) -> bool:
        """Фаза 1: Валидация компонентов"""
        print("\n" + "🔵"*40)
        print("ФАЗА 1: ВАЛИДАЦИЯ КОМПОНЕНТОВ")
        print("🔵"*40 + "\n")
        
        # Тестирование индикаторов
        print("📊 Тестирование индикаторов...")
        indicator_validator = IndicatorValidator()
        indicator_results = indicator_validator.run_all_tests()
        
        # Тестирование стратегий
        print("\n🎯 Тестирование стратегий...")
        strategy_validator = StrategyValidator()
        strategy_results = strategy_validator.run_all_tests()
        
        # Проверяем результаты
        all_results = indicator_results + strategy_results
        failed = sum(1 for r in all_results if r['status'] == 'FAIL')
        
        if failed > 0:
            print(f"\n❌ Фаза 1 провалена: {failed} тестов не прошли")
            print("Исправьте ошибки перед продолжением")
            return False
        
        print("\n✅ Фаза 1 успешно завершена!")
        return True
    
    async def run_phase_2(self, duration_minutes: int = 30) -> str:
        """Фаза 2: Live Market Testing"""
        print("\n" + "🟢"*40)
        print("ФАЗА 2: LIVE MARKET TESTING")
        print("🟢"*40 + "\n")
        
        print(f"⏱️ Длительность теста: {duration_minutes} минут")
        print("📊 Тестирование на реальных данных БЕЗ исполнения сделок\n")
        
        tester = LiveMarketTester(test_mode=True)
        await tester.run(duration_minutes=duration_minutes)
        
        # Находим файл с результатами
        result_files = list(self.logs_dir.glob("test_results_*.json"))
        if result_files:
            latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
            print(f"\n✅ Фаза 2 завершена! Результаты: {latest_file.name}")
            return str(latest_file)
        else:
            print("\n⚠️ Фаза 2: результаты не найдены")
            return ""
    
    def run_phase_3(self, results_file: str):
        """Фаза 3: Performance Optimization"""
        print("\n" + "🟡"*40)
        print("ФАЗА 3: PERFORMANCE OPTIMIZATION")
        print("🟡"*40 + "\n")
        
        if not results_file or not Path(results_file).exists():
            print("⚠️ Нет данных для оптимизации")
            return
        
        optimizer = PerformanceOptimizer(results_file)
        report = optimizer.generate_report()
        
        print(report)
        
        # Сохраняем отчёт
        report_file = self.logs_dir / f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n💾 Отчёт оптимизации сохранён: {report_file.name}")
        print("✅ Фаза 3 завершена!")
    
    def run_phase_4(self, results_file: str):
        """Фаза 4: Trade Analysis"""
        print("\n" + "🟣"*40)
        print("ФАЗА 4: TRADE ANALYSIS")
        print("🟣"*40 + "\n")
        
        if not results_file or not Path(results_file).exists():
            print("⚠️ Нет данных для анализа")
            return
        
        analyzer = TradeAnalyzer(results_file)
        report = analyzer.generate_report()
        
        print(report)
        
        # Сохраняем отчёт
        report_file = self.logs_dir / f"trade_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n💾 Отчёт анализа сохранён: {report_file.name}")
        print("✅ Фаза 4 завершена!")
    
    async def run_full_test(self, live_test_duration: int = 30):
        """Запуск полного тестирования"""
        start_time = datetime.now()
        
        # Фаза 1: Валидация
        phase1_success = await self.run_phase_1()
        
        if not phase1_success:
            print("\n❌ Тестирование прервано из-за ошибок в Фазе 1")
            return
        
        # Фаза 2: Live Market Testing
        results_file = await self.run_phase_2(duration_minutes=live_test_duration)
        
        if not results_file:
            print("\n⚠️ Фазы 3 и 4 пропущены (нет данных)")
        else:
            # Фаза 3: Optimization
            self.run_phase_3(results_file)
            
            # Фаза 4: Analysis
            self.run_phase_4(results_file)
        
        # Итоговый отчёт
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print("\n" + "="*80)
        print("🏆 ПОЛНОЕ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print("="*80)
        print(f"⏱️ Общее время: {duration:.1f} минут")
        print(f"📂 Все отчёты сохранены в: {self.logs_dir}")
        print("\n💡 Следующие шаги:")
        print("  1. Изучите отчёты оптимизации и анализа")
        print("  2. Примените рекомендованные параметры в config.py")
        print("  3. Запустите бота на реальном счёте с малыми суммами")
        print("  4. Мониторьте результаты и корректируйте при необходимости")
        print("="*80 + "\n")


async def main():
    """Точка входа"""
    suite = FullTestingSuite()
    
    # Запрашиваем параметры
    print("Настройка тестирования:")
    print("-"*80)
    
    try:
        duration = int(input("Длительность Live Market Testing (минут, по умолчанию 30): ") or "30")
    except ValueError:
        duration = 30
    
    print("\n🚀 Запуск полного тестирования...\n")
    
    await suite.run_full_test(live_test_duration=duration)


if __name__ == "__main__":
    asyncio.run(main())
