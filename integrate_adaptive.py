#!/usr/bin/env python3
"""
🔧 СКРИПТ АВТОМАТИЧЕСКОЙ ИНТЕГРАЦИИ АДАПТИВНОЙ СИСТЕМЫ
Автоматически добавляет адаптивную систему в main.py
"""

import re
from pathlib import Path

def integrate_adaptive_system():
    """Интегрирует адаптивную систему в main.py"""
    
    main_file = Path("main.py")
    if not main_file.exists():
        print("❌ Файл main.py не найден!")
        return False
    
    # Читаем текущий main.py
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Создаём бэкап
    backup_file = Path("main.py.backup_adaptive")
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Создан бэкап: {backup_file}")
    
    # 1. Добавляем импорты
    imports_to_add = """from market_regime_detector import MarketRegimeDetector, MarketRegime
from adaptive_config import AdaptiveConfig
from performance_tracker import PerformanceTracker
"""
    
    # Находим место для вставки импортов (после from daily_reporter)
    import_pattern = r"(from daily_reporter import DailyReporter)"
    if re.search(import_pattern, content):
        content = re.sub(
            import_pattern,
            r"\1\n" + imports_to_add,
            content
        )
        print("✅ Добавлены импорты")
    
    # 2. Добавляем инициализацию в __init__
    init_code = """
        # 🧠 Адаптивная система
        self.regime_detector = MarketRegimeDetector(self.client, self.logger)
        self.adaptive_config = AdaptiveConfig(self.logger)
        self.performance_tracker = PerformanceTracker(self.logger)
        
        # Текущий режим и параметры
        self.current_regime = None
        self.current_adaptive_params = None
"""
    
    # Вставляем после self.daily_reporter
    daily_reporter_pattern = r"(self\.daily_reporter = DailyReporter\(config\.TRADES_LOG_FILE\))"
    if re.search(daily_reporter_pattern, content):
        content = re.sub(
            daily_reporter_pattern,
            r"\1" + init_code,
            content
        )
        print("✅ Добавлена инициализация адаптивной системы")
    
    # 3. Добавляем метод обновления режима
    update_regime_method = """
    def update_market_regime(self):
        \"\"\"Обновляет режим рынка и адаптивные параметры\"\"\"
        try:
            # Определяем режим
            regime, confidence, details = self.regime_detector.detect_regime()
            
            # Если режим изменился - логируем
            if regime != self.current_regime:
                old_regime = self.current_regime.value if self.current_regime else "None"
                self.logger.info(
                    f"🔄 Режим рынка: {old_regime} → {regime.value.upper()} "
                    f"(уверенность: {confidence:.1%})"
                )
                self.current_regime = regime
            
            # Получаем адаптивные параметры
            self.current_adaptive_params = self.adaptive_config.get_params_for_regime(
                regime, confidence
            )
            
            # Применяем параметры
            config.SIGNAL_THRESHOLDS['min_confidence'] = self.current_adaptive_params['min_confidence']
            config.SIGNAL_THRESHOLDS['strong_confidence'] = self.current_adaptive_params['strong_confidence']
            config.MIN_PROFIT_TARGET_PERCENT = self.current_adaptive_params['min_profit_target']
            config.TRAILING_SL_ACTIVATION_PERCENT = self.current_adaptive_params['trailing_sl_activation']
            config.TRAILING_SL_CALLBACK_PERCENT = self.current_adaptive_params['trailing_sl_callback']
            config.SIGNAL_THRESHOLDS['min_volume_ratio'] = self.current_adaptive_params['min_volume_ratio']
            config.MIN_TIMEFRAME_ALIGNMENT = self.current_adaptive_params['min_timeframe_alignment']
            config.MAX_CONCURRENT_POSITIONS = self.current_adaptive_params['max_concurrent_positions']
            config.STOP_LOSS_MAX_USD = self.current_adaptive_params['stop_loss_max_usd']
            
            self.logger.debug(
                f"📊 Применены параметры для {regime.value}: "
                f"confidence={self.current_adaptive_params['min_confidence']:.0%}, "
                f"target={self.current_adaptive_params['min_profit_target']:.0f}%"
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка обновления режима рынка: {e}")
"""
    
    # Вставляем перед async def scan_and_trade
    scan_pattern = r"(    async def scan_and_trade\(self\):)"
    if re.search(scan_pattern, content):
        content = re.sub(
            scan_pattern,
            update_regime_method + "\n" + r"\1",
            content
        )
        print("✅ Добавлен метод update_market_regime()")
    
    # 4. Добавляем вызов обновления режима в scan_and_trade
    scan_start_code = """
        # 🧠 Обновляем режим рынка и адаптивные параметры
        self.update_market_regime()
        
"""
    
    # Вставляем после "Сканирование рынка и открытие позиций"
    scan_start_pattern = r'(async def scan_and_trade\(self\):\n        """Сканирование рынка и открытие позиций""")'
    if re.search(scan_start_pattern, content):
        content = re.sub(
            scan_start_pattern,
            r'\1\n' + scan_start_code,
            content
        )
        print("✅ Добавлен вызов update_market_regime() в scan_and_trade()")
    
    # 5. Добавляем запись результатов в close_position
    record_code = """
            # 🧠 Записываем результат для адаптивной системы
            if hasattr(self, 'adaptive_config') and self.current_regime:
                self.adaptive_config.record_trade_result(
                    regime=self.current_regime,
                    pnl=pnl_usd,
                    duration_seconds=duration,
                    params_used=self.current_adaptive_params or {}
                )
            
            if hasattr(self, 'performance_tracker'):
                self.performance_tracker.record_trade({
                    'symbol': symbol,
                    'regime': self.current_regime.value if self.current_regime else 'unknown',
                    'direction': pos['direction'],
                    'pnl': pnl_usd,
                    'pnl_percent': pnl_percent,
                    'duration': duration,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price if exit_price else pos['entry_price'],
                    'params': self.current_adaptive_params or {},
                    'config_name': f"{self.current_regime.value}_adaptive" if self.current_regime else 'unknown',
                    'reason': reason
                })
            
"""
    
    # Вставляем перед удалением позиции
    delete_pattern = r"(                # Удаляем из списка\n                del self\.open_positions\[symbol\])"
    if re.search(delete_pattern, content):
        content = re.sub(
            delete_pattern,
            record_code + r"\1",
            content
        )
        print("✅ Добавлена запись результатов в close_position()")
    
    # Сохраняем изменённый файл
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n" + "="*60)
    print("✅ ИНТЕГРАЦИЯ ЗАВЕРШЕНА!")
    print("="*60)
    print(f"\n📝 Бэкап сохранён: {backup_file}")
    print("\n🎯 Следующие шаги:")
    print("1. Проверьте main.py на наличие ошибок")
    print("2. Добавьте Telegram команды (см. ADAPTIVE_INTEGRATION_GUIDE.md)")
    print("3. Протестируйте локально")
    print("4. Разверните на сервере")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("🔧 ИНТЕГРАЦИЯ АДАПТИВНОЙ СИСТЕМЫ")
    print("="*60)
    print()
    
    success = integrate_adaptive_system()
    
    if success:
        print("\n✅ Готово!")
    else:
        print("\n❌ Ошибка интеграции")
