"""
⚙️ ADAPTIVE CONFIG MANAGER - Динамическое управление параметрами
Автоматически адаптирует параметры под текущий режим рынка
"""

from typing import Dict, Any
import logging
from market_regime_detector import MarketRegime
import json
from pathlib import Path
from datetime import datetime, timezone


class AdaptiveConfig:
    """
    Менеджер адаптивной конфигурации
    
    Автоматически подстраивает параметры торговли под:
    - Режим рынка (Bull/Bear/Sideways/Volatile)
    - Историческую производительность
    - Текущую волатильность
    """
    
    # Базовые конфигурации для каждого режима
    REGIME_CONFIGS = {
        MarketRegime.BULL: {
            'name': 'БЫЧИЙ РЫНОК 🟢',
            'description': 'Агрессивная торговля, высокие цели',
            'params': {
                'min_confidence': 0.70,           # Можно рисковать
                'strong_confidence': 0.82,
                'min_profit_target': 15.0,        # Высокие цели
                'trailing_sl_activation': 2.0,    # Раньше активируем
                'trailing_sl_callback': 1.8,
                'min_volume_ratio': 1.3,          # Мягче требования
                'min_timeframe_alignment': 3,     # 3 из 5
                'max_concurrent_positions': 3,    # Максимум позиций
                'stop_loss_max_usd': 1.0,
            }
        },
        
        MarketRegime.BEAR: {
            'name': 'МЕДВЕЖИЙ РЫНОК 🔴',
            'description': 'Консервативная торговля, защита капитала',
            'params': {
                'min_confidence': 0.82,           # Очень строго!
                'strong_confidence': 0.90,
                'min_profit_target': 8.0,         # Быстрые цели
                'trailing_sl_activation': 1.5,    # Защищаем прибыль
                'trailing_sl_callback': 2.5,      # Широкий откат
                'min_volume_ratio': 1.8,          # Строгие требования
                'min_timeframe_alignment': 5,     # Все таймфреймы!
                'max_concurrent_positions': 1,    # Минимум риска
                'stop_loss_max_usd': 0.8,         # Меньше риск
            }
        },
        
        MarketRegime.SIDEWAYS: {
            'name': 'БОКОВОЙ РЫНОК 🟡',
            'description': 'Сбалансированная торговля',
            'params': {
                'min_confidence': 0.75,
                'strong_confidence': 0.85,
                'min_profit_target': 12.0,
                'trailing_sl_activation': 2.5,
                'trailing_sl_callback': 2.0,
                'min_volume_ratio': 1.5,
                'min_timeframe_alignment': 4,
                'max_concurrent_positions': 2,
                'stop_loss_max_usd': 1.0,
            }
        },
        
        MarketRegime.VOLATILE: {
            'name': 'ВОЛАТИЛЬНЫЙ РЫНОК ⚡',
            'description': 'Осторожная торговля, быстрые выходы',
            'params': {
                'min_confidence': 0.80,           # Высокая уверенность
                'strong_confidence': 0.88,
                'min_profit_target': 10.0,        # Средние цели
                'trailing_sl_activation': 1.8,    # Быстро защищаем
                'trailing_sl_callback': 2.2,
                'min_volume_ratio': 1.6,
                'min_timeframe_alignment': 4,
                'max_concurrent_positions': 2,
                'stop_loss_max_usd': 0.9,
            }
        }
    }
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.current_regime = None
        self.current_params = None
        self.performance_history = []
        
        # Файл для сохранения истории производительности
        self.perf_file = Path("logs/adaptive_performance.json")
        self.perf_file.parent.mkdir(exist_ok=True)
        
        # Загружаем историю
        self._load_performance_history()
    
    def get_params_for_regime(self, regime: MarketRegime, 
                              confidence: float = 1.0) -> Dict[str, Any]:
        """
        Возвращает параметры для текущего режима
        
        Args:
            regime: Режим рынка
            confidence: Уверенность в режиме (0-1)
        
        Returns:
            Словарь с параметрами
        """
        # Базовые параметры для режима
        base_params = self.REGIME_CONFIGS[regime]['params'].copy()
        
        # Применяем динамическую оптимизацию на основе истории
        optimized_params = self._optimize_params(regime, base_params, confidence)
        
        # Сохраняем текущие параметры
        self.current_regime = regime
        self.current_params = optimized_params
        
        return optimized_params
    
    def _optimize_params(self, regime: MarketRegime, 
                        base_params: Dict, confidence: float) -> Dict:
        """
        Оптимизирует параметры на основе исторической производительности
        """
        # Если уверенность низкая - используем более консервативные настройки
        if confidence < 0.7:
            # Смешиваем с консервативными настройками
            conservative = self.REGIME_CONFIGS[MarketRegime.BEAR]['params']
            
            for key in base_params:
                if key in conservative:
                    # Интерполяция между базовыми и консервативными
                    base_val = base_params[key]
                    cons_val = conservative[key]
                    
                    # Чем ниже уверенность, тем ближе к консервативным
                    weight = confidence / 0.7  # 0.7 -> 1.0, <0.7 -> <1.0
                    base_params[key] = base_val * weight + cons_val * (1 - weight)
        
        # Анализируем историю производительности для этого режима
        regime_performance = self._get_regime_performance(regime)
        
        if regime_performance:
            win_rate = regime_performance.get('win_rate', 0.5)
            avg_pnl = regime_performance.get('avg_pnl', 0.0)
            
            # Если Win Rate низкий - усиливаем фильтры
            if win_rate < 0.45:
                base_params['min_confidence'] = min(base_params['min_confidence'] + 0.05, 0.90)
                base_params['min_timeframe_alignment'] = min(base_params['min_timeframe_alignment'] + 1, 5)
                self.logger.info(f"📊 Усилены фильтры для {regime.value} (Win Rate: {win_rate:.1%})")
            
            # Если Win Rate высокий - можно ослабить
            elif win_rate > 0.60:
                base_params['min_confidence'] = max(base_params['min_confidence'] - 0.03, 0.65)
                self.logger.info(f"📊 Ослаблены фильтры для {regime.value} (Win Rate: {win_rate:.1%})")
            
            # Если средний PnL отрицательный - увеличиваем цели
            if avg_pnl < 0:
                base_params['min_profit_target'] = min(base_params['min_profit_target'] + 2.0, 20.0)
                self.logger.info(f"📊 Увеличена цель прибыли для {regime.value} (Avg PnL: ${avg_pnl:.2f})")
        
        return base_params
    
    def _get_regime_performance(self, regime: MarketRegime) -> Dict:
        """Возвращает статистику производительности для режима"""
        if not self.performance_history:
            return {}
        
        # Фильтруем сделки для этого режима
        regime_trades = [t for t in self.performance_history if t.get('regime') == regime.value]
        
        if not regime_trades:
            return {}
        
        # Считаем статистику
        total = len(regime_trades)
        wins = sum(1 for t in regime_trades if t.get('pnl', 0) > 0)
        total_pnl = sum(t.get('pnl', 0) for t in regime_trades)
        
        return {
            'total_trades': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': wins / total if total > 0 else 0.5,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / total if total > 0 else 0.0
        }
    
    def record_trade_result(self, regime: MarketRegime, pnl: float, 
                           duration_seconds: float, params_used: Dict):
        """
        Записывает результат сделки для обучения системы
        
        Args:
            regime: Режим рынка во время сделки
            pnl: Прибыль/убыток в USD
            duration_seconds: Длительность сделки
            params_used: Параметры, использованные для сделки
        """
        trade_record = {
            'regime': regime.value,
            'pnl': pnl,
            'duration': duration_seconds,
            'params': params_used,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        self.performance_history.append(trade_record)
        
        # Ограничиваем размер истории
        if len(self.performance_history) > 500:
            self.performance_history = self.performance_history[-500:]
        
        # Сохраняем
        self._save_performance_history()
        
        self.logger.info(
            f"📝 Записан результат сделки: {regime.value} | "
            f"PnL: ${pnl:.2f} | Длительность: {duration_seconds/3600:.1f}ч"
        )
    
    def _save_performance_history(self):
        """Сохраняет историю производительности"""
        try:
            with open(self.perf_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения истории производительности: {e}")
    
    def _load_performance_history(self):
        """Загружает историю производительности"""
        try:
            if self.perf_file.exists():
                with open(self.perf_file, 'r') as f:
                    self.performance_history = json.load(f)
                self.logger.info(f"📊 Загружено {len(self.performance_history)} записей производительности")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки истории производительности: {e}")
    
    def get_regime_info(self, regime: MarketRegime) -> Dict:
        """Возвращает информацию о режиме"""
        config = self.REGIME_CONFIGS[regime]
        performance = self._get_regime_performance(regime)
        
        return {
            'name': config['name'],
            'description': config['description'],
            'params': config['params'],
            'performance': performance
        }
    
    def get_all_regimes_stats(self) -> Dict:
        """Возвращает статистику по всем режимам"""
        stats = {}
        
        for regime in MarketRegime:
            stats[regime.value] = {
                'name': self.REGIME_CONFIGS[regime]['name'],
                'performance': self._get_regime_performance(regime)
            }
        
        return stats
    
    def suggest_manual_override(self, current_performance: Dict) -> Dict:
        """
        Предлагает ручную корректировку параметров на основе текущей производительности
        
        Returns:
            Словарь с рекомендациями
        """
        suggestions = []
        
        win_rate = current_performance.get('win_rate', 0.5)
        avg_pnl = current_performance.get('avg_pnl', 0.0)
        
        if win_rate < 0.40:
            suggestions.append({
                'issue': 'Низкий Win Rate',
                'current': f'{win_rate:.1%}',
                'suggestion': 'Увеличить min_confidence до 0.80+',
                'priority': 'HIGH'
            })
        
        if avg_pnl < -0.05:
            suggestions.append({
                'issue': 'Отрицательный средний PnL',
                'current': f'${avg_pnl:.2f}',
                'suggestion': 'Увеличить min_profit_target или уменьшить max_positions',
                'priority': 'HIGH'
            })
        
        if win_rate > 0.65 and avg_pnl > 0.15:
            suggestions.append({
                'issue': 'Отличные результаты',
                'current': f'Win Rate: {win_rate:.1%}, Avg PnL: ${avg_pnl:.2f}',
                'suggestion': 'Можно немного ослабить фильтры для большего количества сделок',
                'priority': 'LOW'
            })
        
        return {
            'suggestions': suggestions,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
