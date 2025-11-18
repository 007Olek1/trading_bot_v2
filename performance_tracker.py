"""
📊 PERFORMANCE TRACKER - A/B тестирование и аналитика
Отслеживает эффективность разных конфигураций и режимов
"""

import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import logging
from collections import defaultdict


class PerformanceTracker:
    """Трекер производительности для A/B тестирования"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.trades_file = Path("logs/trades_detailed.json")
        self.trades_file.parent.mkdir(exist_ok=True)
        
        self.trades = []
        self._load_trades()
    
    def record_trade(self, trade_data: Dict):
        """Записывает детали сделки"""
        trade_data['recorded_at'] = datetime.now(timezone.utc).isoformat()
        self.trades.append(trade_data)
        
        if len(self.trades) > 1000:
            self.trades = self.trades[-1000:]
        
        self._save_trades()
    
    def get_performance_by_regime(self, days: int = 7) -> Dict:
        """Статистика по режимам за последние N дней"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent_trades = [t for t in self.trades 
                        if datetime.fromisoformat(t['recorded_at']) > cutoff]
        
        stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'total_pnl': 0.0})
        
        for trade in recent_trades:
            regime = trade.get('regime', 'unknown')
            stats[regime]['trades'] += 1
            if trade.get('pnl', 0) > 0:
                stats[regime]['wins'] += 1
            stats[regime]['total_pnl'] += trade.get('pnl', 0)
        
        # Рассчитываем метрики
        for regime, data in stats.items():
            if data['trades'] > 0:
                data['win_rate'] = data['wins'] / data['trades']
                data['avg_pnl'] = data['total_pnl'] / data['trades']
        
        return dict(stats)
    
    def compare_configs(self, config_a: str, config_b: str, days: int = 7) -> Dict:
        """Сравнивает две конфигурации"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        a_trades = [t for t in self.trades 
                   if t.get('config_name') == config_a 
                   and datetime.fromisoformat(t['recorded_at']) > cutoff]
        
        b_trades = [t for t in self.trades 
                   if t.get('config_name') == config_b 
                   and datetime.fromisoformat(t['recorded_at']) > cutoff]
        
        def calc_stats(trades):
            if not trades:
                return {}
            total = len(trades)
            wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            return {
                'total_trades': total,
                'win_rate': wins / total if total > 0 else 0,
                'total_pnl': total_pnl,
                'avg_pnl': total_pnl / total if total > 0 else 0
            }
        
        return {
            'config_a': {'name': config_a, 'stats': calc_stats(a_trades)},
            'config_b': {'name': config_b, 'stats': calc_stats(b_trades)},
            'winner': config_a if calc_stats(a_trades).get('avg_pnl', 0) > 
                               calc_stats(b_trades).get('avg_pnl', 0) else config_b
        }
    
    def get_best_params_for_regime(self, regime: str) -> Dict:
        """Находит лучшие параметры для режима"""
        regime_trades = [t for t in self.trades if t.get('regime') == regime]
        
        if not regime_trades:
            return {}
        
        # Группируем по параметрам
        param_groups = defaultdict(list)
        for trade in regime_trades:
            params_key = json.dumps(trade.get('params', {}), sort_keys=True)
            param_groups[params_key].append(trade.get('pnl', 0))
        
        # Находим лучшую группу
        best_params = None
        best_avg = float('-inf')
        
        for params_str, pnls in param_groups.items():
            avg_pnl = sum(pnls) / len(pnls)
            if avg_pnl > best_avg and len(pnls) >= 3:  # Минимум 3 сделки
                best_avg = avg_pnl
                best_params = json.loads(params_str)
        
        return {
            'params': best_params,
            'avg_pnl': best_avg,
            'sample_size': len(param_groups.get(json.dumps(best_params, sort_keys=True), []))
        }
    
    def _save_trades(self):
        """Сохраняет сделки"""
        try:
            with open(self.trades_file, 'w') as f:
                json.dump(self.trades, f, indent=2)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения сделок: {e}")
    
    def _load_trades(self):
        """Загружает сделки"""
        try:
            if self.trades_file.exists():
                with open(self.trades_file, 'r') as f:
                    self.trades = json.load(f)
                self.logger.info(f"📊 Загружено {len(self.trades)} сделок")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки сделок: {e}")
