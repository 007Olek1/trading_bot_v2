"""
3️⃣ PERFORMANCE OPTIMIZATION ENGINE
Автоматический анализ производительности и оптимизация параметров
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np


class PerformanceOptimizer:
    """Оптимизация параметров торговли"""
    
    def __init__(self, results_file: str = None):
        """
        Args:
            results_file: Путь к файлу с результатами тестирования
        """
        self.results_file = results_file
        self.results = []
        
        if results_file and Path(results_file).exists():
            with open(results_file, 'r') as f:
                self.results = json.load(f)
    
    def analyze_performance(self) -> Dict:
        """Анализ производительности"""
        if not self.results:
            return {'error': 'Нет данных для анализа'}
        
        df = pd.DataFrame(self.results)
        
        analysis = {
            'total_trades': len(df),
            'winning_trades': len(df[df['pnl_percent'] > 0]),
            'losing_trades': len(df[df['pnl_percent'] <= 0]),
            'win_rate': len(df[df['pnl_percent'] > 0]) / len(df) * 100 if len(df) > 0 else 0,
            'avg_pnl': df['pnl_percent'].mean(),
            'max_profit': df['pnl_percent'].max(),
            'max_loss': df['pnl_percent'].min(),
            'avg_confidence': df['confidence'].mean(),
            'avg_timeframes_aligned': df['timeframes_aligned'].mean(),
        }
        
        # Анализ по направлениям
        if 'direction' in df.columns:
            long_trades = df[df['direction'] == 'LONG']
            short_trades = df[df['direction'] == 'SHORT']
            
            analysis['long_trades'] = len(long_trades)
            analysis['short_trades'] = len(short_trades)
            
            if len(long_trades) > 0:
                analysis['long_win_rate'] = len(long_trades[long_trades['pnl_percent'] > 0]) / len(long_trades) * 100
                analysis['long_avg_pnl'] = long_trades['pnl_percent'].mean()
            
            if len(short_trades) > 0:
                analysis['short_win_rate'] = len(short_trades[short_trades['pnl_percent'] > 0]) / len(short_trades) * 100
                analysis['short_avg_pnl'] = short_trades['pnl_percent'].mean()
        
        return analysis
    
    def optimize_confidence_threshold(self) -> Dict:
        """Оптимизация порога уверенности"""
        if not self.results:
            return {'error': 'Нет данных для анализа'}
        
        df = pd.DataFrame(self.results)
        
        thresholds = np.arange(0.50, 0.90, 0.05)
        best_threshold = 0.65
        best_win_rate = 0
        
        results = []
        
        for threshold in thresholds:
            filtered = df[df['confidence'] >= threshold]
            
            if len(filtered) == 0:
                continue
            
            win_rate = len(filtered[filtered['pnl_percent'] > 0]) / len(filtered) * 100
            avg_pnl = filtered['pnl_percent'].mean()
            
            results.append({
                'threshold': threshold,
                'trades': len(filtered),
                'win_rate': win_rate,
                'avg_pnl': avg_pnl
            })
            
            if win_rate > best_win_rate and len(filtered) >= 5:
                best_win_rate = win_rate
                best_threshold = threshold
        
        return {
            'best_threshold': best_threshold,
            'best_win_rate': best_win_rate,
            'analysis': results
        }
    
    def optimize_timeframe_alignment(self) -> Dict:
        """Оптимизация минимального количества таймфреймов"""
        if not self.results:
            return {'error': 'Нет данных для анализа'}
        
        df = pd.DataFrame(self.results)
        
        alignments = range(2, 6)
        best_alignment = 3
        best_win_rate = 0
        
        results = []
        
        for alignment in alignments:
            filtered = df[df['timeframes_aligned'] >= alignment]
            
            if len(filtered) == 0:
                continue
            
            win_rate = len(filtered[filtered['pnl_percent'] > 0]) / len(filtered) * 100
            avg_pnl = filtered['pnl_percent'].mean()
            
            results.append({
                'alignment': alignment,
                'trades': len(filtered),
                'win_rate': win_rate,
                'avg_pnl': avg_pnl
            })
            
            if win_rate > best_win_rate and len(filtered) >= 5:
                best_win_rate = win_rate
                best_alignment = alignment
        
        return {
            'best_alignment': best_alignment,
            'best_win_rate': best_win_rate,
            'analysis': results
        }
    
    def generate_recommendations(self) -> List[str]:
        """Генерация рекомендаций по оптимизации"""
        recommendations = []
        
        perf = self.analyze_performance()
        
        if perf.get('win_rate', 0) < 60:
            recommendations.append(
                "⚠️ Win rate ниже 60%. Рекомендуется:\n"
                "  - Увеличить порог уверенности (min_confidence)\n"
                "  - Увеличить минимальное количество таймфреймов (MIN_TIMEFRAME_ALIGNMENT)\n"
                "  - Пересмотреть параметры индикаторов"
            )
        
        if perf.get('avg_pnl', 0) < 2:
            recommendations.append(
                "⚠️ Средний PnL низкий. Рекомендуется:\n"
                "  - Пересмотреть уровни TP\n"
                "  - Увеличить размер позиции для прибыльных сигналов\n"
                "  - Использовать trailing TP более агрессивно"
            )
        
        # Анализ по направлениям
        if 'long_win_rate' in perf and 'short_win_rate' in perf:
            if abs(perf['long_win_rate'] - perf['short_win_rate']) > 20:
                better_direction = 'LONG' if perf['long_win_rate'] > perf['short_win_rate'] else 'SHORT'
                recommendations.append(
                    f"⚠️ Большая разница в win rate между LONG и SHORT.\n"
                    f"  - Рассмотрите торговлю только в направлении {better_direction}\n"
                    f"  - Или улучшите стратегию для противоположного направления"
                )
        
        # Оптимизация порога уверенности
        conf_opt = self.optimize_confidence_threshold()
        if 'best_threshold' in conf_opt:
            current_threshold = 0.65  # Из config
            if conf_opt['best_threshold'] != current_threshold:
                recommendations.append(
                    f"💡 Оптимальный порог уверенности: {conf_opt['best_threshold']:.2f}\n"
                    f"  - Текущий: {current_threshold:.2f}\n"
                    f"  - Ожидаемый win rate: {conf_opt['best_win_rate']:.1f}%"
                )
        
        # Оптимизация выравнивания таймфреймов
        tf_opt = self.optimize_timeframe_alignment()
        if 'best_alignment' in tf_opt:
            current_alignment = 3  # Из config
            if tf_opt['best_alignment'] != current_alignment:
                recommendations.append(
                    f"💡 Оптимальное количество таймфреймов: {tf_opt['best_alignment']}\n"
                    f"  - Текущее: {current_alignment}\n"
                    f"  - Ожидаемый win rate: {tf_opt['best_win_rate']:.1f}%"
                )
        
        if not recommendations:
            recommendations.append("✅ Параметры оптимальны, продолжайте в том же духе!")
        
        return recommendations
    
    def generate_report(self) -> str:
        """Генерация полного отчёта"""
        report = []
        
        report.append("="*80)
        report.append("3️⃣ PERFORMANCE OPTIMIZATION ENGINE")
        report.append("="*80)
        report.append("")
        
        # Анализ производительности
        perf = self.analyze_performance()
        
        report.append("📊 АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ")
        report.append("-"*80)
        report.append(f"Всего сделок: {perf.get('total_trades', 0)}")
        report.append(f"Прибыльных: {perf.get('winning_trades', 0)}")
        report.append(f"Убыточных: {perf.get('losing_trades', 0)}")
        report.append(f"Win Rate: {perf.get('win_rate', 0):.1f}%")
        report.append(f"Средний PnL: {perf.get('avg_pnl', 0):+.2f}%")
        report.append(f"Макс. прибыль: {perf.get('max_profit', 0):+.2f}%")
        report.append(f"Макс. убыток: {perf.get('max_loss', 0):+.2f}%")
        report.append("")
        
        if 'long_trades' in perf:
            report.append("📈 LONG сделки:")
            report.append(f"  Количество: {perf.get('long_trades', 0)}")
            report.append(f"  Win Rate: {perf.get('long_win_rate', 0):.1f}%")
            report.append(f"  Средний PnL: {perf.get('long_avg_pnl', 0):+.2f}%")
            report.append("")
        
        if 'short_trades' in perf:
            report.append("📉 SHORT сделки:")
            report.append(f"  Количество: {perf.get('short_trades', 0)}")
            report.append(f"  Win Rate: {perf.get('short_win_rate', 0):.1f}%")
            report.append(f"  Средний PnL: {perf.get('short_avg_pnl', 0):+.2f}%")
            report.append("")
        
        # Рекомендации
        report.append("="*80)
        report.append("💡 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ")
        report.append("="*80)
        report.append("")
        
        recommendations = self.generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")
            report.append("")
        
        report.append("="*80)
        
        return "\n".join(report)


def main():
    """Точка входа"""
    print("\n" + "="*80)
    print("3️⃣ PERFORMANCE OPTIMIZATION ENGINE")
    print("="*80 + "\n")
    
    # Ищем последний файл с результатами
    logs_dir = Path(__file__).parent / "logs"
    if logs_dir.exists():
        result_files = list(logs_dir.glob("test_results_*.json"))
        
        if result_files:
            latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
            print(f"📂 Найден файл с результатами: {latest_file.name}\n")
            
            optimizer = PerformanceOptimizer(str(latest_file))
            report = optimizer.generate_report()
            
            print(report)
            
            # Сохраняем отчёт
            report_file = logs_dir / f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"\n💾 Отчёт сохранён: {report_file}")
        else:
            print("⚠️ Файлы с результатами не найдены")
            print("Сначала запустите live_market_test.py для сбора данных")
    else:
        print("⚠️ Директория logs не найдена")
        print("Сначала запустите live_market_test.py для сбора данных")


if __name__ == "__main__":
    main()
