"""
4️⃣ TRADE ANALYSIS AND REPORTING PIPELINE
Пост-анализ закрытых позиций и извлечение инсайтов
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


class TradeAnalyzer:
    """Анализ торговых результатов"""
    
    def __init__(self, results_file: str = None):
        """
        Args:
            results_file: Путь к файлу с результатами
        """
        self.results_file = results_file
        self.trades = []
        
        if results_file and Path(results_file).exists():
            with open(results_file, 'r') as f:
                self.trades = json.load(f)
    
    def calculate_metrics(self) -> Dict:
        """Расчёт ключевых метрик"""
        if not self.trades:
            return {}
        
        df = pd.DataFrame(self.trades)
        
        # Базовые метрики
        total_trades = len(df)
        winning_trades = len(df[df['pnl_percent'] > 0])
        losing_trades = len(df[df['pnl_percent'] <= 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # PnL метрики
        total_pnl = df['pnl_percent'].sum()
        avg_pnl = df['pnl_percent'].mean()
        median_pnl = df['pnl_percent'].median()
        
        avg_win = df[df['pnl_percent'] > 0]['pnl_percent'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl_percent'] <= 0]['pnl_percent'].mean() if losing_trades > 0 else 0
        
        # Profit Factor
        total_profit = df[df['pnl_percent'] > 0]['pnl_percent'].sum()
        total_loss = abs(df[df['pnl_percent'] <= 0]['pnl_percent'].sum())
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
        
        # Максимальные значения
        max_win = df['pnl_percent'].max()
        max_loss = df['pnl_percent'].min()
        
        # Серии
        max_win_streak = self._calculate_max_streak(df, True)
        max_loss_streak = self._calculate_max_streak(df, False)
        
        # Drawdown
        cumulative_pnl = df['pnl_percent'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'median_pnl': median_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_win': max_win,
            'max_loss': max_loss,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'max_drawdown': max_drawdown
        }
    
    def _calculate_max_streak(self, df: pd.DataFrame, winning: bool) -> int:
        """Расчёт максимальной серии побед/поражений"""
        if winning:
            results = (df['pnl_percent'] > 0).astype(int)
        else:
            results = (df['pnl_percent'] <= 0).astype(int)
        
        max_streak = 0
        current_streak = 0
        
        for result in results:
            if result == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def analyze_by_direction(self) -> Dict:
        """Анализ по направлениям (LONG/SHORT)"""
        if not self.trades:
            return {}
        
        df = pd.DataFrame(self.trades)
        
        analysis = {}
        
        for direction in ['LONG', 'SHORT']:
            direction_df = df[df['direction'] == direction]
            
            if len(direction_df) == 0:
                continue
            
            total = len(direction_df)
            winning = len(direction_df[direction_df['pnl_percent'] > 0])
            
            analysis[direction] = {
                'total_trades': total,
                'winning_trades': winning,
                'losing_trades': total - winning,
                'win_rate': (winning / total * 100) if total > 0 else 0,
                'avg_pnl': direction_df['pnl_percent'].mean(),
                'total_pnl': direction_df['pnl_percent'].sum(),
                'max_win': direction_df['pnl_percent'].max(),
                'max_loss': direction_df['pnl_percent'].min()
            }
        
        return analysis
    
    def analyze_by_confidence(self) -> Dict:
        """Анализ по уровням уверенности"""
        if not self.trades:
            return {}
        
        df = pd.DataFrame(self.trades)
        
        # Группируем по диапазонам уверенности
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
        
        df['confidence_range'] = pd.cut(df['confidence'], bins=bins, labels=labels)
        
        analysis = {}
        
        for conf_range in labels:
            range_df = df[df['confidence_range'] == conf_range]
            
            if len(range_df) == 0:
                continue
            
            total = len(range_df)
            winning = len(range_df[range_df['pnl_percent'] > 0])
            
            analysis[conf_range] = {
                'total_trades': total,
                'win_rate': (winning / total * 100) if total > 0 else 0,
                'avg_pnl': range_df['pnl_percent'].mean()
            }
        
        return analysis
    
    def analyze_by_timeframe_alignment(self) -> Dict:
        """Анализ по количеству подтверждающих таймфреймов"""
        if not self.trades:
            return {}
        
        df = pd.DataFrame(self.trades)
        
        analysis = {}
        
        for alignment in sorted(df['timeframes_aligned'].unique()):
            align_df = df[df['timeframes_aligned'] == alignment]
            
            total = len(align_df)
            winning = len(align_df[align_df['pnl_percent'] > 0])
            
            analysis[f'{int(alignment)} TF'] = {
                'total_trades': total,
                'win_rate': (winning / total * 100) if total > 0 else 0,
                'avg_pnl': align_df['pnl_percent'].mean()
            }
        
        return analysis
    
    def find_best_patterns(self) -> List[Dict]:
        """Поиск лучших паттернов для торговли"""
        if not self.trades:
            return []
        
        df = pd.DataFrame(self.trades)
        
        patterns = []
        
        # Паттерн 1: Высокая уверенность + много таймфреймов
        high_conf_multi_tf = df[
            (df['confidence'] >= 0.75) & 
            (df['timeframes_aligned'] >= 4)
        ]
        
        if len(high_conf_multi_tf) > 0:
            win_rate = len(high_conf_multi_tf[high_conf_multi_tf['pnl_percent'] > 0]) / len(high_conf_multi_tf) * 100
            patterns.append({
                'name': 'Высокая уверенность (≥75%) + 4+ таймфреймов',
                'trades': len(high_conf_multi_tf),
                'win_rate': win_rate,
                'avg_pnl': high_conf_multi_tf['pnl_percent'].mean()
            })
        
        # Паттерн 2: Лучшее направление
        direction_analysis = self.analyze_by_direction()
        if direction_analysis:
            best_direction = max(direction_analysis.items(), key=lambda x: x[1]['win_rate'])
            patterns.append({
                'name': f'Только {best_direction[0]} сделки',
                'trades': best_direction[1]['total_trades'],
                'win_rate': best_direction[1]['win_rate'],
                'avg_pnl': best_direction[1]['avg_pnl']
            })
        
        return patterns
    
    def generate_insights(self) -> List[str]:
        """Генерация инсайтов из анализа"""
        insights = []
        
        metrics = self.calculate_metrics()
        
        # Общая производительность
        if metrics.get('win_rate', 0) >= 70:
            insights.append(f"✅ Отличный win rate: {metrics['win_rate']:.1f}%")
        elif metrics.get('win_rate', 0) >= 60:
            insights.append(f"👍 Хороший win rate: {metrics['win_rate']:.1f}%")
        else:
            insights.append(f"⚠️ Низкий win rate: {metrics['win_rate']:.1f}%. Требуется оптимизация")
        
        # Profit Factor
        if metrics.get('profit_factor', 0) >= 2.0:
            insights.append(f"✅ Отличный Profit Factor: {metrics['profit_factor']:.2f}")
        elif metrics.get('profit_factor', 0) >= 1.5:
            insights.append(f"👍 Хороший Profit Factor: {metrics['profit_factor']:.2f}")
        else:
            insights.append(f"⚠️ Низкий Profit Factor: {metrics['profit_factor']:.2f}")
        
        # Средний PnL
        if metrics.get('avg_pnl', 0) >= 3:
            insights.append(f"✅ Отличный средний PnL: {metrics['avg_pnl']:+.2f}%")
        elif metrics.get('avg_pnl', 0) >= 1.5:
            insights.append(f"👍 Хороший средний PnL: {metrics['avg_pnl']:+.2f}%")
        else:
            insights.append(f"⚠️ Низкий средний PnL: {metrics['avg_pnl']:+.2f}%")
        
        # Анализ по направлениям
        direction_analysis = self.analyze_by_direction()
        if 'LONG' in direction_analysis and 'SHORT' in direction_analysis:
            long_wr = direction_analysis['LONG']['win_rate']
            short_wr = direction_analysis['SHORT']['win_rate']
            
            if abs(long_wr - short_wr) > 20:
                better = 'LONG' if long_wr > short_wr else 'SHORT'
                insights.append(
                    f"💡 {better} сделки значительно лучше "
                    f"({direction_analysis[better]['win_rate']:.1f}% vs "
                    f"{direction_analysis['SHORT' if better == 'LONG' else 'LONG']['win_rate']:.1f}%)"
                )
        
        # Лучшие паттерны
        patterns = self.find_best_patterns()
        if patterns:
            best_pattern = max(patterns, key=lambda x: x['win_rate'])
            insights.append(
                f"🎯 Лучший паттерн: {best_pattern['name']} "
                f"(Win Rate: {best_pattern['win_rate']:.1f}%, "
                f"Avg PnL: {best_pattern['avg_pnl']:+.2f}%)"
            )
        
        return insights
    
    def generate_report(self) -> str:
        """Генерация полного отчёта"""
        report = []
        
        report.append("="*80)
        report.append("4️⃣ TRADE ANALYSIS AND REPORTING PIPELINE")
        report.append("="*80)
        report.append("")
        
        # Основные метрики
        metrics = self.calculate_metrics()
        
        report.append("📊 ОСНОВНЫЕ МЕТРИКИ")
        report.append("-"*80)
        report.append(f"Всего сделок: {metrics.get('total_trades', 0)}")
        report.append(f"Прибыльных: {metrics.get('winning_trades', 0)}")
        report.append(f"Убыточных: {metrics.get('losing_trades', 0)}")
        report.append(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
        report.append("")
        report.append(f"Общий PnL: {metrics.get('total_pnl', 0):+.2f}%")
        report.append(f"Средний PnL: {metrics.get('avg_pnl', 0):+.2f}%")
        report.append(f"Медианный PnL: {metrics.get('median_pnl', 0):+.2f}%")
        report.append("")
        report.append(f"Средняя прибыль: {metrics.get('avg_win', 0):+.2f}%")
        report.append(f"Средний убыток: {metrics.get('avg_loss', 0):+.2f}%")
        report.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        report.append("")
        report.append(f"Макс. прибыль: {metrics.get('max_win', 0):+.2f}%")
        report.append(f"Макс. убыток: {metrics.get('max_loss', 0):+.2f}%")
        report.append(f"Макс. серия побед: {metrics.get('max_win_streak', 0)}")
        report.append(f"Макс. серия поражений: {metrics.get('max_loss_streak', 0)}")
        report.append(f"Макс. просадка: {metrics.get('max_drawdown', 0):+.2f}%")
        report.append("")
        
        # Анализ по направлениям
        direction_analysis = self.analyze_by_direction()
        if direction_analysis:
            report.append("="*80)
            report.append("📈 АНАЛИЗ ПО НАПРАВЛЕНИЯМ")
            report.append("-"*80)
            
            for direction, data in direction_analysis.items():
                report.append(f"\n{direction}:")
                report.append(f"  Сделок: {data['total_trades']}")
                report.append(f"  Win Rate: {data['win_rate']:.1f}%")
                report.append(f"  Средний PnL: {data['avg_pnl']:+.2f}%")
                report.append(f"  Общий PnL: {data['total_pnl']:+.2f}%")
            report.append("")
        
        # Анализ по уверенности
        confidence_analysis = self.analyze_by_confidence()
        if confidence_analysis:
            report.append("="*80)
            report.append("🎯 АНАЛИЗ ПО УВЕРЕННОСТИ")
            report.append("-"*80)
            
            for conf_range, data in confidence_analysis.items():
                report.append(f"\n{conf_range}:")
                report.append(f"  Сделок: {data['total_trades']}")
                report.append(f"  Win Rate: {data['win_rate']:.1f}%")
                report.append(f"  Средний PnL: {data['avg_pnl']:+.2f}%")
            report.append("")
        
        # Анализ по таймфреймам
        tf_analysis = self.analyze_by_timeframe_alignment()
        if tf_analysis:
            report.append("="*80)
            report.append("📊 АНАЛИЗ ПО ТАЙМФРЕЙМАМ")
            report.append("-"*80)
            
            for tf, data in tf_analysis.items():
                report.append(f"\n{tf}:")
                report.append(f"  Сделок: {data['total_trades']}")
                report.append(f"  Win Rate: {data['win_rate']:.1f}%")
                report.append(f"  Средний PnL: {data['avg_pnl']:+.2f}%")
            report.append("")
        
        # Инсайты
        report.append("="*80)
        report.append("💡 КЛЮЧЕВЫЕ ИНСАЙТЫ")
        report.append("="*80)
        report.append("")
        
        insights = self.generate_insights()
        for i, insight in enumerate(insights, 1):
            report.append(f"{i}. {insight}")
        
        report.append("")
        report.append("="*80)
        
        return "\n".join(report)


def main():
    """Точка входа"""
    print("\n" + "="*80)
    print("4️⃣ TRADE ANALYSIS AND REPORTING PIPELINE")
    print("="*80 + "\n")
    
    # Ищем последний файл с результатами
    logs_dir = Path(__file__).parent / "logs"
    if logs_dir.exists():
        result_files = list(logs_dir.glob("test_results_*.json"))
        
        if result_files:
            latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
            print(f"📂 Найден файл с результатами: {latest_file.name}\n")
            
            analyzer = TradeAnalyzer(str(latest_file))
            report = analyzer.generate_report()
            
            print(report)
            
            # Сохраняем отчёт
            report_file = logs_dir / f"trade_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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
