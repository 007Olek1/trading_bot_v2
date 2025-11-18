"""
📊 DAILY REPORTER - Ежедневные отчёты
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List
import json
from pathlib import Path


class DailyReporter:
    """Генератор ежедневных отчётов"""
    
    def __init__(self, trades_log_file: Path):
        self.trades_log_file = trades_log_file
    
    def generate_daily_report(self) -> str:
        """Генерация ежедневного отчёта"""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)
        
        # Читаем сделки за последние 24 часа
        trades_24h = self._get_trades_last_24h()
        
        # Статистика
        total_trades = len(trades_24h)
        winning_trades = sum(1 for t in trades_24h if t.get('pnl_usd', 0) > 0)
        losing_trades = sum(1 for t in trades_24h if t.get('pnl_usd', 0) < 0)
        
        total_pnl = sum(t.get('pnl_usd', 0) for t in trades_24h)
        total_pnl_percent = sum(t.get('pnl_percent', 0) for t in trades_24h)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Лучшая и худшая сделка
        best_trade = max(trades_24h, key=lambda x: x.get('pnl_usd', 0)) if trades_24h else None
        worst_trade = min(trades_24h, key=lambda x: x.get('pnl_usd', 0)) if trades_24h else None
        
        # Формируем отчёт
        report = (
            f"📊 <b>ЕЖЕДНЕВНЫЙ ОТЧЁТ</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 {yesterday.strftime('%d.%m.%Y')} → {now.strftime('%d.%m.%Y')}\n"
            f"⏰ {now.strftime('%H:%M:%S UTC')}\n\n"
            f"<b>📈 ТОРГОВЛЯ ЗА 24 ЧАСА</b>\n"
            f"💼 Всего сделок: {total_trades}\n"
            f"✅ Прибыльных: {winning_trades}\n"
            f"❌ Убыточных: {losing_trades}\n"
            f"📊 Win Rate: {win_rate:.1f}%\n\n"
            f"<b>💰 ФИНАНСЫ</b>\n"
            f"💵 Общий PnL: ${total_pnl:.2f}\n"
            f"📊 Общий PnL: {total_pnl_percent:+.2f}%\n"
        )
        
        if best_trade:
            report += (
                f"\n<b>🏆 ЛУЧШАЯ СДЕЛКА</b>\n"
                f"💎 {best_trade.get('symbol', 'N/A')}\n"
                f"💰 PnL: ${best_trade.get('pnl_usd', 0):.2f} ({best_trade.get('pnl_percent', 0):+.2f}%)\n"
            )
        
        if worst_trade:
            report += (
                f"\n<b>📉 ХУДШАЯ СДЕЛКА</b>\n"
                f"💎 {worst_trade.get('symbol', 'N/A')}\n"
                f"💰 PnL: ${worst_trade.get('pnl_usd', 0):.2f} ({worst_trade.get('pnl_percent', 0):+.2f}%)\n"
            )
        
        report += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        return report
    
    def _get_trades_last_24h(self) -> List[Dict]:
        """Получение сделок за последние 24 часа"""
        if not self.trades_log_file.exists():
            return []
        
        trades = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=1)
        
        try:
            with open(self.trades_log_file, 'r') as f:
                for line in f:
                    try:
                        trade = json.loads(line.strip())
                        
                        # Проверяем время
                        trade_time_str = trade.get('close_time') or trade.get('timestamp')
                        if not trade_time_str:
                            continue
                        
                        trade_time = datetime.fromisoformat(trade_time_str.replace('Z', '+00:00'))
                        
                        if trade_time >= cutoff_time:
                            trades.append(trade)
                    except:
                        continue
        except Exception:
            pass
        
        return trades
    
    def get_weekly_summary(self) -> str:
        """Недельная сводка"""
        # Аналогично дневному отчёту, но за 7 дней
        pass
    
    def get_monthly_summary(self) -> str:
        """Месячная сводка"""
        # Аналогично дневному отчёту, но за 30 дней
        pass
