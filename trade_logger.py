"""
📊 TRADE LOGGER - Система логирования сделок
Сохраняет историю всех сделок с детальной информацией
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import gzip
import shutil


class TradeLogger:
    """Логирование и хранение истории сделок"""
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Файлы
        self.trades_file = self.logs_dir / "trades.json"
        self.trades_archive_dir = self.logs_dir / "trades_archive"
        self.trades_archive_dir.mkdir(exist_ok=True)
        
        # Статистика
        self.stats_file = self.logs_dir / "trading_stats.json"
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Инициализация
        self._init_files()
    
    def _init_files(self):
        """Инициализация файлов"""
        if not self.trades_file.exists():
            self._save_trades([])
        
        if not self.stats_file.exists():
            self._save_stats({
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl_usd": 0.0,
                "total_pnl_percent": 0.0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "last_updated": datetime.now(timezone.utc).isoformat()
            })
    
    def log_trade_open(self, symbol: str, direction: str, entry_price: float, 
                       quantity: float, leverage: int, confidence: float,
                       strategy: str = "unknown", mode: str = "swing") -> str:
        """
        Логирование открытия позиции
        
        Returns:
            trade_id: Уникальный ID сделки
        """
        trade_id = f"{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        trade = {
            "trade_id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "mode": mode,  # swing или scalping
            "strategy": strategy,
            "entry_price": entry_price,
            "quantity": quantity,
            "leverage": leverage,
            "confidence": confidence,
            "position_size_usd": (entry_price * quantity),
            "open_time": datetime.now(timezone.utc).isoformat(),
            "close_time": None,
            "exit_price": None,
            "pnl_usd": None,
            "pnl_percent": None,
            "roe_percent": None,
            "duration_minutes": None,
            "status": "open",
            "close_reason": None,
            "tp_hits": [],
            "max_profit_percent": 0.0,
            "max_drawdown_percent": 0.0
        }
        
        # Сохраняем
        trades = self._load_trades()
        trades.append(trade)
        self._save_trades(trades)
        
        self.logger.info(
            f"📝 Trade opened: {trade_id} | {symbol} {direction} @ ${entry_price:.4f} | "
            f"Mode: {mode} | Leverage: {leverage}x | Confidence: {confidence:.0%}"
        )
        
        return trade_id
    
    def log_trade_close(self, trade_id: str, exit_price: float, 
                       close_reason: str = "manual", tp_hits: List[str] = None):
        """Логирование закрытия позиции"""
        trades = self._load_trades()
        
        for trade in trades:
            if trade["trade_id"] == trade_id and trade["status"] == "open":
                # Рассчитываем результаты
                entry_price = trade["entry_price"]
                direction = trade["direction"]
                leverage = trade["leverage"]
                
                # PnL в процентах
                if direction == "LONG":
                    pnl_percent = ((exit_price - entry_price) / entry_price) * 100
                else:  # SHORT
                    pnl_percent = ((entry_price - exit_price) / entry_price) * 100
                
                # ROE с учетом плеча
                roe_percent = pnl_percent * leverage
                
                # PnL в USD
                position_size = trade["position_size_usd"]
                pnl_usd = position_size * (roe_percent / 100)
                
                # Длительность
                open_time = datetime.fromisoformat(trade["open_time"])
                close_time = datetime.now(timezone.utc)
                duration = (close_time - open_time).total_seconds() / 60
                
                # Обновляем сделку
                trade["exit_price"] = exit_price
                trade["pnl_usd"] = pnl_usd
                trade["pnl_percent"] = pnl_percent
                trade["roe_percent"] = roe_percent
                trade["close_time"] = close_time.isoformat()
                trade["duration_minutes"] = duration
                trade["status"] = "closed"
                trade["close_reason"] = close_reason
                trade["tp_hits"] = tp_hits or []
                
                self._save_trades(trades)
                self._update_stats()
                
                result = "✅ WIN" if pnl_usd > 0 else "❌ LOSS"
                self.logger.info(
                    f"📝 Trade closed: {trade_id} | {result} | "
                    f"PnL: ${pnl_usd:.2f} ({pnl_percent:+.2f}%) | "
                    f"ROE: {roe_percent:+.1f}% | Duration: {duration:.0f}m | "
                    f"Reason: {close_reason}"
                )
                
                return trade
        
        self.logger.warning(f"⚠️ Trade not found: {trade_id}")
        return None
    
    def update_trade_metrics(self, trade_id: str, max_profit: float = None, 
                            max_drawdown: float = None):
        """Обновление метрик открытой сделки"""
        trades = self._load_trades()
        
        for trade in trades:
            if trade["trade_id"] == trade_id and trade["status"] == "open":
                if max_profit is not None:
                    trade["max_profit_percent"] = max(
                        trade.get("max_profit_percent", 0), max_profit
                    )
                if max_drawdown is not None:
                    trade["max_drawdown_percent"] = min(
                        trade.get("max_drawdown_percent", 0), max_drawdown
                    )
                
                self._save_trades(trades)
                break
    
    def get_open_trades(self) -> List[Dict]:
        """Получить все открытые сделки"""
        trades = self._load_trades()
        return [t for t in trades if t["status"] == "open"]
    
    def get_closed_trades(self, limit: int = None) -> List[Dict]:
        """Получить закрытые сделки"""
        trades = self._load_trades()
        closed = [t for t in trades if t["status"] == "closed"]
        
        # Сортируем по времени закрытия (новые первые)
        closed.sort(key=lambda x: x["close_time"], reverse=True)
        
        if limit:
            return closed[:limit]
        return closed
    
    def get_stats(self) -> Dict:
        """Получить статистику"""
        return self._load_stats()
    
    def _update_stats(self):
        """Обновление статистики"""
        trades = self._load_trades()
        closed = [t for t in trades if t["status"] == "closed"]
        
        if not closed:
            return
        
        wins = [t for t in closed if t["pnl_usd"] > 0]
        losses = [t for t in closed if t["pnl_usd"] <= 0]
        
        total_pnl = sum(t["pnl_usd"] for t in closed)
        total_pnl_percent = sum(t["pnl_percent"] for t in closed)
        
        stats = {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "total_pnl_usd": total_pnl,
            "total_pnl_percent": total_pnl_percent,
            "win_rate": (len(wins) / len(closed) * 100) if closed else 0,
            "avg_win": (sum(t["pnl_usd"] for t in wins) / len(wins)) if wins else 0,
            "avg_loss": (sum(t["pnl_usd"] for t in losses) / len(losses)) if losses else 0,
            "best_trade": max((t["pnl_usd"] for t in closed), default=0),
            "worst_trade": min((t["pnl_usd"] for t in closed), default=0),
            "avg_duration_minutes": sum(t["duration_minutes"] for t in closed) / len(closed) if closed else 0,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        self._save_stats(stats)
    
    def _load_trades(self) -> List[Dict]:
        """Загрузка сделок из файла"""
        try:
            with open(self.trades_file, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def _save_trades(self, trades: List[Dict]):
        """Сохранение сделок в файл"""
        with open(self.trades_file, 'w') as f:
            json.dump(trades, f, indent=2)
    
    def _load_stats(self) -> Dict:
        """Загрузка статистики"""
        try:
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def _save_stats(self, stats: Dict):
        """Сохранение статистики"""
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def archive_old_trades(self, days: int = 30):
        """
        Архивирование старых сделок
        Переносит сделки старше N дней в архив
        """
        trades = self._load_trades()
        cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
        
        active_trades = []
        archive_trades = []
        
        for trade in trades:
            if trade["status"] == "open":
                active_trades.append(trade)
            else:
                close_time = datetime.fromisoformat(trade["close_time"])
                if close_time.timestamp() < cutoff:
                    archive_trades.append(trade)
                else:
                    active_trades.append(trade)
        
        if archive_trades:
            # Сохраняем архив
            archive_file = self.trades_archive_dir / f"trades_{datetime.now().strftime('%Y%m')}.json.gz"
            
            # Загружаем существующий архив если есть
            existing = []
            if archive_file.exists():
                with gzip.open(archive_file, 'rt') as f:
                    existing = json.load(f)
            
            # Добавляем новые
            existing.extend(archive_trades)
            
            # Сохраняем сжатый архив
            with gzip.open(archive_file, 'wt') as f:
                json.dump(existing, f)
            
            # Обновляем активные сделки
            self._save_trades(active_trades)
            
            self.logger.info(f"📦 Archived {len(archive_trades)} trades to {archive_file.name}")
    
    def cleanup_logs(self, keep_days: int = 7):
        """Очистка старых логов"""
        # Архивируем старые сделки
        self.archive_old_trades(days=30)
        
        # Удаляем старые архивы (старше keep_days)
        cutoff = datetime.now().timestamp() - (keep_days * 86400)
        
        for archive_file in self.trades_archive_dir.glob("*.json.gz"):
            if archive_file.stat().st_mtime < cutoff:
                archive_file.unlink()
                self.logger.info(f"🗑️ Deleted old archive: {archive_file.name}")
