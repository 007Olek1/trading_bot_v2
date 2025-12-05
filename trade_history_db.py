#!/usr/bin/env python3
"""
Trade History Database - История сделок с авторотацией 72 часа
"""

import sqlite3
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from threading import RLock

logger = logging.getLogger(__name__)

DB_PATH = '/opt/bot/data/trade_history.db'
RETENTION_HOURS = 72  # Хранить историю 72 часа


class TradeHistoryDB:
    """База данных истории сделок с авторотацией"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = RLock()
        self._ensure_db_exists()
        self._conn = self._create_connection()
        self._create_tables()
        self._add_missing_columns()
        self._cleanup_old_records()
        logger.info(f"TradeHistoryDB инициализирована: {db_path}")
    
    def _ensure_db_exists(self):
        """Создать директорию если не существует"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _create_connection(self):
        """Создать единое соединение с БД"""
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _get_connection(self):
        return self._conn
    
    def _create_tables(self):
        """Создать таблицы"""
        conn = self._get_connection()
        with self._lock:
            cursor = conn.cursor()
            # Таблица сделок
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    pnl_usd REAL,
                    pnl_pct REAL,
                    reason TEXT,
                    signal_strength INTEGER,
                    disco_confidence REAL,
                    trailing_activated INTEGER DEFAULT 0,
                    duration_sec REAL,
                    status TEXT DEFAULT 'open',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    closed_at REAL
                )
            ''')
            # Таблица статистики по дням
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    best_trade REAL DEFAULT 0,
                    worst_trade REAL DEFAULT 0,
                    avg_duration_sec REAL DEFAULT 0
                )
            ''')
            # Индексы для быстрого поиска
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)')
            conn.commit()
    
    def _add_missing_columns(self):
        """Добавить недостающие колонки при обновлении схемы"""
        conn = self._get_connection()
        with self._lock:
            cursor = conn.cursor()
            cursor.execute('PRAGMA table_info(trades)')
            columns = {row[1] for row in cursor.fetchall()}

            if 'closed_at' not in columns:
                cursor.execute('ALTER TABLE trades ADD COLUMN closed_at REAL')
                logger.info("Добавлена колонка closed_at в таблицу trades")

            conn.commit()

    def _cleanup_old_records(self):
        """Удалить записи старше 72 часов"""
        conn = self._get_connection()
        with self._lock:
            cursor = conn.cursor()
            cutoff_time = time.time() - (RETENTION_HOURS * 3600)
            cursor.execute('DELETE FROM trades WHERE timestamp < ?', (cutoff_time,))
            deleted = cursor.rowcount
            conn.commit()
        if deleted > 0:
            logger.info(f"Удалено {deleted} старых записей (старше {RETENTION_HOURS}ч)")
    
    def add_trade_open(self, symbol: str, side: str, entry_price: float, 
                       quantity: float, signal_strength: int = 0, 
                       disco_confidence: float = 0) -> int:
        """Добавить открытую сделку"""
        conn = self._get_connection()
        with self._lock:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades (timestamp, symbol, side, entry_price, quantity, 
                                   signal_strength, disco_confidence, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'open')
            ''', (time.time(), symbol, side, entry_price, quantity, 
                  signal_strength, disco_confidence))
            trade_id = cursor.lastrowid
            conn.commit()
        logger.debug(f"Сделка открыта в БД: {symbol} {side} #{trade_id}")
        return trade_id
    
    def close_trade(self, symbol: str, exit_price: float, pnl_usd: float, 
                    reason: str, trailing_activated: bool = False):
        """Закрыть сделку"""
        conn = self._get_connection()
        with self._lock:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, timestamp, entry_price FROM trades 
                WHERE symbol = ? AND status = 'open' 
                ORDER BY timestamp DESC LIMIT 1
            ''', (symbol,))
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Открытая сделка не найдена: {symbol}")
                return

            trade_id, open_time, entry_price = row
            duration = time.time() - open_time
            pnl_pct = (exit_price - entry_price) / entry_price * 100 if entry_price > 0 else 0

            cursor.execute('''
                UPDATE trades SET 
                    exit_price = ?,
                    pnl_usd = ?,
                    pnl_pct = ?,
                    reason = ?,
                    trailing_activated = ?,
                    duration_sec = ?,
                    status = 'closed',
                    closed_at = ?
                WHERE id = ?
            ''', (exit_price, pnl_usd, pnl_pct, reason, 
                  1 if trailing_activated else 0, duration, time.time(), trade_id))
            conn.commit()

        # Обновить дневную статистику (отдельно, чтобы избежать долгой блокировки)
        self._update_daily_stats(pnl_usd, duration)
        logger.debug(f"Сделка закрыта в БД: {symbol} PnL=${pnl_usd:.2f}")
    
    def _update_daily_stats(self, pnl_usd: float, duration: float):
        """Обновить дневную статистику"""
        with self._lock:
            cursor = self._conn.cursor()
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute('''
                INSERT INTO daily_stats (date, total_trades, winning_trades, losing_trades, 
                                        total_pnl, best_trade, worst_trade, avg_duration_sec)
                VALUES (?, 1, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    total_trades = total_trades + 1,
                    winning_trades = winning_trades + ?,
                    losing_trades = losing_trades + ?,
                    total_pnl = total_pnl + ?,
                    best_trade = MAX(best_trade, ?),
                    worst_trade = MIN(worst_trade, ?),
                    avg_duration_sec = (avg_duration_sec * (total_trades - 1) + ?) / total_trades
            ''', (today, 
                  1 if pnl_usd > 0 else 0, 
                  1 if pnl_usd <= 0 else 0,
                  pnl_usd, pnl_usd, pnl_usd, duration,
                  1 if pnl_usd > 0 else 0,
                  1 if pnl_usd <= 0 else 0,
                  pnl_usd, pnl_usd, pnl_usd, duration))
            self._conn.commit()
    
    def get_recent_trades(self, hours: int = 24, limit: int = 50) -> List[Dict]:
        """Получить последние сделки"""
        with self._lock:
            cursor = self._conn.cursor()
            cutoff = time.time() - (hours * 3600)
            cursor.execute('''
                SELECT symbol, side, entry_price, exit_price, pnl_usd, pnl_pct, 
                       reason, duration_sec, timestamp, status
                FROM trades 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (cutoff, limit))
            rows = cursor.fetchall()

        trades = [{
            'symbol': row[0],
            'side': row[1],
            'entry_price': row[2],
            'exit_price': row[3],
            'pnl_usd': row[4],
            'pnl_pct': row[5],
            'reason': row[6],
            'duration_sec': row[7],
            'timestamp': row[8],
            'status': row[9]
        } for row in rows]
        return trades
    
    def get_stats(self, hours: int = 24) -> Dict:
        """Получить статистику за период"""
        with self._lock:
            cursor = self._conn.cursor()
            cutoff = time.time() - (hours * 3600)
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl_usd <= 0 THEN 1 ELSE 0 END) as losses,
                    COALESCE(SUM(pnl_usd), 0) as total_pnl,
                    COALESCE(MAX(pnl_usd), 0) as best,
                    COALESCE(MIN(pnl_usd), 0) as worst,
                    COALESCE(AVG(duration_sec), 0) as avg_duration
                FROM trades 
                WHERE timestamp > ? AND status = 'closed'
            ''', (cutoff,))
            row = cursor.fetchone()
        
        total = row[0] or 0
        wins = row[1] or 0
        
        return {
            'total_trades': total,
            'winning_trades': wins,
            'losing_trades': row[2] or 0,
            'win_rate': (wins / total * 100) if total > 0 else 0,
            'total_pnl': row[3] or 0,
            'best_trade': row[4] or 0,
            'worst_trade': row[5] or 0,
            'avg_duration_min': (row[6] or 0) / 60
        }
    
    def get_open_trades(self) -> List[Dict]:
        """Получить открытые сделки"""
        conn = self._get_connection()
        with self._lock:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT symbol, side, entry_price, quantity, timestamp, signal_strength
                FROM trades WHERE status = 'open'
            ''')
            rows = cursor.fetchall()

        trades = [{
            'symbol': row[0],
            'side': row[1],
            'entry_price': row[2],
            'quantity': row[3],
            'timestamp': row[4],
            'signal_strength': row[5]
        } for row in rows]
        return trades

    def close(self):
        with self._lock:
            if getattr(self, '_conn', None):
                self._conn.close()
                self._conn = None


# Тест
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    db = TradeHistoryDB('./test_trade_history.db')
    
    # Тест добавления сделки
    trade_id = db.add_trade_open('BTC/USDT:USDT', 'long', 95000, 0.001, 4, 0.75)
    print(f"Открыта сделка #{trade_id}")
    
    # Тест закрытия
    db.close_trade('BTC/USDT:USDT', 95500, 0.50, 'TP')
    
    # Тест статистики
    stats = db.get_stats(24)
    print(f"Статистика: {stats}")
    
    # Тест истории
    trades = db.get_recent_trades(24)
    print(f"Последние сделки: {trades}")
