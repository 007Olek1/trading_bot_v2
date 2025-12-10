#!/usr/bin/env python3
"""
Trade History Database - История сделок с авторотацией 72 часа
"""

import logging
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Dict, List

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None

logger = logging.getLogger(__name__)

DB_PATH = '/opt/bot/data/trade_history.db'
RETENTION_HOURS = 72  # Хранить историю 72 часа


class TradeHistoryDB:
    """База данных истории сделок с поддержкой SQLite и PostgreSQL"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_provider = os.getenv('TRADE_DB_PROVIDER', 'sqlite').lower()
        self.using_postgres = self.db_provider == 'postgres'
        self.db_path = os.getenv('SQLITE_DB_PATH', db_path)
        self._lock = RLock()
        self._conn = self._create_connection()
        self._create_tables()
        self._add_missing_columns()
        self._cleanup_old_records()
        provider_str = "PostgreSQL" if self.using_postgres else f"SQLite ({db_path})"
        logger.info(f"TradeHistoryDB активна: {provider_str}")
    
    # --------------------------------------------------------------------- #
    # INITIALIZATION
    # --------------------------------------------------------------------- #
    def _create_connection(self):
        if self.using_postgres:
            if psycopg is None:
                raise RuntimeError(
                    "psycopg не установлен, но TRADE_DB_PROVIDER=postgres. "
                    "Добавьте psycopg[binary] в зависимости."
                )
            return self._create_postgres_connection()
        
        self._ensure_sqlite_dir()
        return self._create_sqlite_connection()
    
    def _ensure_sqlite_dir(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _create_sqlite_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn
    
    def _create_postgres_connection(self):
        host = os.getenv('PG_HOST', '127.0.0.1')
        port = int(os.getenv('PG_PORT', '5432'))
        database = os.getenv('PG_NAME')
        user = os.getenv('PG_USER')
        password = os.getenv('PG_PASSWORD')
        sslmode = os.getenv('PG_SSLMODE', 'disable')
        
        if not all([database, user, password]):
            raise RuntimeError("PG_NAME/PG_USER/PG_PASSWORD должны быть заданы в .env")
        
        conn = psycopg.connect(
            host=host,
            port=port,
            dbname=database,
            user=user,
            password=password,
            sslmode=sslmode,
        )
        conn.autocommit = False
        return conn
    
    # --------------------------------------------------------------------- #
    # SCHEMA MANAGEMENT
    # --------------------------------------------------------------------- #
    def _create_tables(self):
        with self._lock:
            cursor = self._conn.cursor()
            
            if self.using_postgres:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp DOUBLE PRECISION NOT NULL,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        entry_price DOUBLE PRECISION NOT NULL,
                        exit_price DOUBLE PRECISION,
                        quantity DOUBLE PRECISION NOT NULL,
                        pnl_usd DOUBLE PRECISION,
                        pnl_pct DOUBLE PRECISION,
                        reason TEXT,
                        signal_strength INTEGER,
                        disco_confidence DOUBLE PRECISION,
                        trailing_activated INTEGER DEFAULT 0,
                        duration_sec DOUBLE PRECISION,
                        status TEXT DEFAULT 'open',
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        closed_at DOUBLE PRECISION
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS daily_stats (
                        id BIGSERIAL PRIMARY KEY,
                        date DATE UNIQUE NOT NULL,
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        total_pnl DOUBLE PRECISION DEFAULT 0,
                        max_drawdown DOUBLE PRECISION DEFAULT 0,
                        best_trade DOUBLE PRECISION DEFAULT 0,
                        worst_trade DOUBLE PRECISION DEFAULT 0,
                        avg_duration_sec DOUBLE PRECISION DEFAULT 0
                    )
                ''')
            else:
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
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)')
            self._conn.commit()
    
    def _add_missing_columns(self):
        with self._lock:
            cursor = self._conn.cursor()
            if self.using_postgres:
                cursor.execute('''
                    ALTER TABLE trades 
                    ADD COLUMN IF NOT EXISTS closed_at DOUBLE PRECISION
                ''')
            else:
                cursor.execute('PRAGMA table_info(trades)')
                columns = {row[1] for row in cursor.fetchall()}
                if 'closed_at' not in columns:
                    cursor.execute('ALTER TABLE trades ADD COLUMN closed_at REAL')
                    logger.info("Добавлена колонка closed_at в таблицу trades (SQLite)")
            self._conn.commit()
    
    def _cleanup_old_records(self):
        cutoff_time = time.time() - (RETENTION_HOURS * 3600)
        deleted = 0
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                self._prepare_query('DELETE FROM trades WHERE timestamp < %s'),
                (cutoff_time,)
            )
            deleted = cursor.rowcount or 0
            self._conn.commit()
        if deleted > 0:
            logger.info(f"Удалено {deleted} старых записей (старше {RETENTION_HOURS}ч)")
    
    # --------------------------------------------------------------------- #
    # HELPERS
    # --------------------------------------------------------------------- #
    def _prepare_query(self, query: str) -> str:
        """Преобразовать плейсхолдеры для SQLite (%s -> ?)"""
        if self.using_postgres:
            return query
        return query.replace('%s', '?')
    
    # --------------------------------------------------------------------- #
    # CRUD
    # --------------------------------------------------------------------- #
    def add_trade_open(self, symbol: str, side: str, entry_price: float,
                       quantity: float, signal_strength: int = 0,
                       disco_confidence: float = 0) -> int:
        """Добавить открытую сделку"""
        insert_sql = '''
            INSERT INTO trades (
                timestamp, symbol, side, entry_price, quantity,
                signal_strength, disco_confidence, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'open')
            {returning_clause}
        '''
        returning_clause = 'RETURNING id' if self.using_postgres else ''
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                self._prepare_query(insert_sql.format(returning_clause=returning_clause)),
                (
                    time.time(), symbol, side, entry_price, quantity,
                    signal_strength, disco_confidence
                )
            )
            if self.using_postgres:
                trade_id = cursor.fetchone()[0]
            else:
                trade_id = cursor.lastrowid
            self._conn.commit()
        logger.debug(f"Сделка открыта в БД: {symbol} {side} #{trade_id}")
        return trade_id
    
    def close_trade(self, symbol: str, exit_price: float, pnl_usd: float,
                    reason: str, trailing_activated: bool = False):
        """Закрыть сделку"""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                self._prepare_query('''
                    SELECT id, timestamp, entry_price FROM trades
                    WHERE symbol = %s AND status = 'open'
                    ORDER BY timestamp DESC LIMIT 1
                '''),
                (symbol,)
            )
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Открытая сделка не найдена: {symbol}")
                return
            
            trade_id, open_time, entry_price = row
            duration = time.time() - open_time
            pnl_pct = (exit_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
            
            cursor.execute(
                self._prepare_query('''
                    UPDATE trades SET
                        exit_price = %s,
                        pnl_usd = %s,
                        pnl_pct = %s,
                        reason = %s,
                        trailing_activated = %s,
                        duration_sec = %s,
                        status = 'closed',
                        closed_at = %s
                    WHERE id = %s
                '''),
                (
                    exit_price,
                    pnl_usd,
                    pnl_pct,
                    reason,
                    1 if trailing_activated else 0,
                    duration,
                    time.time(),
                    trade_id
                )
            )
            self._conn.commit()
        
        self._update_daily_stats(pnl_usd, duration)
        logger.debug(f"Сделка закрыта в БД: {symbol} PnL=${pnl_usd:.2f}")
    
    def _update_daily_stats(self, pnl_usd: float, duration: float):
        """Обновить дневную статистику"""
        today = datetime.now().strftime('%Y-%m-%d')
        win = 1 if pnl_usd > 0 else 0
        loss = 1 if pnl_usd <= 0 else 0
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                self._prepare_query('''
                    INSERT INTO daily_stats (
                        date, total_trades, winning_trades, losing_trades,
                        total_pnl, best_trade, worst_trade, avg_duration_sec
                    )
                    VALUES (%s, 1, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(date) DO UPDATE SET
                        total_trades = daily_stats.total_trades + 1,
                        winning_trades = daily_stats.winning_trades + %s,
                        losing_trades = daily_stats.losing_trades + %s,
                        total_pnl = daily_stats.total_pnl + %s,
                        best_trade = GREATEST(daily_stats.best_trade, %s),
                        worst_trade = LEAST(daily_stats.worst_trade, %s),
                        avg_duration_sec = (
                            (daily_stats.avg_duration_sec * (daily_stats.total_trades - 1) + %s)
                            / daily_stats.total_trades
                        )
                '''),
                (
                    today,
                    win,
                    loss,
                    pnl_usd,
                    pnl_usd,
                    pnl_usd,
                    duration,
                    win,
                    loss,
                    pnl_usd,
                    pnl_usd,
                    pnl_usd,
                    duration,
                )
            )
            self._conn.commit()
    
    def get_recent_trades(self, hours: int = 24, limit: int = 50) -> List[Dict]:
        cutoff = time.time() - (hours * 3600)
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                self._prepare_query('''
                    SELECT symbol, side, entry_price, exit_price, pnl_usd, pnl_pct,
                           reason, duration_sec, timestamp, status
                    FROM trades
                    WHERE timestamp > %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                '''),
                (cutoff, limit)
            )
            rows = cursor.fetchall()
        
        return [{
            'symbol': row[0],
            'side': row[1],
            'entry_price': row[2],
            'exit_price': row[3],
            'pnl_usd': row[4],
            'pnl_pct': row[5],
            'reason': row[6],
            'duration_sec': row[7],
            'timestamp': row[8],
            'status': row[9],
        } for row in rows]
    
    def get_stats(self, hours: int = 24) -> Dict:
        cutoff = time.time() - (hours * 3600)
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                self._prepare_query('''
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN pnl_usd <= 0 THEN 1 ELSE 0 END) as losses,
                        COALESCE(SUM(pnl_usd), 0) as total_pnl,
                        COALESCE(MAX(pnl_usd), 0) as best,
                        COALESCE(MIN(pnl_usd), 0) as worst,
                        COALESCE(AVG(duration_sec), 0) as avg_duration
                    FROM trades
                    WHERE timestamp > %s AND status = 'closed'
                '''),
                (cutoff,)
            )
            row = cursor.fetchone()
        
        total = row[0] or 0
        wins = row[1] or 0
        losses = row[2] or 0
        win_rate = (wins / total * 100) if total > 0 else 0
        
        return {
            'total_trades': total,
            'winning_trades': wins,
            'losing_trades': losses,
            'win_rate': win_rate,
            'total_pnl': row[3] or 0,
            'best_trade': row[4] or 0,
            'worst_trade': row[5] or 0,
            'avg_duration_min': (row[6] or 0) / 60,
        }
    
    def get_open_trades(self) -> List[Dict]:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute('''
                SELECT symbol, side, entry_price, quantity, timestamp, signal_strength
                FROM trades WHERE status = 'open'
            ''')
            rows = cursor.fetchall()
        
        return [{
            'symbol': row[0],
            'side': row[1],
            'entry_price': row[2],
            'quantity': row[3],
            'timestamp': row[4],
            'signal_strength': row[5],
        } for row in rows]
    
    def close(self):
        with self._lock:
            if getattr(self, '_conn', None):
                self._conn.close()
                self._conn = None
