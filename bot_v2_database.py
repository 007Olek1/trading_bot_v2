"""
🗄️ POSTGRESQL DATABASE MODULE
Хранение истории сделок, ML данных и статистики
"""

import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Менеджер PostgreSQL базы данных"""
    
    def __init__(self):
        self.connection = None
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'trading_bot'),
            'user': os.getenv('DB_USER', 'trading_bot'),
            'password': os.getenv('DB_PASSWORD', 'trading_bot_password')
        }
        
    def connect(self):
        """Подключение к базе данных"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            logger.info("✅ Подключение к PostgreSQL успешно")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к PostgreSQL: {e}")
            return False
    
    def disconnect(self):
        """Отключение от базы данных"""
        if self.connection:
            self.connection.close()
            logger.info("🔌 Отключение от PostgreSQL")
    
    def create_tables(self):
        """Создание таблиц базы данных"""
        try:
            cursor = self.connection.cursor()
            
            # Таблица сделок
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(50) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    entry_price DECIMAL(20, 8) NOT NULL,
                    exit_price DECIMAL(20, 8),
                    amount DECIMAL(20, 8) NOT NULL,
                    leverage INTEGER DEFAULT 5,
                    pnl DECIMAL(20, 8),
                    pnl_percent DECIMAL(10, 2),
                    confidence INTEGER,
                    reason TEXT,
                    duration_minutes INTEGER,
                    success BOOLEAN,
                    open_time TIMESTAMP NOT NULL,
                    close_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_open_time ON trades(open_time);
                CREATE INDEX IF NOT EXISTS idx_success ON trades(success);
            """)
            
            # Таблица ML признаков
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_features (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(50) NOT NULL,
                    signal VARCHAR(10),
                    confidence INTEGER,
                    signal_strength DECIMAL(5, 2),
                    
                    -- Технические индикаторы
                    rsi DECIMAL(10, 2),
                    macd_signal DECIMAL(20, 8),
                    bollinger_position DECIMAL(5, 2),
                    ema_trend DECIMAL(20, 8),
                    volume_ratio DECIMAL(10, 2),
                    stochastic DECIMAL(10, 2),
                    
                    -- Рыночные данные
                    price DECIMAL(20, 8),
                    volume_24h DECIMAL(30, 8),
                    volatility DECIMAL(10, 4),
                    atr DECIMAL(20, 8),
                    
                    -- Временные факторы
                    hour INTEGER,
                    day_of_week INTEGER,
                    is_weekend BOOLEAN,
                    
                    -- Результат
                    actual_success BOOLEAN,
                    actual_pnl DECIMAL(20, 8),
                    trade_id INTEGER REFERENCES trades(id),
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_symbol_ml ON ml_features(symbol);
                CREATE INDEX IF NOT EXISTS idx_created_at_ml ON ml_features(created_at);
                CREATE INDEX IF NOT EXISTS idx_actual_success ON ml_features(actual_success);
            """)
            
            # Таблица статистики
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_statistics (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL UNIQUE,
                    total_trades INTEGER DEFAULT 0,
                    successful_trades INTEGER DEFAULT 0,
                    failed_trades INTEGER DEFAULT 0,
                    total_pnl DECIMAL(20, 8) DEFAULT 0,
                    win_rate DECIMAL(5, 2) DEFAULT 0,
                    avg_pnl DECIMAL(20, 8) DEFAULT 0,
                    max_drawdown DECIMAL(20, 8) DEFAULT 0,
                    balance_start DECIMAL(20, 8),
                    balance_end DECIMAL(20, 8),
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Таблица конфигурации ML модели
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY,
                    model_version VARCHAR(50) NOT NULL,
                    accuracy DECIMAL(5, 4),
                    precision_score DECIMAL(5, 4),
                    recall_score DECIMAL(5, 4),
                    f1_score DECIMAL(5, 4),
                    training_samples INTEGER,
                    test_samples INTEGER,
                    is_active BOOLEAN DEFAULT TRUE,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_active ON ml_models(is_active);
            """)
            
            self.connection.commit()
            logger.info("✅ Таблицы базы данных созданы")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания таблиц: {e}")
            self.connection.rollback()
            return False
    
    def save_trade(self, trade_data: Dict[str, Any]) -> Optional[int]:
        """Сохранение сделки"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO trades (
                    symbol, side, entry_price, exit_price, amount, leverage,
                    pnl, pnl_percent, confidence, reason, duration_minutes,
                    success, open_time, close_time
                ) VALUES (
                    %(symbol)s, %(side)s, %(entry_price)s, %(exit_price)s,
                    %(amount)s, %(leverage)s, %(pnl)s, %(pnl_percent)s,
                    %(confidence)s, %(reason)s, %(duration_minutes)s,
                    %(success)s, %(open_time)s, %(close_time)s
                )
                RETURNING id
            """, {
                'symbol': trade_data.get('symbol'),
                'side': trade_data.get('side'),
                'entry_price': trade_data.get('entry_price'),
                'exit_price': trade_data.get('exit_price'),
                'amount': trade_data.get('amount'),
                'leverage': trade_data.get('leverage', 5),
                'pnl': trade_data.get('pnl'),
                'pnl_percent': trade_data.get('pnl_percent'),
                'confidence': trade_data.get('confidence'),
                'reason': trade_data.get('reason'),
                'duration_minutes': trade_data.get('duration_minutes'),
                'success': trade_data.get('pnl', 0) > 0,
                'open_time': trade_data.get('open_time', datetime.now()),
                'close_time': trade_data.get('close_time', datetime.now())
            })
            
            trade_id = cursor.fetchone()[0]
            self.connection.commit()
            
            logger.debug(f"💾 Сделка сохранена: ID={trade_id}, {trade_data.get('symbol')}")
            return trade_id
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения сделки: {e}")
            self.connection.rollback()
            return None
    
    def save_ml_features(self, feature_data: Dict[str, Any]) -> Optional[int]:
        """Сохранение ML признаков"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO ml_features (
                    symbol, signal, confidence, signal_strength,
                    rsi, macd_signal, bollinger_position, ema_trend,
                    volume_ratio, stochastic, price, volume_24h,
                    volatility, atr, hour, day_of_week, is_weekend,
                    actual_success, actual_pnl, trade_id
                ) VALUES (
                    %(symbol)s, %(signal)s, %(confidence)s, %(signal_strength)s,
                    %(rsi)s, %(macd_signal)s, %(bollinger_position)s, %(ema_trend)s,
                    %(volume_ratio)s, %(stochastic)s, %(price)s, %(volume_24h)s,
                    %(volatility)s, %(atr)s, %(hour)s, %(day_of_week)s, %(is_weekend)s,
                    %(actual_success)s, %(actual_pnl)s, %(trade_id)s
                )
                RETURNING id
            """, feature_data)
            
            feature_id = cursor.fetchone()[0]
            self.connection.commit()
            
            return feature_id
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения ML признаков: {e}")
            self.connection.rollback()
            return None
    
    def get_training_data(self, limit: int = 1000) -> Tuple[List[Dict], List[bool]]:
        """Получение данных для обучения ML"""
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    confidence, signal_strength, rsi, macd_signal,
                    bollinger_position, ema_trend, volume_ratio, stochastic,
                    volume_24h, volatility, atr, hour, day_of_week,
                    actual_success
                FROM ml_features
                WHERE actual_success IS NOT NULL
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
            
            rows = cursor.fetchall()
            
            if not rows:
                return [], []
            
            features = []
            labels = []
            
            for row in rows:
                feature_dict = dict(row)
                labels.append(feature_dict.pop('actual_success'))
                features.append(feature_dict)
            
            return features, labels
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения данных для обучения: {e}")
            return [], []
    
    def get_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Получение статистики торговли"""
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN success = TRUE THEN 1 ELSE 0 END) as successful_trades,
                    SUM(CASE WHEN success = FALSE THEN 1 ELSE 0 END) as failed_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MIN(pnl) as min_pnl,
                    MAX(pnl) as max_pnl,
                    AVG(CASE WHEN success = TRUE THEN 1.0 ELSE 0.0 END) as win_rate
                FROM trades
                WHERE close_time >= CURRENT_DATE - INTERVAL '%s days'
            """, (days,))
            
            stats = cursor.fetchone()
            return dict(stats) if stats else {}
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики: {e}")
            return {}
    
    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Получение последних сделок"""
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT *
                FROM trades
                ORDER BY close_time DESC
                LIMIT %s
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения сделок: {e}")
            return []
    
    def update_daily_statistics(self, date: datetime, stats: Dict[str, Any]):
        """Обновление дневной статистики"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO bot_statistics (
                    date, total_trades, successful_trades, failed_trades,
                    total_pnl, win_rate, avg_pnl, max_drawdown,
                    balance_start, balance_end, updated_at
                ) VALUES (
                    %(date)s, %(total_trades)s, %(successful_trades)s, %(failed_trades)s,
                    %(total_pnl)s, %(win_rate)s, %(avg_pnl)s, %(max_drawdown)s,
                    %(balance_start)s, %(balance_end)s, CURRENT_TIMESTAMP
                )
                ON CONFLICT (date) DO UPDATE SET
                    total_trades = EXCLUDED.total_trades,
                    successful_trades = EXCLUDED.successful_trades,
                    failed_trades = EXCLUDED.failed_trades,
                    total_pnl = EXCLUDED.total_pnl,
                    win_rate = EXCLUDED.win_rate,
                    avg_pnl = EXCLUDED.avg_pnl,
                    max_drawdown = EXCLUDED.max_drawdown,
                    balance_end = EXCLUDED.balance_end,
                    updated_at = CURRENT_TIMESTAMP
            """, {
                'date': date.date(),
                **stats
            })
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления статистики: {e}")
            self.connection.rollback()
    
    def save_ml_model_info(self, model_info: Dict[str, Any]):
        """Сохранение информации о ML модели"""
        try:
            cursor = self.connection.cursor()
            
            # Деактивируем предыдущие модели
            cursor.execute("UPDATE ml_models SET is_active = FALSE")
            
            # Сохраняем новую модель
            cursor.execute("""
                INSERT INTO ml_models (
                    model_version, accuracy, precision_score, recall_score,
                    f1_score, training_samples, test_samples, is_active
                ) VALUES (
                    %(version)s, %(accuracy)s, %(precision)s, %(recall)s,
                    %(f1)s, %(training_samples)s, %(test_samples)s, TRUE
                )
            """, model_info)
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения модели: {e}")
            self.connection.rollback()


# Глобальный экземпляр
db_manager = DatabaseManager()

