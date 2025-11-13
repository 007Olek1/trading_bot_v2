#!/usr/bin/env python3
"""
💾 DATA STORAGE SYSTEM V4.0
============================

Система хранения и управления всеми данными для анализа и принятия решений
- Рыночные данные
- Результаты сделок
- Паттерны обучения
- Универсальные правила
- Статистика производительности
"""

import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path
import pytz

logger = logging.getLogger(__name__)

# Warsaw timezone для синхронизации времени
WARSAW_TZ = pytz.timezone('Europe/Warsaw')

@dataclass
class MarketData:
    """Рыночные данные для анализа"""
    timestamp: str
    symbol: str
    timeframe: str
    price: float
    volume: float
    rsi: float
    macd: float
    bb_position: float
    ema_9: float
    ema_21: float
    ema_50: float
    volume_ratio: float
    momentum: float
    market_condition: str

@dataclass
class TradeDecision:
    """Решение по сделке"""
    timestamp: str
    symbol: str
    decision: str  # 'buy', 'sell', 'hold'
    confidence: float
    strategy_score: float
    reasons: List[str]
    market_data: Dict
    result: Optional[str] = None  # 'win', 'loss', 'pending'
    pnl_percent: Optional[float] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None

@dataclass
class UniversalRule:
    """Универсальное правило обучения"""
    rule_id: str
    rule_type: str  # 'entry', 'exit', 'risk_management'
    conditions: Dict  # Диапазоны условий, не точные значения
    success_rate: float
    total_applications: int
    successful_applications: int
    market_conditions: List[str]  # В каких условиях работает
    created_at: str
    last_updated: str
    confidence_range: Tuple[float, float]  # Диапазон уверенности
    
@dataclass
class LearningPattern:
    """Паттерн для обучения"""
    pattern_id: str
    pattern_type: str
    features: Dict  # Признаки паттерна (диапазоны)
    target: str  # Целевое действие
    success_count: int
    failure_count: int
    market_conditions: List[str]
    generalization_level: float  # Уровень обобщения (0-1)

class DataStorageSystem:
    """💾 Система хранения и управления данными"""
    
    def __init__(self, db_path: Optional[str] = None):
        # Определяем путь к БД адаптивно
        if db_path is None:
            # Проверяем, существует ли /opt/bot/ (сервер)
            if Path("/opt/bot").exists():
                db_path = "/opt/bot/trading_data.db"
            else:
                # Используем текущую директорию или директорию скрипта
                base_dir = Path(__file__).parent
                data_dir = base_dir / "data"
                data_dir.mkdir(exist_ok=True)
                db_path = str(data_dir / "trading_data.db")
        
        self.db_path = db_path
        
        # Создаем директорию для БД, если её нет
        db_parent = Path(self.db_path).parent
        db_parent.mkdir(parents=True, exist_ok=True)
        
        self.init_database()
        logger.info(f"💾 DataStorageSystem инициализирована (БД: {self.db_path})")
    
    def init_database(self):
        """Инициализация базы данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Таблица рыночных данных
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume REAL NOT NULL,
                    rsi REAL NOT NULL,
                    macd REAL NOT NULL,
                    bb_position REAL NOT NULL,
                    ema_9 REAL NOT NULL,
                    ema_21 REAL NOT NULL,
                    ema_50 REAL NOT NULL,
                    volume_ratio REAL NOT NULL,
                    momentum REAL NOT NULL,
                    market_condition TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица решений по сделкам
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    strategy_score REAL NOT NULL,
                    reasons TEXT NOT NULL,
                    market_data TEXT NOT NULL,
                    result TEXT,
                    pnl_percent REAL,
                    entry_price REAL,
                    exit_price REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица универсальных правил
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS universal_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id TEXT UNIQUE NOT NULL,
                    rule_type TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    success_rate REAL NOT NULL,
                    total_applications INTEGER NOT NULL,
                    successful_applications INTEGER NOT NULL,
                    market_conditions TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    confidence_range TEXT NOT NULL
                )
            ''')
            
            # Таблица паттернов обучения
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id TEXT UNIQUE NOT NULL,
                    pattern_type TEXT NOT NULL,
                    features TEXT NOT NULL,
                    target TEXT NOT NULL,
                    success_count INTEGER NOT NULL,
                    failure_count INTEGER NOT NULL,
                    market_conditions TEXT NOT NULL,
                    generalization_level REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица статистики производительности
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    losing_trades INTEGER NOT NULL,
                    total_pnl REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    avg_win REAL NOT NULL,
                    avg_loss REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    market_condition TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ База данных инициализирована")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации БД: {e}")
    
    def store_market_data(self, data: MarketData):
        """Сохранить рыночные данные"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO market_data 
                (timestamp, symbol, timeframe, price, volume, rsi, macd, bb_position,
                 ema_9, ema_21, ema_50, volume_ratio, momentum, market_condition)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.timestamp, data.symbol, data.timeframe, data.price, data.volume,
                data.rsi, data.macd, data.bb_position, data.ema_9, data.ema_21,
                data.ema_50, data.volume_ratio, data.momentum, data.market_condition
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения рыночных данных: {e}")
    
    def store_trade_decision(self, decision: TradeDecision):
        """Сохранить решение по сделке"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trade_decisions 
                (timestamp, symbol, decision, confidence, strategy_score, reasons,
                 market_data, result, pnl_percent, entry_price, exit_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                decision.timestamp, decision.symbol, decision.decision,
                decision.confidence, decision.strategy_score,
                json.dumps(decision.reasons), json.dumps(decision.market_data),
                decision.result, decision.pnl_percent, decision.entry_price, decision.exit_price
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения решения: {e}")
    
    def store_universal_rule(self, rule: UniversalRule):
        """Сохранить универсальное правило"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO universal_rules 
                (rule_id, rule_type, conditions, success_rate, total_applications,
                 successful_applications, market_conditions, created_at, last_updated, confidence_range)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.rule_id, rule.rule_type, json.dumps(rule.conditions),
                rule.success_rate, rule.total_applications, rule.successful_applications,
                json.dumps(rule.market_conditions), rule.created_at, rule.last_updated,
                json.dumps(rule.confidence_range)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения правила: {e}")
    
    def get_market_data(self, symbol: str, timeframe: str, hours: int = 24) -> List[MarketData]:
        """Получить рыночные данные за период"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            since = (datetime.now(WARSAW_TZ) - timedelta(hours=hours)).isoformat()
            
            cursor.execute('''
                SELECT * FROM market_data 
                WHERE symbol = ? AND timeframe = ? AND timestamp > ?
                ORDER BY timestamp DESC
            ''', (symbol, timeframe, since))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Преобразуем в объекты MarketData
            data = []
            for row in rows:
                data.append(MarketData(
                    timestamp=row[1], symbol=row[2], timeframe=row[3],
                    price=row[4], volume=row[5], rsi=row[6], macd=row[7],
                    bb_position=row[8], ema_9=row[9], ema_21=row[10],
                    ema_50=row[11], volume_ratio=row[12], momentum=row[13],
                    market_condition=row[14]
                ))
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения рыночных данных: {e}")
            return []
    
    def get_universal_rules(self, rule_type: str = None, min_success_rate: float = 0.6) -> List[UniversalRule]:
        """Получить универсальные правила"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT * FROM universal_rules 
                WHERE success_rate >= ?
            '''
            params = [min_success_rate]
            
            if rule_type:
                query += ' AND rule_type = ?'
                params.append(rule_type)
            
            query += ' ORDER BY success_rate DESC'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            # Преобразуем в объекты UniversalRule
            rules = []
            for row in rows:
                rules.append(UniversalRule(
                    rule_id=row[1], rule_type=row[2],
                    conditions=json.loads(row[3]), success_rate=row[4],
                    total_applications=row[5], successful_applications=row[6],
                    market_conditions=json.loads(row[7]), created_at=row[8],
                    last_updated=row[9], confidence_range=tuple(json.loads(row[10]))
                ))
            
            return rules
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения правил: {e}")
            return []
    
    def analyze_decision_patterns(self) -> Dict[str, Any]:
        """Анализ паттернов принятия решений"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Статистика решений
            df_decisions = pd.read_sql_query('''
                SELECT decision, confidence, strategy_score, result, pnl_percent,
                       market_data, timestamp
                FROM trade_decisions 
                WHERE timestamp > datetime('now', '-7 days')
            ''', conn)
            
            # Статистика по рыночным условиям
            df_market = pd.read_sql_query('''
                SELECT market_condition, COUNT(*) as count,
                       AVG(rsi) as avg_rsi, AVG(bb_position) as avg_bb,
                       AVG(volume_ratio) as avg_volume
                FROM market_data 
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY market_condition
            ''', conn)
            
            conn.close()
            
            analysis = {
                'total_decisions': len(df_decisions),
                'decision_distribution': df_decisions['decision'].value_counts().to_dict(),
                'avg_confidence': df_decisions['confidence'].mean(),
                'avg_strategy_score': df_decisions['strategy_score'].mean(),
                'market_conditions': df_market.to_dict('records'),
                'win_rate': len(df_decisions[df_decisions['result'] == 'win']) / len(df_decisions) if len(df_decisions) > 0 else 0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа паттернов: {e}")
            return {}
    
    def create_universal_rule_from_patterns(self, patterns: List[Dict]) -> UniversalRule:
        """Создать универсальное правило из паттернов"""
        try:
            # Анализируем паттерны для создания универсального правила
            rule_id = f"rule_{datetime.now(WARSAW_TZ).strftime('%Y%m%d_%H%M%S')}"
            
            # Определяем диапазоны условий (НЕ точные значения!)
            rsi_values = [p.get('rsi', 50) for p in patterns if 'rsi' in p]
            bb_values = [p.get('bb_position', 50) for p in patterns if 'bb_position' in p]
            vol_values = [p.get('volume_ratio', 1.0) for p in patterns if 'volume_ratio' in p]
            
            conditions = {
                'rsi_range': (min(rsi_values) - 5, max(rsi_values) + 5) if rsi_values else (30, 70),
                'bb_range': (min(bb_values) - 10, max(bb_values) + 10) if bb_values else (20, 80),
                'volume_range': (min(vol_values) * 0.8, max(vol_values) * 1.2) if vol_values else (0.5, 2.0),
                'pattern_count': len(patterns)
            }
            
            # Рассчитываем успешность
            successful = len([p for p in patterns if p.get('result') == 'win'])
            total = len(patterns)
            success_rate = successful / total if total > 0 else 0
            
            # Определяем рыночные условия
            market_conditions = list(set([p.get('market_condition', 'NEUTRAL') for p in patterns]))
            
            rule = UniversalRule(
                rule_id=rule_id,
                rule_type='entry',
                conditions=conditions,
                success_rate=success_rate,
                total_applications=total,
                successful_applications=successful,
                market_conditions=market_conditions,
                created_at=datetime.now(WARSAW_TZ).isoformat(),
                last_updated=datetime.now(WARSAW_TZ).isoformat(),
                confidence_range=(40.0, 90.0)
            )
            
            # Сохраняем правило
            self.store_universal_rule(rule)
            
            logger.info(f"✅ Создано универсальное правило {rule_id} с успешностью {success_rate:.1%}")
            
            return rule
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания универсального правила: {e}")
            return None
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Получить инсайты обучения"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Анализ эволюции правил
            df_rules = pd.read_sql_query('''
                SELECT rule_type, success_rate, total_applications,
                       created_at, last_updated
                FROM universal_rules
                ORDER BY created_at DESC
            ''', conn)
            
            # Анализ производительности по времени
            df_performance = pd.read_sql_query('''
                SELECT date, win_rate, total_trades, market_condition
                FROM performance_stats
                ORDER BY date DESC
                LIMIT 30
            ''', conn)
            
            conn.close()
            
            insights = {
                'total_rules': len(df_rules),
                'avg_success_rate': df_rules['success_rate'].mean() if len(df_rules) > 0 else 0,
                'best_rule_success_rate': df_rules['success_rate'].max() if len(df_rules) > 0 else 0,
                'rules_by_type': df_rules['rule_type'].value_counts().to_dict(),
                'recent_performance': df_performance.to_dict('records'),
                'learning_trend': 'improving' if len(df_performance) > 1 and df_performance.iloc[0]['win_rate'] > df_performance.iloc[-1]['win_rate'] else 'stable'
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения инсайтов: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 30):
        """Очистка старых данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now(WARSAW_TZ) - timedelta(days=days)).isoformat()
            
            # Удаляем старые рыночные данные
            cursor.execute('DELETE FROM market_data WHERE timestamp < ?', (cutoff_date,))
            deleted_market = cursor.rowcount
            
            # Удаляем старые решения (кроме результативных)
            cursor.execute('''
                DELETE FROM trade_decisions 
                WHERE timestamp < ? AND result IS NULL
            ''', (cutoff_date,))
            deleted_decisions = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"🧹 Очищено: {deleted_market} рыночных записей, {deleted_decisions} решений")
            
        except Exception as e:
            logger.error(f"❌ Ошибка очистки данных: {e}")

# Тестирование системы
if __name__ == "__main__":
    storage = DataStorageSystem()
    
    # Тест сохранения рыночных данных
    test_market_data = MarketData(
        timestamp=datetime.now(WARSAW_TZ).isoformat(),
        symbol="BTCUSDT",
        timeframe="30m",
        price=35000.0,
        volume=1000000.0,
        rsi=45.0,
        macd=0.5,
        bb_position=30.0,
        ema_9=35100.0,
        ema_21=35000.0,
        ema_50=34900.0,
        volume_ratio=1.2,
        momentum=2.5,
        market_condition="BULLISH"
    )
    
    storage.store_market_data(test_market_data)
    
    # Тест анализа паттернов
    analysis = storage.analyze_decision_patterns()
    print("📊 Анализ паттернов:", analysis)
    
    # Тест инсайтов обучения
    insights = storage.get_learning_insights()
    print("🧠 Инсайты обучения:", insights)
    
    print("✅ DataStorageSystem протестирована")

