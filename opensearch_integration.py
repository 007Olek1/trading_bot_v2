#!/usr/bin/env python3
"""
🔍 OPENSEARCH ИНТЕГРАЦИЯ
========================
Реальный поиск и анализ данных торгового бота в режиме реального времени
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)

try:
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from opensearch_dsl import Search, Q
    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False
    logger.warning("⚠️ OpenSearch не установлен: pip install opensearch-py opensearch-dsl")


@dataclass
class TradeDocument:
    """Документ сделки для OpenSearch"""
    timestamp: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    profit_usd: float
    profit_percent: float
    confidence: float
    strategy_score: float
    market_condition: str
    tp_level: Optional[int]
    stop_loss_hit: bool
    indicators: Dict[str, Any]
    reasons: List[str]


class OpenSearchIntegration:
    """🔍 Интеграция с OpenSearch для анализа данных"""
    
    def __init__(self, 
                 host: str = None,
                 port: int = 9200,
                 use_ssl: bool = False,
                 verify_certs: bool = False):
        """
        Инициализация OpenSearch клиента
        
        Args:
            host: Адрес OpenSearch сервера (default: из env или localhost)
            port: Порт (default: 9200)
            use_ssl: Использовать SSL
            verify_certs: Проверять сертификаты
        """
        if not OPENSEARCH_AVAILABLE:
            self.client = None
            self.available = False
            logger.warning("⚠️ OpenSearch недоступен")
            return
        
        # Настройки из переменных окружения
        self.host = host or os.getenv('OPENSEARCH_HOST', 'localhost')
        self.port = port or int(os.getenv('OPENSEARCH_PORT', '9200'))
        self.use_ssl = use_ssl or os.getenv('OPENSEARCH_USE_SSL', 'false').lower() == 'true'
        self.verify_certs = verify_certs
        
        # Аутентификация (если есть)
        http_auth = None
        if os.getenv('OPENSEARCH_USER') and os.getenv('OPENSEARCH_PASSWORD'):
            http_auth = (os.getenv('OPENSEARCH_USER'), os.getenv('OPENSEARCH_PASSWORD'))
        
        try:
            self.client = OpenSearch(
                hosts=[{'host': self.host, 'port': self.port}],
                http_auth=http_auth,
                use_ssl=self.use_ssl,
                verify_certs=self.verify_certs,
                connection_class=RequestsHttpConnection,
                timeout=30
            )
            
            # Проверка подключения
            if self.client.ping():
                self.available = True
                logger.info(f"✅ OpenSearch подключен: {self.host}:{self.port}")
                self._ensure_index_exists()
            else:
                self.available = False
                logger.warning("⚠️ OpenSearch не отвечает")
                self.client = None
                
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к OpenSearch: {e}")
            self.client = None
            self.available = False
    
    def _ensure_index_exists(self):
        """Создать индекс если не существует"""
        if not self.available:
            return
        
        index_name = "trading_bot_trades"
        
        try:
            if not self.client.indices.exists(index=index_name):
                # Создаем индекс с маппингом
                index_body = {
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    },
                    "mappings": {
                        "properties": {
                            "timestamp": {"type": "date"},
                            "symbol": {"type": "keyword"},
                            "direction": {"type": "keyword"},
                            "entry_price": {"type": "float"},
                            "exit_price": {"type": "float"},
                            "profit_usd": {"type": "float"},
                            "profit_percent": {"type": "float"},
                            "confidence": {"type": "float"},
                            "strategy_score": {"type": "float"},
                            "market_condition": {"type": "keyword"},
                            "tp_level": {"type": "integer"},
                            "stop_loss_hit": {"type": "boolean"},
                            "indicators": {"type": "object"},
                            "reasons": {"type": "keyword"}
                        }
                    }
                }
                self.client.indices.create(index=index_name, body=index_body)
                logger.info(f"✅ Индекс {index_name} создан")
        except Exception as e:
            logger.error(f"❌ Ошибка создания индекса: {e}")
    
    def index_trade(self, trade: TradeDocument) -> bool:
        """Добавить сделку в OpenSearch"""
        if not self.available:
            return False
        
        try:
            index_name = "trading_bot_trades"
            doc = asdict(trade)
            doc['@timestamp'] = trade.timestamp
            
            response = self.client.index(
                index=index_name,
                body=doc,
                refresh=True  # Обновляем индекс сразу
            )
            
            logger.debug(f"✅ Сделка проиндексирована: {response['_id']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка индексации сделки: {e}")
            return False
    
    def search_trades(self, 
                     symbol: Optional[str] = None,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     min_profit: Optional[float] = None,
                     market_condition: Optional[str] = None,
                     limit: int = 100) -> List[Dict]:
        """Поиск сделок в OpenSearch"""
        if not self.available:
            return []
        
        try:
            index_name = "trading_bot_trades"
            query = {"bool": {"must": []}}
            
            # Фильтры
            if symbol:
                query["bool"]["must"].append({"term": {"symbol.keyword": symbol}})
            
            if start_time or end_time:
                time_range = {}
                if start_time:
                    time_range["gte"] = start_time.isoformat()
                if end_time:
                    time_range["lte"] = end_time.isoformat()
                query["bool"]["must"].append({"range": {"timestamp": time_range}})
            
            if min_profit is not None:
                query["bool"]["must"].append({"range": {"profit_usd": {"gte": min_profit}}})
            
            if market_condition:
                query["bool"]["must"].append({"term": {"market_condition.keyword": market_condition}})
            
            # Поиск
            search_body = {
                "query": query,
                "sort": [{"timestamp": {"order": "desc"}}],
                "size": limit
            }
            
            response = self.client.search(
                index=index_name,
                body=search_body
            )
            
            trades = [hit["_source"] for hit in response["hits"]["hits"]]
            return trades
            
        except Exception as e:
            logger.error(f"❌ Ошибка поиска: {e}")
            return []
    
    def get_trading_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Аналитика за период"""
        if not self.available:
            return {}
        
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            index_name = "trading_bot_trades"
            
            # Агрегации
            search_body = {
                "query": {
                    "range": {
                        "timestamp": {
                            "gte": start_time.isoformat(),
                            "lte": end_time.isoformat()
                        }
                    }
                },
                "aggs": {
                    "total_profit": {"sum": {"field": "profit_usd"}},
                    "avg_profit": {"avg": {"field": "profit_usd"}},
                    "win_rate": {
                        "filter": {"range": {"profit_usd": {"gt": 0}}},
                        "aggs": {"count": {"value_count": {"field": "profit_usd"}}}
                    },
                    "by_symbol": {
                        "terms": {"field": "symbol.keyword", "size": 10},
                        "aggs": {
                            "total_profit": {"sum": {"field": "profit_usd"}}
                        }
                    },
                    "by_market_condition": {
                        "terms": {"field": "market_condition.keyword"},
                        "aggs": {
                            "total_profit": {"sum": {"field": "profit_usd"}}
                        }
                    }
                },
                "size": 0
            }
            
            response = self.client.search(
                index=index_name,
                body=search_body
            )
            
            aggs = response.get("aggregations", {})
            
            total_trades = response["hits"]["total"]["value"]
            total_profit = aggs.get("total_profit", {}).get("value", 0)
            avg_profit = aggs.get("avg_profit", {}).get("value", 0)
            
            win_count = aggs.get("win_rate", {}).get("count", {}).get("value", 0)
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
            
            analytics = {
                "period_days": days,
                "total_trades": total_trades,
                "total_profit_usd": total_profit,
                "avg_profit_usd": avg_profit,
                "win_rate_percent": win_rate,
                "top_symbols": [
                    {"symbol": bucket["key"], "profit": bucket["total_profit"]["value"]}
                    for bucket in aggs.get("by_symbol", {}).get("buckets", [])
                ],
                "by_market_condition": [
                    {"condition": bucket["key"], "profit": bucket["total_profit"]["value"]}
                    for bucket in aggs.get("by_market_condition", {}).get("buckets", [])
                ]
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"❌ Ошибка аналитики: {e}")
            return {}


if __name__ == "__main__":
    # Пример использования
    logging.basicConfig(level=logging.INFO)
    
    opensearch = OpenSearchIntegration()
    
    if opensearch.available:
        # Тестовая сделка
        test_trade = TradeDocument(
            timestamp=datetime.now().isoformat(),
            symbol="BTCUSDT",
            direction="buy",
            entry_price=50000.0,
            exit_price=51000.0,
            profit_usd=50.0,
            profit_percent=2.0,
            confidence=80.0,
            strategy_score=15.0,
            market_condition="BULLISH",
            tp_level=1,
            stop_loss_hit=False,
            indicators={"rsi": 60, "macd": 0.5},
            reasons=["Strong trend", "Volume spike"]
        )
        
        opensearch.index_trade(test_trade)
        
        # Поиск
        trades = opensearch.search_trades(symbol="BTCUSDT", limit=10)
        print(f"✅ Найдено сделок: {len(trades)}")
        
        # Аналитика
        analytics = opensearch.get_trading_analytics(days=7)
        print(f"📊 Аналитика: {json.dumps(analytics, indent=2)}")
    else:
        print("⚠️ OpenSearch недоступен")


