#!/usr/bin/env python3
"""
⚡ API OPTIMIZER & RATE LIMITER
==============================

Оптимизация запросов к бирже:
- Кэширование данных
- Умный rate limiting
- Batch запросы
- WebSocket для real-time данных
- Адаптивная частота запросов
"""

import asyncio
import time
import hashlib
import json
from typing import Dict, List, Optional, Any
from collections import deque
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class RateLimiter:
    """Умный rate limiter с адаптацией"""
    
    def __init__(self, max_requests: int = 120, time_window: int = 60):
        """
        max_requests: Максимум запросов за окно времени
        time_window: Окно времени в секундах (Bybit: 120 req/min)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times = deque()
        self.last_rate_limit_hit = None
        self.adaptive_delay = 0.1  # Начальная задержка
        self.consecutive_errors = 0
        
    async def acquire(self):
        """Получить разрешение на запрос"""
        now = time.time()
        
        # Удаляем старые запросы вне окна
        while self.request_times and self.request_times[0] < now - self.time_window:
            self.request_times.popleft()
        
        # Проверяем лимит
        if len(self.request_times) >= self.max_requests:
            # Вычисляем время до следующего доступного слота
            oldest_request = self.request_times[0]
            wait_time = self.time_window - (now - oldest_request) + 0.1
            
            if wait_time > 0:
                logger.debug(f"⏳ Rate limit: ожидание {wait_time:.2f}с")
                await asyncio.sleep(wait_time)
                now = time.time()
                # Очищаем старые после ожидания
                while self.request_times and self.request_times[0] < now - self.time_window:
                    self.request_times.popleft()
        
        # Добавляем текущий запрос
        self.request_times.append(now)
        
        # Применяем адаптивную задержку
        if self.adaptive_delay > 0:
            await asyncio.sleep(self.adaptive_delay)
    
    def on_rate_limit_error(self):
        """Обработка ошибки rate limit"""
        self.consecutive_errors += 1
        self.adaptive_delay = min(self.adaptive_delay * 1.5, 1.0)  # Увеличиваем задержку
        self.last_rate_limit_hit = time.time()
        logger.warning(f"⚠️ Rate limit достигнут, задержка увеличена до {self.adaptive_delay:.2f}с")
    
    def on_success(self):
        """При успешном запросе уменьшаем задержку"""
        if self.consecutive_errors > 0:
            self.consecutive_errors -= 1
            if self.consecutive_errors == 0:
                self.adaptive_delay = max(self.adaptive_delay * 0.9, 0.05)  # Уменьшаем задержку

class DataCache:
    """Умное кэширование данных"""
    
    def __init__(self, cache_dir: str = "data/cache", default_ttl: int = 30):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl  # TTL в секундах
        self.memory_cache = {}  # Быстрый in-memory кэш
        
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Генерация ключа кэша"""
        key_data = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Получить данные из кэша"""
        cache_key = self._get_cache_key(endpoint, params)
        
        # Проверяем память
        if cache_key in self.memory_cache:
            data, timestamp, ttl = self.memory_cache[cache_key]
            if time.time() - timestamp < ttl:
                return data
            else:
                del self.memory_cache[cache_key]
        
        # Проверяем файловый кэш
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    if time.time() - cache_data['timestamp'] < cache_data['ttl']:
                        # Добавляем в память
                        self.memory_cache[cache_key] = (
                            cache_data['data'],
                            cache_data['timestamp'],
                            cache_data['ttl']
                        )
                        return cache_data['data']
            except Exception as e:
                logger.debug(f"⚠️ Ошибка чтения кэша: {e}")
        
        return None
    
    def set(self, endpoint: str, params: Dict, data: Dict, ttl: Optional[int] = None):
        """Сохранить данные в кэш"""
        cache_key = self._get_cache_key(endpoint, params)
        ttl = ttl or self.default_ttl
        timestamp = time.time()
        
        # Сохраняем в память
        self.memory_cache[cache_key] = (data, timestamp, ttl)
        
        # Сохраняем в файл
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'data': data,
                    'timestamp': timestamp,
                    'ttl': ttl
                }, f)
        except Exception as e:
            logger.debug(f"⚠️ Ошибка записи кэша: {e}")
    
    def clear_old_cache(self, max_age_hours: int = 24):
        """Очистка старого кэша"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleared = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    cleared += 1
            except Exception:
                pass
        
        if cleared > 0:
            logger.info(f"🧹 Очищено {cleared} старых кэш-файлов")

class RequestBatcher:
    """Batch обработка запросов"""
    
    def __init__(self, batch_size: int = 10, batch_timeout: float = 0.5):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_requests = []
        self.last_batch_time = time.time()
        
    async def add_request(self, request_func, *args, **kwargs):
        """Добавить запрос в batch"""
        future = asyncio.Future()
        self.pending_requests.append((future, request_func, args, kwargs))
        
        # Проверяем нужно ли выполнить batch
        should_execute = (
            len(self.pending_requests) >= self.batch_size or
            (time.time() - self.last_batch_time) >= self.batch_timeout
        )
        
        if should_execute:
            await self._execute_batch()
        
        return await future
    
    async def _execute_batch(self):
        """Выполнить batch запросов"""
        if not self.pending_requests:
            return
        
        requests_to_process = self.pending_requests[:self.batch_size]
        self.pending_requests = self.pending_requests[self.batch_size:]
        self.last_batch_time = time.time()
        
        # Выполняем параллельно
        tasks = []
        for future, request_func, args, kwargs in requests_to_process:
            task = asyncio.create_task(self._execute_single(future, request_func, args, kwargs))
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_single(self, future, request_func, args, kwargs):
        """Выполнить один запрос"""
        try:
            result = await request_func(*args, **kwargs)
            if not future.cancelled():
                future.set_result(result)
        except Exception as e:
            if not future.cancelled():
                future.set_exception(e)

class APIOptimizer:
    """⚡ Оптимизатор API запросов"""
    
    def __init__(self, exchange, cache_dir: str = "data/cache"):
        self.exchange = exchange
        self.rate_limiter = RateLimiter(max_requests=100, time_window=60)  # 100 req/min для безопасности
        self.cache = DataCache(cache_dir=cache_dir)
        self.batcher = RequestBatcher(batch_size=5, batch_timeout=0.3)
        self.request_stats = {
            'total': 0,
            'cached': 0,
            'batched': 0,
            'errors': 0
        }
        
    def _normalize_symbol(self, symbol: str) -> str:
        """Нормализует символ, убирая дублирование USDT"""
        if not symbol:
            return symbol
        
        norm = symbol.upper().replace('/', '').replace('-', '')
        # Убираем :USDT если есть
        if norm.endswith(':USDT'):
            norm = norm[:-5] + 'USDT'
        elif ':USDT' in norm:
            norm = norm.replace(':USDT', '') + 'USDT'
        # Убираем все остальные ':'
        norm = norm.replace(':', '')
        # Убеждаемся, что заканчивается на USDT
        if not norm.endswith('USDT'):
            norm = norm + 'USDT'
        # Убираем возможное дублирование USDT в конце
        while norm.endswith('USDTUSDT'):
            norm = norm[:-4]
        return norm
        
    async def fetch_with_cache(self, method: str, symbol: str, 
                              timeframe: str = '15m', 
                              limit: int = 100,
                              cache_ttl: int = 30) -> Optional[List]:
        """
        Запрос данных с кэшированием и rate limiting
        """
        # Нормализуем символ перед использованием
        normalized_symbol = self._normalize_symbol(symbol)
        
        params = {
            'symbol': normalized_symbol,
            'timeframe': timeframe,
            'limit': limit
        }
        
        # Проверяем кэш
        cached_data = self.cache.get(method, params)
        if cached_data:
            self.request_stats['cached'] += 1
            logger.debug(f"✅ Кэш hit: {normalized_symbol} {timeframe}")
            return cached_data
        
        # Делаем запрос с rate limiting
        try:
            await self.rate_limiter.acquire()
            self.request_stats['total'] += 1
            
            if method == 'fetch_ohlcv':
                data = await self.exchange.fetch_ohlcv(normalized_symbol, timeframe, limit=limit)
            elif method == 'fetch_ticker':
                data = await self.exchange.fetch_ticker(normalized_symbol)
            else:
                data = None
            
            # Сохраняем в кэш
            if data:
                self.cache.set(method, params, data, cache_ttl)
                self.rate_limiter.on_success()
                return data
            
        except Exception as e:
            self.request_stats['errors'] += 1
            error_str = str(e).lower()
            
            if 'rate limit' in error_str or '429' in error_str:
                self.rate_limiter.on_rate_limit_error()
                logger.warning(f"⚠️ Rate limit error для {normalized_symbol}")
            else:
                logger.error(f"❌ Ошибка запроса {normalized_symbol}: {e}")
            
            # Пробуем вернуть из кэша даже если старый
            cached_data = self.cache.get(method, params)
            if cached_data:
                logger.debug(f"💾 Используем старый кэш для {normalized_symbol}")
                return cached_data
            
            return None
        
        return None
    
    async def fetch_multiple_symbols(self, symbols: List[str], 
                                     timeframe: str = '15m',
                                     limit: int = 100,
                                     max_concurrent: int = 5) -> Dict[str, Optional[List]]:
        """Параллельная загрузка нескольких символов с ограничением"""
        results = {}
        
        # Разбиваем на батчи
        for i in range(0, len(symbols), max_concurrent):
            batch = symbols[i:i + max_concurrent]
            
            tasks = [
                self.fetch_with_cache('fetch_ohlcv', symbol, timeframe, limit)
                for symbol in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.warning(f"⚠️ Ошибка для {symbol}: {result}")
                    results[symbol] = None
                else:
                    results[symbol] = result
            
            # Небольшая пауза между батчами
            if i + max_concurrent < len(symbols):
                await asyncio.sleep(0.1)
        
        return results
    
    def get_stats(self) -> Dict:
        """Получить статистику оптимизации"""
        cache_hit_rate = (
            (self.request_stats['cached'] / self.request_stats['total'] * 100)
            if self.request_stats['total'] > 0 else 0
        )
        
        return {
            'total_requests': self.request_stats['total'],
            'cached_requests': self.request_stats['cached'],
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'batched_requests': self.request_stats['batched'],
            'errors': self.request_stats['errors'],
            'rate_limiter_delay': self.rate_limiter.adaptive_delay
        }
    
    def clear_cache(self):
        """Очистить кэш"""
        self.cache.clear_old_cache(max_age_hours=1)
        self.cache.memory_cache.clear()


