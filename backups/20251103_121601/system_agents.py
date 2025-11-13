#!/usr/bin/env python3
"""
🤖 СИСТЕМА АГЕНТОВ МОНИТОРИНГА И УПРАВЛЕНИЯ
==========================================

Агенты для автономной работы:
- Agent Cleaner: уборка логов и данных
- Agent Security: проверка безопасности
- Agent Stability: мониторинг стабильности
- Agent Recovery: автоматическое восстановление

Все агенты самообучаются через универсальные правила!
"""

import asyncio
import os
import psutil
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import shutil

# Импорт интеллектуальной системы
try:
    from intelligent_agents import IntelligentAgent, IntelligentAgentsSystem
    INTELLIGENT_AGENTS_AVAILABLE = True
except ImportError:
    INTELLIGENT_AGENTS_AVAILABLE = False
    logger.warning("⚠️ Интеллектуальная система агентов недоступна")

logger = logging.getLogger(__name__)

class AgentCleaner(IntelligentAgent if INTELLIGENT_AGENTS_AVAILABLE else object):
    """🧹 Агент уборки: очистка логов, кэша, старых данных (с самообучением)"""
    
    def __init__(self, bot_dir: str = "/opt/bot"):
        if INTELLIGENT_AGENTS_AVAILABLE:
            super().__init__('cleaner_agent', 'cleaner')
        
        self.bot_dir = Path(bot_dir)
        self.log_dir = self.bot_dir / "logs"
        self.cache_dir = self.bot_dir / "data" / "cache"
        self.storage_dir = self.bot_dir / "data" / "storage"
        self.db_path = self.bot_dir / "trading_data.db"
        
    async def clean_logs(self, max_age_days: int = 7, max_size_mb: int = 100):
        """Очистка старых логов (с самообучением оптимальных параметров)"""
        try:
            # Применяем изученные правила
            if INTELLIGENT_AGENTS_AVAILABLE:
                context = {
                    'features': {
                        'disk_usage': self._get_disk_usage(),
                        'log_dir_size_mb': self._get_dir_size(self.log_dir),
                        'days_since_last_clean': self._get_days_since_last_clean()
                    },
                    'task': 'clean_logs'
                }
                
                learned_rule = self.apply_learned_rules(context)
                if learned_rule:
                    # Используем параметры из правила (диапазоны)
                    if 'feature_ranges' in learned_rule:
                        max_age_range = learned_rule['feature_ranges'].get('max_age_days', (7, 14))
                        max_age_days = (max_age_range[0] + max_age_range[1]) / 2
            
            if not self.log_dir.exists():
                return
            
            total_size = 0
            cleaned_files = 0
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            for log_file in self.log_dir.rglob("*.log"):
                try:
                    file_stat = log_file.stat()
                    file_date = datetime.fromtimestamp(file_stat.st_mtime)
                    
                    # Удаляем старые файлы
                    if file_date < cutoff_date:
                        total_size += file_stat.st_size
                        log_file.unlink()
                        cleaned_files += 1
                    
                    # Проверяем размер (если файл слишком большой, ротируем)
                    elif file_stat.st_size > max_size_mb * 1024 * 1024:
                        # Ротируем логи
                        rotated_file = log_file.parent / f"{log_file.stem}_{datetime.now().strftime('%Y%m%d')}.log"
                        shutil.move(str(log_file), str(rotated_file))
                        logger.info(f"📦 Лог ротирован: {log_file.name}")
                
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка обработки {log_file}: {e}")
            
            if cleaned_files > 0:
                logger.info(f"🧹 Очищено {cleaned_files} лог-файлов ({total_size/1024/1024:.1f} MB)")
                
                # Сохраняем опыт для обучения
                if INTELLIGENT_AGENTS_AVAILABLE:
                    experience = {
                        'features': {
                            'disk_usage_before': context.get('features', {}).get('disk_usage', 0),
                            'log_dir_size_before': context.get('features', {}).get('log_dir_size_mb', 0),
                            'max_age_days_used': max_age_days,
                            'files_cleaned': cleaned_files,
                            'size_freed_mb': total_size / 1024 / 1024
                        },
                        'result': 'success' if cleaned_files > 0 else 'failure',
                        'performance': cleaned_files / max(1, max_age_days)  # Метрика эффективности
                    }
                    self.learn_from_experience(experience)
        
        except Exception as e:
            logger.error(f"❌ Ошибка очистки логов: {e}")
            if INTELLIGENT_AGENTS_AVAILABLE:
                self.learn_from_experience({
                    'features': {'error_type': type(e).__name__},
                    'result': 'failure'
                })
    
    async def clean_cache(self, max_age_hours: int = 24):
        """Очистка кэша"""
        try:
            if not self.cache_dir.exists():
                return
            
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            cleaned = 0
            total_size = 0
            
            for cache_file in self.cache_dir.glob("*"):
                try:
                    if cache_file.is_file():
                        file_date = datetime.fromtimestamp(cache_file.stat().st_mtime)
                        if file_date < cutoff_time:
                            total_size += cache_file.stat().st_size
                            cache_file.unlink()
                            cleaned += 1
                except Exception:
                    pass
            
            if cleaned > 0:
                logger.info(f"🧹 Очищено {cleaned} кэш-файлов ({total_size/1024:.1f} KB)")
        
        except Exception as e:
            logger.error(f"❌ Ошибка очистки кэша: {e}")
    
    async def clean_database(self, max_age_days: int = 30):
        """Очистка старых данных из БД"""
        try:
            if not self.db_path.exists():
                return
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=max_age_days)).isoformat()
            
            # Удаляем старые рыночные данные
            cursor.execute('''
                DELETE FROM market_data 
                WHERE timestamp < ?
            ''', (cutoff_date,))
            market_deleted = cursor.rowcount
            
            # Удаляем старые решения без результата
            cursor.execute('''
                DELETE FROM trade_decisions 
                WHERE timestamp < ? AND result IS NULL
            ''', (cutoff_date,))
            decisions_deleted = cursor.rowcount
            
            conn.commit()
            
            # Оптимизация БД
            cursor.execute('VACUUM')
            conn.close()
            
            if market_deleted > 0 or decisions_deleted > 0:
                logger.info(f"🧹 БД: удалено {market_deleted} market_data, {decisions_deleted} решений")
        
        except Exception as e:
            logger.error(f"❌ Ошибка очистки БД: {e}")
    
    def _get_disk_usage(self) -> float:
        """Получить использование диска в процентах"""
        try:
            return psutil.disk_usage(str(self.bot_dir)).percent
        except:
            return 0.0
    
    def _get_dir_size(self, path: Path) -> float:
        """Получить размер директории в MB"""
        try:
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / 1024 / 1024
        except:
            return 0.0
    
    def _get_days_since_last_clean(self) -> float:
        """Получить дни с последней очистки"""
        try:
            if hasattr(self, 'last_clean_time'):
                return (datetime.now() - self.last_clean_time).total_seconds() / 86400
            return 7.0
        except:
            return 7.0
    
    async def run(self):
        """Запустить уборку (с самообучением)"""
        logger.info("🧹 Агент уборки запущен")
        
        results = await asyncio.gather(
            self.clean_logs(),
            self.clean_cache(),
            self.clean_database(),
            return_exceptions=True
        )
        
        # Учимся на результатах
        if INTELLIGENT_AGENTS_AVAILABLE:
            success_count = sum(1 for r in results if r is not None and not isinstance(r, Exception))
            if success_count > 0:
                self.last_clean_time = datetime.now()
                
                # Обмен знаниями с другими агентами
                self.share_knowledge_with_peers()

class AgentSecurity:
    """🔒 Агент безопасности: проверка API ключей, соединений"""
    
    def __init__(self, bot_dir: str = "/opt/bot"):
        self.bot_dir = Path(bot_dir)
        self.env_file = self.bot_dir / ".env"
        
    async def check_api_keys(self) -> Dict[str, bool]:
        """Проверка наличия и валидности API ключей"""
        checks = {
            'bybit_key_exists': False,
            'bybit_secret_exists': False,
            'telegram_token_exists': False,
            'openai_key_exists': False
        }
        
        try:
            if self.env_file.exists():
                env_content = self.env_file.read_text()
                
                checks['bybit_key_exists'] = 'BYBIT_API_KEY' in env_content and len(os.getenv('BYBIT_API_KEY', '')) > 0
                checks['bybit_secret_exists'] = 'BYBIT_API_SECRET' in env_content and len(os.getenv('BYBIT_API_SECRET', '')) > 0
                checks['telegram_token_exists'] = 'TELEGRAM' in env_content and len(os.getenv('TELEGRAM_BOT_TOKEN', '') or os.getenv('TELEGRAM_TOKEN', '')) > 0
                checks['openai_key_exists'] = 'OPENAI_API_KEY' in env_content and len(os.getenv('OPENAI_API_KEY', '')) > 0
            
            return checks
        
        except Exception as e:
            logger.error(f"❌ Ошибка проверки API ключей: {e}")
            return checks
    
    async def check_file_permissions(self) -> Dict[str, bool]:
        """Проверка прав доступа к файлам"""
        checks = {}
        
        critical_files = [
            self.bot_dir / ".env",
            self.bot_dir / "super_bot_v4_mtf.py",
            self.bot_dir / "trading_data.db"
        ]
        
        for file_path in critical_files:
            if file_path.exists():
                stat = file_path.stat()
                # Проверяем что файл не доступен всем (не 777)
                mode = stat.st_mode & 0o777
                checks[str(file_path)] = mode < 0o777
            else:
                checks[str(file_path)] = False
        
        return checks
    
    async def check_connections(self) -> Dict[str, bool]:
        """Проверка сетевых соединений"""
        checks = {
            'bybit_api': False,
            'telegram_api': False
        }
        
        # Простая проверка доступности (можно расширить)
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                # Проверка Telegram API
                try:
                    async with session.get('https://api.telegram.org', timeout=5) as resp:
                        checks['telegram_api'] = resp.status < 500
                except:
                    checks['telegram_api'] = False
                
                # Проверка Bybit API
                try:
                    async with session.get('https://api.bybit.com', timeout=5) as resp:
                        checks['bybit_api'] = resp.status < 500
                except:
                    checks['bybit_api'] = False
        
        except Exception as e:
            logger.debug(f"⚠️ Ошибка проверки соединений: {e}")
        
        return checks
    
    async def run(self) -> Dict[str, any]:
        """Запустить проверки безопасности"""
        logger.info("🔒 Агент безопасности запущен")
        
        results = {
            'api_keys': await self.check_api_keys(),
            'file_permissions': await self.check_file_permissions(),
            'connections': await self.check_connections()
        }
        
        # Логируем проблемы
        issues = []
        if not all(results['api_keys'].values()):
            issues.append("Неполные API ключи")
        if not all(results['file_permissions'].values()):
            issues.append("Проблемы с правами доступа")
        if not all(results['connections'].values()):
            issues.append("Проблемы с соединениями")
        
        if issues:
            logger.warning(f"⚠️ Найдены проблемы безопасности: {', '.join(issues)}")
        else:
            logger.info("✅ Все проверки безопасности пройдены")
        
        return results

class AgentStability:
    """⚖️ Агент стабильности: мониторинг производительности"""
    
    def __init__(self, bot_pid: Optional[int] = None):
        self.bot_pid = bot_pid
        self.metrics_history = []
        
    async def check_memory_usage(self) -> Dict[str, float]:
        """Проверка использования памяти"""
        try:
            if self.bot_pid:
                process = psutil.Process(self.bot_pid)
                process_memory = process.memory_info().rss / 1024 / 1024  # MB
            else:
                process_memory = 0
            
            system_memory = psutil.virtual_memory()
            
            return {
                'process_mb': process_memory,
                'system_percent': system_memory.percent,
                'system_available_mb': system_memory.available / 1024 / 1024,
                'warning': process_memory > 500 or system_memory.percent > 85
            }
        
        except Exception as e:
            logger.debug(f"⚠️ Ошибка проверки памяти: {e}")
            return {}
    
    async def check_disk_usage(self, bot_dir: str = "/opt/bot") -> Dict[str, float]:
        """Проверка использования диска"""
        try:
            disk = psutil.disk_usage(bot_dir)
            
            return {
                'total_gb': disk.total / 1024 / 1024 / 1024,
                'used_gb': disk.used / 1024 / 1024 / 1024,
                'free_gb': disk.free / 1024 / 1024 / 1024,
                'percent': disk.percent,
                'warning': disk.percent > 90
            }
        
        except Exception as e:
            logger.debug(f"⚠️ Ошибка проверки диска: {e}")
            return {}
    
    async def check_cpu_usage(self) -> Dict[str, float]:
        """Проверка использования CPU"""
        try:
            if self.bot_pid:
                process = psutil.Process(self.bot_pid)
                process_cpu = process.cpu_percent(interval=1)
            else:
                process_cpu = 0
            
            system_cpu = psutil.cpu_percent(interval=1)
            
            return {
                'process_percent': process_cpu,
                'system_percent': system_cpu,
                'warning': process_cpu > 80 or system_cpu > 90
            }
        
        except Exception as e:
            logger.debug(f"⚠️ Ошибка проверки CPU: {e}")
            return {}
    
    async def check_database_size(self, db_path: str = "/opt/bot/trading_data.db") -> Dict[str, float]:
        """Проверка размера БД"""
        try:
            db_file = Path(db_path)
            if db_file.exists():
                size_mb = db_file.stat().st_size / 1024 / 1024
                return {
                    'size_mb': size_mb,
                    'warning': size_mb > 1000  # > 1GB
                }
            return {'size_mb': 0, 'warning': False}
        
        except Exception as e:
            logger.debug(f"⚠️ Ошибка проверки БД: {e}")
            return {}
    
    async def run(self) -> Dict[str, any]:
        """Запустить проверки стабильности"""
        logger.info("⚖️ Агент стабильности запущен")
        
        results = {
            'memory': await self.check_memory_usage(),
            'disk': await self.check_disk_usage(),
            'cpu': await self.check_cpu_usage(),
            'database': await self.check_database_size(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Проверяем на предупреждения
        warnings = []
        if results.get('memory', {}).get('warning'):
            warnings.append("Высокое использование памяти")
        if results.get('disk', {}).get('warning'):
            warnings.append("Мало места на диске")
        if results.get('cpu', {}).get('warning'):
            warnings.append("Высокая загрузка CPU")
        if results.get('database', {}).get('warning'):
            warnings.append("Большая БД")
        
        if warnings:
            logger.warning(f"⚠️ Проблемы стабильности: {', '.join(warnings)}")
        
        # Сохраняем историю (последние 100 метрик)
        self.metrics_history.append(results)
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
        
        return results

class AgentRecovery:
    """🔄 Агент восстановления: автоматическое исправление проблем"""
    
    def __init__(self, bot_dir: str = "/opt/bot"):
        self.bot_dir = Path(bot_dir)
        self.recovery_count = 0
        
    async def recover_database(self) -> bool:
        """Восстановление БД при проблемах"""
        try:
            db_path = self.bot_dir / "trading_data.db"
            if not db_path.exists():
                logger.warning("⚠️ БД не найдена, требуется пересоздание")
                return False
            
            # Проверяем целостность
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            try:
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()
                
                if result[0] != 'ok':
                    logger.error(f"❌ БД повреждена: {result[0]}")
                    # Пробуем восстановить
                    cursor.execute("VACUUM")
                    logger.info("🔧 БД восстановлена через VACUUM")
                    return True
                else:
                    return True
            
            finally:
                conn.close()
        
        except Exception as e:
            logger.error(f"❌ Ошибка восстановления БД: {e}")
            return False
    
    async def recover_logs(self) -> bool:
        """Восстановление структуры логов"""
        try:
            log_dirs = [
                self.bot_dir / "logs",
                self.bot_dir / "logs" / "trading",
                self.bot_dir / "logs" / "system",
                self.bot_dir / "logs" / "ml"
            ]
            
            for log_dir in log_dirs:
                log_dir.mkdir(parents=True, exist_ok=True)
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Ошибка восстановления логов: {e}")
            return False
    
    async def recover_cache(self) -> bool:
        """Очистка поврежденного кэша"""
        try:
            cache_dir = self.bot_dir / "data" / "cache"
            if cache_dir.exists():
                # Удаляем поврежденные файлы
                for cache_file in cache_dir.glob("*.json"):
                    try:
                        import json
                        with open(cache_file, 'r') as f:
                            json.load(f)  # Проверка валидности
                    except:
                        cache_file.unlink()
                        logger.info(f"🗑️ Удален поврежденный кэш: {cache_file.name}")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Ошибка восстановления кэша: {e}")
            return False
    
    async def run(self) -> Dict[str, bool]:
        """Запустить восстановление"""
        logger.info("🔄 Агент восстановления запущен")
        
        results = {
            'database': await self.recover_database(),
            'logs': await self.recover_logs(),
            'cache': await self.recover_cache()
        }
        
        if any(results.values()):
            self.recovery_count += 1
            logger.info(f"✅ Восстановление выполнено (всего: {self.recovery_count})")
        
        return results

class SystemAgentsManager:
    """🤖 Менеджер всех агентов"""
    
    def __init__(self, bot_dir: str = "/opt/bot", bot_pid: Optional[int] = None):
        self.cleaner = AgentCleaner(bot_dir)
        self.security = AgentSecurity(bot_dir)
        self.stability = AgentStability(bot_pid)
        self.recovery = AgentRecovery(bot_dir)
        
        self.running = False
        
    async def run_all_agents(self):
        """Запустить всех агентов"""
        self.running = True
        logger.info("🤖 Запуск всех системных агентов")
        
        results = await asyncio.gather(
            self.cleaner.run(),
            self.security.run(),
            self.stability.run(),
            self.recovery.run(),
            return_exceptions=True
        )
        
        return {
            'cleaner': results[0] if not isinstance(results[0], Exception) else None,
            'security': results[1] if not isinstance(results[1], Exception) else None,
            'stability': results[2] if not isinstance(results[2], Exception) else None,
            'recovery': results[3] if not isinstance(results[3], Exception) else None
        }
    
    async def run_periodic(self, cleaner_interval: int = 3600,  # 1 час
                          security_interval: int = 1800,       # 30 мин
                          stability_interval: int = 300,        # 5 мин
                          recovery_interval: int = 7200):       # 2 часа
        """Запуск агентов по расписанию"""
        while self.running:
            await asyncio.sleep(stability_interval)
            
            try:
                # Стабильность каждые 5 минут
                await self.stability.run()
                
                # Безопасность каждые 30 минут
                if int(time.time()) % security_interval < stability_interval:
                    await self.security.run()
                
                # Уборка каждый час
                if int(time.time()) % cleaner_interval < stability_interval:
                    await self.cleaner.run()
                
                # Восстановление каждые 2 часа
                if int(time.time()) % recovery_interval < stability_interval:
                    await self.recovery.run()
            
            except Exception as e:
                logger.error(f"❌ Ошибка в периодических агентах: {e}")
    
    def stop(self):
        """Остановить агентов"""
        self.running = False

# Для использования
import time
if __name__ == "__main__":
    # Пример использования
    async def test():
        manager = SystemAgentsManager(bot_dir="/opt/bot")
        results = await manager.run_all_agents()
        print("Результаты:", results)
    
    asyncio.run(test())

