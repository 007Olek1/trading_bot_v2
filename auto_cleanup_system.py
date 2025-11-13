#!/usr/bin/env python3
"""
🧹 СИСТЕМА АВТООЧИСТКИ ЛОГОВ И БАЗЫ ДАННЫХ
==========================================

Автоматически очищает:
- Старые логи (>7 дней)
- Старые записи в БД (>30 дней для market_data, >90 дней для trade_decisions)
- Временные файлы и кэши
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class AutoCleanupSystem:
    """🧹 Система автоочистки"""
    
    def __init__(self):
        # Определяем базовый путь
        if Path("/opt/bot").exists():
            self.base_dir = Path("/opt/bot")
        else:
            self.base_dir = Path(__file__).parent
        
        self.logs_dir = self.base_dir / "logs"
        self.data_dir = self.base_dir / "data"
        self.cache_dir = self.data_dir / "cache"
        self.db_path = self.base_dir / "trading_data.db"
        if not self.db_path.exists():
            self.db_path = self.data_dir / "trading_data.db"
        
        # Настройки очистки
        self.log_retention_days = 7  # Логи старше 7 дней
        self.cache_retention_hours = 24  # Кэш старше 24 часов
        self.market_data_retention_days = 30  # Рыночные данные старше 30 дней
        self.trade_decisions_retention_days = 90  # Торговые решения старше 90 дней (важные данные)
        self.temp_files_retention_hours = 12  # Временные файлы старше 12 часов
        
        self.stats = {
            'logs_deleted': 0,
            'logs_freed_mb': 0.0,
            'cache_deleted': 0,
            'cache_freed_mb': 0.0,
            'db_records_deleted': 0,
            'temp_files_deleted': 0,
            'temp_freed_mb': 0.0,
            'total_freed_mb': 0.0
        }
    
    def cleanup_logs(self) -> bool:
        """🧹 Очистка старых логов и больших файлов"""
        logger.info("\n" + "="*60)
        logger.info("🧹 ОЧИСТКА СТАРЫХ ЛОГОВ")
        logger.info("="*60)
        
        try:
            if not self.logs_dir.exists():
                logger.warning("  ⚠️ Папка logs не найдена")
                return False
            
            cutoff_date = datetime.now() - timedelta(days=self.log_retention_days)
            cutoff_timestamp = cutoff_date.timestamp()
            
            logs_deleted = 0
            freed_size = 0
            
            # 1. Удаляем старые .log файлы (>7 дней)
            for log_file in self.logs_dir.rglob("*.log"):
                try:
                    if log_file.is_file() and log_file.stat().st_mtime < cutoff_timestamp:
                        size = log_file.stat().st_size
                        log_file.unlink()
                        logs_deleted += 1
                        freed_size += size
                        logger.debug(f"  🗑️ Удален старый: {log_file.name} ({size / 1024 / 1024:.1f} MB)")
                except Exception as e:
                    logger.warning(f"  ⚠️ Ошибка удаления {log_file}: {e}")
            
            # 2. Удаляем очень большие файлы (>500MB) старше 1 дня
            large_cutoff = datetime.now() - timedelta(days=1)
            large_cutoff_timestamp = large_cutoff.timestamp()
            for log_file in self.logs_dir.rglob("*.log"):
                try:
                    if log_file.is_file():
                        size = log_file.stat().st_size
                        # Если файл >500MB и старше 1 дня
                        if size > 500 * 1024 * 1024 and log_file.stat().st_mtime < large_cutoff_timestamp:
                            log_file.unlink()
                            logs_deleted += 1
                            freed_size += size
                            logger.info(f"  🗑️ Удален большой файл: {log_file.name} ({size / 1024 / 1024:.1f} MB)")
                        # Если файл >2GB (критично) - удаляем независимо от возраста
                        elif size > 2 * 1024 * 1024 * 1024:
                            log_file.unlink()
                            logs_deleted += 1
                            freed_size += size
                            logger.warning(f"  ⚠️ Удален критично большой файл: {log_file.name} ({size / 1024 / 1024 / 1024:.2f} GB)")
                except Exception as e:
                    logger.warning(f"  ⚠️ Ошибка удаления {log_file}: {e}")
            
            # 3. Удаляем дубликаты ротированных логов (с длинными именами с датами)
            for log_file in self.logs_dir.rglob("*2025*.log"):
                try:
                    if log_file.is_file() and log_file.stat().st_mtime < large_cutoff_timestamp:
                        size = log_file.stat().st_size
                        # Ротированные логи с датами в имени старше 1 дня
                        if "2025" in log_file.name and size > 10 * 1024 * 1024:  # >10MB
                            log_file.unlink()
                            logs_deleted += 1
                            freed_size += size
                            logger.debug(f"  🗑️ Удален ротированный: {log_file.name} ({size / 1024 / 1024:.1f} MB)")
                except Exception as e:
                    logger.debug(f"  ⚠️ Ошибка удаления {log_file}: {e}")
            
            self.stats['logs_deleted'] = logs_deleted
            self.stats['logs_freed_mb'] = round(freed_size / (1024**2), 2)
            
            if logs_deleted > 0:
                logger.info(f"  ✅ Удалено логов: {logs_deleted} файлов")
                logger.info(f"  💾 Освобождено: {self.stats['logs_freed_mb']:.2f} MB")
            else:
                logger.info(f"  ✅ Старых логов не найдено (все свежее {self.log_retention_days} дней)")
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Ошибка очистки логов: {e}")
            return False
    
    def cleanup_cache(self) -> bool:
        """🗂️ Очистка старых кэшей"""
        logger.info("\n" + "="*60)
        logger.info("🗂️ ОЧИСТКА СТАРЫХ КЭШЕЙ")
        logger.info("="*60)
        
        try:
            if not self.cache_dir.exists():
                logger.warning("  ⚠️ Папка cache не найдена")
                return False
            
            cutoff_date = datetime.now() - timedelta(hours=self.cache_retention_hours)
            cutoff_timestamp = cutoff_date.timestamp()
            
            cache_deleted = 0
            freed_size = 0
            
            # Удаляем старые файлы кэша
            for cache_file in self.cache_dir.rglob("*"):
                try:
                    if cache_file.is_file() and cache_file.stat().st_mtime < cutoff_timestamp:
                        size = cache_file.stat().st_size
                        cache_file.unlink()
                        cache_deleted += 1
                        freed_size += size
                except Exception as e:
                    logger.debug(f"  ⚠️ Ошибка удаления {cache_file}: {e}")
            
            self.stats['cache_deleted'] = cache_deleted
            self.stats['cache_freed_mb'] = round(freed_size / (1024**2), 2)
            
            if cache_deleted > 0:
                logger.info(f"  ✅ Удалено кэшей: {cache_deleted} файлов")
                logger.info(f"  💾 Освобождено: {self.stats['cache_freed_mb']:.2f} MB")
            else:
                logger.info(f"  ✅ Старых кэшей не найдено (все свежее {self.cache_retention_hours} часов)")
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Ошибка очистки кэша: {e}")
            return False
    
    def cleanup_database(self) -> bool:
        """💾 Очистка старой базы данных"""
        logger.info("\n" + "="*60)
        logger.info("💾 ОЧИСТКА БАЗЫ ДАННЫХ")
        logger.info("="*60)
        
        try:
            if not self.db_path.exists():
                logger.warning("  ⚠️ База данных не найдена")
                return False
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            total_deleted = 0
            
            # 1. Очистка старых market_data (старше 30 дней)
            try:
                cutoff_date = datetime.now() - timedelta(days=self.market_data_retention_days)
                cutoff_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')
                
                cursor.execute("""
                    SELECT COUNT(*) FROM market_data
                    WHERE datetime(timestamp) < datetime(?)
                """, (cutoff_str,))
                
                count_before = cursor.fetchone()[0]
                
                if count_before > 0:
                    cursor.execute("""
                        DELETE FROM market_data
                        WHERE datetime(timestamp) < datetime(?)
                    """, (cutoff_str,))
                    
                    deleted_market = cursor.rowcount
                    total_deleted += deleted_market
                    logger.info(f"  ✅ Удалено старых market_data: {deleted_market} записей (старше {self.market_data_retention_days} дней)")
                else:
                    logger.info(f"  ✅ Старых market_data не найдено (все свежее {self.market_data_retention_days} дней)")
                    
            except Exception as e:
                logger.warning(f"  ⚠️ Ошибка очистки market_data: {e}")
            
            # 2. Очистка старых trade_decisions (старше 90 дней, но сохраняем успешные сделки)
            try:
                cutoff_date = datetime.now() - timedelta(days=self.trade_decisions_retention_days)
                cutoff_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')
                
                # Удаляем только неудачные сделки старше 90 дней, успешные сохраняем дольше
                cursor.execute("""
                    SELECT COUNT(*) FROM trade_decisions
                    WHERE datetime(timestamp) < datetime(?)
                    AND (result = 'loss' OR result IS NULL OR result = '')
                """, (cutoff_str,))
                
                count_before = cursor.fetchone()[0]
                
                if count_before > 0:
                    cursor.execute("""
                        DELETE FROM trade_decisions
                        WHERE datetime(timestamp) < datetime(?)
                        AND (result = 'loss' OR result IS NULL OR result = '')
                    """, (cutoff_str,))
                    
                    deleted_trades = cursor.rowcount
                    total_deleted += deleted_trades
                    logger.info(f"  ✅ Удалено старых trade_decisions (loss): {deleted_trades} записей (старше {self.trade_decisions_retention_days} дней)")
                else:
                    logger.info(f"  ✅ Старых trade_decisions (loss) не найдено")
                
                # Также удаляем очень старые успешные сделки (старше 180 дней)
                old_cutoff = datetime.now() - timedelta(days=180)
                old_cutoff_str = old_cutoff.strftime('%Y-%m-%d %H:%M:%S')
                
                cursor.execute("""
                    SELECT COUNT(*) FROM trade_decisions
                    WHERE datetime(timestamp) < datetime(?)
                """, (old_cutoff_str,))
                
                count_very_old = cursor.fetchone()[0]
                
                if count_very_old > 0:
                    cursor.execute("""
                        DELETE FROM trade_decisions
                        WHERE datetime(timestamp) < datetime(?)
                    """, (old_cutoff_str,))
                    
                    deleted_old = cursor.rowcount
                    total_deleted += deleted_old
                    logger.info(f"  ✅ Удалено очень старых trade_decisions (все): {deleted_old} записей (старше 180 дней)")
                    
            except Exception as e:
                logger.warning(f"  ⚠️ Ошибка очистки trade_decisions: {e}")
            
            # 3. Оптимизация БД (VACUUM)
            try:
                logger.info("  🔄 Оптимизация базы данных...")
                cursor.execute("VACUUM")
                logger.info("  ✅ База данных оптимизирована")
            except Exception as e:
                logger.warning(f"  ⚠️ Ошибка оптимизации БД: {e}")
            
            conn.commit()
            conn.close()
            
            self.stats['db_records_deleted'] = total_deleted
            
            if total_deleted > 0:
                logger.info(f"\n  ✅ ИТОГО удалено из БД: {total_deleted} записей")
            else:
                logger.info(f"\n  ✅ База данных чистая, удалять нечего")
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Ошибка очистки базы данных: {e}")
            return False
    
    def cleanup_temp_files(self) -> bool:
        """🗑️ Очистка временных файлов"""
        logger.info("\n" + "="*60)
        logger.info("🗑️ ОЧИСТКА ВРЕМЕННЫХ ФАЙЛОВ")
        logger.info("="*60)
        
        try:
            cutoff_date = datetime.now() - timedelta(hours=self.temp_files_retention_hours)
            cutoff_timestamp = cutoff_date.timestamp()
            
            temp_patterns = ['*.tmp', '*.temp', '*~', '*.swp', '*.bak']
            temp_files_deleted = 0
            freed_size = 0
            
            for pattern in temp_patterns:
                for temp_file in self.base_dir.rglob(pattern):
                    try:
                        if temp_file.is_file() and temp_file.stat().st_mtime < cutoff_timestamp:
                            size = temp_file.stat().st_size
                            temp_file.unlink()
                            temp_files_deleted += 1
                            freed_size += size
                    except Exception as e:
                        logger.debug(f"  ⚠️ Ошибка удаления {temp_file}: {e}")
            
            # Удаляем __pycache__
            pycache_deleted = 0
            for pycache_dir in self.base_dir.rglob('__pycache__'):
                try:
                    if pycache_dir.is_dir():
                        size = sum(f.stat().st_size for f in pycache_dir.rglob('*') if f.is_file())
                        shutil.rmtree(pycache_dir)
                        pycache_deleted += 1
                        freed_size += size
                except Exception as e:
                    logger.debug(f"  ⚠️ Ошибка удаления {pycache_dir}: {e}")
            
            self.stats['temp_files_deleted'] = temp_files_deleted + pycache_deleted
            self.stats['temp_freed_mb'] = round(freed_size / (1024**2), 2)
            
            if self.stats['temp_files_deleted'] > 0:
                logger.info(f"  ✅ Удалено временных файлов: {temp_files_deleted} файлов, {pycache_deleted} __pycache__")
                logger.info(f"  💾 Освобождено: {self.stats['temp_freed_mb']:.2f} MB")
            else:
                logger.info(f"  ✅ Временных файлов не найдено")
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Ошибка очистки временных файлов: {e}")
            return False
    
    def run_full_cleanup(self):
        """🚀 Полная автоочистка"""
        logger.info("\n" + "="*60)
        logger.info("🚀 ЗАПУСК ПОЛНОЙ АВТООЧИСТКИ")
        logger.info("="*60)
        logger.info(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Запускаем все виды очистки
        self.cleanup_logs()
        self.cleanup_cache()
        self.cleanup_database()
        self.cleanup_temp_files()
        
        # Подсчитываем общее освобожденное место
        self.stats['total_freed_mb'] = (
            self.stats['logs_freed_mb'] +
            self.stats['cache_freed_mb'] +
            self.stats['temp_freed_mb']
        )
        
        # Итоговый отчет
        logger.info("\n" + "="*60)
        logger.info("📊 ИТОГОВЫЙ ОТЧЕТ АВТООЧИСТКИ")
        logger.info("="*60)
        logger.info(f"🧹 Удалено логов: {self.stats['logs_deleted']} файлов ({self.stats['logs_freed_mb']:.2f} MB)")
        logger.info(f"🗂️ Удалено кэшей: {self.stats['cache_deleted']} файлов ({self.stats['cache_freed_mb']:.2f} MB)")
        logger.info(f"💾 Удалено из БД: {self.stats['db_records_deleted']} записей")
        logger.info(f"🗑️ Удалено временных: {self.stats['temp_files_deleted']} объектов ({self.stats['temp_freed_mb']:.2f} MB)")
        logger.info(f"\n💾 ВСЕГО ОСВОБОЖДЕНО: {self.stats['total_freed_mb']:.2f} MB")
        logger.info("="*60)
        
        return self.stats


def main():
    """Главная функция"""
    cleanup = AutoCleanupSystem()
    cleanup.run_full_cleanup()


if __name__ == "__main__":
    main()

