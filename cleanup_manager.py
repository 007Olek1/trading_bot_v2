#!/usr/bin/env python3
"""
Cleanup Manager - Управление очисткой и ротацией файлов
Авторотация логов, кэша, бэкапов для экономии места
"""

import os
import shutil
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Конфигурация ротации - ВСЕ ПАПКИ
CONFIG = {
    'logs': {
        'path': '/opt/bot/logs',
        'max_age_hours': 72,
        'max_size_mb': 200,
        'extensions': ['.log', '.txt']
    },
    'logs_archive': {
        'path': '/opt/bot/logs/archive',
        'max_age_hours': 48,
        'max_size_mb': 100
    },
    'logs_ml': {
        'path': '/opt/bot/logs/ml',
        'max_age_hours': 72,
        'max_size_mb': 50
    },
    'logs_system': {
        'path': '/opt/bot/logs/system',
        'max_age_hours': 72,
        'max_size_mb': 50
    },
    'cache': {
        'path': '/opt/bot/data/cache',
        'max_age_hours': 24,
        'max_size_mb': 50
    },
    'models': {
        'path': '/opt/bot/data/models',
        'max_age_hours': 168,  # 7 дней для моделей
        'max_size_mb': 100
    },
    'knowledge': {
        'path': '/opt/bot/data/knowledge',
        'max_age_hours': 168,
        'max_size_mb': 50
    },
    'backups': {
        'path': '/opt/bot/backups',
        'max_age_hours': 72,  # 3 дня для бэкапов
        'max_size_mb': 100
    },
    'disco57': {
        'path': '/opt/bot/disco57',
        'max_age_hours': 168,  # 7 дней для RL моделей
        'max_size_mb': 100
    },
    'reports': {
        'path': '/opt/bot/reports',
        'max_age_hours': 168,
        'max_size_mb': 50
    },
    'state': {
        'path': '/opt/bot/state',
        'max_age_hours': 72,
        'max_size_mb': 20
    },
    'pycache': {
        'path': '/opt/bot/__pycache__',
        'max_age_hours': 24,
        'max_size_mb': 20
    },
    'log_txt': {
        'path': '/opt/bot/log.txt',
        'max_size_mb': 10,
        'rotate_count': 3
    },
    'tradegpt_log': {
        'path': '/opt/bot/tradegpt_v5.log',
        'max_size_mb': 10,
        'rotate_count': 3
    }
}


class CleanupManager:
    """Менеджер очистки и ротации файлов"""
    
    def __init__(self):
        self.stats = {
            'files_deleted': 0,
            'bytes_freed': 0,
            'errors': []
        }
    
    def run_full_cleanup(self):
        """Запустить полную очистку ВСЕХ папок"""
        logger.info("=" * 50)
        logger.info("Запуск полной очистки...")
        logger.info("=" * 50)
        
        self.stats = {'files_deleted': 0, 'bytes_freed': 0, 'errors': []}
        
        # Очистка всех директорий из конфига
        for name, cfg in CONFIG.items():
            if 'rotate_count' in cfg:
                # Это лог-файл для ротации
                self._rotate_log_file(
                    cfg['path'],
                    cfg['max_size_mb'],
                    cfg['rotate_count']
                )
            elif 'path' in cfg and 'max_age_hours' in cfg:
                # Это директория для очистки
                self._cleanup_directory(
                    cfg['path'],
                    cfg['max_age_hours'],
                    cfg['max_size_mb'],
                    cfg.get('extensions')
                )
        
        # Очистка старых .py.backup файлов
        self._cleanup_backup_files('/opt/bot', 72)
        
        # Очистка старых .backup_ файлов в корне
        self._cleanup_backup_files('/opt/bot', 48)
        
        # Итоги
        freed_mb = self.stats['bytes_freed'] / (1024 * 1024)
        logger.info("=" * 50)
        logger.info(f"Очистка завершена:")
        logger.info(f"  Удалено файлов: {self.stats['files_deleted']}")
        logger.info(f"  Освобождено: {freed_mb:.2f} MB")
        if self.stats['errors']:
            logger.warning(f"  Ошибок: {len(self.stats['errors'])}")
        logger.info("=" * 50)
        
        return self.stats
    
    def _cleanup_directory(self, dir_path: str, max_age_hours: int, 
                          max_size_mb: int, extensions: List[str] = None):
        """Очистить директорию от старых файлов"""
        if not os.path.exists(dir_path):
            return
        
        logger.info(f"Очистка: {dir_path}")
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        total_size = 0
        files_to_delete = []
        
        try:
            for root, dirs, files in os.walk(dir_path):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    
                    # Проверка расширения
                    if extensions:
                        if not any(filename.endswith(ext) for ext in extensions):
                            continue
                    
                    try:
                        stat = os.stat(filepath)
                        file_age = stat.st_mtime
                        file_size = stat.st_size
                        total_size += file_size
                        
                        # Удалить если старше лимита
                        if file_age < cutoff_time:
                            files_to_delete.append((filepath, file_size))
                    except OSError:
                        continue
            
            # Если размер превышает лимит, удалить самые старые
            max_size_bytes = max_size_mb * 1024 * 1024
            if total_size > max_size_bytes:
                # Сортируем по времени модификации
                all_files = []
                for root, dirs, files in os.walk(dir_path):
                    for filename in files:
                        filepath = os.path.join(root, filename)
                        try:
                            stat = os.stat(filepath)
                            all_files.append((filepath, stat.st_mtime, stat.st_size))
                        except OSError:
                            continue
                
                all_files.sort(key=lambda x: x[1])  # По времени
                
                # Удаляем пока не уложимся в лимит
                current_size = total_size
                for filepath, mtime, size in all_files:
                    if current_size <= max_size_bytes:
                        break
                    if (filepath, size) not in files_to_delete:
                        files_to_delete.append((filepath, size))
                    current_size -= size
            
            # Удаляем файлы
            for filepath, size in files_to_delete:
                try:
                    os.remove(filepath)
                    self.stats['files_deleted'] += 1
                    self.stats['bytes_freed'] += size
                    logger.debug(f"  Удален: {filepath}")
                except OSError as e:
                    self.stats['errors'].append(str(e))
            
            if files_to_delete:
                logger.info(f"  Удалено {len(files_to_delete)} файлов")
                
        except Exception as e:
            logger.error(f"Ошибка очистки {dir_path}: {e}")
            self.stats['errors'].append(str(e))
    
    def _rotate_log_file(self, log_path: str, max_size_mb: int, rotate_count: int):
        """Ротация лог-файла"""
        if not os.path.exists(log_path):
            return
        
        try:
            size = os.path.getsize(log_path)
            max_size_bytes = max_size_mb * 1024 * 1024
            
            if size > max_size_bytes:
                logger.info(f"Ротация лога: {log_path} ({size / 1024 / 1024:.1f} MB)")
                
                # Удалить самый старый
                oldest = f"{log_path}.{rotate_count}"
                if os.path.exists(oldest):
                    os.remove(oldest)
                    self.stats['bytes_freed'] += os.path.getsize(oldest) if os.path.exists(oldest) else 0
                
                # Сдвинуть остальные
                for i in range(rotate_count - 1, 0, -1):
                    old_name = f"{log_path}.{i}"
                    new_name = f"{log_path}.{i + 1}"
                    if os.path.exists(old_name):
                        os.rename(old_name, new_name)
                
                # Переименовать текущий
                os.rename(log_path, f"{log_path}.1")
                
                # Создать новый пустой
                open(log_path, 'w').close()
                
                self.stats['files_deleted'] += 1
                self.stats['bytes_freed'] += size
                logger.info(f"  Лог ротирован, освобождено {size / 1024 / 1024:.1f} MB")
                
        except Exception as e:
            logger.error(f"Ошибка ротации {log_path}: {e}")
            self.stats['errors'].append(str(e))
    
    def _cleanup_backup_files(self, dir_path: str, max_age_hours: int):
        """Очистить старые .backup файлы"""
        if not os.path.exists(dir_path):
            return
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        try:
            for filename in os.listdir(dir_path):
                if '.backup' in filename or filename.endswith('.bak'):
                    filepath = os.path.join(dir_path, filename)
                    try:
                        stat = os.stat(filepath)
                        if stat.st_mtime < cutoff_time:
                            size = stat.st_size
                            os.remove(filepath)
                            self.stats['files_deleted'] += 1
                            self.stats['bytes_freed'] += size
                            logger.info(f"  Удален backup: {filename}")
                    except OSError:
                        continue
        except Exception as e:
            logger.error(f"Ошибка очистки backup файлов: {e}")
    
    def get_disk_usage(self) -> dict:
        """Получить информацию об использовании диска"""
        bot_path = '/opt/bot'
        
        usage = {
            'total_mb': 0,
            'logs_mb': 0,
            'data_mb': 0,
            'cache_mb': 0,
            'backups_mb': 0
        }
        
        def get_dir_size(path):
            total = 0
            if os.path.exists(path):
                for root, dirs, files in os.walk(path):
                    for f in files:
                        try:
                            total += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            pass
            return total
        
        usage['logs_mb'] = get_dir_size('/opt/bot/logs') / (1024 * 1024)
        usage['data_mb'] = get_dir_size('/opt/bot/data') / (1024 * 1024)
        usage['cache_mb'] = get_dir_size('/opt/bot/data/cache') / (1024 * 1024)
        usage['backups_mb'] = get_dir_size('/opt/bot/backups') / (1024 * 1024)
        usage['total_mb'] = get_dir_size(bot_path) / (1024 * 1024)
        
        return usage


def setup_cron_cleanup():
    """Настроить cron для автоматической очистки"""
    cron_line = "0 */6 * * * cd /opt/bot && python3 cleanup_manager.py >> /opt/bot/logs/cleanup.log 2>&1"
    
    print("Добавьте в crontab следующую строку для автоочистки каждые 6 часов:")
    print(cron_line)
    print("\nКоманда: crontab -e")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    manager = CleanupManager()
    
    # Показать текущее использование
    usage = manager.get_disk_usage()
    print("\n📊 Текущее использование диска:")
    print(f"  Всего: {usage['total_mb']:.1f} MB")
    print(f"  Логи: {usage['logs_mb']:.1f} MB")
    print(f"  Данные: {usage['data_mb']:.1f} MB")
    print(f"  Кэш: {usage['cache_mb']:.1f} MB")
    print(f"  Бэкапы: {usage['backups_mb']:.1f} MB")
    
    # Запустить очистку
    print("\n🧹 Запуск очистки...")
    stats = manager.run_full_cleanup()
    
    # Показать результат
    print(f"\n✅ Очистка завершена!")
    print(f"  Удалено файлов: {stats['files_deleted']}")
    print(f"  Освобождено: {stats['bytes_freed'] / 1024 / 1024:.1f} MB")
