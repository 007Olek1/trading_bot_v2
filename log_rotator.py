"""
🔄 LOG ROTATOR - Автоматическая ротация и очистка логов
Экономит место на сервере
"""

import gzip
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import logging


class LogRotator:
    """Ротация и сжатие логов"""
    
    def __init__(self, logs_dir: str = "logs", max_size_mb: int = 10, keep_days: int = 7):
        self.logs_dir = Path(logs_dir)
        self.logger = logging.getLogger(__name__)
        self.max_size_mb = max_size_mb
        self.keep_days = keep_days
    
    def rotate_logs(self, max_size_mb: int = 10, keep_days: int = 7):
        """
        Ротация логов
        
        Args:
            max_size_mb: Максимальный размер файла в MB
            keep_days: Сколько дней хранить старые логи
        """
        self.logger.info("🔄 Starting log rotation...")
        
        # Файлы для ротации
        log_files = [
            "trading_bot_v4.log",
            "bot_output.log",
            "critical_alerts.log",
            "uptime.log"
        ]
        
        for log_file in log_files:
            log_path = self.logs_dir / log_file
            
            if not log_path.exists():
                continue
            
            # Проверяем размер
            size_mb = log_path.stat().st_size / (1024 * 1024)
            
            if size_mb > max_size_mb:
                self._rotate_file(log_path)
        
        # Очистка старых логов
        self._cleanup_old_logs(keep_days)
        
        self.logger.info("✅ Log rotation completed")
    
    def _rotate_file(self, log_path: Path):
        """Ротация одного файла"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Новое имя с timestamp
        rotated_name = f"{log_path.stem}_{timestamp}.log"
        rotated_path = self.logs_dir / rotated_name
        
        # Копируем файл
        shutil.copy2(log_path, rotated_path)
        
        # Сжимаем
        compressed_path = rotated_path.with_suffix('.log.gz')
        with open(rotated_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Удаляем несжатую копию
        rotated_path.unlink()
        
        # Очищаем оригинальный файл
        with open(log_path, 'w') as f:
            f.write(f"# Log rotated at {datetime.now().isoformat()}\n")
        
        size_mb = compressed_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"📦 Rotated and compressed: {log_path.name} -> {compressed_path.name} ({size_mb:.2f} MB)")
    
    def _cleanup_old_logs(self, keep_days: int):
        """Удаление старых сжатых логов"""
        cutoff = datetime.now() - timedelta(days=keep_days)
        
        deleted_count = 0
        freed_mb = 0
        
        for gz_file in self.logs_dir.glob("*.log.gz"):
            # Проверяем дату модификации
            mtime = datetime.fromtimestamp(gz_file.stat().st_mtime)
            
            if mtime < cutoff:
                size_mb = gz_file.stat().st_size / (1024 * 1024)
                gz_file.unlink()
                deleted_count += 1
                freed_mb += size_mb
        
        if deleted_count > 0:
            self.logger.info(f"🗑️ Deleted {deleted_count} old logs, freed {freed_mb:.2f} MB")
    
    def get_old_files(self) -> list:
        """Получить список старых файлов для удаления"""
        cutoff = datetime.now() - timedelta(days=self.keep_days)
        old_files = []
        
        for gz_file in self.logs_dir.glob("*.log.gz"):
            mtime = datetime.fromtimestamp(gz_file.stat().st_mtime)
            if mtime < cutoff:
                old_files.append(gz_file)
        
        return old_files
    
    def get_logs_info(self) -> dict:
        """Информация о логах"""
        total_size = 0
        files_info = []
        
        for log_file in self.logs_dir.glob("*"):
            if log_file.is_file():
                size = log_file.stat().st_size
                total_size += size
                
                files_info.append({
                    "name": log_file.name,
                    "size_mb": size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                })
        
        return {
            "total_size_mb": total_size / (1024 * 1024),
            "files_count": len(files_info),
            "files": sorted(files_info, key=lambda x: x["size_mb"], reverse=True)
        }
