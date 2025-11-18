#!/usr/bin/env python3
"""
⏰ UPTIME MONITOR
Мониторинг работоспособности бота и отправка статуса в Uptime Robot
"""

import os
import sys
import time
import requests
import psutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()


class UptimeMonitor:
    """Мониторинг uptime и health check"""
    
    def __init__(self):
        # Uptime Robot API (опционально)
        self.uptime_robot_api_key = os.getenv("UPTIME_ROBOT_API_KEY", "")
        self.uptime_robot_monitor_id = os.getenv("UPTIME_ROBOT_MONITOR_ID", "")
        
        # Telegram для алертов
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        
        # Пути
        self.bot_dir = Path(__file__).parent
        self.log_file = self.bot_dir / "logs" / "trading_bot_v4.log"
        self.uptime_log = self.bot_dir / "logs" / "uptime.log"
        
        # Последний статус
        self.last_status = None
        self.last_alert_time = None
    
    def check_bot_process(self) -> dict:
        """
        Проверяет процесс бота
        
        Returns:
            Словарь со статусом процесса
        """
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'cpu_percent', 'memory_percent']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'python' in cmdline.lower() and 'main.py' in cmdline:
                    uptime_seconds = time.time() - proc.info['create_time']
                    
                    return {
                        'running': True,
                        'pid': proc.info['pid'],
                        'uptime_seconds': uptime_seconds,
                        'uptime_hours': uptime_seconds / 3600,
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent'],
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return {'running': False}
    
    def check_log_activity(self) -> dict:
        """
        Проверяет активность в логах
        
        Returns:
            Словарь со статусом логов
        """
        if not self.log_file.exists():
            return {'active': False, 'reason': 'Log file not found'}
        
        # Проверяем время последней записи
        last_modified = self.log_file.stat().st_mtime
        time_since_update = time.time() - last_modified
        
        # Если логи не обновлялись более 10 минут - проблема
        if time_since_update > 600:
            return {
                'active': False,
                'reason': f'No log updates for {time_since_update/60:.1f} minutes',
                'last_update': datetime.fromtimestamp(last_modified).isoformat()
            }
        
        # Проверяем наличие ошибок в последних строках
        try:
            with open(self.log_file, 'r') as f:
                last_lines = f.readlines()[-50:]
            
            error_count = sum(1 for line in last_lines if 'ERROR' in line or 'CRITICAL' in line)
            
            return {
                'active': True,
                'last_update': datetime.fromtimestamp(last_modified).isoformat(),
                'recent_errors': error_count,
                'time_since_update': time_since_update
            }
        except Exception as e:
            return {'active': False, 'reason': f'Error reading log: {e}'}
    
    def check_system_resources(self) -> dict:
        """
        Проверяет системные ресурсы
        
        Returns:
            Словарь с метриками системы
        """
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3),
        }
    
    def get_health_status(self) -> dict:
        """
        Получает полный статус здоровья бота
        
        Returns:
            Словарь со статусом
        """
        process = self.check_bot_process()
        logs = self.check_log_activity()
        resources = self.check_system_resources()
        
        # Определяем общий статус
        is_healthy = True
        issues = []
        
        if not process['running']:
            is_healthy = False
            issues.append('Bot process not running')
        
        if not logs.get('active', False):
            is_healthy = False
            issues.append(logs.get('reason', 'Log inactive'))
        
        if logs.get('recent_errors', 0) > 10:
            is_healthy = False
            issues.append(f"Many recent errors: {logs['recent_errors']}")
        
        if resources['cpu_percent'] > 80:
            is_healthy = False
            issues.append(f"High CPU: {resources['cpu_percent']:.1f}%")
        
        if resources['memory_percent'] > 90:
            is_healthy = False
            issues.append(f"High memory: {resources['memory_percent']:.1f}%")
        
        if resources['disk_free_gb'] < 2:
            is_healthy = False
            issues.append(f"Low disk space: {resources['disk_free_gb']:.1f}GB")
        
        return {
            'healthy': is_healthy,
            'timestamp': datetime.now().isoformat(),
            'process': process,
            'logs': logs,
            'resources': resources,
            'issues': issues
        }
    
    def send_telegram_alert(self, message: str):
        """
        Отправляет алерт в Telegram
        
        Args:
            message: Текст сообщения
        """
        if not self.telegram_token or not self.telegram_chat_id:
            print("⚠️ Telegram credentials not configured")
            return
        
        # Ограничение частоты алертов (не чаще раза в час)
        if self.last_alert_time:
            time_since_last = time.time() - self.last_alert_time
            if time_since_last < 3600:
                print(f"⏳ Skipping alert (last sent {time_since_last/60:.1f} min ago)")
                return
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                print("✅ Alert sent to Telegram")
                self.last_alert_time = time.time()
            else:
                print(f"❌ Failed to send alert: {response.status_code}")
        except Exception as e:
            print(f"❌ Error sending alert: {e}")
    
    def update_uptime_robot(self, status: dict):
        """
        Обновляет статус в Uptime Robot
        
        Args:
            status: Статус здоровья
        """
        if not self.uptime_robot_api_key or not self.uptime_robot_monitor_id:
            return
        
        try:
            # Uptime Robot API для обновления монитора
            url = "https://api.uptimerobot.com/v2/editMonitor"
            
            data = {
                'api_key': self.uptime_robot_api_key,
                'id': self.uptime_robot_monitor_id,
                'status': 2 if status['healthy'] else 9,  # 2=up, 9=down
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                print("✅ Uptime Robot updated")
            else:
                print(f"⚠️ Failed to update Uptime Robot: {response.status_code}")
        except Exception as e:
            print(f"❌ Error updating Uptime Robot: {e}")
    
    def log_uptime(self, status: dict):
        """
        Записывает uptime в лог
        
        Args:
            status: Статус здоровья
        """
        self.uptime_log.parent.mkdir(exist_ok=True)
        
        with open(self.uptime_log, 'a') as f:
            f.write(f"{status['timestamp']} | ")
            f.write(f"Healthy: {status['healthy']} | ")
            
            if status['process']['running']:
                f.write(f"Uptime: {status['process']['uptime_hours']:.1f}h | ")
                f.write(f"CPU: {status['process']['cpu_percent']:.1f}% | ")
                f.write(f"Mem: {status['process']['memory_percent']:.1f}% | ")
            else:
                f.write("Process: DOWN | ")
            
            if status['issues']:
                f.write(f"Issues: {', '.join(status['issues'])}")
            
            f.write("\n")
    
    def run_check(self):
        """Запускает проверку и отправляет алерты при необходимости"""
        print(f"🔍 Running health check at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        status = self.get_health_status()
        
        # Выводим статус
        if status['healthy']:
            print("✅ Bot is healthy")
            if status['process']['running']:
                print(f"   Uptime: {status['process']['uptime_hours']:.1f} hours")
                print(f"   CPU: {status['process']['cpu_percent']:.1f}%")
                print(f"   Memory: {status['process']['memory_percent']:.1f}%")
        else:
            print("❌ Bot has issues:")
            for issue in status['issues']:
                print(f"   - {issue}")
        
        # Логируем
        self.log_uptime(status)
        
        # Отправляем алерт если статус изменился
        if self.last_status is not None and status['healthy'] != self.last_status:
            if not status['healthy']:
                # Бот упал
                message = (
                    "🚨 <b>КРИТИЧЕСКИЙ АЛЕРТ</b>\n\n"
                    "❌ Trading Bot НЕ РАБОТАЕТ!\n\n"
                    "<b>Проблемы:</b>\n"
                )
                for issue in status['issues']:
                    message += f"• {issue}\n"
                
                message += f"\n⏰ Время: {datetime.now().strftime('%H:%M:%S %d.%m.%Y')}"
                
                self.send_telegram_alert(message)
            else:
                # Бот восстановился
                message = (
                    "✅ <b>БОТ ВОССТАНОВЛЕН</b>\n\n"
                    "Trading Bot снова работает!\n\n"
                    f"⏰ Время: {datetime.now().strftime('%H:%M:%S %d.%m.%Y')}"
                )
                self.send_telegram_alert(message)
        
        # Обновляем Uptime Robot
        self.update_uptime_robot(status)
        
        self.last_status = status['healthy']
        
        return status


def main():
    """Точка входа"""
    monitor = UptimeMonitor()
    
    # Если запущен с аргументом --daemon, работаем в фоне
    if len(sys.argv) > 1 and sys.argv[1] == '--daemon':
        print("🔄 Running in daemon mode (check every 5 minutes)")
        
        while True:
            try:
                monitor.run_check()
            except Exception as e:
                print(f"❌ Error in check: {e}")
            
            # Ждём 5 минут
            time.sleep(300)
    else:
        # Одиночная проверка
        status = monitor.run_check()
        
        # Возвращаем код выхода
        sys.exit(0 if status['healthy'] else 1)


if __name__ == "__main__":
    main()
