"""
🚨 КРИТИЧЕСКИЕ АЛЕРТЫ В TELEGRAM
Мониторинг критических ошибок и отправка уведомлений
"""

import os
import re
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import requests
from dotenv import load_dotenv

load_dotenv()


class CriticalAlertsMonitor:
    """Мониторинг критических ошибок"""
    
    def __init__(self):
        # Telegram
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        
        # Пути
        self.bot_dir = Path(__file__).parent
        self.log_file = self.bot_dir / "logs" / "trading_bot_v4.log"
        self.state_file = self.bot_dir / "logs" / "alerts_state.txt"
        
        # Паттерны критических ошибок
        self.critical_patterns = [
            (r'CRITICAL', 'Критическая ошибка'),
            (r'ERROR.*API.*failed', 'Ошибка API'),
            (r'ERROR.*Connection', 'Ошибка подключения'),
            (r'ERROR.*Insufficient', 'Недостаточно средств'),
            (r'Exception.*Traceback', 'Необработанное исключение'),
            (r'ERROR.*Position.*failed', 'Ошибка управления позицией'),
            (r'ERROR.*Order.*failed', 'Ошибка создания ордера'),
        ]
        
        # Последняя проверенная позиция в логе
        self.last_position = self._load_last_position()
        
        # Ограничение частоты алертов
        self.alert_cooldown = {}  # {pattern: last_alert_time}
        self.cooldown_seconds = 3600  # 1 час между одинаковыми алертами
    
    def _load_last_position(self) -> int:
        """Загружает последнюю проверенную позицию"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return int(f.read().strip())
            except:
                pass
        return 0
    
    def _save_last_position(self, position: int):
        """Сохраняет последнюю проверенную позицию"""
        self.state_file.parent.mkdir(exist_ok=True)
        with open(self.state_file, 'w') as f:
            f.write(str(position))
    
    def send_telegram_alert(self, title: str, message: str, priority: str = "high"):
        """
        Отправляет алерт в Telegram
        
        Args:
            title: Заголовок
            message: Текст сообщения
            priority: Приоритет (high, medium, low)
        """
        if not self.telegram_token or not self.telegram_chat_id:
            print("⚠️ Telegram credentials not configured")
            return False
        
        # Эмодзи по приоритету
        emoji = {
            'high': '🚨',
            'medium': '⚠️',
            'low': 'ℹ️'
        }.get(priority, '⚠️')
        
        full_message = (
            f"{emoji} <b>{title}</b>\n\n"
            f"{message}\n\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S %d.%m.%Y')}"
        )
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': full_message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ Alert sent: {title}")
                return True
            else:
                print(f"❌ Failed to send alert: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error sending alert: {e}")
            return False
    
    def check_cooldown(self, pattern: str) -> bool:
        """
        Проверяет можно ли отправить алерт (не в cooldown)
        
        Args:
            pattern: Паттерн ошибки
            
        Returns:
            True если можно отправить
        """
        if pattern not in self.alert_cooldown:
            return True
        
        last_time = self.alert_cooldown[pattern]
        elapsed = time.time() - last_time
        
        return elapsed >= self.cooldown_seconds
    
    def update_cooldown(self, pattern: str):
        """Обновляет время последнего алерта"""
        self.alert_cooldown[pattern] = time.time()
    
    def scan_log_for_errors(self) -> List[Dict]:
        """
        Сканирует лог на наличие новых критических ошибок
        
        Returns:
            Список найденных ошибок
        """
        if not self.log_file.exists():
            return []
        
        errors = []
        
        try:
            with open(self.log_file, 'r') as f:
                # Переходим к последней проверенной позиции
                f.seek(self.last_position)
                
                line_number = self.last_position
                
                for line in f:
                    line_number += len(line.encode('utf-8'))
                    
                    # Проверяем каждый паттерн
                    for pattern, description in self.critical_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            errors.append({
                                'pattern': pattern,
                                'description': description,
                                'line': line.strip(),
                                'timestamp': self._extract_timestamp(line)
                            })
                            break
                
                # Сохраняем новую позицию
                self.last_position = line_number
                self._save_last_position(line_number)
        
        except Exception as e:
            print(f"❌ Error scanning log: {e}")
        
        return errors
    
    def _extract_timestamp(self, line: str) -> str:
        """Извлекает timestamp из строки лога"""
        # Формат: 2025-11-17 11:40:08 | INFO | ...
        match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        if match:
            return match.group(1)
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def process_errors(self, errors: List[Dict]):
        """
        Обрабатывает найденные ошибки и отправляет алерты
        
        Args:
            errors: Список ошибок
        """
        if not errors:
            return
        
        # Группируем по типу
        grouped = {}
        for error in errors:
            desc = error['description']
            if desc not in grouped:
                grouped[desc] = []
            grouped[desc].append(error)
        
        # Отправляем алерты
        for description, error_list in grouped.items():
            # Проверяем cooldown
            if not self.check_cooldown(description):
                print(f"⏳ Skipping alert for '{description}' (cooldown)")
                continue
            
            # Формируем сообщение
            if len(error_list) == 1:
                error = error_list[0]
                message = (
                    f"<b>Тип:</b> {description}\n"
                    f"<b>Время:</b> {error['timestamp']}\n\n"
                    f"<code>{error['line'][:500]}</code>"
                )
            else:
                message = (
                    f"<b>Тип:</b> {description}\n"
                    f"<b>Количество:</b> {len(error_list)} ошибок\n\n"
                    f"Последняя:\n"
                    f"<code>{error_list[-1]['line'][:500]}</code>"
                )
            
            # Отправляем
            if self.send_telegram_alert("КРИТИЧЕСКАЯ ОШИБКА", message, "high"):
                self.update_cooldown(description)
    
    def run_monitoring(self):
        """Запускает мониторинг"""
        print(f"🔍 Scanning log for critical errors...")
        
        errors = self.scan_log_for_errors()
        
        if errors:
            print(f"⚠️ Found {len(errors)} critical errors")
            self.process_errors(errors)
        else:
            print("✅ No critical errors found")
        
        return len(errors)


def main():
    """Точка входа"""
    import sys
    
    monitor = CriticalAlertsMonitor()
    
    # Если запущен с --daemon, работаем в фоне
    if len(sys.argv) > 1 and sys.argv[1] == '--daemon':
        print("🔄 Running in daemon mode (check every 2 minutes)")
        
        while True:
            try:
                monitor.run_monitoring()
            except Exception as e:
                print(f"❌ Error in monitoring: {e}")
            
            # Ждём 2 минуты
            time.sleep(120)
    else:
        # Одиночная проверка
        error_count = monitor.run_monitoring()
        sys.exit(0 if error_count == 0 else 1)


if __name__ == "__main__":
    main()
