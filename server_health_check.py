#!/usr/bin/env python3
"""
🏥 КОМПЛЕКСНАЯ ПРОВЕРКА ЗДОРОВЬЯ СЕРВЕРА
========================================

Проверяет:
1. Место на диске и уборка мусора
2. Все библиотеки и зависимости
3. OpenAI API подключение
4. Прибыльность бота (PnL, статистика)
"""

import os
import sys
import shutil
import subprocess
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional
import requests

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ServerHealthCheck:
    """🏥 Комплексная проверка здоровья сервера"""
    
    def __init__(self):
        self.results = {
            'disk_space': {},
            'cleanup': {},
            'libraries': {},
            'openai_api': {},
            'profitability': {},
            'overall': 'PENDING'
        }
        
        # Определяем базовый путь
        if Path("/opt/bot").exists():
            self.base_dir = Path("/opt/bot")
            logger.info("📂 Работаем на сервере: /opt/bot")
        else:
            self.base_dir = Path(__file__).parent
            logger.info(f"📂 Работаем локально: {self.base_dir}")
        
        self.data_dir = self.base_dir / "data"
        self.logs_dir = self.base_dir / "logs"
        self.cache_dir = self.data_dir / "cache"
    
    def check_disk_space(self) -> Dict[str, Any]:
        """💾 Проверка места на диске"""
        logger.info("\n" + "="*60)
        logger.info("💾 ПРОВЕРКА МЕСТА НА ДИСКЕ")
        logger.info("="*60)
        
        results = {}
        
        try:
            # Получаем статистику диска
            stat = shutil.disk_usage(self.base_dir)
            
            total_gb = stat.total / (1024**3)
            used_gb = stat.used / (1024**3)
            free_gb = stat.free / (1024**3)
            used_percent = (stat.used / stat.total) * 100
            
            results = {
                'total_gb': round(total_gb, 2),
                'used_gb': round(used_gb, 2),
                'free_gb': round(free_gb, 2),
                'used_percent': round(used_percent, 2),
                'status': '✅' if used_percent < 80 else '⚠️' if used_percent < 90 else '❌'
            }
            
            logger.info(f"  📊 Всего: {total_gb:.2f} GB")
            logger.info(f"  📊 Использовано: {used_gb:.2f} GB ({used_percent:.1f}%)")
            logger.info(f"  📊 Свободно: {free_gb:.2f} GB")
            logger.info(f"  {results['status']} Статус: {'Норма' if used_percent < 80 else 'Внимание' if used_percent < 90 else 'Критично'}")
            
            # Проверяем размер папок бота
            if self.base_dir.exists():
                bot_size = sum(f.stat().st_size for f in self.base_dir.rglob('*') if f.is_file())
                bot_size_gb = bot_size / (1024**3)
                results['bot_size_gb'] = round(bot_size_gb, 2)
                logger.info(f"  📁 Размер папки бота: {bot_size_gb:.2f} GB")
                
                # Размеры подпапок
                for subdir in ['data', 'logs']:
                    subdir_path = self.base_dir / subdir
                    if subdir_path.exists():
                        subdir_size = sum(f.stat().st_size for f in subdir_path.rglob('*') if f.is_file())
                        subdir_size_mb = subdir_size / (1024**2)
                        results[f'{subdir}_size_mb'] = round(subdir_size_mb, 2)
                        logger.info(f"    📂 {subdir}/: {subdir_size_mb:.2f} MB")
        
        except Exception as e:
            results = {'error': str(e), 'status': '❌'}
            logger.error(f"  ❌ Ошибка проверки диска: {e}")
        
        self.results['disk_space'] = results
        return results
    
    def cleanup_junk(self) -> Dict[str, Any]:
        """🧹 Уборка мусора"""
        logger.info("\n" + "="*60)
        logger.info("🧹 УБОРКА МУСОРА")
        logger.info("="*60)
        
        results = {
            'cleaned': {},
            'freed_mb': 0,
            'total_cleaned': 0
        }
        
        try:
            # 1. Удаление старых логов (>7 дней)
            if self.logs_dir.exists():
                log_files = list(self.logs_dir.rglob('*.log'))
                old_logs = []
                cutoff_date = datetime.now() - timedelta(days=7)
                freed_size = 0
                
                for log_file in log_files:
                    try:
                        if log_file.stat().st_mtime < cutoff_date.timestamp():
                            size = log_file.stat().st_size
                            log_file.unlink()
                            old_logs.append(str(log_file))
                            freed_size += size
                    except:
                        pass
                
                results['cleaned']['old_logs'] = {
                    'count': len(old_logs),
                    'freed_mb': round(freed_size / (1024**2), 2)
                }
                logger.info(f"  ✅ Удалено старых логов: {len(old_logs)} файлов, освобождено {freed_size / (1024**2):.2f} MB")
            
            # 2. Очистка кэша (>24 часов)
            if self.cache_dir.exists():
                cache_files = list(self.cache_dir.rglob('*'))
                old_cache = []
                cutoff_date = datetime.now() - timedelta(hours=24)
                freed_size = 0
                
                for cache_file in cache_files:
                    try:
                        if cache_file.is_file() and cache_file.stat().st_mtime < cutoff_date.timestamp():
                            size = cache_file.stat().st_size
                            cache_file.unlink()
                            old_cache.append(str(cache_file))
                            freed_size += size
                    except:
                        pass
                
                results['cleaned']['old_cache'] = {
                    'count': len(old_cache),
                    'freed_mb': round(freed_size / (1024**2), 2)
                }
                logger.info(f"  ✅ Удалено старых кэшей: {len(old_cache)} файлов, освобождено {freed_size / (1024**2):.2f} MB")
            
            # 3. Удаление временных файлов
            temp_patterns = ['*.tmp', '*.temp', '*~', '*.swp', '*.bak']
            temp_files = []
            freed_size = 0
            
            for pattern in temp_patterns:
                for temp_file in self.base_dir.rglob(pattern):
                    try:
                        if temp_file.is_file():
                            size = temp_file.stat().st_size
                            temp_file.unlink()
                            temp_files.append(str(temp_file))
                            freed_size += size
                    except:
                        pass
            
            results['cleaned']['temp_files'] = {
                'count': len(temp_files),
                'freed_mb': round(freed_size / (1024**2), 2)
            }
            logger.info(f"  ✅ Удалено временных файлов: {len(temp_files)} файлов, освобождено {freed_size / (1024**2):.2f} MB")
            
            # 4. Удаление __pycache__
            pycache_dirs = []
            freed_size = 0
            
            for pycache_dir in self.base_dir.rglob('__pycache__'):
                try:
                    if pycache_dir.is_dir():
                        size = sum(f.stat().st_size for f in pycache_dir.rglob('*') if f.is_file())
                        shutil.rmtree(pycache_dir)
                        pycache_dirs.append(str(pycache_dir))
                        freed_size += size
                except:
                    pass
            
            results['cleaned']['pycache'] = {
                'count': len(pycache_dirs),
                'freed_mb': round(freed_size / (1024**2), 2)
            }
            logger.info(f"  ✅ Удалено __pycache__: {len(pycache_dirs)} директорий, освобождено {freed_size / (1024**2):.2f} MB")
            
            # Подсчитываем общее освобожденное место
            total_freed = sum(
                item.get('freed_mb', 0) for item in results['cleaned'].values()
                if isinstance(item, dict)
            )
            results['freed_mb'] = round(total_freed, 2)
            results['total_cleaned'] = sum(
                item.get('count', 0) for item in results['cleaned'].values()
                if isinstance(item, dict)
            )
            results['status'] = '✅'
            
            logger.info(f"\n  🎉 ИТОГО: Освобождено {total_freed:.2f} MB, удалено {results['total_cleaned']} объектов")
        
        except Exception as e:
            results['error'] = str(e)
            results['status'] = '❌'
            logger.error(f"  ❌ Ошибка уборки: {e}")
        
        self.results['cleanup'] = results
        return results
    
    def check_libraries(self) -> Dict[str, Any]:
        """📚 Проверка всех библиотек"""
        logger.info("\n" + "="*60)
        logger.info("📚 ПРОВЕРКА БИБЛИОТЕК")
        logger.info("="*60)
        
        results = {
            'required': {},
            'missing': [],
            'outdated': [],
            'status': '✅'
        }
        
        # Критически важные библиотеки
        required_libs = {
            'ccxt': 'ccxt',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'scikit-learn': 'sklearn',
            'tensorflow': 'tensorflow',
            'requests': 'requests',
            'python-telegram-bot': 'telegram',
            'pybit': 'pybit',
            'asyncio': 'asyncio',
            'pytz': 'pytz',
            'sqlite3': 'sqlite3',
        }
        
        for lib_name, import_name in required_libs.items():
            try:
                if import_name == 'sqlite3':
                    import sqlite3
                    version = sqlite3.sqlite_version
                elif import_name == 'asyncio':
                    import asyncio
                    version = 'built-in'
                else:
                    module = __import__(import_name)
                    version = getattr(module, '__version__', 'unknown')
                
                results['required'][lib_name] = {
                    'status': '✅',
                    'version': version,
                    'installed': True
                }
                logger.info(f"  ✅ {lib_name}: {version}")
                
            except ImportError:
                results['required'][lib_name] = {
                    'status': '❌',
                    'installed': False
                }
                results['missing'].append(lib_name)
                logger.error(f"  ❌ {lib_name}: НЕ УСТАНОВЛЕН")
                results['status'] = '❌'
            except Exception as e:
                results['required'][lib_name] = {
                    'status': '⚠️',
                    'error': str(e)
                }
                logger.warning(f"  ⚠️ {lib_name}: Ошибка проверки - {e}")
        
        # Проверка дополнительных библиотек
        optional_libs = {
            'openai': 'openai',
            'joblib': 'joblib',
            'matplotlib': 'matplotlib',
        }
        
        for lib_name, import_name in optional_libs.items():
            try:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'unknown')
                logger.info(f"  ✅ {lib_name} (опционально): {version}")
            except ImportError:
                logger.warning(f"  ⚠️ {lib_name} (опционально): НЕ УСТАНОВЛЕН")
        
        if results['missing']:
            logger.error(f"\n  ❌ Отсутствуют библиотеки: {', '.join(results['missing'])}")
        else:
            logger.info(f"\n  ✅ Все необходимые библиотеки установлены")
        
        self.results['libraries'] = results
        return results
    
    def check_openai_api(self) -> Dict[str, Any]:
        """🤖 Проверка OpenAI API"""
        logger.info("\n" + "="*60)
        logger.info("🤖 ПРОВЕРКА OPENAI API")
        logger.info("="*60)
        
        results = {
            'api_key_set': False,
            'connection': False,
            'status': '❌'
        }
        
        try:
            # Проверяем наличие API ключа
            api_key = os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                # Пробуем загрузить из .env файла
                env_file = self.base_dir / ".env"
                if env_file.exists():
                    with open(env_file, 'r') as f:
                        for line in f:
                            if line.startswith('OPENAI_API_KEY='):
                                api_key = line.split('=', 1)[1].strip().strip('"\'')
                                break
            
            if api_key:
                results['api_key_set'] = True
                logger.info("  ✅ OPENAI_API_KEY найден")
                
                # Пробуем подключиться к API
                try:
                    import openai
                    openai.api_key = api_key
                    
                    # Тестовый запрос (простой, чтобы не тратить токены)
                    response = requests.get(
                        'https://api.openai.com/v1/models',
                        headers={'Authorization': f'Bearer {api_key}'},
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        results['connection'] = True
                        results['status'] = '✅'
                        logger.info("  ✅ Подключение к OpenAI API успешно")
                        
                        # Показываем доступные модели
                        models_data = response.json()
                        model_count = len(models_data.get('data', []))
                        logger.info(f"  📊 Доступно моделей: {model_count}")
                    else:
                        logger.error(f"  ❌ Ошибка подключения: статус {response.status_code}")
                        results['error'] = f"HTTP {response.status_code}"
                except ImportError:
                    logger.warning("  ⚠️ Библиотека openai не установлена")
                    results['error'] = "Library not installed"
                except Exception as e:
                    logger.error(f"  ❌ Ошибка подключения: {e}")
                    results['error'] = str(e)
            else:
                logger.warning("  ⚠️ OPENAI_API_KEY не найден")
                results['error'] = "API key not found"
        
        except Exception as e:
            results['error'] = str(e)
            logger.error(f"  ❌ Ошибка проверки OpenAI API: {e}")
        
        self.results['openai_api'] = results
        return results
    
    def check_profitability(self) -> Dict[str, Any]:
        """💰 Проверка прибыльности"""
        logger.info("\n" + "="*60)
        logger.info("💰 ПРОВЕРКА ПРИБЫЛЬНОСТИ")
        logger.info("="*60)
        
        results = {
            'total_pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'status': '⚠️'
        }
        
        try:
            # Проверяем базу данных
            db_path = self.base_dir / "data" / "trading_data.db"
            if not db_path.exists():
                db_path = self.base_dir / "trading_data.db"
            
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Получаем статистику из trade_decisions
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losses,
                        AVG(CASE WHEN result = 'win' AND pnl_percent IS NOT NULL THEN pnl_percent ELSE NULL END) as avg_win,
                        AVG(CASE WHEN result = 'loss' AND pnl_percent IS NOT NULL THEN pnl_percent ELSE NULL END) as avg_loss,
                        SUM(CASE WHEN pnl_percent IS NOT NULL THEN pnl_percent ELSE 0 END) as total_pnl_pct
                    FROM trade_decisions
                    WHERE result IN ('win', 'loss')
                """)
                
                row = cursor.fetchone()
                if row and row[0]:
                    total, wins, losses, avg_win, avg_loss, total_pnl_pct = row
                    
                    results['total_trades'] = total or 0
                    results['winning_trades'] = wins or 0
                    results['losing_trades'] = losses or 0
                    results['win_rate'] = round((wins / total * 100) if total > 0 else 0, 2)
                    results['avg_profit'] = round(avg_win if avg_win else 0, 2)
                    results['avg_loss'] = round(avg_loss if avg_loss else 0, 2)
                    results['total_pnl_percent'] = round(total_pnl_pct if total_pnl_pct else 0, 2)
                    
                    # Оцениваем прибыль в USD (примерно, на основе позиций $25)
                    position_size = 25.0
                    results['estimated_total_pnl_usd'] = round(
                        (total_pnl_pct / 100) * position_size if total_pnl_pct else 0, 2
                    )
                    
                    logger.info(f"  📊 Всего сделок: {total}")
                    logger.info(f"  ✅ Выигрышных: {wins} ({results['win_rate']}%)")
                    logger.info(f"  ❌ Проигрышных: {losses}")
                    logger.info(f"  💵 Средняя прибыль: +{results['avg_profit']:.2f}%")
                    logger.info(f"  💸 Средний убыток: {results['avg_loss']:.2f}%")
                    logger.info(f"  📈 Общий PnL: {results['total_pnl_percent']:.2f}% (≈${results['estimated_total_pnl_usd']:.2f})")
                    
                    if results['win_rate'] > 50 and results['estimated_total_pnl_usd'] > 0:
                        results['status'] = '✅'
                    elif results['win_rate'] > 40:
                        results['status'] = '⚠️'
                else:
                    logger.warning("  ⚠️ Нет данных о сделках в базе")
                    results['status'] = '⚠️'
                
                # Проверяем последние сделки
                cursor.execute("""
                    SELECT symbol, decision, result, pnl_percent, timestamp
                    FROM trade_decisions
                    WHERE result IN ('win', 'loss')
                    ORDER BY timestamp DESC
                    LIMIT 5
                """)
                
                recent_trades = cursor.fetchall()
                if recent_trades:
                    logger.info(f"\n  📋 Последние 5 сделок:")
                    for trade in recent_trades:
                        symbol, decision, result, pnl, ts = trade
                        emoji = '✅' if result == 'win' else '❌'
                        logger.info(f"    {emoji} {symbol} {decision.upper()} | {result} | PnL: {pnl if pnl else 'N/A'}% | {ts}")
                
                conn.close()
            else:
                logger.warning("  ⚠️ База данных не найдена")
                results['error'] = "Database not found"
                results['status'] = '⚠️'
        
        except Exception as e:
            results['error'] = str(e)
            results['status'] = '❌'
            logger.error(f"  ❌ Ошибка проверки прибыльности: {e}")
        
        self.results['profitability'] = results
        return results
    
    def generate_report(self) -> str:
        """📊 Генерация итогового отчета"""
        logger.info("\n" + "="*60)
        logger.info("📊 ИТОГОВЫЙ ОТЧЕТ")
        logger.info("="*60)
        
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("🏥 КОМПЛЕКСНАЯ ПРОВЕРКА ЗДОРОВЬЯ СЕРВЕРА")
        report_lines.append("="*60)
        report_lines.append(f"\nДата проверки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Сервер: {self.base_dir}")
        report_lines.append("\n" + "-"*60)
        
        # Диск
        disk = self.results.get('disk_space', {})
        if disk.get('status'):
            status = disk['status']
            report_lines.append(f"\n💾 Место на диске: {status}")
            if 'used_percent' in disk:
                report_lines.append(f"   Использовано: {disk['used_percent']:.1f}%")
                report_lines.append(f"   Свободно: {disk.get('free_gb', 0):.2f} GB")
        
        # Уборка
        cleanup = self.results.get('cleanup', {})
        if cleanup.get('status'):
            status = cleanup['status']
            report_lines.append(f"\n🧹 Уборка: {status}")
            if 'freed_mb' in cleanup:
                report_lines.append(f"   Освобождено: {cleanup['freed_mb']:.2f} MB")
                report_lines.append(f"   Удалено объектов: {cleanup.get('total_cleaned', 0)}")
        
        # Библиотеки
        libs = self.results.get('libraries', {})
        if libs.get('status'):
            status = libs['status']
            report_lines.append(f"\n📚 Библиотеки: {status}")
            if libs.get('missing'):
                report_lines.append(f"   Отсутствуют: {', '.join(libs['missing'])}")
            else:
                report_lines.append(f"   Все необходимые библиотеки установлены")
        
        # OpenAI
        openai_result = self.results.get('openai_api', {})
        status = openai_result.get('status', '⚠️')
        report_lines.append(f"\n🤖 OpenAI API: {status}")
        if openai_result.get('api_key_set'):
            report_lines.append(f"   API ключ: ✅ Найден")
        else:
            report_lines.append(f"   API ключ: ❌ Не найден")
        if openai_result.get('connection'):
            report_lines.append(f"   Подключение: ✅ Успешно")
        
        # Прибыльность
        profit = self.results.get('profitability', {})
        status = profit.get('status', '⚠️')
        report_lines.append(f"\n💰 Прибыльность: {status}")
        if profit.get('total_trades', 0) > 0:
            report_lines.append(f"   Всего сделок: {profit['total_trades']}")
            report_lines.append(f"   Win Rate: {profit.get('win_rate', 0):.1f}%")
            report_lines.append(f"   Общий PnL: ≈${profit.get('estimated_total_pnl_usd', 0):.2f}")
        
        report_lines.append("\n" + "-"*60)
        report_lines.append("="*60)
        
        report = "\n".join(report_lines)
        logger.info(report)
        
        return report
    
    def run_all_checks(self):
        """🚀 Запуск всех проверок"""
        logger.info("\n" + "="*60)
        logger.info("🚀 ЗАПУСК КОМПЛЕКСНОЙ ПРОВЕРКИ")
        logger.info("="*60)
        
        # 1. Проверка места на диске
        self.check_disk_space()
        
        # 2. Уборка мусора
        self.cleanup_junk()
        
        # 3. Проверка библиотек
        self.check_libraries()
        
        # 4. Проверка OpenAI API
        self.check_openai_api()
        
        # 5. Проверка прибыльности
        self.check_profitability()
        
        # Генерация отчета
        report = self.generate_report()
        
        # Сохранение отчета
        report_file = self.base_dir / "logs" / "system" / f"health_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            f.write("\n\nДетальные результаты:\n")
            f.write(json.dumps(self.results, indent=2, ensure_ascii=False, default=str))
        
        logger.info(f"\n📄 Отчет сохранен: {report_file}")
        
        return self.results


def main():
    """Главная функция"""
    checker = ServerHealthCheck()
    checker.run_all_checks()


if __name__ == "__main__":
    main()










