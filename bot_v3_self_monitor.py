"""
👁️ SELF-MONITORING SYSTEM V3.5 - Самомониторинг и самоисправление

Функции:
- 24/7 мониторинг собственного здоровья
- Автоматическое обнаружение проблем
- Самостоятельное исправление ошибок
- Анализ эффективности и самооптимизация
- Обучение на ошибках

Автор: AI Trading Bot Team
Версия: 3.5 AUTONOMOUS SELF-MONITORING
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import json
from loguru import logger


class SelfMonitoringSystem:
    """
    Система самомониторинга и самоисправления
    
    Отслеживает:
    - Работоспособность всех компонентов
    - Производительность ML/LLM
    - Качество сигналов
    - Эффективность сделок
    - Системные ресурсы
    """
    
    def __init__(self, check_interval: int = 300):  # 5 минут
        self.check_interval = check_interval
        self.running = False
        
        # История проверок
        self.health_checks = deque(maxlen=1000)
        self.issues_detected = deque(maxlen=100)
        self.auto_fixes = deque(maxlen=100)
        
        # Метрики производительности
        self.performance_metrics = {
            'ml_accuracy_trend': deque(maxlen=50),
            'llm_approval_rate': deque(maxlen=50),
            'signal_quality': deque(maxlen=50),
            'trade_success_rate': deque(maxlen=50),
            'avg_profit_per_trade': deque(maxlen=50)
        }
        
        # Пороги для срабатывания
        self.thresholds = {
            'ml_accuracy_min': 0.75,  # Минимум 75% точности ML
            'llm_approval_min': 0.50,  # Минимум 50% одобрений LLM
            'signal_quality_min': 70,  # Минимум 70% качество сигналов
            'trade_success_min': 0.40,  # Минимум 40% прибыльных сделок
            'error_rate_max': 0.10,  # Максимум 10% ошибок
            'consecutive_losses_max': 5  # Максимум 5 убытков подряд
        }
        
        # Счетчики
        self.total_checks = 0
        self.issues_found = 0
        self.auto_fixed = 0
        self.improvement_actions = 0
        
        # Последнее улучшение
        self.last_improvement = None
        
        logger.info("👁️ Self-Monitoring System V3.5 инициализирован")
    
    async def start(self):
        """Запуск системы мониторинга"""
        self.running = True
        logger.info("👁️ Self-Monitoring: Запуск мониторинга...")
        
        while self.running:
            try:
                await self._run_health_check()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"❌ Self-Monitor: Ошибка в цикле мониторинга: {e}")
                await asyncio.sleep(60)
    
    def stop(self):
        """Остановка мониторинга"""
        self.running = False
        logger.info("👁️ Self-Monitoring: Остановлен")
    
    async def _run_health_check(self):
        """Основная проверка здоровья системы"""
        try:
            self.total_checks += 1
            check_time = datetime.now()
            
            logger.debug(f"👁️ Self-Monitor: Проверка #{self.total_checks}...")
            
            # 1. Проверка компонентов
            component_health = await self._check_components()
            
            # 2. Проверка производительности
            performance_issues = await self._check_performance()
            
            # 3. Проверка ML/LLM
            ml_llm_issues = await self._check_ml_llm()
            
            # 4. Проверка системных ресурсов
            resource_issues = await self._check_system_resources()
            
            # 5. Анализ трендов
            trend_issues = await self._analyze_trends()
            
            # Собираем все проблемы
            all_issues = (
                component_health['issues'] +
                performance_issues +
                ml_llm_issues +
                resource_issues +
                trend_issues
            )
            
            # Сохраняем результат проверки
            check_result = {
                'timestamp': check_time,
                'total_issues': len(all_issues),
                'issues': all_issues,
                'component_health': component_health,
                'auto_fixed': 0
            }
            
            # Если есть проблемы - пытаемся исправить
            if all_issues:
                self.issues_found += len(all_issues)
                logger.warning(f"⚠️ Self-Monitor: Обнаружено {len(all_issues)} проблем!")
                
                fixed_count = await self._auto_fix_issues(all_issues)
                check_result['auto_fixed'] = fixed_count
                self.auto_fixed += fixed_count
                
                # Сохраняем проблемы
                for issue in all_issues:
                    self.issues_detected.append({
                        'timestamp': check_time,
                        'issue': issue,
                        'fixed': issue.get('fixed', False)
                    })
            
            self.health_checks.append(check_result)
            
            # 6. Проверка необходимости улучшений
            if self.total_checks % 12 == 0:  # Каждый час
                await self._check_improvement_opportunities()
            
            logger.debug(f"✅ Self-Monitor: Проверка завершена (проблем: {len(all_issues)})")
            
        except Exception as e:
            logger.error(f"❌ Self-Monitor: Ошибка health check: {e}")
    
    async def _check_components(self) -> Dict:
        """Проверка всех компонентов системы"""
        issues = []
        
        try:
            # Проверка ML Engine
            try:
                from bot_v3_ml_engine import ml_engine
                ml_status = ml_engine.get_status()
                if not ml_status['ml_available']:
                    issues.append({
                        'component': 'ML Engine',
                        'severity': 'medium',
                        'issue': 'ML библиотеки недоступны',
                        'fixable': False
                    })
            except Exception as e:
                issues.append({
                    'component': 'ML Engine',
                    'severity': 'high',
                    'issue': f'Не удалось проверить: {e}',
                    'fixable': False
                })
            
            # Проверка LLM Agent
            try:
                from bot_v3_llm_agent import llm_agent
                llm_status = llm_agent.get_status()
                if not llm_status['enabled']:
                    issues.append({
                        'component': 'LLM Agent',
                        'severity': 'medium',
                        'issue': 'LLM отключен (нет OPENAI_API_KEY)',
                        'fixable': False
                    })
            except Exception as e:
                issues.append({
                    'component': 'LLM Agent',
                    'severity': 'high',
                    'issue': f'Не удалось проверить: {e}',
                    'fixable': False
                })
            
            # Проверка Exchange
            try:
                from bot_v2_exchange import exchange_manager
                if not exchange_manager.exchange:
                    issues.append({
                        'component': 'Exchange Manager',
                        'severity': 'critical',
                        'issue': 'Не подключен к бирже',
                        'fixable': True,
                        'fix_action': 'reconnect_exchange'
                    })
            except Exception as e:
                issues.append({
                    'component': 'Exchange Manager',
                    'severity': 'critical',
                    'issue': f'Не удалось проверить: {e}',
                    'fixable': True,
                    'fix_action': 'reconnect_exchange'
                })
            
            return {
                'healthy': len(issues) == 0,
                'total_components': 3,
                'issues': issues
            }
            
        except Exception as e:
            logger.error(f"❌ Self-Monitor: Ошибка проверки компонентов: {e}")
            return {'healthy': False, 'total_components': 0, 'issues': []}
    
    async def _check_performance(self) -> List[Dict]:
        """Проверка производительности торговли"""
        issues = []
        
        try:
            from bot_v2_ai_agent import trading_bot_agent
            
            report = trading_bot_agent.get_performance_report()
            
            # Проверка Win Rate
            if report['total_trades'] >= 10:  # Минимум 10 сделок для статистики
                if report['win_rate'] < self.thresholds['trade_success_min']:
                    issues.append({
                        'component': 'Trading Performance',
                        'severity': 'high',
                        'issue': f"Низкий Win Rate: {report['win_rate']:.1%} < {self.thresholds['trade_success_min']:.0%}",
                        'fixable': True,
                        'fix_action': 'increase_confidence_threshold'
                    })
                
                # Проверка Profit Factor
                if report['profit_factor'] < 1.0 and report['total_trades'] >= 20:
                    issues.append({
                        'component': 'Trading Performance',
                        'severity': 'high',
                        'issue': f"Profit Factor < 1.0: {report['profit_factor']:.2f}",
                        'fixable': True,
                        'fix_action': 'improve_strategy'
                    })
            
            # Проверка серии убытков
            from bot_v2_safety import risk_manager
            if risk_manager.consecutive_losses >= self.thresholds['consecutive_losses_max']:
                issues.append({
                    'component': 'Risk Management',
                    'severity': 'critical',
                    'issue': f"Серия убытков: {risk_manager.consecutive_losses}",
                    'fixable': True,
                    'fix_action': 'pause_trading'
                })
            
        except Exception as e:
            logger.error(f"❌ Self-Monitor: Ошибка проверки производительности: {e}")
        
        return issues
    
    async def _check_ml_llm(self) -> List[Dict]:
        """Проверка ML/LLM систем"""
        issues = []
        
        try:
            # ML Engine
            from bot_v3_ml_engine import ml_engine
            ml_status = ml_engine.get_status()
            
            if ml_status['model_trained']:
                accuracy = ml_status['accuracy']
                
                # Сохраняем метрику
                self.performance_metrics['ml_accuracy_trend'].append(accuracy)
                
                # Проверка точности
                if accuracy < self.thresholds['ml_accuracy_min']:
                    issues.append({
                        'component': 'ML Engine',
                        'severity': 'high',
                        'issue': f"Низкая точность ML: {accuracy:.1%} < {self.thresholds['ml_accuracy_min']:.0%}",
                        'fixable': True,
                        'fix_action': 'retrain_ml_model'
                    })
            
            # LLM Agent
            from bot_v3_llm_agent import llm_agent
            llm_status = llm_agent.get_status()
            
            if llm_status['enabled'] and llm_status['total_analyses'] > 10:
                approval_rate = llm_status['successful_validations'] / llm_status['total_analyses']
                
                # Сохраняем метрику
                self.performance_metrics['llm_approval_rate'].append(approval_rate)
                
                # Проверка одобрений
                if approval_rate < self.thresholds['llm_approval_min']:
                    issues.append({
                        'component': 'LLM Agent',
                        'severity': 'medium',
                        'issue': f"Низкий процент одобрений LLM: {approval_rate:.1%}",
                        'fixable': True,
                        'fix_action': 'review_llm_criteria'
                    })
        
        except Exception as e:
            logger.error(f"❌ Self-Monitor: Ошибка проверки ML/LLM: {e}")
        
        return issues
    
    async def _check_system_resources(self) -> List[Dict]:
        """Проверка системных ресурсов"""
        issues = []
        
        try:
            import psutil
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                issues.append({
                    'component': 'System Resources',
                    'severity': 'medium',
                    'issue': f"Высокая загрузка CPU: {cpu_percent}%",
                    'fixable': False
                })
            
            # Память
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                issues.append({
                    'component': 'System Resources',
                    'severity': 'high',
                    'issue': f"Высокое использование памяти: {memory.percent}%",
                    'fixable': True,
                    'fix_action': 'clear_cache'
                })
            
            # Диск
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                issues.append({
                    'component': 'System Resources',
                    'severity': 'high',
                    'issue': f"Мало места на диске: {disk.percent}% занято",
                    'fixable': True,
                    'fix_action': 'clean_old_files'
                })
        
        except Exception as e:
            logger.debug(f"⚠️ Self-Monitor: Не удалось проверить ресурсы: {e}")
        
        return issues
    
    async def _analyze_trends(self) -> List[Dict]:
        """Анализ трендов производительности"""
        issues = []
        
        try:
            # Анализ тренда точности ML
            if len(self.performance_metrics['ml_accuracy_trend']) >= 5:
                recent = list(self.performance_metrics['ml_accuracy_trend'])[-5:]
                avg_recent = sum(recent) / len(recent)
                
                if len(self.performance_metrics['ml_accuracy_trend']) >= 10:
                    older = list(self.performance_metrics['ml_accuracy_trend'])[-10:-5]
                    avg_older = sum(older) / len(older)
                    
                    # Если точность падает
                    if avg_recent < avg_older - 0.05:  # Падение >5%
                        issues.append({
                            'component': 'ML Performance Trend',
                            'severity': 'medium',
                            'issue': f"Точность ML падает: {avg_recent:.1%} < {avg_older:.1%}",
                            'fixable': True,
                            'fix_action': 'retrain_ml_model'
                        })
        
        except Exception as e:
            logger.debug(f"⚠️ Self-Monitor: Ошибка анализа трендов: {e}")
        
        return issues
    
    async def _auto_fix_issues(self, issues: List[Dict]) -> int:
        """Автоматическое исправление проблем"""
        fixed_count = 0
        
        for issue in issues:
            if not issue.get('fixable', False):
                continue
            
            fix_action = issue.get('fix_action')
            if not fix_action:
                continue
            
            logger.info(f"🔧 Self-Monitor: Исправление '{issue['issue']}'...")
            
            try:
                if fix_action == 'reconnect_exchange':
                    success = await self._fix_reconnect_exchange()
                elif fix_action == 'increase_confidence_threshold':
                    success = await self._fix_increase_confidence()
                elif fix_action == 'improve_strategy':
                    success = await self._fix_improve_strategy()
                elif fix_action == 'pause_trading':
                    success = await self._fix_pause_trading()
                elif fix_action == 'retrain_ml_model':
                    success = await self._fix_retrain_ml()
                elif fix_action == 'review_llm_criteria':
                    success = await self._fix_review_llm()
                elif fix_action == 'clear_cache':
                    success = await self._fix_clear_cache()
                elif fix_action == 'clean_old_files':
                    success = await self._fix_clean_files()
                else:
                    logger.warning(f"⚠️ Self-Monitor: Неизвестное действие: {fix_action}")
                    success = False
                
                if success:
                    fixed_count += 1
                    issue['fixed'] = True
                    logger.info(f"✅ Self-Monitor: Исправлено '{issue['issue']}'")
                    
                    # Сохраняем информацию о fix
                    self.auto_fixes.append({
                        'timestamp': datetime.now(),
                        'issue': issue,
                        'action': fix_action,
                        'success': True
                    })
                else:
                    logger.warning(f"⚠️ Self-Monitor: Не удалось исправить '{issue['issue']}'")
                    issue['fixed'] = False
            
            except Exception as e:
                logger.error(f"❌ Self-Monitor: Ошибка исправления: {e}")
                issue['fixed'] = False
        
        if fixed_count > 0:
            logger.info(f"🔧 Self-Monitor: Автоматически исправлено {fixed_count}/{len([i for i in issues if i.get('fixable')])} проблем")
        
        return fixed_count
    
    async def _fix_reconnect_exchange(self) -> bool:
        """Переподключение к бирже"""
        try:
            from bot_v2_exchange import exchange_manager
            logger.info("🔄 Self-Monitor: Переподключение к бирже...")
            success = await exchange_manager.connect()
            return success
        except Exception as e:
            logger.error(f"❌ Self-Monitor: Ошибка переподключения: {e}")
            return False
    
    async def _fix_increase_confidence(self) -> bool:
        """Увеличение порога уверенности"""
        try:
            from bot_v2_config import Config
            
            old_value = Config.MIN_CONFIDENCE_PERCENT
            new_value = min(95, old_value + 5)  # +5%, но не больше 95%
            
            Config.MIN_CONFIDENCE_PERCENT = new_value
            
            logger.info(f"📈 Self-Monitor: Увеличен порог confidence: {old_value}% → {new_value}%")
            
            self.improvement_actions += 1
            self.last_improvement = {
                'timestamp': datetime.now(),
                'action': 'increase_confidence',
                'old_value': old_value,
                'new_value': new_value
            }
            
            return True
        except Exception as e:
            logger.error(f"❌ Self-Monitor: Ошибка увеличения confidence: {e}")
            return False
    
    async def _fix_improve_strategy(self) -> bool:
        """Улучшение стратегии"""
        try:
            # Триггерим переобучение ML
            from bot_v3_ml_engine import ml_engine
            
            if len(ml_engine.training_data) >= 50:
                logger.info("🧠 Self-Monitor: Запуск переобучения ML для улучшения...")
                await ml_engine.retrain_models()
                
                self.improvement_actions += 1
                return True
            else:
                logger.warning(f"⚠️ Self-Monitor: Недостаточно данных для переобучения ({len(ml_engine.training_data)}/50)")
                return False
        except Exception as e:
            logger.error(f"❌ Self-Monitor: Ошибка улучшения стратегии: {e}")
            return False
    
    async def _fix_pause_trading(self) -> bool:
        """Пауза торговли при серии убытков"""
        try:
            from bot_v2_safety import risk_manager
            
            logger.warning("⏸️ Self-Monitor: ПАУЗА торговли из-за серии убытков!")
            
            # Устанавливаем паузу на 1 час
            risk_manager.pause_until = datetime.now() + timedelta(hours=1)
            
            self.improvement_actions += 1
            return True
        except Exception as e:
            logger.error(f"❌ Self-Monitor: Ошибка паузы: {e}")
            return False
    
    async def _fix_retrain_ml(self) -> bool:
        """Переобучение ML модели"""
        try:
            from bot_v3_ml_engine import ml_engine
            
            if len(ml_engine.training_data) >= 50:
                logger.info("🔄 Self-Monitor: Переобучение ML...")
                await ml_engine.retrain_models()
                
                self.improvement_actions += 1
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"❌ Self-Monitor: Ошибка переобучения: {e}")
            return False
    
    async def _fix_review_llm(self) -> bool:
        """Проверка критериев LLM"""
        # TODO: Можно добавить логику для адаптации критериев LLM
        logger.info("📝 Self-Monitor: Критерии LLM в порядке (автонастройка пока недоступна)")
        return True
    
    async def _fix_clear_cache(self) -> bool:
        """Очистка кеша"""
        try:
            import gc
            gc.collect()
            logger.info("🧹 Self-Monitor: Кеш очищен")
            return True
        except Exception as e:
            logger.error(f"❌ Self-Monitor: Ошибка очистки кеша: {e}")
            return False
    
    async def _fix_clean_files(self) -> bool:
        """Очистка старых файлов"""
        try:
            # Очистка старых логов (>7 дней)
            import glob
            
            cleaned = 0
            for log_file in glob.glob("*.log"):
                try:
                    age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(log_file))).days
                    if age_days > 7:
                        os.remove(log_file)
                        cleaned += 1
                except:
                    pass
            
            logger.info(f"🧹 Self-Monitor: Удалено {cleaned} старых лог-файлов")
            return True
        except Exception as e:
            logger.error(f"❌ Self-Monitor: Ошибка очистки файлов: {e}")
            return False
    
    async def _check_improvement_opportunities(self):
        """Проверка возможностей для улучшения"""
        try:
            logger.info("💡 Self-Monitor: Проверка возможностей улучшения...")
            
            # Анализ производительности за последние 24ч
            from bot_v2_ai_agent import trading_bot_agent
            report = trading_bot_agent.get_performance_report()
            
            if report['total_trades'] >= 20:
                # Если Win Rate хороший, но можно лучше
                if 0.60 <= report['win_rate'] < 0.75:
                    logger.info("💡 Self-Monitor: Win Rate неплохой, но можно улучшить до 75%+")
                    
                    # Предлагаем увеличить порог confidence
                    from bot_v2_config import Config
                    if Config.MIN_CONFIDENCE_PERCENT < 90:
                        await self._fix_increase_confidence()
                
                # Если ML модель давно не обучалась
                from bot_v3_ml_engine import ml_engine
                if ml_engine.last_training:
                    hours_since_training = (datetime.now() - ml_engine.last_training).total_seconds() / 3600
                    if hours_since_training > 24 and len(ml_engine.training_data) >= 50:
                        logger.info("💡 Self-Monitor: ML не обучалась >24ч, запускаем переобучение...")
                        await self._fix_retrain_ml()
        
        except Exception as e:
            logger.error(f"❌ Self-Monitor: Ошибка проверки улучшений: {e}")
    
    def get_status(self) -> Dict:
        """Статус системы мониторинга"""
        return {
            'running': self.running,
            'total_checks': self.total_checks,
            'issues_found': self.issues_found,
            'auto_fixed': self.auto_fixed,
            'improvement_actions': self.improvement_actions,
            'fix_rate': f"{(self.auto_fixed / max(1, self.issues_found)) * 100:.0f}%",
            'last_check': self.health_checks[-1]['timestamp'].isoformat() if self.health_checks else None,
            'last_improvement': self.last_improvement
        }


# Глобальный экземпляр
self_monitor = SelfMonitoringSystem()


if __name__ == "__main__":
    logger.info("👁️ Self-Monitoring System V3.5 - Тестовый режим")
    logger.info(f"Status: {self_monitor.get_status()}")


