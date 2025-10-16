"""
🤖 LLM AGENT V3.5 - Интеллектуальный агент на базе GPT-4

Функции:
- Анализ рыночного контекста
- Интерпретация новостей и настроений
- Валидация ML предсказаний
- Генерация торговых инсайтов
- Адаптация стратегии под условия рынка

Автор: AI Trading Bot Team
Версия: 3.5 AUTONOMOUS LLM
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger
import json

# OpenAI API
try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ OpenAI не установлен. Используйте: pip install openai")
    OPENAI_AVAILABLE = False


class LLMTradingAgent:
    """
    LLM агент для интеллектуального анализа рынка
    
    Использует GPT-4 для:
    - Анализа рыночного контекста
    - Валидации сигналов
    - Генерации стратегических решений
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.warning("⚠️ OPENAI_API_KEY не найден в переменных окружения")
            self.enabled = False
        else:
            self.enabled = OPENAI_AVAILABLE
            if self.enabled:
                self.client = AsyncOpenAI(api_key=self.api_key)
                logger.info("🤖 LLM Agent V3.5 инициализирован (GPT-4)")
        
        # История анализов
        self.analysis_history = []
        self.max_history = 100
        
        # Статистика
        self.total_analyses = 0
        self.successful_validations = 0
        self.rejected_signals = 0
    
    async def analyze_market_context(
        self,
        symbol: str,
        current_price: float,
        signal_result: Dict,
        ml_result: Dict,
        market_conditions: Dict
    ) -> Dict[str, any]:
        """
        Глубокий анализ рыночного контекста с помощью LLM
        
        Возвращает:
        - recommendation: 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
        - confidence: 0-100
        - reasoning: текстовое объяснение
        - risk_level: 'low', 'medium', 'high'
        """
        try:
            if not self.enabled:
                return self._fallback_analysis(signal_result, ml_result)
            
            # Формируем промпт для GPT-4
            prompt = self._build_analysis_prompt(
                symbol, current_price, signal_result, ml_result, market_conditions
            )
            
            # Запрос к GPT-4
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": """Ты - эксперт по криптовалютной торговле с 10+ летним опытом.
Твоя задача - ПОДТВЕРДИТЬ или ОТКЛОНИТЬ торговый сигнал.

ВАЖНО:
- Сигнал УЖЕ прошел строгую фильтрацию (≥90% уверенность, все индикаторы)
- Твоя роль: финальная проверка, НЕ первичный анализ
- Используй "hold" ТОЛЬКО если есть СЕРЬЕЗНЫЕ красные флаги (противоречия, аномалии)
- Если индикаторы согласованы и нет явных опасностей → ОДОБРЯЙ ("buy" или "sell")
- Целевая точность: 85-99% прибыльных сделок

КРИТЕРИИ ДЛЯ "HOLD":
- Противоречивые индикаторы (тренд вверх, но MACD вниз)
- Экстремальная перекупленность/перепроданность (RSI >95 или <5)
- Критические уровни сопротивления/поддержки без подтверждения
- Нестабильный или манипулируемый рынок

КРИТЕРИИ ДЛЯ "BUY"/"SELL":
- Индикаторы согласованы с сигналом
- Нет явных красных флагов
- Разумные уровни RSI/MACD
- Тренд или дивергенция подтверждают сигнал

ФОРМАТ ОТВЕТА (JSON):
{
  "recommendation": "buy|sell|hold",
  "confidence": 70-95,
  "reasoning": "краткое объяснение (1-2 предложения)",
  "risk_level": "low|medium|high",
  "key_factors": ["фактор1", "фактор2"],
  "concerns": [] или ["опасение1"]
}"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Низкая температура для более детерминированных ответов
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            # Парсим ответ
            content = response.choices[0].message.content
            analysis = json.loads(content)
            
            # Валидация
            if not self._validate_llm_response(analysis):
                logger.warning("⚠️ Некорректный ответ LLM, используем fallback")
                return self._fallback_analysis(signal_result, ml_result)
            
            # Статистика
            self.total_analyses += 1
            
            # Сохраняем в историю
            self.analysis_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'analysis': analysis,
                'signal_result': signal_result,
                'ml_result': ml_result
            })
            
            if len(self.analysis_history) > self.max_history:
                self.analysis_history = self.analysis_history[-self.max_history:]
            
            logger.info(f"🤖 LLM: {symbol} {analysis['recommendation'].upper()} "
                       f"({analysis['confidence']:.0f}%) - {analysis['reasoning'][:100]}...")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ LLM ошибка: {e}")
            return self._fallback_analysis(signal_result, ml_result)
    
    def _build_analysis_prompt(
        self,
        symbol: str,
        current_price: float,
        signal_result: Dict,
        ml_result: Dict,
        market_conditions: Dict
    ) -> str:
        """Построение промпта для анализа"""
        
        prompt = f"""Проанализируй торговую возможность для {symbol}:

ТЕКУЩИЕ ДАННЫЕ:
- Цена: ${current_price:.4f}
- Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

ТЕХНИЧЕСКИЙ АНАЛИЗ:
- Базовый сигнал: {signal_result.get('signal', 'None')}
- Уверенность: {signal_result.get('confidence', 0):.1f}%
- RSI: {signal_result.get('rsi', 'N/A')}
- MACD: {signal_result.get('macd', 'N/A')}
- Trend: {signal_result.get('trend', 'N/A')}

ML ПРЕДСКАЗАНИЕ:
- ML сигнал: {ml_result.get('ml_signal', 'None')}
- ML уверенность: {ml_result.get('ml_score', 0):.1f}%
- Финальный сигнал: {ml_result.get('signal', 'None')}
- Финальная уверенность: {ml_result.get('confidence', 0):.1f}%
"""
        
        # Предсказание цены LSTM
        if ml_result.get('price_prediction'):
            price_pred = ml_result['price_prediction']
            price_change = ((price_pred - current_price) / current_price) * 100
            prompt += f"- Предсказание цены (LSTM): ${price_pred:.4f} ({price_change:+.2f}%)\n"
        
        # Вероятности
        if ml_result.get('probabilities'):
            prob = ml_result['probabilities']
            prompt += f"- Вероятности: BUY={prob['buy']:.1f}% HOLD={prob['hold']:.1f}% SELL={prob['sell']:.1f}%\n"
        
        # Рыночные условия
        prompt += f"""
РЫНОЧНЫЕ УСЛОВИЯ:
- Волатильность: {market_conditions.get('volatility', 'N/A')}
- Тренд: {market_conditions.get('trend', 'N/A')}
- Общее настроение: {market_conditions.get('sentiment', 'N/A')}

ЗАДАЧА:
Проанализируй всю информацию и дай рекомендацию с учетом:
1. Целевая точность 85-99% - быть очень консервативным
2. Согласованность базового сигнала и ML
3. Рыночные условия и волатильность
4. Соотношение риск/прибыль
5. Потенциальные риски и опасности

Ответь в формате JSON.
"""
        
        return prompt
    
    def _validate_llm_response(self, analysis: Dict) -> bool:
        """Валидация ответа LLM"""
        required_fields = ['recommendation', 'confidence', 'reasoning', 'risk_level']
        
        # Проверка наличия полей
        if not all(field in analysis for field in required_fields):
            return False
        
        # Проверка значений
        valid_recommendations = ['strong_buy', 'buy', 'hold', 'sell', 'strong_sell']
        if analysis['recommendation'] not in valid_recommendations:
            return False
        
        if not (0 <= analysis['confidence'] <= 100):
            return False
        
        valid_risk_levels = ['low', 'medium', 'high']
        if analysis['risk_level'] not in valid_risk_levels:
            return False
        
        return True
    
    def _fallback_analysis(self, signal_result: Dict, ml_result: Dict) -> Dict:
        """Запасной анализ если LLM недоступен"""
        
        signal = ml_result.get('signal') or signal_result.get('signal')
        confidence = ml_result.get('confidence', signal_result.get('confidence', 0))
        
        # Маппинг сигнала в рекомендацию
        if signal == 'buy':
            if confidence >= 95:
                recommendation = 'strong_buy'
            else:
                recommendation = 'buy'
        elif signal == 'sell':
            if confidence >= 95:
                recommendation = 'strong_sell'
            else:
                recommendation = 'sell'
        else:
            recommendation = 'hold'
        
        # Уровень риска
        if confidence >= 90:
            risk_level = 'low'
        elif confidence >= 85:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'reasoning': f"Fallback analysis: {signal} signal with {confidence:.1f}% confidence",
            'risk_level': risk_level,
            'key_factors': ['Technical indicators', 'ML prediction'],
            'concerns': [] if confidence >= 85 else ['Low confidence']
        }
    
    async def validate_trade_decision(
        self,
        symbol: str,
        signal: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        llm_analysis: Dict
    ) -> bool:
        """
        Финальная валидация торгового решения перед открытием
        
        Возвращает True если сделка одобрена
        """
        try:
            # Проверка рекомендации LLM
            recommendation = llm_analysis.get('recommendation', 'hold')
            confidence = llm_analysis.get('confidence', 0)
            risk_level = llm_analysis.get('risk_level', 'high')
            
            # Правила валидации
            
            # 1. Рекомендация должна соответствовать сигналу
            if signal == 'buy' and recommendation not in ['buy', 'strong_buy']:
                logger.warning(f"🚫 LLM не рекомендует BUY: {recommendation}")
                self.rejected_signals += 1
                return False
            
            if signal == 'sell' and recommendation not in ['sell', 'strong_sell']:
                logger.warning(f"🚫 LLM не рекомендует SELL: {recommendation}")
                self.rejected_signals += 1
                return False
            
            # 2. Уверенность должна быть высокой (≥85%)
            if confidence < 85:
                logger.warning(f"🚫 LLM уверенность слишком низкая: {confidence:.1f}%")
                self.rejected_signals += 1
                return False
            
            # 3. Риск не должен быть слишком высоким
            if risk_level == 'high':
                logger.warning(f"🚫 LLM определил высокий риск")
                self.rejected_signals += 1
                return False
            
            # 4. Проверка соотношения риск/прибыль
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            if risk_reward_ratio < 2.0:  # Минимум 1:2
                logger.warning(f"🚫 Плохое соотношение Risk/Reward: 1:{risk_reward_ratio:.2f}")
                self.rejected_signals += 1
                return False
            
            # 5. Проверка concerns
            concerns = llm_analysis.get('concerns', [])
            if len(concerns) > 2:
                logger.warning(f"🚫 Слишком много опасений LLM: {concerns}")
                self.rejected_signals += 1
                return False
            
            # Все проверки пройдены
            logger.info(f"✅ LLM одобрил сделку: {symbol} {signal.upper()} "
                       f"({confidence:.0f}%, {risk_level} risk, RR 1:{risk_reward_ratio:.1f})")
            self.successful_validations += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка валидации LLM: {e}")
            return False
    
    def get_status(self) -> Dict:
        """Статус LLM агента"""
        return {
            'enabled': self.enabled,
            'openai_available': OPENAI_AVAILABLE,
            'total_analyses': self.total_analyses,
            'successful_validations': self.successful_validations,
            'rejected_signals': self.rejected_signals,
            'validation_rate': f"{(self.successful_validations / max(1, self.total_analyses)) * 100:.1f}%",
            'history_size': len(self.analysis_history)
        }


# Глобальный экземпляр
llm_agent = LLMTradingAgent()


if __name__ == "__main__":
    logger.info("🤖 LLM Agent V3.5 - Тестовый режим")
    logger.info(f"LLM Status: {llm_agent.get_status()}")


