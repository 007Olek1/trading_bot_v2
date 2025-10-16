#!/usr/bin/env python3
"""
🎓 ML Model Trainer
Обучает DistilBERT на исторических данных для классификации РОСТ/ПАДЕНИЕ/БОКОВИК
"""

import asyncio
import logging
from typing import List, Dict, Tuple
import json
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

try:
    from transformers import (
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
        Trainer,
        TrainingArguments
    )
    from datasets import Dataset
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("⚠️ transformers недоступен")


class MarketDatasetBuilder:
    """
    Строит датасет для обучения из исторических данных
    """
    
    def __init__(self):
        self.data = []
    
    async def build_from_history(
        self,
        exchange_manager,
        symbols: List[str],
        days: int = 30
    ) -> List[Dict]:
        """
        Собирает исторические данные и создаёт датасет
        
        Формат:
        {
            'text': "price rising strongly, volume increasing...",
            'label': 0/1/2  # 0=ПАДЕНИЕ, 1=БОКОВИК, 2=РОСТ
        }
        """
        
        logger.info(f"📊 Собираем данные по {len(symbols)} символам за {days} дней...")
        
        from bot_v2_nlp_analyzer import nlp_analyzer
        
        for symbol in symbols:
            try:
                # Получаем исторические свечи
                candles = await exchange_manager.fetch_ohlcv(
                    symbol,
                    timeframe='1h',
                    limit=days * 24
                )
                
                if len(candles) < 50:
                    continue
                
                # Создаём окна данных
                window_size = 20
                for i in range(window_size, len(candles) - 5):
                    window = candles[i-window_size:i]
                    
                    # Будущее движение (метка)
                    current_price = candles[i]['close']
                    future_price = candles[i+5]['close']  # Через 5 часов
                    
                    price_change = (future_price - current_price) / current_price * 100
                    
                    # Определяем метку
                    if price_change > 2:
                        label = 2  # РОСТ
                    elif price_change < -2:
                        label = 0  # ПАДЕНИЕ
                    else:
                        label = 1  # БОКОВИК
                    
                    # Генерируем описание
                    indicators = self._calculate_indicators(window)
                    description = nlp_analyzer.generate_market_description(
                        candles=window,
                        indicators=indicators,
                        current_price=current_price
                    )
                    
                    self.data.append({
                        'text': description,
                        'label': label,
                        'symbol': symbol,
                        'price_change': price_change
                    })
                
                logger.info(f"✅ {symbol}: собрано {len(self.data)} примеров")
                
            except Exception as e:
                logger.error(f"❌ Ошибка сбора данных {symbol}: {e}")
                continue
        
        logger.info(f"📦 Всего собрано {len(self.data)} примеров")
        return self.data
    
    def _calculate_indicators(self, candles: List[Dict]) -> Dict[str, float]:
        """Быстрый расчёт основных индикаторов"""
        import pandas as pd
        
        df = pd.DataFrame(candles)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        return {
            'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
            'macd': macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0,
            'macd_signal': signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0
        }
    
    def save_dataset(self, filename: str = 'market_dataset.json'):
        """Сохраняет датасет в файл"""
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=2)
        logger.info(f"💾 Датасет сохранён: {filename}")
    
    def load_dataset(self, filename: str = 'market_dataset.json') -> List[Dict]:
        """Загружает датасет из файла"""
        with open(filename, 'r') as f:
            self.data = json.load(f)
        logger.info(f"📂 Датасет загружен: {len(self.data)} примеров")
        return self.data


class DistilBERTTrainer:
    """
    Обучает DistilBERT на классификацию рынка
    """
    
    def __init__(self):
        self.model_name = "distilbert-base-uncased"
        self.model = None
        self.tokenizer = None
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("❌ transformers не установлен! Установите: pip install transformers datasets")
            return
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        logger.info("✅ Tokenizer загружен")
    
    def prepare_dataset(self, data: List[Dict]) -> Dataset:
        """Подготавливает данные для обучения"""
        
        # Балансировка классов
        from collections import Counter
        label_counts = Counter([d['label'] for d in data])
        logger.info(f"📊 Распределение классов: {label_counts}")
        
        # Токенизация
        texts = [d['text'] for d in data]
        labels = [d['label'] for d in data]
        
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # Создаём Dataset
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Разделяем на train/test
        split = dataset.train_test_split(test_size=0.2, seed=42)
        
        logger.info(f"📚 Train: {len(split['train'])}, Test: {len(split['test'])}")
        
        return split
    
    def train(
        self,
        dataset,
        output_dir: str = "./distilbert_market_classifier",
        epochs: int = 3,
        batch_size: int = 16
    ):
        """Обучает модель"""
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("❌ transformers недоступен")
            return
        
        # Инициализируем модель
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3  # ПАДЕНИЕ, БОКОВИК, РОСТ
        )
        
        # Настройки обучения
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
        )
        
        logger.info("🎓 Начинаем обучение DistilBERT...")
        
        # Обучение
        trainer.train()
        
        # Сохранение
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"✅ Модель обучена и сохранена в {output_dir}")
        
        # Оценка
        results = trainer.evaluate()
        logger.info(f"📊 Результаты: {results}")
        
        return results
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Предсказывает класс для текста"""
        
        if not self.model:
            logger.error("❌ Модель не загружена")
            return "БОКОВИК", 0.0
        
        # Токенизация
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # Предсказание
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Маппинг
        class_names = {0: "ПАДЕНИЕ", 1: "БОКОВИК", 2: "РОСТ"}
        
        return class_names[predicted_class], confidence


async def main():
    """Запуск обучения"""
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Установите библиотеки:")
        print("pip install transformers datasets torch")
        return
    
    print("\n" + "="*60)
    print("🎓 ОБУЧЕНИЕ DISTILBERT ДЛЯ КЛАССИФИКАЦИИ РЫНКА")
    print("="*60)
    
    # 1. Собираем данные
    print("\n📊 Шаг 1: Сбор исторических данных...")
    
    from bot_v2_exchange import ExchangeManager
    
    exchange = ExchangeManager()
    await exchange.connect()
    
    builder = MarketDatasetBuilder()
    
    # Выбираем популярные монеты
    symbols = [
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
        'BNB/USDT:USDT', 'XRP/USDT:USDT', 'DOGE/USDT:USDT'
    ]
    
    data = await builder.build_from_history(exchange, symbols, days=7)
    builder.save_dataset()
    
    await exchange.disconnect()
    
    if len(data) < 100:
        print("❌ Недостаточно данных для обучения!")
        return
    
    # 2. Обучаем модель
    print("\n🎓 Шаг 2: Обучение DistilBERT...")
    
    trainer = DistilBERTTrainer()
    dataset = trainer.prepare_dataset(data)
    
    results = trainer.train(
        dataset,
        epochs=3,
        batch_size=8
    )
    
    print("\n✅ Обучение завершено!")
    print(f"📊 Точность: {results.get('eval_accuracy', 'N/A')}")
    
    # 3. Тестируем
    print("\n🧪 Шаг 3: Тестирование...")
    
    test_texts = [
        "price rising strongly, volume surging, bullish momentum",
        "price falling sharply, volume declining, bearish momentum",
        "price stable, volume normal, sideways movement"
    ]
    
    for text in test_texts:
        prediction, confidence = trainer.predict(text)
        print(f"📝 '{text[:40]}...'")
        print(f"   → {prediction} ({confidence:.0%})")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    asyncio.run(main())

