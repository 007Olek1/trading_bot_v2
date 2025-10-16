#!/usr/bin/env python3
"""
🎓 ML Trainer V2.0 - Enhanced with Early Stopping & AUC
Обучение DistilBERT с продвинутыми техниками

УЛУЧШЕНИЯ:
- ✅ Early Stopping (предотвращение переобучения)
- ✅ AUC метрика (важнее Accuracy для трейдинга!)
- ✅ Class Balancing (учёт дисбаланса классов)
- ✅ Learning Rate Scheduling
- ✅ Model Checkpointing
- ✅ Validation на отложенной выборке
- ✅ Экспорт метрик и визуализация
"""

import asyncio
import logging
from typing import List, Dict, Tuple, Any
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import pickle

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s"
)

try:
    from transformers import (
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback
    )
    from datasets import Dataset
    import torch
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
        confusion_matrix,
        classification_report
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.error("❌ transformers не установлен! pip install transformers datasets torch scikit-learn")
    # Mock классы для избежания ошибок импорта
    Dataset = object
    torch = None


@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    model_name: str = "distilbert-base-uncased"
    output_dir: str = "./models/distilbert_market_v2"
    
    # Обучение
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Early Stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Валидация
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model: bool = True
    metric_for_best_model: str = "eval_auc"  # ← КЛЮЧЕВОЕ!
    
    # Оптимизация
    fp16: bool = True  # Mixed precision (GPU)
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Данные
    max_length: int = 128
    test_size: float = 0.2
    val_size: float = 0.1


class EnhancedDatasetBuilder:
    """
    Улучшенный сборщик датасета с балансировкой классов
    """
    
    def __init__(self):
        self.data = []
        logger.info("📦 Enhanced Dataset Builder инициализирован")
    
    async def build_from_historical_data(
        self,
        exchange_manager,
        symbols: List[str],
        days: int = 60,
        balance_classes: bool = True
    ) -> pd.DataFrame:
        """
        Собирает данные с балансировкой классов
        """
        from bot_v2_nlp_analyzer_v2 import OptimizedFeatureExtractor
        
        extractor = OptimizedFeatureExtractor()
        all_data = []
        
        logger.info(f"📊 Сбор данных по {len(symbols)} символам за {days} дней...")
        
        for symbol in symbols:
            try:
                # Загружаем свечи
                candles = await exchange_manager.fetch_ohlcv(
                    symbol,
                    timeframe='1h',
                    limit=days * 24
                )
                
                if len(candles) < 50:
                    continue
                
                # Преобразуем в DataFrame
                df = pd.DataFrame(candles)
                
                # Вычисляем признаки
                features = extractor._calculate_features_vectorized(df)
                
                # Генерируем текст
                features['text'] = extractor._features_to_text_vectorized(features)
                
                # Целевая переменная (будущее движение)
                # Смотрим на 5 свечей вперёд
                future_price = df['close'].shift(-5)
                current_price = df['close']
                price_change = ((future_price - current_price) / current_price) * 100
                
                # Классы: 0 = ПАДЕНИЕ, 1 = БОКОВИК, 2 = РОСТ
                features['label'] = 1  # По умолчанию боковик
                features.loc[price_change > 2, 'label'] = 2  # Рост
                features.loc[price_change < -2, 'label'] = 0  # Падение
                
                features['symbol'] = symbol
                features['price_change'] = price_change
                
                # Убираем NaN
                features = features.dropna(subset=['text', 'label'])
                
                all_data.append(features[['text', 'label', 'symbol', 'price_change']])
                
                logger.info(f"✅ {symbol}: {len(features)} примеров")
                
            except Exception as e:
                logger.error(f"❌ Ошибка {symbol}: {e}")
                continue
        
        if not all_data:
            logger.error("❌ Нет данных!")
            return pd.DataFrame()
        
        # Объединяем
        df = pd.concat(all_data, ignore_index=True)
        
        # Балансировка классов
        if balance_classes:
            df = self._balance_classes(df)
        
        logger.info(f"📦 Всего собрано {len(df)} примеров")
        
        # Статистика
        label_dist = df['label'].value_counts().to_dict()
        logger.info(f"📊 Распределение классов: {label_dist}")
        
        return df
    
    def _balance_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Балансирует классы (undersampling/oversampling)"""
        
        # Считаем распределение
        class_counts = df['label'].value_counts()
        min_count = class_counts.min()
        
        logger.info(f"⚖️ Балансировка классов до {min_count} примеров каждого")
        
        # Undersample к минимальному классу
        balanced_dfs = []
        for label in df['label'].unique():
            class_df = df[df['label'] == label]
            sampled = class_df.sample(n=min(min_count, len(class_df)), random_state=42)
            balanced_dfs.append(sampled)
        
        balanced = pd.concat(balanced_dfs, ignore_index=True)
        balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        logger.info(f"✅ Сбалансировано: {len(balanced)} примеров")
        
        return balanced
    
    def save_dataset(self, df: pd.DataFrame, filename: str = 'market_dataset_v2.csv'):
        """Сохраняет датасет"""
        df.to_csv(filename, index=False)
        logger.info(f"💾 Датасет сохранён: {filename}")
    
    def load_dataset(self, filename: str = 'market_dataset_v2.csv') -> pd.DataFrame:
        """Загружает датасет"""
        df = pd.read_csv(filename)
        logger.info(f"📂 Датасет загружен: {len(df)} примеров")
        return df


class EnhancedDistilBERTTrainer:
    """
    Улучшенный тренер с Early Stopping и AUC
    """
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.metrics_history = []
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("❌ transformers недоступен!")
            return
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.config.model_name)
        logger.info(f"🤖 Enhanced Trainer инициализирован: {self.config.model_name}")
    
    def prepare_dataset(self, df: pd.DataFrame) -> Dict[str, Dataset]:
        """
        Подготавливает данные с train/val/test split
        """
        from sklearn.model_selection import train_test_split
        
        # Извлекаем тексты и метки
        texts = df['text'].tolist()
        labels = df['label'].astype(int).tolist()
        
        # Токенизация
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length
        )
        
        # Train/Temp split
        X_train, X_temp, y_train, y_temp = train_test_split(
            encodings['input_ids'],
            labels,
            test_size=self.config.test_size + self.config.val_size,
            random_state=42,
            stratify=labels
        )
        
        # Val/Test split
        val_test_ratio = self.config.val_size / (self.config.test_size + self.config.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=1 - val_test_ratio,
            random_state=42,
            stratify=y_temp
        )
        
        # Создаём datasets
        def create_dataset(input_ids, labels):
            # Преобразуем обратно через tokenizer
            texts_subset = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            encodings_subset = self.tokenizer(
                texts_subset,
                truncation=True,
                padding=True,
                max_length=self.config.max_length
            )
            
            return Dataset.from_dict({
                'input_ids': encodings_subset['input_ids'],
                'attention_mask': encodings_subset['attention_mask'],
                'labels': labels
            })
        
        datasets = {
            'train': create_dataset(X_train, y_train),
            'validation': create_dataset(X_val, y_val),
            'test': create_dataset(X_test, y_test)
        }
        
        logger.info(f"📚 Datasets готовы:")
        logger.info(f"   Train: {len(datasets['train'])}")
        logger.info(f"   Validation: {len(datasets['validation'])}")
        logger.info(f"   Test: {len(datasets['test'])}")
        
        return datasets
    
    def compute_metrics(self, eval_pred):
        """
        Вычисляет метрики включая AUC
        """
        predictions, labels = eval_pred
        
        # Вероятности
        probs = torch.softmax(torch.from_numpy(predictions), dim=1).numpy()
        
        # Предсказанные классы
        preds = np.argmax(predictions, axis=1)
        
        # Accuracy
        accuracy = accuracy_score(labels, preds)
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )
        
        # AUC (one-vs-rest для multiclass)
        try:
            auc = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
        except ValueError:
            auc = 0.5
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc  # ← ВАЖНЕЙШАЯ МЕТРИКА!
        }
    
    def train(self, datasets: Dict[str, Dataset]):
        """
        Обучает модель с Early Stopping
        """
        # Инициализация модели
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=3,  # ПАДЕНИЕ, БОКОВИК, РОСТ
            problem_type="single_label_classification"
        )
        
        # Подавляем предупреждения
        import logging as transformers_logging
        transformers_logging.getLogger("transformers.modeling_utils").setLevel(transformers_logging.ERROR)
        
        # Аргументы обучения
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size * 2,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            
            # Оптимизация
            fp16=self.config.fp16 and torch.cuda.is_available(),
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_grad_norm=self.config.max_grad_norm,
            
            # Валидация
            eval_strategy=self.config.eval_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=True,  # AUC больше = лучше
            
            # Логирование
            logging_dir=f'{self.config.output_dir}/logs',
            logging_steps=50,
            report_to="none",
            
            # Сохранение
            save_total_limit=2,  # Только 2 лучших checkpoint
        )
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                )
            ]
        )
        
        logger.info("🎓 Начинаем обучение с Early Stopping...")
        logger.info(f"   Epochs: {self.config.num_epochs}")
        logger.info(f"   Batch size: {self.config.batch_size}")
        logger.info(f"   Learning rate: {self.config.learning_rate}")
        logger.info(f"   Early stopping patience: {self.config.early_stopping_patience}")
        logger.info(f"   Metric: {self.config.metric_for_best_model}")
        
        # Обучение
        train_result = self.trainer.train()
        
        # Сохранение лучшей модели
        self.trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info(f"✅ Обучение завершено!")
        logger.info(f"💾 Модель сохранена: {self.config.output_dir}")
        
        return train_result
    
    def evaluate(self, datasets: Dict[str, Dataset], split: str = 'test') -> Dict[str, float]:
        """
        Оценивает модель на тестовой выборке
        """
        logger.info(f"📊 Оценка на {split} выборке...")
        
        results = self.trainer.evaluate(datasets[split])
        
        logger.info(f"✅ Результаты на {split}:")
        for metric, value in results.items():
            logger.info(f"   {metric}: {value:.4f}")
        
        return results
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказывает классы и вероятности
        
        Returns:
            (predicted_classes, probabilities)
        """
        if not self.model:
            logger.error("❌ Модель не загружена!")
            return np.array([]), np.array([])
        
        # Токенизация
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Предсказание
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).numpy()
            preds = np.argmax(probs, axis=1)
        
        return preds, probs
    
    def save_metrics(self, metrics: Dict, filename: str = 'training_metrics.json'):
        """Сохраняет метрики"""
        metrics['timestamp'] = datetime.now().isoformat()
        
        with open(f"{self.config.output_dir}/{filename}", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"📊 Метрики сохранены: {filename}")
    
    def load_model(self, model_dir: str = None):
        """Загружает обученную модель"""
        model_dir = model_dir or self.config.output_dir
        
        try:
            self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
            logger.info(f"✅ Модель загружена: {model_dir}")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")


async def main():
    """Полный цикл обучения"""
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Установите: pip install transformers datasets torch scikit-learn")
        return
    
    print("\n" + "="*70)
    print("🎓 ML TRAINER V2.0 - ENHANCED WITH EARLY STOPPING & AUC")
    print("="*70)
    
    # Конфигурация
    config = TrainingConfig(
        num_epochs=5,
        batch_size=8,
        early_stopping_patience=2,
        metric_for_best_model="eval_auc"
    )
    
    # 1. Сбор данных
    print("\n📊 ШАГ 1: Сбор исторических данных...")
    
    from bot_v2_exchange import ExchangeManager
    
    exchange = ExchangeManager()
    await exchange.connect()
    
    builder = EnhancedDatasetBuilder()
    
    # Топ символы
    symbols = [
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
        'BNB/USDT:USDT', 'XRP/USDT:USDT'
    ]
    
    df = await builder.build_from_historical_data(exchange, symbols, days=14, balance_classes=True)
    
    await exchange.disconnect()
    
    if df.empty or len(df) < 100:
        print("❌ Недостаточно данных!")
        return
    
    builder.save_dataset(df)
    
    # 2. Подготовка данных
    print("\n📚 ШАГ 2: Подготовка datasets...")
    
    trainer = EnhancedDistilBERTTrainer(config)
    datasets = trainer.prepare_dataset(df)
    
    # 3. Обучение
    print("\n🎓 ШАГ 3: Обучение модели...")
    
    train_result = trainer.train(datasets)
    
    # 4. Оценка
    print("\n📊 ШАГ 4: Оценка на тестовой выборке...")
    
    test_metrics = trainer.evaluate(datasets, split='test')
    trainer.save_metrics(test_metrics)
    
    # 5. Тестирование предсказаний
    print("\n🧪 ШАГ 5: Тестовые предсказания...")
    
    test_texts = [
        "price rising strongly volume surging near resistance",
        "price falling sharply volume declining oversold",
        "price consolidating sideways movement volume stable"
    ]
    
    preds, probs = trainer.predict(test_texts)
    
    class_names = ['ПАДЕНИЕ', 'БОКОВИК', 'РОСТ']
    
    for i, text in enumerate(test_texts):
        pred_class = class_names[preds[i]]
        confidence = probs[i][preds[i]]
        print(f"\n   '{text[:50]}...'")
        print(f"   → {pred_class} ({confidence:.1%} confidence)")
        print(f"   Probabilities: {dict(zip(class_names, probs[i]))}")
    
    print("\n" + "="*70)
    print("✅ ВСЁ ГОТОВО!")
    print(f"📊 Test AUC: {test_metrics.get('eval_auc', 0):.3f}")
    print(f"📊 Test Accuracy: {test_metrics.get('eval_accuracy', 0):.3f}")
    print(f"💾 Модель сохранена: {config.output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

