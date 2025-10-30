#!/usr/bin/env python3
"""
🎯 СИСТЕМА АВТОМАТИЧЕСКОГО ВЫБОРА МОНЕТ ДЛЯ ТОРГОВЛИ
- Динамический выбор лучших монет по объему и активности
- Фильтрация по ликвидности и волатильности
- Адаптивное количество монет в зависимости от рыночных условий
"""

import asyncio
import ccxt
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class SmartCoinSelector:
    """Умный селектор монет для торговли"""
    
    def __init__(self):
        self.min_volume_24h = 1000000  # Минимальный объем $1M
        self.min_price = 0.001  # Минимальная цена $0.001
        self.max_price = 500000  # Максимальная цена $500K (для BTC и других дорогих монет)
        self.min_change_24h = -50  # Минимальное изменение -50% (манипуляции вниз)
        self.max_change_24h = 200   # Максимальное изменение +200% (манипуляции вверх)
        
        # Популярные мемкоины и мем-токены (расширенный список самых известных)
        self.allowed_memecoins = [
            # Топ-мемкоины (очень известные) - используем правильные форматы фьючерсов
            'DOGEUSDT',      # Dogecoin - самый известный мемкоин
            'SHIBUSDT',      # Shiba Inu - второй по популярности
            'PEPEUSDT',      # Pepe - популярный лягушонок
            '1000FLOKIUSDT', # Floki - викинг-мемкоин (правильный формат фьючерса)
            'BONKUSDT',   # Bonk - Solana мемкоин
            'WIFUSDT',    # Dogwifhat - очень популярный
            'BOMEUSDT',   # Book of Meme
            'MYROUSDT',   # Myro
            'POPCATUSDT', # Popcat
            'MEWUSDT',    # Mew
            
            # Другие популярные мемкоины
            'FLOKIUSDT', 'BABYDOGEUSDT', 'ELONUSDT', 'SAFEMOONUSDT',
            'DOBOUSDT', 'SHIBUSDT', 'FEGUSDT', 'KISHUUSDT',
            'HOKKUSDT', 'AKITAUSDT', 'SAMOUSDT', 'VOLTUSDT',
            
            # Новые популярные мем-токены (2024-2025)
            'BRETTUSDT', 'GIGACHADUSDT', 'TREMPUSDT', 'GOATSUUSDT',
            'MICHIUSDT', 'OWLUSDT', 'NEIROUSDT', 'ANDYUSDT',
            'TURBOUSDT', 'LADYSUSDT', 'AIDOGEUSDT', 'PSPSUSDT',
        ]
        
        # Паттерны для автоматического определения мемкоинов
        self.memecoin_patterns = [
            'DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'BOME',
            'ELON', 'SAFE', 'DOBO', 'FEG', 'KISHU', 'HOKK', 'AKITA',
            'SAMO', 'VOLT', 'BRETT', 'GIGA', 'TREMP', 'GOATS', 'MICHI',
            'OWL', 'NEIRO', 'ANDY', 'TURBO', 'LADY', 'AIDOGE', 'PSPS',
            'BABY', 'MEW', 'POPCAT', 'MYRO'
        ]
        
        # Явный бан-лист токенов из «Зоны инноваций» и высокорисковых листингов
        # ВНИМАНИЕ: символы указывать в формате без разделителей, например TURTLEUSDT
        self.innovation_zone_blacklist = set([
            'TURTLEUSDT',
            # можно расширять из env/файла, см. ниже
        ])
        
        # Маркеры «Зоны инноваций»/высокого риска из полей биржи (ccxt market.info/ticker.info)
        self.innovation_markers = {
            'innovation', 'newlisting', 'new_listing', 'hot', 'risk', 'specialtreatment', 'st',
            'seed', 'launchpad', 'trial', 'isolated_only'
        }
        
        # Расширяем blacklist из окружения и локального файла
        try:
            import os
            env_list = os.getenv('INNOVATION_BLACKLIST', '')
            for token in [x.strip().upper().replace('/', '').replace('-', '') for x in env_list.split(',') if x.strip()]:
                if token.endswith(':USDT'):
                    token = token[:-5] + 'USDT'
                token = token.replace(':', '')
                if not token.endswith('USDT') and token:
                    token = token + 'USDT'
                self.innovation_zone_blacklist.add(token)
            # Локальный конфиг
            from pathlib import Path
            fpath = Path(__file__).parent / 'config' / 'blacklist_symbols.txt'
            if fpath.exists():
                for line in fpath.read_text(encoding='utf-8').splitlines():
                    token = line.strip().upper().replace('/', '').replace('-', '')
                    if not token:
                        continue
                    if token.endswith(':USDT'):
                        token = token[:-5] + 'USDT'
                    token = token.replace(':', '')
                    if not token.endswith('USDT'):
                        token = token + 'USDT'
                    self.innovation_zone_blacklist.add(token)
        except Exception as _:
            pass
    
    def filter_symbols(self, symbols_data: List[Dict]) -> List[Dict]:
        """🔍 Фильтрация символов по базовым критериям (обновлено для манипуляций и мемкоинов)"""
        filtered = []
        
        for symbol_data in symbols_data:
            try:
                symbol = symbol_data.get('symbol', '').upper()
                
                # Проверяем объем
                volume_24h = symbol_data.get('volume_24h', 0)
                if volume_24h < self.min_volume_24h:
                    continue
                
                # Проверяем цену (исключение для BTC и других топ-монет)
                price = symbol_data.get('price', 0)
                # BTC и ETH могут быть выше $100K, пропускаем их
                if 'BTC' in symbol or 'ETH' in symbol:
                    # Для BTC/ETH проверяем только минимум
                    if price < self.min_price:
                        continue
                else:
                    # Для остальных - стандартный диапазон
                    if price < self.min_price or price > self.max_price:
                        continue
                
                # Проверяем изменение за 24h (расширенный диапазон для поиска манипуляций)
                change_24h = symbol_data.get('change_24h', 0)
                # Для мемкоинов более мягкие ограничения (они более волатильны)
                is_memecoin = symbol in self.allowed_memecoins or self._is_memecoin_by_pattern(symbol)
                if is_memecoin:
                    # Мемкоины: до -70% и до +300% (очень волатильные)
                    if change_24h < -70 or change_24h > 300:
                        continue
                else:
                    # Обычные монеты: расширенный диапазон для манипуляций
                    if change_24h < self.min_change_24h or change_24h > self.max_change_24h:
                        continue
                
                filtered.append(symbol_data)
                
            except Exception as e:
                logger.debug(f"⚠️ Ошибка фильтрации {symbol_data}: {e}")
                continue
        
        return filtered
    
    def analyze_market(self, symbols_data: List[Dict]) -> Dict:
        """📊 Анализ общего состояния рынка"""
        try:
            if not symbols_data:
                return {
                    'condition': 'NEUTRAL',
                    'score': 0,
                    'btc_change': 0.0,
                    'rising_count': 0,
                    'falling_count': 0,
                    'total_count': 0
                }
            
            # Подсчитываем растущие/падающие монеты
            rising = 0
            falling = 0
            total_change = 0
            btc_change = 0
            
            for symbol_data in symbols_data:
                change_24h = symbol_data.get('change_24h', 0)
                total_change += change_24h
                
                if change_24h > 0:
                    rising += 1
                elif change_24h < 0:
                    falling += 1
                
                # Ищем BTC для отдельного анализа
                if 'BTC' in symbol_data.get('symbol', ''):
                    btc_change = change_24h
            
            total_count = len(symbols_data)
            avg_change = total_change / total_count if total_count > 0 else 0
            
            # Рассчитываем score
            score = (rising - falling) * 5 + avg_change * 2
            
            # Определяем условие рынка
            if score > 30:
                condition = 'BULLISH'
            elif score < -30:
                condition = 'BEARISH'
            elif abs(score) > 15:
                condition = 'VOLATILE'
            else:
                condition = 'NEUTRAL'
            
            return {
                'condition': condition,
                'score': score,
                'btc_change': btc_change,
                'rising_count': rising,
                'falling_count': falling,
                'total_count': total_count
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа рынка: {e}")
            return {
                'condition': 'NEUTRAL',
                'score': 0,
                'btc_change': 0.0,
                'rising_count': 0,
                'falling_count': 0,
                'total_count': 0
            }
        
    async def get_smart_symbols(self, exchange, market_condition: str = "normal") -> List[str]:
        """
        Получить умно отобранные символы для торговли
        
        Args:
            exchange: Объект биржи
            market_condition: Условие рынка (BULLISH, BEARISH, NEUTRAL, VOLATILE или bull, bear, normal, volatile)
        """
        try:
            logger.info("🎯 Начинаю умный отбор монет...")
            
            # Нормализуем условие рынка
            market_condition = market_condition.lower()
            
            # Получаем тикеры через markets (более надежно для Bybit)
            tickers = {}
            
            try:
                # Загружаем markets если еще не загружены
                if not hasattr(exchange, 'markets') or not exchange.markets:
                    await exchange.load_markets()
                
                # Получаем ТОЛЬКО линейные фьючерсы (linear) из markets
                all_symbols = list(exchange.markets.keys())
                usdt_symbols = []
                for s in all_symbols:
                    if s.startswith('.'):
                        continue
                    market = exchange.markets.get(s, {})
                    # ТОЛЬКО линейные контракты (фьючерсы)
                    if market.get('linear', False) or market.get('type', '') == 'linear':
                        if 'USDT' in s and (':USDT' in s or s.endswith('USDT')):
                            usdt_symbols.append(s)
                
                logger.info(f"📊 Найдено USDT символов в markets: {len(usdt_symbols)}")
                
                # Для Bybit используем правильный формат категории
                # Пытаемся получить тикеры через правильный API метод
                try:
                    # Для Bybit v5 API используем правильные параметры
                    if hasattr(exchange, 'api') and 'bybit' in exchange.id.lower():
                        # Получаем топ символов по объему через правильный API endpoint
                        # Используем метод бота для получения топ символов
                        tickers_data = await exchange.fetch_tickers(params={'category': 'linear'})
                        if tickers_data:
                            tickers.update(tickers_data)
                except Exception as e:
                    logger.debug(f"⚠️ fetch_tickers с параметрами не сработал: {e}")
                
                # Если тикеров мало, получаем данные из markets и формируем тикеры
                if len(tickers) < 50:
                    logger.info(f"📊 Получаем данные из markets для {min(200, len(usdt_symbols))} символов...")
                    for symbol in usdt_symbols[:200]:
                        try:
                            # Получаем тикер для конкретного символа (ТОЛЬКО фьючерсы)
                            ticker = await exchange.fetch_ticker(symbol, params={'category': 'linear'})
                            if ticker:
                                tickers[symbol] = ticker
                        except:
                            # Если не получили тикер, используем базовую info из market
                            if symbol in exchange.markets:
                                market = exchange.markets[symbol]
                                tickers[symbol] = {
                                    'symbol': symbol,
                                    'quoteVolume': market.get('info', {}).get('volume24h', 0) or 1000000,
                                    'last': market.get('last', 0),
                                    'percentage': market.get('percentage', 0),
                                    'high': market.get('high', 0),
                                    'low': market.get('low', 0),
                                    'open': market.get('open', 0),
                                    'close': market.get('close', 0)
                                }
                            
                            # Ограничиваем количество запросов
                            if len(tickers) >= 200:
                                break
                            
                            # Небольшая пауза чтобы не перегрузить API
                            if len(tickers) % 50 == 0:
                                await asyncio.sleep(0.1)
                    
                    logger.info(f"📊 Получено тикеров: {len(tickers)}")
                
            except Exception as e:
                logger.error(f"❌ Ошибка получения тикеров: {e}")
                # Используем fallback
                if not tickers or len(tickers) < 50:
                    return await self._get_fallback_symbols(exchange)
            
            if not tickers or len(tickers) < 10:
                logger.warning(f"⚠️ Тикеров слишком мало ({len(tickers)}), используем fallback")
                return await self._get_fallback_symbols(exchange)
            
            # Фильтруем USDT пары
            usdt_pairs = self._filter_usdt_pairs(tickers)
            
            if not usdt_pairs:
                logger.warning("⚠️ USDT пары не найдены, используем fallback")
                return await self._get_fallback_symbols(exchange)
            
            # Применяем базовые фильтры
            filtered_pairs = self._apply_basic_filters(usdt_pairs)
            
            if not filtered_pairs:
                # Если после фильтров пусто, но сырые пары есть — возьмём топ по объёму без жёстких фильтров, чтобы обеспечить минимум 100-200 монет
                if usdt_pairs:
                    logger.warning("⚠️ После фильтрации пусто — выбираю топ по объёму без жёстких фильтров")
                    target_count = self._get_target_count(market_condition)
                    rough_sorted = sorted(usdt_pairs, key=lambda x: x[1].get('quoteVolume', 0) or 0, reverse=True)
                    symbols = [pair[0] for pair in rough_sorted[:target_count]]
                    logger.info(f"🎯 Выбран rough-топ: {len(symbols)} монет (мин-гар.)")
                    return symbols
                logger.warning("⚠️ После фильтрации монет не осталось, используем fallback")
                return await self._get_fallback_symbols(exchange)
            
            # Сортируем по объему
            sorted_pairs = sorted(filtered_pairs, key=lambda x: x[1].get('quoteVolume', 0) or 0, reverse=True)
            
            # Адаптируем количество под рыночные условия
            target_count = self._get_target_count(market_condition)
            
            # Выбираем топ монеты
            selected_pairs = sorted_pairs[:target_count]
            
            symbols = [pair[0] for pair in selected_pairs]
            
            logger.info(f"🎯 Умный селектор отобрал {len(symbols)} монет из {len(usdt_pairs)} доступных")
            logger.info(f"📊 Топ-5 по объему: {', '.join(symbols[:5])}")
            
            return symbols
            
        except Exception as e:
            logger.error(f"❌ Ошибка отбора монет: {e}", exc_info=True)
            # Fallback к стандартному списку
            return await self._get_fallback_symbols(exchange)
    
    def _filter_usdt_pairs(self, tickers: Dict) -> List[Tuple[str, Dict]]:
        """Фильтруем только USDT пары"""
        usdt_pairs = []
        for symbol, ticker in tickers.items():
            try:
                # Пропускаем системные символы
                if symbol.startswith('.') or not symbol:
                    continue
                
                # Поддерживаем разные форматы:
                # - BTC/USDT:USDT -> BTCUSDT
                # - BTCUSDT -> BTCUSDT
                # - BTC/USDT -> BTCUSDT
                # - PHBUSDT:USDT -> PHBUSDT (убираем :USDT, не заменяем!)
                normalized_symbol = symbol.replace('/', '').upper()
                # Если есть двоеточие в середине (формат вида BASE:USDCUSDT), берём часть слева от ':' и приклеиваем USDT
                if ':' in normalized_symbol:
                    base_part = normalized_symbol.split(':', 1)[0]
                    normalized_symbol = (base_part if base_part.endswith('USDT') else base_part + 'USDT')
                # Убеждаемся, что заканчивается на USDT
                if not normalized_symbol.endswith('USDT'):
                    normalized_symbol = normalized_symbol + 'USDT'
                # Убираем возможное дублирование USDT в конце
                while normalized_symbol.endswith('USDTUSDT'):
                    normalized_symbol = normalized_symbol[:-4]
                
                # Должно заканчиваться на USDT
                if normalized_symbol.endswith('USDT') and len(normalized_symbol) > 4:
                    usdt_pairs.append((normalized_symbol, ticker))
            except Exception as e:
                logger.debug(f"⚠️ Ошибка обработки символа {symbol}: {e}")
                continue
        
        logger.info(f"📊 Найдено USDT пар: {len(usdt_pairs)}")
        return usdt_pairs
    
    def _apply_basic_filters(self, pairs: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """Применяем базовые фильтры (обновлено для манипуляций и мемкоинов)"""
        filtered = []
        
        for symbol, ticker in pairs:
            try:
                # Нормализуем символ (предотвращаем дублирование USDT)
                normalized_symbol = symbol.upper().replace('/', '').replace('-', '')
                # Если присутствует формат с ':' (например BTCUSDC:USDCUSDT), берём левую часть и приводим к ...USDT
                if ':' in normalized_symbol:
                    base_part = normalized_symbol.split(':', 1)[0]
                    normalized_symbol = (base_part if base_part.endswith('USDT') else base_part + 'USDT')
                # Убеждаемся, что заканчивается на USDT
                if not normalized_symbol.endswith('USDT'):
                    normalized_symbol = normalized_symbol + 'USDT'
                # Убираем возможное дублирование USDT в конце
                while normalized_symbol.endswith('USDTUSDT'):
                    normalized_symbol = normalized_symbol[:-4]
                
                # Жестко исключаем токены из «Зоны инноваций»/высокого риска (явный список)
                if normalized_symbol in self.innovation_zone_blacklist:
                    logger.debug(f"🚫 Исключен по Innovation Zone blacklist: {normalized_symbol}")
                    continue
                
                # Автодетекция по маркерам биржи (market/ticker info)
                try:
                    info_obj = ticker.get('info') or {}
                    # объединяем ключи и значения в одну строку
                    blob = ' '.join(list(info_obj.keys()) + [str(v) for v in info_obj.values()]).lower()
                    if any(mrk in blob for mrk in self.innovation_markers):
                        logger.debug(f"🚫 Исключен по маркерам биржи (Innovation/Risk): {normalized_symbol}")
                        continue
                except Exception:
                    pass
                
                # Проверяем объем
                volume_24h = ticker.get('quoteVolume', 0)
                if volume_24h < self.min_volume_24h:
                    continue
                
                # Проверяем цену (исключение для BTC и ETH)
                price = ticker.get('last', 0)
                # BTC и ETH могут быть выше $100K, пропускаем их через фильтр цены
                if 'BTC' in normalized_symbol or 'ETH' in normalized_symbol:
                    # Для BTC/ETH проверяем только минимум
                    if price < self.min_price:
                        continue
                else:
                    # Для остальных - стандартный диапазон (обновлен до $500K)
                    if price < self.min_price or price > self.max_price:
                        continue
                
                # Проверяем изменение цены (расширенный диапазон для манипуляций)
                change_24h = ticker.get('percentage', 0)
                # Для мемкоинов более мягкие ограничения (автоматическое определение)
                is_memecoin = normalized_symbol in self.allowed_memecoins or self._is_memecoin_by_pattern(normalized_symbol)
                if is_memecoin:
                    # Мемкоины: до -70% и до +300% (очень волатильные, манипуляции)
                    if change_24h < -70 or change_24h > 300:
                        continue
                    logger.debug(f"🎭 Мемкоин {symbol} включен (изменение: {change_24h:.1f}%)")
                else:
                    # Обычные монеты: расширенный диапазон для поиска манипуляций
                    if change_24h < self.min_change_24h or change_24h > self.max_change_24h:
                        continue
                
                # Проверяем наличие данных
                if not all([ticker.get('high'), ticker.get('low'), ticker.get('open'), ticker.get('close')]):
                    continue
                
                filtered.append((symbol, ticker))
                
            except Exception as e:
                logger.debug(f"⚠️ Ошибка фильтрации {symbol}: {e}")
                continue
        
        logger.info(f"📊 После фильтрации: {len(filtered)} монет из {len(pairs)} (включая мемкоины и манипуляции)")
        return filtered
    
    def _get_target_count(self, market_condition: str) -> int:
        """Определяем целевое количество монет в зависимости от рыночных условий"""
        # Нормализуем условие
        condition = market_condition.lower()
        if condition in ['bullish', 'bull']:
            condition = 'bull'
        elif condition in ['bearish', 'bear']:
            condition = 'bear'
        elif condition in ['volatile']:
            condition = 'volatile'
        else:
            condition = 'normal'
        
        base_counts = {
            'bull': 200,      # Бычий рынок - больше монет
            'bear': 100,      # Медвежий рынок - меньше монет
            'volatile': 150,  # Волатильный рынок - среднее количество
            'normal': 145     # Обычные условия - 145 монет!
        }
        
        count = base_counts.get(condition, 145)
        logger.info(f"🎯 Целевое количество монет для {condition.upper()}: {count}")
        return count
    
    async def _get_fallback_symbols(self, exchange) -> List[str]:
        """Fallback список популярных монет и мемкоинов при ошибке (расширенный список)"""
        # Пытаемся динамически построить fallback из markets (топ по объёму) до целевого количества
        try:
            target_count = self._get_target_count('normal')
            if hasattr(exchange, 'markets') and exchange.markets:
                items = []
                for k, m in exchange.markets.items():
                    try:
                        if (m.get('linear') or m.get('type')=='linear') and ('USDT' in k):
                            sym = k.replace('/', '')
                            if sym.endswith(':USDT'):
                                sym = sym[:-5] + 'USDT'
                            vol = m.get('info', {}).get('volume24h', 0) or 0
                            items.append((sym, float(vol)))
                    except Exception:
                        continue
                items = sorted(items, key=lambda x: x[1], reverse=True)
                symbols = [s for s,_ in items[:target_count]]
                if symbols:
                    logger.warning(f"⚠️ Использую динамический fallback из markets: {len(symbols)} монет")
                    return symbols
        except Exception:
            pass
        # Статический резерв на крайний случай (≈145 не гарантируем, но покроем основные)
        fallback_symbols = [
            'BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT','ADAUSDT','AVAXUSDT','LINKUSDT','DOTUSDT','LTCUSDT',
            'ATOMUSDT','ETCUSDT','XLMUSDT','NEARUSDT','ICPUSDT','FILUSDT','APTUSDT','ARBUSDT','OPUSDT','SUIUSDT',
            'TIAUSDT','SEIUSDT','DOGEUSDT','SHIBUSDT','PEPEUSDT','1000FLOKIUSDT','BONKUSDT','WIFUSDT','BOMEUSDT','MYROUSDT',
            'POPCATUSDT','MEWUSDT','TRXUSDT','TONUSDT','AAVEUSDT','AAVEUSDT','HBARUSDT','BCHUSDT','AAVEUSDT','UNIUSDT'
        ]
        logger.warning(f"⚠️ Использую статический fallback: {len(fallback_symbols)} монет")
        return list(dict.fromkeys(fallback_symbols))
    
    def analyze_market_condition(self, btc_change: float, market_trend: str) -> str:
        """
        Анализируем рыночные условия для адаптации выбора монет
        
        Args:
            btc_change: Изменение BTC за 24ч
            market_trend: Общий тренд рынка
        """
        if btc_change > 5:
            return 'bull'
        elif btc_change < -5:
            return 'bear'
        elif abs(btc_change) > 3:
            return 'volatile'
        else:
            return 'normal'

# Пример использования
async def test_smart_selector():
    """Тестируем умный селектор"""
    exchange = ccxt.bybit({
        'apiKey': 'test',
        'secret': 'test',
        'sandbox': True
    })
    
    selector = SmartCoinSelector()
    
    # Тестируем разные рыночные условия
    conditions = ['bull', 'bear', 'volatile', 'normal']
    
    for condition in conditions:
        symbols = await selector.get_smart_symbols(exchange, condition)
        print(f"🎯 {condition.upper()}: {len(symbols)} монет")
        print(f"   Топ-3: {symbols[:3]}")

if __name__ == "__main__":
    asyncio.run(test_smart_selector())






