"""
📊 МЕНЕДЖЕР ПОЗИЦИЙ V3.0
✅ Множественные Take Profit уровни
✅ Trailing Stop после первого TP
✅ Частичное закрытие позиций
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class AdvancedPositionManager:
    """Продвинутое управление позициями"""
    
    def __init__(self):
        self.positions: Dict[str, Dict[str, Any]] = {}
    
    def create_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        amount: float,
        sl_price: float,
        tp_levels: List[float],
        leverage: int,
        signal_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Создание позиции с множественными TP
        
        Args:
            symbol: Символ торговой пары
            side: 'buy' или 'sell'
            entry_price: Цена входа
            amount: Размер позиции
            sl_price: Stop Loss цена
            tp_levels: [TP1, TP2, TP3, TP4, TP5]
            leverage: Плечо
            signal_data: Данные сигнала
        
        Returns:
            Position dict
        """
        position = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'initial_amount': amount,
            'current_amount': amount,
            'sl_price': sl_price,
            'tp_levels': tp_levels,
            'tp_hit': [False] * len(tp_levels),  # Какие TP достигнуты
            'leverage': leverage,
            'trailing_stop_active': False,
            'trailing_stop_price': None,
            'signal_confidence': signal_data.get('confidence', 0),
            'signal_reason': signal_data.get('reason', ''),
            'opened_at': datetime.now(),
            'closed_tp_count': 0,
            'total_realized_pnl': 0.0,
            'status': 'open',
            'levels': signal_data.get('levels', {}),
            'indicators': signal_data.get('indicators', {})
        }
        
        self.positions[symbol] = position
        logger.info(
            f"📊 Позиция создана: {symbol} {side.upper()}\n"
            f"   Вход: ${entry_price:.4f}\n"
            f"   Размер: {amount:.6f}\n"
            f"   SL: ${sl_price:.4f}\n"
            f"   TP уровни: {[f'${tp:.4f}' for tp in tp_levels]}"
        )
        
        return position
    
    def update_position(
        self,
        symbol: str,
        current_price: float,
        exchange_manager: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Обновление позиции и проверка TP/SL
        
        Returns:
            Action dict or None
            {
                'action': 'close_partial' | 'activate_trailing' | 'update_trailing' | 'hit_sl',
                'amount_to_close': float,
                'tp_level': int,
                'new_sl': float (для trailing)
            }
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Проверка Stop Loss
        if self._check_stop_loss(position, current_price):
            return {
                'action': 'hit_sl',
                'close_all': True,
                'reason': 'Stop Loss достигнут'
            }
        
        # Проверка Take Profit уровней
        tp_action = self._check_take_profit(position, current_price)
        if tp_action:
            return tp_action
        
        # Обновление Trailing Stop
        if position['trailing_stop_active']:
            trailing_action = self._update_trailing_stop(position, current_price)
            if trailing_action:
                return trailing_action
        
        return None
    
    def _check_stop_loss(self, position: Dict[str, Any], current_price: float) -> bool:
        """Проверка Stop Loss"""
        if position['side'] == 'buy':
            return current_price <= position['sl_price']
        else:  # sell
            return current_price >= position['sl_price']
    
    def _check_take_profit(
        self,
        position: Dict[str, Any],
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """
        Проверка достижения TP уровней
        
        Стратегия:
        - TP1 (25%): Закрываем 25% позиции + активируем trailing stop
        - TP2 (25%): Закрываем еще 25%
        - TP3 (25%): Закрываем еще 25%
        - TP4 (15%): Закрываем 15%
        - TP5 (10%): Закрываем остаток
        """
        tp_percentages = [0.25, 0.25, 0.25, 0.15, 0.10]  # Процент закрытия на каждом TP
        
        for i, (tp_price, hit) in enumerate(zip(position['tp_levels'], position['tp_hit'])):
            if hit:
                continue
            
            # Проверяем достижение TP
            tp_reached = False
            if position['side'] == 'buy':
                tp_reached = current_price >= tp_price
            else:  # sell
                tp_reached = current_price <= tp_price
            
            if tp_reached:
                # Отмечаем TP как достигнутый
                position['tp_hit'][i] = True
                position['closed_tp_count'] += 1
                
                # Рассчитываем сумму для закрытия
                close_percentage = tp_percentages[i]
                amount_to_close = position['initial_amount'] * close_percentage
                
                # Обновляем текущий размер позиции
                position['current_amount'] -= amount_to_close
                
                # Рассчитываем прибыль
                if position['side'] == 'buy':
                    pnl = (tp_price - position['entry_price']) * amount_to_close * position['leverage']
                else:
                    pnl = (position['entry_price'] - tp_price) * amount_to_close * position['leverage']
                
                position['total_realized_pnl'] += pnl
                
                logger.info(
                    f"🎯 TP{i+1} достигнут для {position['symbol']}!\n"
                    f"   Цена TP: ${tp_price:.4f}\n"
                    f"   Закрыто: {close_percentage*100:.0f}% позиции ({amount_to_close:.6f})\n"
                    f"   PnL: ${pnl:.2f}\n"
                    f"   Осталось: {position['current_amount']:.6f}"
                )
                
                action = {
                    'action': 'close_partial',
                    'amount_to_close': amount_to_close,
                    'tp_level': i + 1,
                    'tp_price': tp_price,
                    'pnl': pnl
                }
                
                # После первого TP активируем trailing stop
                if i == 0 and not position['trailing_stop_active']:
                    position['trailing_stop_active'] = True
                    position['trailing_stop_price'] = position['entry_price']  # Безубыток
                    action['activate_trailing'] = True
                    action['initial_trailing_price'] = position['entry_price']
                    logger.info(f"🔄 Trailing Stop активирован на уровне безубытка: ${position['entry_price']:.4f}")
                
                return action
        
        return None
    
    def _update_trailing_stop(
        self,
        position: Dict[str, Any],
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """
        Обновление Trailing Stop
        
        Trailing Stop перемещается вслед за ценой с отступом в 1 ATR
        """
        if not position.get('trailing_stop_active'):
            return None
        
        atr = position['indicators'].get('atr', 0)
        trailing_distance = atr * 1.5  # Отступ = 1.5 ATR
        
        if position['side'] == 'buy':
            # Для LONG: trailing stop движется вверх
            new_trailing_stop = current_price - trailing_distance
            
            if new_trailing_stop > position['trailing_stop_price']:
                old_trailing = position['trailing_stop_price']
                position['trailing_stop_price'] = new_trailing_stop
                logger.info(
                    f"🔄 Trailing Stop обновлен для {position['symbol']}: "
                    f"${old_trailing:.4f} -> ${new_trailing_stop:.4f}"
                )
                return {
                    'action': 'update_trailing',
                    'new_trailing_stop': new_trailing_stop,
                    'old_trailing_stop': old_trailing
                }
            
            # Проверка срабатывания trailing stop
            if current_price <= position['trailing_stop_price']:
                logger.info(f"🛑 Trailing Stop сработал для {position['symbol']} на ${position['trailing_stop_price']:.4f}")
                return {
                    'action': 'hit_trailing_stop',
                    'close_all': True,
                    'price': position['trailing_stop_price'],
                    'reason': 'Trailing Stop'
                }
        
        else:  # sell (SHORT)
            # Для SHORT: trailing stop движется вниз
            new_trailing_stop = current_price + trailing_distance
            
            if new_trailing_stop < position['trailing_stop_price']:
                old_trailing = position['trailing_stop_price']
                position['trailing_stop_price'] = new_trailing_stop
                logger.info(
                    f"🔄 Trailing Stop обновлен для {position['symbol']}: "
                    f"${old_trailing:.4f} -> ${new_trailing_stop:.4f}"
                )
                return {
                    'action': 'update_trailing',
                    'new_trailing_stop': new_trailing_stop,
                    'old_trailing_stop': old_trailing
                }
            
            # Проверка срабатывания trailing stop
            if current_price >= position['trailing_stop_price']:
                logger.info(f"🛑 Trailing Stop сработал для {position['symbol']} на ${position['trailing_stop_price']:.4f}")
                return {
                    'action': 'hit_trailing_stop',
                    'close_all': True,
                    'price': position['trailing_stop_price'],
                    'reason': 'Trailing Stop'
                }
        
        return None
    
    def close_position(self, symbol: str, reason: str = "Manual close") -> Optional[Dict[str, Any]]:
        """Закрытие позиции"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        position['status'] = 'closed'
        position['closed_at'] = datetime.now()
        position['close_reason'] = reason
        
        logger.info(
            f"🔚 Позиция закрыта: {symbol}\n"
            f"   Причина: {reason}\n"
            f"   Реализованный PnL: ${position['total_realized_pnl']:.2f}\n"
            f"   TP достигнуто: {position['closed_tp_count']}/{len(position['tp_levels'])}"
        )
        
        closed_position = self.positions.pop(symbol)
        return closed_position
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Получить позицию"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Получить все позиции"""
        return self.positions.copy()
    
    def format_position_info(self, symbol: str) -> str:
        """Форматирование информации о позиции для Telegram"""
        position = self.get_position(symbol)
        if not position:
            return "Позиция не найдена"
        
        tp_status = []
        for i, (tp_price, hit) in enumerate(zip(position['tp_levels'], position['tp_hit'])):
            status = "✅" if hit else "🎯"
            tp_status.append(f"{status} TP{i+1}: ${tp_price:.4f}")
        
        info = (
            f"📊 *Позиция {symbol}*\n\n"
            f"{'🟢 LONG' if position['side'] == 'buy' else '🔴 SHORT'}\n"
            f"💰 Вход: ${position['entry_price']:.4f}\n"
            f"📊 Размер: {position['current_amount']:.6f} ({position['current_amount']/position['initial_amount']*100:.0f}%)\n"
            f"🛡️ SL: ${position['sl_price']:.4f}\n\n"
            f"*Take Profit уровни:*\n"
            + "\n".join(tp_status) + "\n\n"
        )
        
        if position['trailing_stop_active']:
            info += f"🔄 *Trailing Stop:* ${position['trailing_stop_price']:.4f}\n\n"
        
        info += (
            f"💵 Реализованный PnL: ${position['total_realized_pnl']:.2f}\n"
            f"🎯 Достигнуто TP: {position['closed_tp_count']}/{len(position['tp_levels'])}\n"
            f"⭐ Уверенность: {position['signal_confidence']:.0f}%"
        )
        
        return info


# Глобальный экземпляр
position_manager = AdvancedPositionManager()

