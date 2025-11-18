"""
🎯 ADAPTIVE TAKE PROFIT MANAGER
Управление адаптивным TP: от +10% до +100% ROI
"""

import config
from typing import Dict, Optional
import logging


class AdaptiveTPManager:
    """Менеджер адаптивного Take Profit"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.position_tps: Dict[str, float] = {}  # {symbol: current_tp_roi}
    
    def initialize_tp(self, symbol: str) -> float:
        """Инициализация TP для новой позиции"""
        self.position_tps[symbol] = config.MIN_TP_ROI
        self.logger.info(f"📊 {symbol}: Начальный TP = +{config.MIN_TP_ROI}% ROI")
        return config.MIN_TP_ROI
    
    def update_tp(self, symbol: str, current_roi: float) -> Optional[float]:
        """
        Обновление TP на основе текущего ROI
        
        Args:
            symbol: Символ монеты
            current_roi: Текущий ROI в процентах
            
        Returns:
            Новый TP ROI если изменился, иначе None
        """
        if symbol not in self.position_tps:
            return None
        
        current_tp = self.position_tps[symbol]
        
        # Если достигли текущего TP, подтягиваем на шаг вверх
        if current_roi >= current_tp:
            new_tp = min(current_tp + config.TP_TRAIL_STEP_ROI, config.MAX_TP_ROI)
            
            if new_tp > current_tp:
                self.position_tps[symbol] = new_tp
                self.logger.info(
                    f"🎯 {symbol}: TP подтянут {current_tp}% → {new_tp}% ROI "
                    f"(текущий ROI: {current_roi:.1f}%)"
                )
                return new_tp
        
        return None
    
    def get_tp_price(self, symbol: str, entry_price: float, direction: str, leverage: int) -> float:
        """
        Рассчитать цену TP на основе текущего ROI
        
        Args:
            symbol: Символ монеты
            entry_price: Цена входа
            direction: LONG или SHORT
            leverage: Плечо
            
        Returns:
            Цена TP
        """
        if symbol not in self.position_tps:
            self.initialize_tp(symbol)
        
        tp_roi = self.position_tps[symbol]
        price_change_percent = tp_roi / leverage  # ROI / leverage = % изменения цены
        
        if direction == 'LONG':
            tp_price = entry_price * (1 + price_change_percent / 100)
        else:
            tp_price = entry_price * (1 - price_change_percent / 100)
        
        return tp_price
    
    def get_current_tp_roi(self, symbol: str) -> float:
        """Получить текущий TP ROI для позиции"""
        return self.position_tps.get(symbol, config.MIN_TP_ROI)
    
    def remove_position(self, symbol: str):
        """Удалить позицию из трекинга"""
        if symbol in self.position_tps:
            del self.position_tps[symbol]
            self.logger.info(f"📊 {symbol}: TP трекинг удален")
    
    def get_tp_info(self, symbol: str, entry_price: float, current_price: float, 
                    direction: str, leverage: int) -> Dict:
        """
        Получить полную информацию о TP
        
        Returns:
            {
                'current_tp_roi': float,
                'tp_price': float,
                'distance_to_tp_percent': float,
                'next_tp_roi': float,
                'can_trail': bool
            }
        """
        if symbol not in self.position_tps:
            self.initialize_tp(symbol)
        
        current_tp_roi = self.position_tps[symbol]
        tp_price = self.get_tp_price(symbol, entry_price, direction, leverage)
        
        # Расстояние до TP
        if direction == 'LONG':
            distance_percent = ((tp_price - current_price) / current_price) * 100
        else:
            distance_percent = ((current_price - tp_price) / current_price) * 100
        
        # Следующий уровень TP
        next_tp_roi = min(current_tp_roi + config.TP_TRAIL_STEP_ROI, config.MAX_TP_ROI)
        can_trail = current_tp_roi < config.MAX_TP_ROI
        
        return {
            'current_tp_roi': current_tp_roi,
            'tp_price': tp_price,
            'distance_to_tp_percent': distance_percent,
            'next_tp_roi': next_tp_roi if can_trail else None,
            'can_trail': can_trail,
            'max_reached': current_tp_roi >= config.MAX_TP_ROI
        }
