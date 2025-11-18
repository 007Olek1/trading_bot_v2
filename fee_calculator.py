"""
💰 FEE CALCULATOR - Расчёт комиссий и реальной прибыли
"""


class FeeCalculator:
    """Калькулятор комиссий Bybit"""
    
    # Комиссии Bybit для фьючерсов
    MAKER_FEE = 0.0002  # 0.02% для maker
    TAKER_FEE = 0.0006  # 0.06% для taker
    
    @staticmethod
    def calculate_entry_fee(position_value: float, is_maker: bool = False) -> float:
        """Расчёт комиссии при входе"""
        fee_rate = FeeCalculator.MAKER_FEE if is_maker else FeeCalculator.TAKER_FEE
        return position_value * fee_rate
    
    @staticmethod
    def calculate_exit_fee(position_value: float, is_maker: bool = False) -> float:
        """Расчёт комиссии при выходе"""
        fee_rate = FeeCalculator.MAKER_FEE if is_maker else FeeCalculator.TAKER_FEE
        return position_value * fee_rate
    
    @staticmethod
    def calculate_total_fees(position_value: float, is_maker_entry: bool = False, is_maker_exit: bool = False) -> float:
        """Расчёт общей комиссии (вход + выход)"""
        entry_fee = FeeCalculator.calculate_entry_fee(position_value, is_maker_entry)
        exit_fee = FeeCalculator.calculate_exit_fee(position_value, is_maker_exit)
        return entry_fee + exit_fee
    
    @staticmethod
    def calculate_net_pnl(gross_pnl: float, position_value: float, is_maker_entry: bool = False, is_maker_exit: bool = False) -> float:
        """
        Расчёт чистой прибыли с учётом комиссий
        
        Args:
            gross_pnl: Валовая прибыль
            position_value: Размер позиции в USD
            is_maker_entry: Вход по maker ордеру
            is_maker_exit: Выход по maker ордеру
        
        Returns:
            Чистая прибыль после комиссий
        """
        total_fees = FeeCalculator.calculate_total_fees(position_value, is_maker_entry, is_maker_exit)
        return gross_pnl - total_fees
    
    @staticmethod
    def adjust_tp_for_fees(tp_percent: float, position_value: float, leverage: int = 10) -> float:
        """
        Корректировка TP с учётом комиссий
        
        Увеличивает целевой процент TP чтобы покрыть комиссии
        """
        # Комиссии в процентах от позиции
        total_fee_percent = (FeeCalculator.TAKER_FEE * 2) * 100  # Вход + выход
        
        # С учётом плеча комиссии влияют сильнее
        adjusted_fee_percent = total_fee_percent * leverage
        
        # Добавляем комиссии к целевому TP
        return tp_percent + adjusted_fee_percent
    
    @staticmethod
    def get_min_profitable_percent(leverage: int = 10) -> float:
        """Минимальный процент прибыли для безубытка"""
        # Комиссии вход + выход
        total_fee_percent = (FeeCalculator.TAKER_FEE * 2) * 100
        # С учётом плеча
        return total_fee_percent * leverage
