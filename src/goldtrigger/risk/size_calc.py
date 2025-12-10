from dataclasses import dataclass


@dataclass
class PositionSizeInputs:
    balance: float
    risk_pct: float
    sl_distance_pct: float
    entry_price: float
    fee_pct: float = 0.00075


def calc_position_size(inputs: PositionSizeInputs) -> float:
    """
    Kelly-style fixed-fraction sizing.
    Returns contract size (USDT notionals) respecting risk % and SL distance.
    """
    if inputs.balance <= 0 or inputs.entry_price <= 0:
        return 0.0
    risk_amount = inputs.balance * inputs.risk_pct
    sl_buffer_pct = inputs.sl_distance_pct + inputs.fee_pct * 2
    if sl_buffer_pct <= 0:
        return 0.0
    notional = risk_amount / sl_buffer_pct
    contracts = notional / inputs.entry_price
    return max(contracts, 0.0)
