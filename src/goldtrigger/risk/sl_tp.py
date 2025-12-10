from dataclasses import dataclass
from typing import Dict


@dataclass
class SltpInputs:
    entry_price: float
    sl_distance_pct: float
    rr_multiplier: float
    direction: str = "long"
    tp_cap_pct: float = 0.12
    tp_floor_pct: float = 0.04


def calc_sl_tp_targets(inputs: SltpInputs) -> Dict[str, float]:
    """
    Skeleton SL/TP calculation supporting TP1/TP2 placeholders.
    """
    tp_base = inputs.sl_distance_pct * inputs.rr_multiplier
    tp_pct = max(min(tp_base, inputs.tp_cap_pct), inputs.tp_floor_pct)
    direction = inputs.direction.lower()
    if direction == "short":
        sl_price = inputs.entry_price * (1 + inputs.sl_distance_pct)
        tp_price = inputs.entry_price * (1 - tp_pct)
        tp_mid = inputs.entry_price * (1 - tp_pct / 2)
    else:
        sl_price = inputs.entry_price * (1 - inputs.sl_distance_pct)
        tp_price = inputs.entry_price * (1 + tp_pct)
        tp_mid = inputs.entry_price * (1 + tp_pct / 2)
    return {
        "sl": sl_price,
        "tp_final": tp_price,
        "tp1": tp_mid,
        "tp2": tp_price,
    }
