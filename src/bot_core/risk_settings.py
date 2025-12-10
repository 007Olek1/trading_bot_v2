"""
Centralized risk-management constants for the constrained trading mode.
"""

FIXED_MARGIN_USD = 1.0  # margin allocated per position
FIXED_LEVERAGE = 20
MAX_OPEN_POSITIONS = 3

FIXED_SL_USD = 0.30
RR_RATIO = 1.5
FIXED_TP_USD = FIXED_SL_USD * RR_RATIO  # 0.45

SL_TOLERANCE = 1e-4
TP_TOLERANCE = 1e-4

TELEGRAM_STARTUP_MESSAGE = (
    "🚀 <b>GoldTrigger_bot запущен! Стартуем на реальной торговле 💎</b>\n"
    "• Леверидж: 20x | Маржа: $1\n"
    "• Фикс. риск: SL $0.30 | TP $0.45 (R/R 1:1.5)\n"
    "• Скальпинг: 5m / 15m свечи\n"
    "• Одновременно позиций: 3"
)

PARTIAL_TAKES = (0.0, 0.0, 0.0)

__all__ = [
    "FIXED_MARGIN_USD",
    "FIXED_LEVERAGE",
    "FIXED_SL_USD",
    "FIXED_TP_USD",
    "RR_RATIO",
    "MAX_OPEN_POSITIONS",
    "SL_TOLERANCE",
    "TP_TOLERANCE",
    "TELEGRAM_STARTUP_MESSAGE",
    "PARTIAL_TAKES",
]
