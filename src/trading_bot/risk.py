from __future__ import annotations


def calculate_position_size(
    equity: float,
    risk_per_trade_pct: float,
    entry_price: float,
    stop_price: float,
) -> float:
    risk_amount = equity * risk_per_trade_pct
    price_risk = entry_price - stop_price
    if price_risk <= 0:
        return 0.0
    return risk_amount / price_risk
