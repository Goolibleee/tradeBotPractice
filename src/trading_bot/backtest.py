from __future__ import annotations

from dataclasses import dataclass

from trading_bot.config import StrategyConfig
from trading_bot.data import Candle
from trading_bot.risk import calculate_position_size
from trading_bot.strategy import SignalSnapshot, build_signal_snapshots


@dataclass(frozen=True)
class Trade:
    entry_timestamp: str
    exit_timestamp: str
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    net_pnl: float
    exit_reason: str


@dataclass
class OpenPosition:
    entry_timestamp: str
    entry_price: float
    stop_price: float
    target_price: float
    quantity: float
    entry_fee: float


@dataclass(frozen=True)
class BacktestResult:
    trades: list[Trade]
    equity_curve: list[float]
    ending_equity: float
    max_drawdown_pct: float


def run_backtest(candles: list[Candle], config: StrategyConfig) -> BacktestResult:
    snapshots = build_signal_snapshots(candles, config)
    equity = config.starting_equity
    equity_curve: list[float] = [equity]
    trades: list[Trade] = []
    open_position: OpenPosition | None = None
    cooldown_remaining = 0
    active_day: str | None = None
    daily_start_equity = equity
    daily_realized_pnl = 0.0

    for snapshot in snapshots:
        candle = snapshot.candle
        candle_day = candle.timestamp.split("T", 1)[0]
        if candle_day != active_day:
            active_day = candle_day
            daily_start_equity = equity
            daily_realized_pnl = 0.0

        if open_position is not None:
            trade = _maybe_exit_position(open_position, snapshot, config)
            if trade is not None:
                open_position = None
                trades.append(trade)
                equity += trade.net_pnl
                daily_realized_pnl += trade.net_pnl
                cooldown_remaining = config.cooldown_bars

        allowed_daily_loss = daily_start_equity * config.max_daily_loss_pct
        max_loss_reached = daily_realized_pnl <= -allowed_daily_loss

        if open_position is None and cooldown_remaining > 0:
            cooldown_remaining -= 1

        if (
            open_position is None
            and cooldown_remaining == 0
            and not max_loss_reached
            and snapshot.is_long_entry
        ):
            open_position = _open_long_position(snapshot, equity, config)

        marked_equity = equity
        if open_position is not None:
            unrealized = (candle.close - open_position.entry_price) * open_position.quantity
            marked_equity += unrealized - open_position.entry_fee
        equity_curve.append(marked_equity)

    if open_position is not None:
        final_candle = candles[-1]
        exit_price = _apply_sell_slippage(final_candle.close, config.slippage_rate)
        gross_pnl = (exit_price - open_position.entry_price) * open_position.quantity
        exit_fee = exit_price * open_position.quantity * config.fee_rate
        net_pnl = gross_pnl - open_position.entry_fee - exit_fee
        trade = Trade(
            entry_timestamp=open_position.entry_timestamp,
            exit_timestamp=final_candle.timestamp,
            entry_price=open_position.entry_price,
            exit_price=exit_price,
            quantity=open_position.quantity,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            exit_reason="end_of_data",
        )
        trades.append(trade)
        equity += trade.net_pnl
        equity_curve[-1] = equity

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        ending_equity=equity,
        max_drawdown_pct=_calculate_max_drawdown_pct(equity_curve),
    )


def _open_long_position(snapshot: SignalSnapshot, equity: float, config: StrategyConfig) -> OpenPosition | None:
    assert snapshot.atr is not None
    entry_price = _apply_buy_slippage(snapshot.candle.close, config.slippage_rate)
    stop_price = entry_price - (snapshot.atr * config.atr_stop_multiple)
    target_price = entry_price + ((entry_price - stop_price) * config.risk_reward_ratio)
    quantity = calculate_position_size(
        equity=equity,
        risk_per_trade_pct=config.risk_per_trade_pct,
        entry_price=entry_price,
        stop_price=stop_price,
    )
    if quantity <= 0:
        return None
    entry_fee = entry_price * quantity * config.fee_rate
    return OpenPosition(
        entry_timestamp=snapshot.candle.timestamp,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        quantity=quantity,
        entry_fee=entry_fee,
    )


def _maybe_exit_position(
    position: OpenPosition,
    snapshot: SignalSnapshot,
    config: StrategyConfig,
) -> Trade | None:
    candle = snapshot.candle
    exit_reason: str | None = None
    raw_exit_price: float | None = None

    # When both stop and target hit inside one candle, assume the stop hit first.
    if candle.low <= position.stop_price:
        raw_exit_price = position.stop_price
        exit_reason = "stop_loss"
    elif candle.high >= position.target_price:
        raw_exit_price = position.target_price
        exit_reason = "take_profit"
    elif snapshot.is_trend_break:
        raw_exit_price = candle.close
        exit_reason = "trend_break"

    if raw_exit_price is None or exit_reason is None:
        return None

    exit_price = _apply_sell_slippage(raw_exit_price, config.slippage_rate)
    gross_pnl = (exit_price - position.entry_price) * position.quantity
    exit_fee = exit_price * position.quantity * config.fee_rate
    net_pnl = gross_pnl - position.entry_fee - exit_fee
    return Trade(
        entry_timestamp=position.entry_timestamp,
        exit_timestamp=candle.timestamp,
        entry_price=position.entry_price,
        exit_price=exit_price,
        quantity=position.quantity,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        exit_reason=exit_reason,
    )


def _calculate_max_drawdown_pct(equity_curve: list[float]) -> float:
    peak = equity_curve[0]
    max_drawdown = 0.0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak if peak else 0.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def _apply_buy_slippage(price: float, slippage_rate: float) -> float:
    return price * (1.0 + slippage_rate)


def _apply_sell_slippage(price: float, slippage_rate: float) -> float:
    return price * (1.0 - slippage_rate)
