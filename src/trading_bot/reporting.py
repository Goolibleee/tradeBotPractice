from __future__ import annotations

from math import sqrt

from trading_bot.backtest import BacktestResult
from trading_bot.config import StrategyConfig


def build_report(result: BacktestResult, config: StrategyConfig) -> str:
    trades = result.trades
    trade_count = len(trades)
    wins = [trade for trade in trades if trade.net_pnl > 0]
    losses = [trade for trade in trades if trade.net_pnl < 0]
    total_net_pnl = sum(trade.net_pnl for trade in trades)
    total_return_pct = ((result.ending_equity / config.starting_equity) - 1.0) * 100.0
    win_rate = (len(wins) / trade_count * 100.0) if trade_count else 0.0
    gross_profit = sum(trade.net_pnl for trade in wins)
    gross_loss = abs(sum(trade.net_pnl for trade in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss else 0.0
    average_trade = (total_net_pnl / trade_count) if trade_count else 0.0
    trade_returns = [trade.net_pnl / config.starting_equity for trade in trades]
    sharpe_like = _calculate_sharpe_like(trade_returns)

    lines = [
        f"Symbol: {config.symbol}",
        f"Timeframe: {config.timeframe}",
        f"Starting equity: {config.starting_equity:,.2f}",
        f"Ending equity: {result.ending_equity:,.2f}",
        f"Net PnL: {total_net_pnl:,.2f}",
        f"Total return: {total_return_pct:.2f}%",
        f"Max drawdown: {result.max_drawdown_pct * 100.0:.2f}%",
        f"Trades: {trade_count}",
        f"Win rate: {win_rate:.2f}%",
        f"Profit factor: {profit_factor:.2f}",
        f"Average trade: {average_trade:,.2f}",
        f"Sharpe-like: {sharpe_like:.2f}",
    ]

    if trades:
        lines.append("")
        lines.append("Recent trades:")
        for trade in trades[-5:]:
            lines.append(
                " | ".join(
                    [
                        trade.entry_timestamp,
                        trade.exit_timestamp,
                        trade.exit_reason,
                        f"net={trade.net_pnl:,.2f}",
                    ]
                )
            )

    return "\n".join(lines)


def _calculate_sharpe_like(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1)
    if variance <= 0:
        return 0.0
    return (mean_return / sqrt(variance)) * sqrt(len(returns))
