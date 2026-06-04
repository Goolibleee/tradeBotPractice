from __future__ import annotations

from trading_bot.backtest import run_backtest
from trading_bot.config import StrategyConfig
from trading_bot.data import Candle
from trading_bot.dataset import build_dataset
from trading_bot.model_filter import ModelSignalFilter
from trading_bot.reporting import build_report
from trading_bot.train import train_classifier


def run_walkforward_test(
    train_candles: list[Candle],
    test_candles: list[Candle],
    config: StrategyConfig,
    model_type: str = "random_forest",
) -> None:
    """Run a walk-forward test: train on train_candles, test on test_candles."""
    print("=" * 60)
    print("WALK-FORWARD TEST")
    print("=" * 60)

    # Step 1: Generate training dataset
    print(f"\n1. Building training dataset from {len(train_candles)} candles...")
    train_rows = build_dataset(train_candles, config)
    print(f"   Training rows: {len(train_rows)}")

    # Step 2: Train model
    print(f"\n2. Training {model_type} model...")
    from trading_bot.train import FEATURE_COLUMNS, LABEL_COLUMN
    import numpy as np

    X_train = []
    y_train = []
    for row in train_rows:
        X_train.append([getattr(row, col) for col in FEATURE_COLUMNS])
        y_train.append(getattr(row, LABEL_COLUMN))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    model = train_classifier(X_train, y_train, model_type=model_type)
    print(f"   Training complete. Class distribution: {np.bincount(y_train)}")

    # Step 3: Baseline backtest on test data
    print(f"\n3. Running baseline backtest on {len(test_candles)} test candles...")
    base_result = run_backtest(test_candles, config)
    print(f"   Base strategy return: {((base_result.ending_equity / config.starting_equity) - 1) * 100:.2f}%")
    print(f"   Base strategy trades: {len(base_result.trades)}")

    # Step 4: Model-filtered backtest on test data
    print(f"\n4. Running model-filtered backtest...")
    import pickle
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(model, f)
        model_path = f.name

    try:
        filter_obj = ModelSignalFilter(model_path, config)
        filter_obj.prepare(test_candles)
        filtered_result = run_backtest(test_candles, config, model_filter=filter_obj.should_enter)
        print(f"   Filtered strategy return: {((filtered_result.ending_equity / config.starting_equity) - 1) * 100:.2f}%")
        print(f"   Filtered strategy trades: {len(filtered_result.trades)}")
    finally:
        os.unlink(model_path)

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("WALK-FORWARD RESULTS SUMMARY")
    print("=" * 60)
    print(f"Base strategy:")
    print(f"  Return:     {((base_result.ending_equity / config.starting_equity) - 1) * 100:.2f}%")
    print(f"  Trades:     {len(base_result.trades)}")
    print(f"  Win rate:   {len([t for t in base_result.trades if t.net_pnl > 0]) / len(base_result.trades) * 100:.1f}%" if base_result.trades else "  Win rate:   N/A")
    print(f"  Max DD:     {base_result.max_drawdown_pct * 100:.2f}%")

    print(f"\nModel-filtered strategy:")
    print(f"  Return:     {((filtered_result.ending_equity / config.starting_equity) - 1) * 100:.2f}%")
    print(f"  Trades:     {len(filtered_result.trades)}")
    print(f"  Win rate:   {len([t for t in filtered_result.trades if t.net_pnl > 0]) / len(filtered_result.trades) * 100:.1f}%" if filtered_result.trades else "  Win rate:   N/A")
    print(f"  Max DD:     {filtered_result.max_drawdown_pct * 100:.2f}%")

    base_ret = (base_result.ending_equity / config.starting_equity) - 1
    filtered_ret = (filtered_result.ending_equity / config.starting_equity) - 1
    improvement = (filtered_ret - base_ret) * 100
    print(f"\nImprovement: {improvement:+.2f} percentage points")
