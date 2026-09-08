"""PerpEngine: liquidation, funding, and rebalancing tests."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from backtest.engines.perp import PerpEngine


def test_basic_long_profits():
    """Simple upward trend → long makes profit."""
    bars = [[900_000 * i, 100.0, 101.0, 99.0, 100.0 + i * 0.5, 10.0] for i in range(30)]

    def signal(sym, window, pos, cap):
        return 1.0

    engine = PerpEngine({"initial_cash": 10_000, "leverage": 1.0})
    result = engine.run({"BTC/USDT": bars}, signal)
    assert result["metrics"]["total_trades"] > 0
    assert result["metrics"]["total_return_pct"] > 0


def test_liquidation_triggers():
    """10x long on a crash → gets liquidated."""
    bars = [
        [900_000 * i, 100.0 - i * 6.0, 101.0 - i * 6.0, 99.0 - i * 6.0, 100.0 - i * 6.0, 10.0]
        for i in range(20)
    ]

    def signal(sym, window, pos, cap):
        if len(window) < 2:
            return 0.0
        return 1.0  # long

    engine = PerpEngine({"initial_cash": 10_000, "leverage": 10.0})
    engine.run({"BTC/USDT": bars}, signal)

    liq_trades = [t for t in engine.trades if t.exit_reason == "liquidation"]
    assert len(liq_trades) >= 1, "Expected at least 1 liquidation"


def test_short_profits_on_downtrend():
    """Short position profits on a downtrend."""
    bars = [[900_000 * i, 100.0, 101.0, 99.0, 100.0 - i * 0.5, 10.0] for i in range(30)]

    def signal(sym, window, pos, cap):
        return -1.0  # always short

    engine = PerpEngine({"initial_cash": 10_000, "leverage": 1.0})
    result = engine.run({"BTC/USDT": bars}, signal)
    assert result["metrics"]["total_trades"] > 0
    assert result["metrics"]["total_return_pct"] > 0, "Short should profit on downtrend"


def test_funding_fee_applies():
    """Explicit funding events debit a long position once."""
    bars = [[900_000 * i, 100.0, 101.0, 99.0, 100.0, 10.0] for i in range(100)]

    def signal(sym, window, pos, cap):
        if len(window) < 2:
            return 0.0
        return 0.8

    cfg = {
        "initial_cash": 10_000,
        "leverage": 3.0,
        "maker_rate": 0.0,
        "taker_rate": 0.0,
        "slippage": 0.0,
    }
    engine = PerpEngine(cfg)
    settlement_timestamp = int(bars[3][0])
    engine.run(
        {"BTC/USDT": bars},
        signal,
        funding_events_by_symbol={"BTC/USDT": [(settlement_timestamp, 0.001)]},
    )

    assert engine.capital == pytest.approx(9_976.0)
    assert engine._funding_applied == {("BTC/USDT", settlement_timestamp)}


def test_current_bar_entry_does_not_pay_earlier_or_shared_timestamp_funding():
    bars = [
        [0, 100.0, 100.0, 100.0, 100.0, 10.0],
        [86_400_000, 200.0, 200.0, 200.0, 200.0, 10.0],
    ]

    engine = PerpEngine(
        {
            "initial_cash": 10_000,
            "leverage": 1.0,
            "maker_rate": 0.0,
            "taker_rate": 0.0,
            "slippage": 0.0,
        }
    )
    engine.run(
        {"BTC/USDT": bars},
        lambda *_args: 1.0,
        funding_events_by_symbol={
            "BTC/USDT": [(43_200_000, 0.01), (86_400_000, 0.01)]
        },
    )

    assert engine.capital == pytest.approx(10_000.0)


def test_current_bar_exit_pays_elapsed_and_shared_timestamp_funding():
    bars = [
        [0, 100.0, 100.0, 100.0, 100.0, 10.0],
        [86_400_000, 100.0, 100.0, 100.0, 100.0, 10.0],
        [172_800_000, 200.0, 200.0, 200.0, 200.0, 10.0],
    ]

    def signal(_symbol, window, _positions, _account):
        return 1.0 if len(window) == 1 else 0.0

    engine = PerpEngine(
        {
            "initial_cash": 10_000,
            "leverage": 1.0,
            "maker_rate": 0.0,
            "taker_rate": 0.0,
            "slippage": 0.0,
        }
    )
    engine.run(
        {"BTC/USDT": bars},
        signal,
        funding_events_by_symbol={
            "BTC/USDT": [(129_600_000, 0.01), (172_800_000, 0.01)]
        },
    )

    # The elapsed settlement uses the prior close (100), while a settlement
    # exactly at the next bar open uses that known open (200), before exit.
    assert engine.capital == pytest.approx(19_700.0)


def test_run_exports_immutable_fee_and_applied_funding_events(tmp_path):
    bars = [
        [0, 100.0, 100.0, 100.0, 100.0, 10.0],
        [86_400_000, 100.0, 100.0, 100.0, 100.0, 10.0],
        [172_800_000, 100.0, 100.0, 100.0, 100.0, 10.0],
    ]

    def signal(_symbol, window, _positions, _account):
        return 0.5 if len(window) == 1 else 0.0

    engine = PerpEngine(
        {
            "initial_cash": 10_000,
            "leverage": 1.0,
            "maker_rate": 0.0,
            "taker_rate": 0.01,
            "slippage": 0.0,
        }
    )
    result = engine.run(
        {"BTC/USDT": bars},
        signal,
        run_id="cost-events",
        runs_dir=tmp_path,
        funding_events_by_symbol={"BTC/USDT": [(129_600_000, 0.001), (172_800_000, 0.001)]},
    )

    assert isinstance(engine.entry_fee_events, tuple)
    assert isinstance(engine.exit_fee_events, tuple)
    assert isinstance(engine.applied_funding_events, tuple)
    with pytest.raises(FrozenInstanceError):
        engine.entry_fee_events[0].amount = 0.0
    with pytest.raises(FrozenInstanceError):
        engine.exit_fee_events[0].application_bar_timestamp_ms = 0
    with pytest.raises(FrozenInstanceError):
        engine.applied_funding_events[0].application_bar_timestamp_ms = 0

    assert result["cost_events"] == {
        "entry_fee": [
            {
                "symbol": "BTC/USDT",
                "timestamp_ms": 86_400_000,
                "application_bar_timestamp_ms": 86_400_000,
                "size": 50.0,
                "price": 100.0,
                "rate": 0.01,
                "liquidity": "taker",
                "amount": -50.0,
            }
        ],
        "exit_fee": [
            {
                "symbol": "BTC/USDT",
                "timestamp_ms": 172_800_000,
                "application_bar_timestamp_ms": 172_800_000,
                "size": 50.0,
                "price": 100.0,
                "rate": 0.01,
                "liquidity": "taker",
                "amount": -50.0,
            }
        ],
        "applied_funding": [
            {
                "symbol": "BTC/USDT",
                "timestamp_ms": 129_600_000,
                "application_bar_timestamp_ms": 172_800_000,
                "direction": 1,
                "size": 50.0,
                "mark_price": 100.0,
                "rate": 0.001,
                "notional": 5_000.0,
                "amount": -5.0,
            },
            {
                "symbol": "BTC/USDT",
                "timestamp_ms": 172_800_000,
                "application_bar_timestamp_ms": 172_800_000,
                "direction": 1,
                "size": 50.0,
                "mark_price": 100.0,
                "rate": 0.001,
                "notional": 5_000.0,
                "amount": -5.0,
            },
        ],
    }
    persisted = json.loads(
        (tmp_path / "backtests/cost-events/summary.json").read_text(encoding="utf-8")
    )
    assert persisted["cost_events"] == result["cost_events"]


def test_forced_close_reconciles_terminal_equity_with_all_cost_events(tmp_path):
    bars = [
        [0, 100.0, 100.0, 100.0, 100.0, 10.0],
        [86_400_000, 100.0, 100.0, 100.0, 100.0, 10.0],
        [172_800_000, 110.0, 110.0, 110.0, 110.0, 10.0],
    ]
    engine = PerpEngine(
        {
            "initial_cash": 10_000,
            "leverage": 1.0,
            "maker_rate": 0.0,
            "taker_rate": 0.01,
            "slippage": 0.0,
        }
    )

    result = engine.run(
        {"BTC/USDT": bars},
        lambda *_args: 1.0,
        run_id="forced-close-reconciliation",
        runs_dir=tmp_path,
        funding_events_by_symbol={"BTC/USDT": [(129_600_000, 0.001)]},
    )

    trade = engine.trades[-1]
    cost_amounts = [
        event["amount"]
        for events in result["cost_events"].values()
        for event in events
    ]
    expected_final_equity = engine.initial_cash + trade.pnl + sum(cost_amounts)

    assert trade.exit_reason == "end_of_backtest"
    assert engine.exit_fee_events[-1].timestamp_ms == bars[-1][0]
    assert engine.exit_fee_events[-1].application_bar_timestamp_ms == bars[-1][0]
    assert trade.commission == pytest.approx(
        -(engine.entry_fee_events[-1].amount + engine.exit_fee_events[-1].amount)
    )
    assert result["final_equity"] == pytest.approx(expected_final_equity)
    assert engine.snapshots[-1].equity == pytest.approx(expected_final_equity)
    assert engine.snapshots[-1].capital == pytest.approx(expected_final_equity)
    assert engine.snapshots[-1].position_count == 0
    assert engine.snapshots[-1].timestamp == bars[-1][0]
    assert len(engine.snapshots) == len(bars)


def test_progress_callback_failure_aborts_with_step_context():
    bars = [[0, 100.0, 100.0, 100.0, 100.0, 10.0]]

    def fail_progress(_index, _total, _snapshot):
        raise OSError("progress store unavailable")

    engine = PerpEngine({"initial_cash": 10_000})
    with pytest.raises(RuntimeError, match="progress_callback.*bar 0.*timestamp 0") as exc_info:
        engine.run({"BTC/USDT": bars}, lambda *_args: 0.0, progress_callback=fail_progress)

    assert isinstance(exc_info.value.__cause__, OSError)


def test_direction_change_flips_position():
    """Changing signal direction closes old → opens new."""
    bars = [[900_000 * i, 100.0, 101.0, 99.0, 100.0, 10.0] for i in range(20)]

    calls = []

    def signal(sym, window, pos, cap):
        calls.append(len(window))
        if len(window) < 3:
            return 0.0
        return 1.0 if len(window) < 10 else -1.0

    engine = PerpEngine({"initial_cash": 10_000, "leverage": 3.0})
    engine.run({"BTC/USDT": bars}, signal)

    trade_count = len(engine.trades)
    assert trade_count >= 2, f"Expected >= 2 trades for flip, got {trade_count}"


def test_no_signal_no_trades():
    """Signal stays 0 → no positions opened."""
    bars = [[900_000 * i, 100.0, 101.0, 99.0, 100.0, 10.0] for i in range(20)]

    def signal(sym, window, pos, cap):
        return 0.0

    engine = PerpEngine({"initial_cash": 10_000})
    engine.run({"BTC/USDT": bars}, signal)

    assert len(engine.trades) == 0
    assert engine.capital == pytest.approx(10_000.0, abs=1e-6)


def test_multi_symbol_backtest():
    """Multi-symbol alignment works."""
    bars_btc = [[900_000 * i, 100.0, 101.0, 99.0, 100.0 + i * 0.3, 10.0] for i in range(20)]
    bars_eth = [[900_000 * i, 10.0, 10.1, 9.9, 10.0 + i * 0.05, 100.0] for i in range(20)]

    def signal(sym, window, pos, cap):
        return 0.5

    engine = PerpEngine({"initial_cash": 10_000, "leverage": 1.0})
    result = engine.run({"BTC/USDT": bars_btc, "ETH/USDT": bars_eth}, signal)

    assert result["metrics"]["total_trades"] > 0


def test_ohlcv_window_grows_per_step():
    """Regression: signal_fn receives only *completed* bars (no look-ahead).

    Bar 0 has no completed history → no signal call. Bar i uses bars 0..i-1.
    """
    bars = [[900_000 * i, 100.0, 101.0, 99.0, 100.0 + i * 0.1, 10.0] for i in range(30)]
    window_lengths: list[int] = []

    def signal(sym, window, pos, cap):
        window_lengths.append(len(window))
        return 0.0

    engine = PerpEngine({"initial_cash": 10_000, "leverage": 1.0})
    engine.run({"TEST/USDT": bars}, signal)

    # Bar 0 skipped; bars 1..29 each invoke signal with 1..29 completed bars.
    assert len(window_lengths) == 29, f"Expected 29 signal calls, got {len(window_lengths)}"
    for i, n in enumerate(window_lengths):
        assert n == i + 1, f"Call {i + 1}: expected {i + 1} completed bars, got {n}"
    closes_step = [bars[i][4] for i in range(29)]
    assert len(set(closes_step)) > 1, (
        "Only one unique last-close across windows — window may not be advancing"
    )


def test_no_lookahead_signal_cannot_trade_same_bar_intrabar_move():
    """Signal must not use the current bar's OHLC before that bar completes."""
    bars = [
        [900_000, 100.0, 101.0, 99.0, 100.0, 10.0],
        [900_000 * 2, 100.0, 210.0, 95.0, 200.0, 10.0],
        [900_000 * 3, 200.0, 205.0, 195.0, 202.0, 10.0],
    ]

    def signal(sym, window, pos, cap):
        if len(window) < 1:
            return 0.0
        o = float(window[-1][1])
        c = float(window[-1][4])
        # Buy when the last *completed* bar rallied >50% close vs open.
        if c > o * 1.5:
            return 1.0
        return 0.0

    engine = PerpEngine({"initial_cash": 10_000, "leverage": 1.0})
    engine.run({"BTC/USDT": bars}, signal)
    # Old model entered at bar 1 open using bar 1's close=200 before it was known.
    assert all(t.entry_bar_index != 1 for t in engine.trades), (
        "Look-ahead: filled on bar 1 open using that bar's intrabar close"
    )
