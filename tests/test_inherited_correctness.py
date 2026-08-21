from __future__ import annotations

import inspect
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ["NEXUS_DISABLE"] = "1"

import pytest

from backtest.engines.perp import PerpEngine, Position
from nexus_data.historical.provider import HistoricalNexusProvider


def _bar(timestamp_ms: int, price: float) -> list[float]:
    return [timestamp_ms, price, price, price, price, 1.0]


def _utc_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _overview(provider: HistoricalNexusProvider, as_of: datetime) -> dict[str, object]:
    bundle = provider.get_bundle(
        as_of_ms=_utc_ms(as_of),
        universe=["BTC/USDT"],
        primary="BTC/USDT",
    )
    return (bundle["endpoints"].get("market_overview") or {}).get("data") or {}


def _write_daily_sources(root: Path, fred_rows: str, fear_greed_rows: str) -> None:
    macro = root / "macro"
    macro.mkdir(parents=True)
    (macro / "fred_daily.csv").write_text(
        "date,vix,source\n" + fred_rows,
        encoding="utf-8",
    )
    (macro / "fear_greed_daily.csv").write_text(
        "date,value,label\n" + fear_greed_rows,
        encoding="utf-8",
    )


def _engine(**extra: object) -> PerpEngine:
    config: dict[str, object] = {
        "initial_cash": 10_000,
        "leverage": 1.0,
        "maker_rate": 0.001,
        "taker_rate": 0.02,
        "slippage": 0.0,
        "funding_rate": 0.0,
    }
    config.update(extra)
    return PerpEngine(config)


def test_d1_staggered_listing_does_not_fabricate_bars(tmp_path: Path) -> None:
    start = 1_700_000_000_000
    hour = 3_600_000
    btc = [_bar(start, 100.0), _bar(start + 2 * hour, 102.0), _bar(start + 3 * hour, 103.0)]
    eth = [_bar(start + 2 * hour, 20.0), _bar(start + 3 * hour, 21.0)]
    calls: dict[str, list[list[list[float]]]] = {"BTC/USDT": [], "ETH/USDT": []}

    def hold(symbol, window, _positions, _account):
        calls[symbol].append([list(row) for row in window])
        return 0.0

    engine = _engine(interval_sec=3600)
    engine.run(
        {"BTC/USDT": btc, "ETH/USDT": eth},
        hold,
        runs_dir=tmp_path,
    )

    eth_history = [row for window in calls["ETH/USDT"] for row in window]
    btc_timestamps = {int(row[0]) for window in calls["BTC/USDT"] for row in window}
    assert not eth_history or min(int(row[0]) for row in eth_history) >= start + 2 * hour, (
        "D1_NO_FABRICATED_PRELISTING_BAR: ETH history existed before its first real bar"
    )
    assert start + 2 * hour in btc_timestamps, (
        "D1_NO_FABRICATED_PRELISTING_BAR: the union clock discarded valid BTC history"
    )


def test_d2_daily_sources_do_not_admit_same_day_rows(tmp_path: Path) -> None:
    day = datetime(2024, 1, 3, tzinfo=timezone.utc)
    previous = (day - timedelta(days=1)).date().isoformat()
    current = day.date().isoformat()
    _write_daily_sources(
        tmp_path,
        f"{previous},10.0,fred_public_csv\n{current},20.0,fred_public_csv\n",
        f"{previous},30,Fear\n{current},70,Greed\n",
    )
    provider = HistoricalNexusProvider(root=tmp_path)

    at_midnight = _overview(provider, day)
    before_fred = _overview(provider, day.replace(hour=21, minute=14, second=59))
    at_fred = _overview(provider, day.replace(hour=21, minute=15))

    missing_root = tmp_path / "missing"
    _write_daily_sources(
        missing_root,
        f"{current},20.0,fred_public_csv\n",
        f"{current},70,Greed\n",
    )
    missing_provider = HistoricalNexusProvider(root=missing_root)
    missing_fred = _overview(missing_provider, day.replace(hour=21, minute=14, second=59))

    assert (
        at_midnight.get("fear_greed_index") == 70
        and before_fred.get("vix") == 10.0
        and at_fred.get("vix") == 20.0
        and missing_fred.get("vix") is None
    ), "D2_SOURCE_PUBLICATION_GATE: daily source rows violated their UTC availability gate"


def test_d3_funding_uses_symbol_events_sign_and_settlement_timestamp(tmp_path: Path) -> None:
    run_parameters = inspect.signature(PerpEngine.run).parameters
    assert "funding_events_by_symbol" in run_parameters, (
        "D3_EXPLICIT_FUNDING_EVENTS: PerpEngine.run has no explicit funding event input"
    )

    start = 1_704_067_200_000
    day = 86_400_000
    event_offsets = (8 * 3_600_000, 16 * 3_600_000, day)
    rates = (0.01, -0.005, 0.015)
    events = [(start + offset, rate) for offset, rate in zip(event_offsets, rates, strict=True)]
    duplicate_events = [event for event in events for _ in range(2)]
    bars = {
        "BTC/USDT": [_bar(start, 100.0), _bar(start + day, 100.0)],
        "ETH/USDT": [_bar(start, 50.0), _bar(start + day, 50.0)],
    }
    engine = _engine()
    engine.positions = {
        "BTC/USDT": Position("BTC/USDT", 1, 100.0, 1.0, 1.0, 0.0, 0),
        "ETH/USDT": Position("ETH/USDT", -1, 50.0, 4.0, 1.0, 0.0, 0),
    }

    engine.run(
        bars,
        lambda symbol, _window, _positions, _account: 1.0 if symbol == "BTC/USDT" else -1.0,
        runs_dir=tmp_path,
        funding_events_by_symbol={
            "BTC/USDT": duplicate_events,
            "ETH/USDT": duplicate_events,
        },
    )

    expected_delta = -(100.0 * sum(rates)) + (200.0 * sum(rates))
    assert engine.capital == pytest.approx(10_000 + expected_delta), (
        "D3_EXPLICIT_FUNDING_EVENTS: signed per-symbol settlements were not applied once"
    )
    assert engine._funding_applied == {
        (symbol, timestamp)
        for symbol in ("BTC/USDT", "ETH/USDT")
        for timestamp, _rate in events
    }, "D3_EXPLICIT_FUNDING_EVENTS: settlements were not keyed by their real timestamps"


def _assert_taker_commission(trade) -> None:
    expected = trade.size * trade.entry_price * 0.02 + trade.size * trade.exit_price * 0.02
    assert trade.commission == pytest.approx(expected), (
        f"D4_TAKER_EXIT_COMMISSION: {trade.exit_reason} exit did not use taker liquidity"
    )


def test_d4_market_exits_use_taker_commission(tmp_path: Path) -> None:
    commission_parameters = inspect.signature(PerpEngine.calc_commission).parameters
    assert "liquidity" in commission_parameters, (
        "D4_TAKER_EXIT_COMMISSION: commission attribution still depends on open versus close"
    )

    start = 1_700_000_000_000
    day = 86_400_000
    flat_bars = [_bar(start + index * day, 100.0) for index in range(3)]

    signal_engine = _engine()
    signal_engine.run(
        {"BTC/USDT": flat_bars},
        lambda _symbol, window, _positions, _account: 1.0 if len(window) == 1 else 0.0,
        runs_dir=tmp_path / "signal",
    )
    signal_trade = next(trade for trade in signal_engine.trades if trade.exit_reason == "signal")

    stop_engine = _engine(stop_loss_pct=5.0)
    stop_bars = [
        _bar(start, 100.0),
        _bar(start + day, 100.0),
        [start + 2 * day, 100.0, 101.0, 90.0, 100.0, 1.0],
    ]
    stop_engine.run(
        {"BTC/USDT": stop_bars},
        lambda _symbol, _window, _positions, _account: 1.0,
        runs_dir=tmp_path / "stop",
    )
    stop_trade = next(trade for trade in stop_engine.trades if trade.exit_reason == "stop_loss")

    end_engine = _engine()
    end_engine.run(
        {"BTC/USDT": flat_bars[:2]},
        lambda _symbol, _window, _positions, _account: 1.0,
        runs_dir=tmp_path / "end",
    )
    end_trade = next(trade for trade in end_engine.trades if trade.exit_reason == "end_of_backtest")

    for trade in (signal_trade, stop_trade, end_trade):
        _assert_taker_commission(trade)
