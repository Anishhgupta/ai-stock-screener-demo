"""The walk-forward test must find a real signal, find nothing in noise, and never train on a day it predicts."""
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from ml import walk_forward as wf

IST = timezone(timedelta(hours=5, minutes=30))
NAMES = ["smma_spread_pct", "smma_slope_fast", "ltq_change_pct", "ltq_acceleration", "bid_ask_imbalance",
         "imbalance_trend", "spread_pct", "price_change_pct_1m", "price_change_pct_5m", "ltq_avg_recent",
         "volume_last_candle"]
START = datetime(2026, 10, 15, 10, 0, tzinfo=IST).timestamp()


def make_db(path, n_days=14, per_day=15, signal=0.0, current="random", seed=0, day_effect=0.0,
            mark_every=0, bad_every=0):
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(path)
    con.execute("""create table crossover_events (id integer primary key, user_id integer, symbol text,
        signal text, ts float, price float, decision text, probability float, features text)""")
    con.execute("""create table paper_positions (id integer primary key, user_id integer, book text, symbol text,
        signal text, qty integer, entry_price float, entry_ts float, entry_prob float, exit_price float,
        exit_ts float, exit_reason text, pnl float, pnl_pct float)""")
    n = 0
    for d in range(n_days):
        shift = rng.normal() * day_effect
        for i in range(per_day):
            n += 1
            ts = START + d * 86400 + i * 600
            f = rng.normal(size=11)
            profitable = bool(signal * f[0] + shift + rng.normal() > 0.2)
            if current == "random":
                prob = float(rng.uniform())
            elif current == "inverse":
                prob = float(np.clip(0.7 - 0.4 * profitable + rng.normal() * 0.1, 0.01, 0.99))
            else:
                prob = float(np.clip(0.3 + 0.4 * profitable + rng.normal() * 0.1, 0.01, 0.99))
            features = {k: float(v) for k, v in zip(NAMES, f)}
            text = "not json" if bad_every and n % bad_every == 0 else json.dumps(features)
            sym = f"NSE:S{n}-EQ"
            con.execute("insert into crossover_events (user_id, symbol, signal, ts, price, decision, probability, features)"
                        " values (1,?,?,?,?,?,?,?)", (sym, "BUY", ts, 100.0, "ACCEPT" if prob >= 0.62 else "AVOID", prob, text))
            reason = "Session end (marked at last price)" if mark_every and n % mark_every == 0 else "Max hold time reached"
            pnl = 20.0 if profitable else (0.0 if n % 7 == 0 else -20.0)
            con.execute("insert into paper_positions (user_id, book, symbol, signal, qty, entry_price, entry_ts, exit_price,"
                        " exit_ts, exit_reason, pnl, pnl_pct) values (1,'basic',?,?,10,100,?,100,?,?,?,0)",
                        (sym, "BUY", ts, ts + 2700, reason, pnl))
            con.execute("insert into paper_positions (user_id, book, symbol, signal, qty, entry_price, entry_ts, exit_price,"
                        " exit_ts, exit_reason, pnl, pnl_pct) values (1,'filtered',?,?,10,100,?,100,?,'x',999,0)",
                        (sym, "BUY", ts, ts + 60))
    con.commit()
    con.close()
    return path


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------- the AUC itself
def test_auc_known_values():
    assert wf.auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert wf.auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert wf.auc([1, 0], [0.5, 0.5]) == 0.5
    assert wf.auc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(0.75)
    assert np.isnan(wf.auc([1, 1], [0.1, 0.9])) and np.isnan(wf.auc([0, 0], [0.1, 0.9]))


def test_features_must_be_finite_numbers():
    assert wf.parse_features('{"a": 1, "b": 2.5}') == {"a": 1.0, "b": 2.5}
    for bad in ("not json", "[]", "{}", '{"a": null}', '{"a": "x"}', '{"a": NaN}', '{"a": true}', None):
        assert wf.parse_features(bad) is None


# ---------------------------------------------------------------- loading and linking
def test_a_crossover_is_linked_to_its_basic_trade_and_labelled_by_profit(tmp_path):
    db = make_db(tmp_path / "a.db", n_days=2, per_day=7)
    rows, counts = wf.load_rows(db)
    assert counts["crossovers"] == 14 and len(rows) == 14 and counts["no_basic_trade"] == 0
    zero = [r for r in rows if r.symbol == "NSE:S7-EQ"][0]
    assert zero.profitable is False                       # an exact breakeven is not a win
    assert {r.day for r in rows} == {"2026-10-15", "2026-10-16"}


def test_marks_are_left_out_unless_asked_for_and_bad_features_are_skipped(tmp_path):
    db = make_db(tmp_path / "a.db", n_days=2, per_day=10, mark_every=5, bad_every=4)
    rows, counts = wf.load_rows(db)
    assert counts["left_out_marks"] == 4 and counts["bad_features"] >= 3
    with_marks, counts2 = wf.load_rows(db, include_marks=True)
    assert counts2["left_out_marks"] == 0 and len(with_marks) > len(rows)


def test_a_trade_for_a_different_time_is_not_linked(tmp_path):
    db = make_db(tmp_path / "a.db", n_days=1, per_day=3)
    con = sqlite3.connect(db)
    con.execute("update paper_positions set entry_ts = entry_ts + 5 where book = 'basic'")
    con.commit()
    con.close()
    rows, counts = wf.load_rows(db)
    assert rows == [] and counts["no_basic_trade"] == 3


def test_the_database_is_never_changed(tmp_path):
    db = make_db(tmp_path / "a.db")
    before = sha(db)
    wf.load_rows(db)
    wf.main([str(db), "--resamples", "200"])
    assert sha(db) == before


# ---------------------------------------------------------------- no leakage
def _rows(tmp_path, **kwargs):
    rows, _ = wf.load_rows(make_db(tmp_path / "w.db", **kwargs))
    return rows, wf.feature_names(rows)


def test_every_prediction_comes_from_a_model_trained_on_earlier_days_only(tmp_path):
    rows, names = _rows(tmp_path, signal=1.5)
    folds, scores, y, days = wf.walk_forward(rows, names, min_train=40)
    assert folds and all(f.last_train_day < f.test_day for f in folds)
    assert all(f.n_train >= 40 for f in folds)
    candidate = next(iter(wf.candidates()))
    predicted = ~np.isnan(scores[candidate])
    assert set(days[predicted]) == {f.test_day for f in folds}
    assert not predicted[days == min(days)].any()          # the first day can never be predicted


def test_changing_a_days_outcomes_cannot_change_that_days_predictions(tmp_path):
    rows, names = _rows(tmp_path, signal=1.5)
    _, before, _, days = wf.walk_forward(rows, names, min_train=40)
    last = max(days)
    for r in rows:
        if r.day == last:
            r.profitable = not r.profitable                # flip the outcomes of the final day only
    _, after, _, _ = wf.walk_forward(rows, names, min_train=40)
    mask = days == last
    for name in wf.candidates():
        assert np.array_equal(before[name][mask], after[name][mask])
        assert np.array_equal(before[name][~mask], after[name][~mask], equal_nan=True)


def test_too_little_history_gives_no_folds(tmp_path):
    rows, names = _rows(tmp_path, n_days=3, per_day=5)
    folds, *_ = wf.walk_forward(rows, names, min_train=40)
    assert folds == []


# ---------------------------------------------------------------- the answers it should give
def test_pure_noise_is_not_mistaken_for_skill(tmp_path):
    rows, _ = _rows(tmp_path, signal=0.0, seed=3)
    result = wf.analyse(rows, resamples=600)
    for name in [wf.CURRENT, *wf.candidates()]:
        low, high = result["day_ci"][name]
        assert low <= 0.5 <= high, name
    assert any("No retrained model is distinguishable from chance" in line for line in wf.verdict(result))


def test_a_real_signal_is_found(tmp_path):
    rows, _ = _rows(tmp_path, signal=2.5, seed=1)
    result = wf.analyse(rows, resamples=600)
    gb = next(iter(wf.candidates()))
    assert result["auc"][gb] > 0.7 and result["day_ci"][gb][0] > 0.5
    assert any("Cleared the bar" in line for line in wf.verdict(result))
    assert result["day_ci"][wf.CURRENT][0] <= 0.5 <= result["day_ci"][wf.CURRENT][1]     # the random model stays at chance


def test_a_deployed_model_that_is_backwards_is_called_out(tmp_path):
    rows, _ = _rows(tmp_path, signal=2.5, current="inverse", seed=2)
    result = wf.analyse(rows, resamples=600)
    assert result["auc"][wf.CURRENT] < 0.3 and result["day_ci"][wf.CURRENT][1] < 0.5
    assert any("WORSE than chance" in line for line in wf.verdict(result))
    assert result["day_diff"][next(iter(wf.candidates()))][0] > 0      # the retrained model is clearly ahead


def test_a_deployed_model_that_works_is_not_beaten_by_noise_candidates(tmp_path):
    rows, _ = _rows(tmp_path, signal=0.0, current="good", seed=4)
    result = wf.analyse(rows, resamples=600)
    assert result["day_ci"][wf.CURRENT][0] > 0.5
    assert not any("Cleared the bar" in line for line in wf.verdict(result))


# ---------------------------------------------------------------- the verdict, one condition at a time
def _result(candidate_ci, diff_ci=None, current_ci=(0.40, 0.60), with_current=True):
    name = next(iter(wf.candidates()))
    result = {"auc": {name: 0.6}, "day_ci": {name: candidate_ci}, "day_diff": {name: diff_ci}}
    if with_current:
        result["auc"][wf.CURRENT] = 0.5
        result["day_ci"][wf.CURRENT] = current_ci
    return result


def _cleared(result):
    return any("Cleared the bar" in line for line in wf.verdict(result))


def test_a_candidate_must_have_the_whole_interval_above_chance():
    assert not _cleared(_result((0.45, 0.70), diff_ci=(0.01, 0.2)))
    assert _cleared(_result((0.55, 0.80), diff_ci=(0.01, 0.2)))
    assert not _cleared(_result((0.45, 0.70), with_current=False))
    assert _cleared(_result((0.55, 0.80), with_current=False))


def test_a_candidate_must_also_be_ahead_of_the_deployed_model():
    assert not _cleared(_result((0.55, 0.80), diff_ci=(-0.05, 0.2)))      # above chance, but not clearly ahead
    assert not _cleared(_result((0.55, 0.80), diff_ci=None))              # too few resamples to say


def test_the_verdict_describes_the_deployed_model_by_its_interval():
    worse = " ".join(wf.verdict(_result((0.4, 0.6), current_ci=(0.20, 0.45))))
    better = " ".join(wf.verdict(_result((0.4, 0.6), current_ci=(0.55, 0.70))))
    unclear = " ".join(wf.verdict(_result((0.4, 0.6), current_ci=(0.40, 0.60))))
    assert "WORSE than chance" in worse and "better than chance" in better and "cannot be told apart" in unclear


# ---------------------------------------------------------------- honest intervals
def test_resampling_days_gives_wider_intervals_than_resampling_rows_when_days_move_together():
    rng = np.random.default_rng(0)
    days = np.repeat([f"d{i:02d}" for i in range(12)], 15)
    mood = rng.normal(size=12)
    y = np.array([rng.uniform() < 1 / (1 + np.exp(-3 * mood[int(d[1:])])) for d in days])
    score = np.array([mood[int(d[1:])] + rng.normal() * 0.5 for d in days])
    by_day, _ = wf.bootstrap(y, days, {"m": score}, "day", 1500, 1)
    by_row, _ = wf.bootstrap(y, days, {"m": score}, "row", 1500, 1)
    assert (by_day["m"][1] - by_day["m"][0]) > 1.3 * (by_row["m"][1] - by_row["m"][0])


# ---------------------------------------------------------------- the command line
def test_the_report_names_every_model_and_says_nothing_is_changed(tmp_path, capsys):
    db = make_db(tmp_path / "r.db", signal=1.0)
    assert wf.main([str(db), "--resamples", "300"]) == 0
    out = capsys.readouterr().out
    assert wf.CURRENT in out and "gradient boosting (production settings)" in out and "logistic regression" in out
    assert "Nothing here changes the deployed model." in out and "by row (too narrow)" in out


def test_too_little_data_and_a_missing_file_are_explained(tmp_path, capsys):
    db = make_db(tmp_path / "small.db", n_days=2, per_day=5)
    assert wf.main([str(db)]) == 1
    assert "need at least" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        wf.main([str(tmp_path / "missing.db")])



# ---------------------------------------------------------------- only the model's own features
def test_only_the_models_own_features_are_used_in_the_models_own_order(tmp_path):
    rows, _ = wf.load_rows(make_db(tmp_path / "f.db", n_days=3, per_day=5))
    for r in rows:
        r.features.update({"ts": 1.7e9, "ltp": 123.4, "bid_qty": 5000.0})        # extra numbers the model never sees
    assert wf.feature_names(rows) == wf.FEATURE_ORDER


def test_a_feature_missing_from_some_rows_is_dropped_and_reported(tmp_path):
    rows, counts = wf.load_rows(make_db(tmp_path / "f.db", signal=1.0))
    for r in rows:
        del r.features["spread_pct"]
    assert len(wf.feature_names(rows)) == 10
    report = wf.format_report(wf.analyse(rows, resamples=200), counts, len(rows), False)
    line = [l for l in report.splitlines() if "features used" in l][0]
    assert "10 of 11" in line and "imbalance_trend, spread_pct" not in line and "imbalance_trend, price_change_pct_1m" in line


def test_a_leaked_extra_field_cannot_help_a_candidate(tmp_path):
    rows, _ = wf.load_rows(make_db(tmp_path / "f.db", signal=0.0, seed=3))
    for r in rows:
        r.features["future_peek"] = 1.0 if r.profitable else 0.0                 # a perfect, forbidden clue
    result = wf.analyse(rows, resamples=600)
    for name in wf.candidates():
        low, high = result["day_ci"][name]
        assert low <= 0.5 <= high


def test_nothing_crashes_when_none_of_the_models_features_were_saved(tmp_path):
    rows, counts = wf.load_rows(make_db(tmp_path / "f.db", n_days=6, per_day=10))
    for r in rows:
        r.features = {"unrelated": 1.0}
    result = wf.analyse(rows, resamples=100)
    assert result["folds"] == [] and result["features"] == []
    assert "features used: 0 of 11" in wf.format_report(result, counts, len(rows), False)


def test_a_feature_missing_from_only_some_rows_is_dropped_rather_than_crashing(tmp_path):
    rows, _ = wf.load_rows(make_db(tmp_path / "f.db", signal=1.0))
    for r in rows[::2]:
        del r.features["imbalance_trend"]
    names = wf.feature_names(rows)
    assert "imbalance_trend" not in names and len(names) == 10
    assert wf.analyse(rows, resamples=100)["folds"]