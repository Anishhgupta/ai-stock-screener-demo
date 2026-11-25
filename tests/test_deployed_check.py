"""The deployed model is judged on every day, with intervals that respect days moving together."""
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from ml import deployed_check as dc
from ml.walk_forward import CURRENT, Row

IST = timezone(timedelta(hours=5, minutes=30))
START = datetime(2026, 10, 12, 10, 0, tzinfo=IST)          # a Monday


def day_label(d):
    return (START + timedelta(days=d)).strftime("%Y-%m-%d")


def make_rows(n_days=14, per_day=12, model="random", seed=0, day_rates=None):
    """Rows with a known relation between the deployed model's probability and the outcome."""
    rng = np.random.default_rng(seed)
    moods = rng.normal(0, 0.15, n_days)                     # used by "day_mood": unrelated to the outcomes
    rows = []
    for d in range(n_days):
        rate = 0.45 if day_rates is None else day_rates[d]
        for i in range(per_day):
            profitable = bool(rng.uniform() < rate)
            if model == "random":
                prob = float(rng.uniform())
            elif model == "backwards":
                prob = float(np.clip(0.7 - 0.4 * profitable + rng.normal() * 0.1, 0.01, 0.99))
            elif model == "works":
                prob = float(np.clip(0.3 + 0.4 * profitable + rng.normal() * 0.1, 0.01, 0.99))
            else:                                           # "day_mood": a day-level mood with no link to the outcomes
                prob = float(np.clip(0.5 + moods[d] + rng.normal() * 0.05, 0.01, 0.99))
            rows.append(Row(day=day_label(d), ts=0.0, symbol=f"S{d}_{i}", signal="BUY",
                            probability=prob, features={"f": 0.0}, profitable=profitable))
    return rows


def test_a_backwards_model_is_caught_by_the_day_level_interval():
    result = dc.evaluate(make_rows(model="backwards"), resamples=800)
    assert result["auc"] < 0.2 and result["day_ci"][1] < 0.5
    assert "WORSE than chance" in dc.verdict(result)


def test_a_random_model_cannot_be_told_apart_from_chance():
    result = dc.evaluate(make_rows(model="random", seed=5), resamples=800)
    assert result["day_ci"][0] <= 0.5 <= result["day_ci"][1]
    assert "cannot be told apart from chance" in dc.verdict(result)


def test_a_model_that_works_is_recognised():
    result = dc.evaluate(make_rows(model="works"), resamples=800)
    assert result["day_ci"][0] > 0.5 and "better than chance" in dc.verdict(result)


def test_days_that_move_together_make_the_row_interval_overconfident():
    # a model with NO real link to the outcomes, whose day-to-day mood happens to line up with them by chance
    rates = [0.20, 0.70, 0.35, 0.80, 0.25, 0.65, 0.40, 0.55]
    result = dc.evaluate(make_rows(n_days=8, per_day=20, model="day_mood", seed=18, day_rates=rates), resamples=1500)
    row_low, row_high = result["row_ci"]
    day_low, day_high = result["day_ci"]
    assert row_high < 0.5                          # counting trades as independent, it looks decisively backwards
    assert day_low <= 0.5 <= day_high              # counting days, it cannot be told apart from chance
    assert (day_high - day_low) > 1.8 * (row_high - row_low)
    assert "ignores that trades on the same day move together" in dc.verdict(result)


def test_the_verdict_warns_when_only_the_row_interval_excludes_chance():
    base = {"auc": 0.38, "day_ci": (0.26, 0.56), "row_ci": (0.26, 0.49)}
    assert "ignores that trades on the same day move together" in dc.verdict(base)
    assert "ignores" not in dc.verdict({**base, "row_ci": (0.30, 0.60)})
    assert "WORSE" in dc.verdict({**base, "day_ci": (0.2, 0.45)})
    assert "better than chance" in dc.verdict({**base, "auc": 0.6, "day_ci": (0.52, 0.7)})
    assert "Too few days" in dc.verdict({**base, "day_ci": None})


def test_the_weekly_table_groups_monday_to_sunday_and_hides_small_weeks():
    rows = make_rows(n_days=9, per_day=4)                  # Mon 12 Oct .. Tue 20 Oct: 7 days + 2 days
    weeks = dc.by_week(rows)
    assert [w["week_of"] for w in weeks] == ["2026-10-12", "2026-10-19"]
    assert [w["n"] for w in weeks] == [28, 8]
    assert not np.isnan(weeks[0]["auc"]) and np.isnan(weeks[1]["auc"])        # 8 < the 15-crossover minimum
    assert weeks[0]["profitable"] == sum(r.profitable for r in rows if r.day < "2026-10-19")


def test_rows_without_a_saved_probability_are_skipped_and_none_gives_none():
    rows = make_rows(n_days=4, per_day=5)
    rows[0].probability = None
    assert dc.evaluate(rows, resamples=200)["n"] == 19
    for r in rows:
        r.probability = None
    assert dc.evaluate(rows, resamples=200) is None


def _db(path, n=60):
    con = sqlite3.connect(path)
    con.execute("create table crossover_events (id integer primary key, user_id integer, symbol text, signal text,"
                " ts float, price float, decision text, probability float, features text)")
    con.execute("create table paper_positions (id integer primary key, user_id integer, book text, symbol text,"
                " signal text, qty integer, entry_price float, entry_ts float, entry_prob float, exit_price float,"
                " exit_ts float, exit_reason text, pnl float, pnl_pct float)")
    rng = np.random.default_rng(0)
    for i in range(n):
        ts = START.timestamp() + (i // 6) * 86400 + (i % 6) * 600
        con.execute("insert into crossover_events (user_id, symbol, signal, ts, price, decision, probability, features)"
                    " values (1,?,?,?,100,'AVOID',?,?)", (f"S{i}", "BUY", ts, float(rng.uniform()), json.dumps({"f": 1.0})))
        con.execute("insert into paper_positions (user_id, book, symbol, signal, qty, entry_price, entry_ts, exit_price,"
                    " exit_ts, exit_reason, pnl, pnl_pct) values (1,'basic',?,?,10,100,?,100,?,'Max hold time reached',?,0)",
                    (f"S{i}", "BUY", ts, ts + 60, 10.0 if i % 2 else -10.0))
    con.commit()
    con.close()
    return path


def test_the_command_line_prints_the_report_and_never_changes_the_database(tmp_path, capsys):
    db = _db(tmp_path / "x.db")
    before = hashlib.sha256(open(db, "rb").read()).hexdigest()
    assert dc.main([str(db), "--resamples", "300"]) == 0
    out = capsys.readouterr().out
    assert "Deployed ACCEPT/AVOID model" in out and "the honest one" in out and "By week" in out
    assert "Nothing here changes the deployed model." in out
    assert hashlib.sha256(open(db, "rb").read()).hexdigest() == before


def test_a_missing_file_and_an_empty_database_are_explained(tmp_path, capsys):
    with pytest.raises(SystemExit):
        dc.main([str(tmp_path / "missing.db")])
    empty = _db(tmp_path / "e.db", n=0)
    assert dc.main([str(empty)]) == 0
    assert "nothing to score" in capsys.readouterr().out