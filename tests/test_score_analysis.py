"""The stability analysis: entry score vs later re-scores, and verdict flips."""
import pytest

from storage.analysis import format_report, score_stability, summarize
from storage.db import make_engine
from storage.repository import Repository
from trading.paper_engine import CrossoverRecord, ScoreSample

THRESHOLD = 0.62


@pytest.fixture
def repo(tmp_path):
    return Repository(make_engine(f"sqlite:///{tmp_path / 'analysis.db'}"))


def add_event(repo, uid, symbol, ts, decision, prob, later_scores):
    """later_scores: {seconds_after_crossover: score}"""
    repo.record_crossover(uid, CrossoverRecord(
        symbol=symbol, signal="BUY", ts=ts, price=100.0,
        decision=decision, probability=prob, features={}))
    for seconds, score in later_scores.items():
        repo.record_score_sample(uid, ScoreSample(
            symbol=symbol, signal="BUY", event_ts=ts, ts=ts + seconds,
            seconds_since_event=float(seconds), score=score, price=100.0))


def test_each_crossover_is_compared_with_its_own_later_scores(repo):
    uid = repo.ensure_user()
    add_event(repo, uid, "AAA", 1000.0, "ACCEPT", 0.95,
              {5: 0.99, 20: 0.40, 40: 0.30, 300: 0.20})
    add_event(repo, uid, "BBB", 2000.0, "AVOID", 0.20, {30: 0.25})

    df = score_stability(repo, uid).set_index("symbol")
    assert df.loc["AAA", "entry_prob"] == 0.95
    assert df.loc["AAA", "n_early"] == 2           # 20s and 40s; 5s is before the window
    assert df.loc["AAA", "early_mean"] == pytest.approx(0.35)
    assert df.loc["AAA", "late_mean"] == pytest.approx(0.20)
    assert df.loc["BBB", "early_mean"] == pytest.approx(0.25)
    assert df.loc["BBB", "n_late"] == 0


def test_samples_of_other_crossovers_in_the_same_symbol_are_not_mixed_in(repo):
    uid = repo.ensure_user()
    add_event(repo, uid, "AAA", 1000.0, "ACCEPT", 0.9, {30: 0.10})
    add_event(repo, uid, "AAA", 5000.0, "ACCEPT", 0.9, {30: 0.80})
    df = score_stability(repo, uid)
    assert sorted(df["early_mean"].round(2)) == [0.10, 0.80]


def test_flip_rate_counts_verdicts_a_later_score_would_have_reversed(repo):
    uid = repo.ensure_user()
    add_event(repo, uid, "A1", 1.0, "ACCEPT", 0.95, {30: 0.30})   # would flip to AVOID
    add_event(repo, uid, "A2", 2.0, "ACCEPT", 0.90, {30: 0.80})   # stays ACCEPT
    add_event(repo, uid, "V1", 3.0, "AVOID", 0.20, {30: 0.70})    # would flip to ACCEPT
    add_event(repo, uid, "V2", 4.0, "AVOID", 0.30, {30: 0.35})    # stays AVOID
    add_event(repo, uid, "V3", 5.0, "AVOID", 0.30, {})            # never re-scored

    summary = summarize(score_stability(repo, uid), THRESHOLD)
    assert summary["ACCEPT"]["flip_rate"] == pytest.approx(0.5)
    assert summary["ACCEPT"]["events"] == 2
    assert summary["AVOID"]["flip_rate"] == pytest.approx(0.5)
    assert summary["AVOID"]["events"] == 3 and summary["AVOID"]["measured"] == 2
    assert summary["ACCEPT"]["mean_entry"] == pytest.approx(0.925)
    assert summary["ACCEPT"]["mean_early"] == pytest.approx(0.55)


def test_events_without_a_model_probability_are_left_out(repo):
    uid = repo.ensure_user()
    add_event(repo, uid, "AAA", 1.0, "AVOID", None, {30: 0.5})
    assert summarize(score_stability(repo, uid), THRESHOLD) == {}


def test_the_report_warns_when_there_are_too_few_crossovers(repo):
    uid = repo.ensure_user()
    add_event(repo, uid, "AAA", 1.0, "ACCEPT", 0.9, {30: 0.3})
    text = format_report(score_stability(repo, uid), THRESHOLD)
    assert "ACCEPT: 1 crossovers" in text
    assert "flipped by the later score: 100%" in text
    assert "too few to draw conclusions" in text


def test_an_empty_database_gives_an_empty_frame_and_a_clear_message(repo):
    uid = repo.ensure_user()
    df = score_stability(repo, uid)
    assert len(df) == 0 and "early_mean" in df.columns
    assert format_report(df, THRESHOLD) == "No scored crossovers saved yet."