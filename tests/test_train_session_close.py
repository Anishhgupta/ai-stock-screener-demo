"""Training stops the label replay at the close by default on the command line, and records what it did."""
import json

import joblib

from ml import labeling, train as train_module


def examples(n=60):
    return [labeling.LabeledExample(symbol=f"S{i}", signal="BUY", ts=float(i), features=[float(i % 7)] * 11,
                                    outcome_pct=0.0, label=i % 2) for i in range(n)]


def test_with_a_close_the_labels_are_built_with_it_and_the_effect_is_recorded(monkeypatch, tmp_path, capsys):
    seen = []

    def fake(path, session_close=None, stats=None):
        seen.append(session_close)
        stats.update({"crossovers": 60, "skipped_after_close": 4, "window_crossed_close": 9, "label_changed": 3})
        return examples()

    monkeypatch.setattr(train_module, "build_training_set", fake)
    out = tmp_path / "candidate" / "model.joblib"
    train_module.train(["a.csv", "b.csv"], out, 30, session_close=(15, 15))
    assert seen == [(15, 15), (15, 15)]
    metrics = json.loads((out.parent / "train_metrics.json").read_text())
    assert metrics["session_close"] == "15:15"
    assert metrics["close_effect"]["skipped_after_close"] == 8 and metrics["close_effect"]["label_changed"] == 6
    assert set(joblib.load(out)) == {"model", "feature_order"}
    assert "label(s) changed" in capsys.readouterr().out


def test_without_a_close_the_labeller_is_called_exactly_as_before(monkeypatch, tmp_path):
    def fake(path):                                       # the old call shape: a path and nothing else
        return examples()

    monkeypatch.setattr(train_module, "build_training_set", fake)
    out = tmp_path / "m.joblib"
    train_module.train(["a.csv"], out, 30)
    metrics = json.loads((out.parent / "train_metrics.json").read_text())
    assert metrics["session_close"] is None and metrics["close_effect"] is None


def test_the_command_line_stops_at_the_close_unless_told_not_to(monkeypatch):
    calls = []
    monkeypatch.setattr(train_module, "train", lambda csvs, out, minimum, session_close=None:
                        calls.append((len(csvs), minimum, session_close)))
    train_module.main(["--tick-csv", "a.csv"])
    train_module.main(["--tick-csv", "a.csv", "b.csv", "--min-examples", "10", "--no-session-cutoff"])
    assert calls == [(1, 30, (15, 15)), (2, 10, None)]