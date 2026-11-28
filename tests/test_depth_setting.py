"""A command-line replay must use the depth setting from .env, and must say so, or warn that it has none."""
import json
import os

import pytest

from ml import depth_setting as ds, labeling, train as train_module


def clear_depth(monkeypatch):
    """Remove the variable for this test and restore the original state afterwards."""
    monkeypatch.setenv(ds.DEPTH_ENV, "placeholder")
    monkeypatch.delenv(ds.DEPTH_ENV)


def env_file(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_environment_wins_over_the_env_file(tmp_path):
    environ = {ds.DEPTH_ENV: "3000"}
    path = env_file(tmp_path, f"{ds.DEPTH_ENV}=2000\n")
    assert ds.depth_setting(path, environ) == ("3000", "the environment")
    assert environ[ds.DEPTH_ENV] == "3000"


def test_the_env_file_is_used_when_the_environment_has_nothing_and_is_put_in_the_environment(tmp_path):
    environ = {}
    path = env_file(tmp_path, f"BROKER=fyers\n{ds.DEPTH_ENV}=2000\n")
    assert ds.depth_setting(path, environ) == ("2000", str(path))
    assert environ[ds.DEPTH_ENV] == "2000"                  # so the depth screen, which reads the environment, sees it


def test_an_empty_value_or_a_missing_file_means_not_set(tmp_path):
    assert ds.depth_setting(env_file(tmp_path, f"{ds.DEPTH_ENV}=\n"), {}) == (None, "not set")
    assert ds.depth_setting(tmp_path / "nope.env", {}) == (None, "not set")


def test_the_announcement_names_the_value_or_warns_loudly(tmp_path):
    said = []
    assert ds.announce_depth(env_file(tmp_path, f"{ds.DEPTH_ENV}=2000\n"), {}, said.append) == "2000"
    assert said == [f"Minimum depth: 2000 shares (from {tmp_path / '.env'})."]
    said.clear()
    assert ds.announce_depth(tmp_path / "nope.env", {}, said.append) is None
    assert "WARNING" in said[0] and "10 lakh" in said[0] and "setup_wizard.py" in said[0]


def test_the_labelling_command_announces_the_depth_before_it_starts(monkeypatch, tmp_path, capsys):
    clear_depth(monkeypatch)
    monkeypatch.setattr(ds, "ENV_FILE", env_file(tmp_path, f"{ds.DEPTH_ENV}=1500\n"))
    monkeypatch.setattr(labeling, "build_training_set", lambda path, session_close=None, stats=None: [])
    labeling.main([str(tmp_path / "ticks.csv")])
    out = capsys.readouterr().out
    assert out.index("Minimum depth: 1500") < out.index("Labels with the replay stopped")
    assert os.environ[ds.DEPTH_ENV] == "1500"


def test_the_labelling_command_warns_when_nothing_is_set(monkeypatch, tmp_path, capsys):
    clear_depth(monkeypatch)
    monkeypatch.setattr(ds, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setattr(labeling, "build_training_set", lambda path, session_close=None, stats=None: [])
    labeling.main([str(tmp_path / "ticks.csv")])
    assert "WARNING: SCREENER_MIN_DEPTH_QTY is not set" in capsys.readouterr().out


def test_the_training_command_announces_the_depth_too(monkeypatch, tmp_path, capsys):
    clear_depth(monkeypatch)
    monkeypatch.setattr(ds, "ENV_FILE", env_file(tmp_path, f"{ds.DEPTH_ENV}=2500\n"))
    monkeypatch.setattr(train_module, "train", lambda *args, **kwargs: None)
    train_module.main(["--tick-csv", "a.csv"])
    assert "Minimum depth: 2500" in capsys.readouterr().out


def test_the_depth_in_force_is_recorded_with_the_model(monkeypatch, tmp_path):
    def fake(path, session_close=None, stats=None):
        return [labeling.LabeledExample(symbol=f"S{i}", signal="BUY", ts=float(i), features=[float(i % 7)] * 11,
                                        outcome_pct=0.0, label=i % 2) for i in range(60)]

    monkeypatch.setattr(train_module, "build_training_set", fake)
    monkeypatch.setenv(ds.DEPTH_ENV, "2000")
    out = tmp_path / "m" / "model.joblib"
    train_module.train(["a.csv"], out, 30)
    assert json.loads((out.parent / "train_metrics.json").read_text())["min_depth_qty"] == "2000"
    clear_depth(monkeypatch)
    out2 = tmp_path / "m2" / "model.joblib"
    train_module.train(["a.csv"], out2, 30)
    assert json.loads((out2.parent / "train_metrics.json").read_text())["min_depth_qty"] is None