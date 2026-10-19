"""Shared test fixtures."""
from pathlib import Path

import pytest

from config import DATA_DIR
from data.tick_store import TickStore


@pytest.fixture(autouse=True)
def tests_must_not_write_real_tick_files(monkeypatch):
    """
    Saved tick files are training data. A test that persisted mock ticks
    into data/store once polluted the real files, so any attempt to write
    there from a test now fails loudly. A test that needs persistence can
    pass its own folder: TickStore(out_dir=tmp_path).
    """
    original = TickStore._writer_for_today

    def guarded(self):
        if Path(self.out_dir).resolve() == Path(DATA_DIR).resolve():
            raise AssertionError(
                "A test tried to write ticks into the real data/store folder. "
                "Use CrossoverEngine(persist_ticks=False) or TickStore(out_dir=tmp_path)."
            )
        return original(self)

    monkeypatch.setattr(TickStore, "_writer_for_today", guarded)