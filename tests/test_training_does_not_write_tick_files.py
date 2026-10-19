"""
Building a training set must never write into the real tick files.

build_training_set_from_ticks() used to create CrossoverEngine() with its
default persistence, so every training run re-appended all the ticks it was
reading into data/store/ticks_<today>.csv. That is how the same sessions
ended up duplicated (and the combined files grew to gigabytes).

The autouse guard in conftest.py turns any such write into a test failure.
"""
from brokers.base import Tick
from ml.labeling import build_training_set_from_ticks


def make_ticks(n=300):
    return [Tick(symbol="NSE:SBIN-EQ", ts=1_800_000_000.0 + i, ltp=100.0 + (i % 7) * 0.1,
                 ltq=10, total_traded_qty=1000 + i, bid_price=99.9, bid_qty=5000,
                 ask_price=100.1, ask_qty=5000) for i in range(n)]


def test_building_a_training_set_never_writes_tick_files():
    build_training_set_from_ticks(make_ticks())      # engine=None: the default path