import os

import pytest

import envfile


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def read(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.read()


def test_a_key_that_is_there_is_replaced_and_everything_else_is_kept(tmp_path):
    path = tmp_path / ".env"
    write(path, "# my notes\nFYERS_CLIENT_ID=abc\n\nFYERS_ACCESS_TOKEN=old\nOTHER=keep me # note\n")
    envfile.update_env(path, {"FYERS_ACCESS_TOKEN": "new.token-123"})
    assert read(path) == "# my notes\nFYERS_CLIENT_ID=abc\n\nFYERS_ACCESS_TOKEN=new.token-123\nOTHER=keep me # note\n"


def test_missing_keys_are_added_at_the_end_and_a_missing_final_newline_is_fixed(tmp_path):
    path = tmp_path / ".env"
    write(path, "A=1")
    envfile.update_env(path, {"B": "2", "C": "3"})
    assert read(path) == "A=1\nB=2\nC=3\n"


def test_a_commented_out_key_is_not_mistaken_for_the_real_one(tmp_path):
    path = tmp_path / ".env"
    write(path, "# FYERS_ACCESS_TOKEN=example\n")
    envfile.update_env(path, {"FYERS_ACCESS_TOKEN": "real"})
    assert read(path) == "# FYERS_ACCESS_TOKEN=example\nFYERS_ACCESS_TOKEN=real\n"


def test_windows_line_endings_are_kept(tmp_path):
    path = tmp_path / ".env"
    write(path, "A=1\r\nB=2\r\n")
    envfile.update_env(path, {"B": "3", "C": "4"})
    assert read(path) == "A=1\r\nB=3\r\nC=4\r\n"


def test_a_new_file_is_created(tmp_path):
    path = tmp_path / "sub" / ".env"
    envfile.update_env(path, {"BROKER": "mock"})
    assert read(path) == "BROKER=mock\n"


def test_a_key_listed_twice_is_updated_everywhere(tmp_path):
    path = tmp_path / ".env"
    write(path, "K=1\nX=9\nK=2\n")
    envfile.update_env(path, {"K": "7"})
    assert read(path) == "K=7\nX=9\nK=7\n"


@pytest.mark.parametrize("value", [
    "plain", "with space", "has # hash", "cost$5", 'say "hi"', "it's", "a=b=c",
    "NSE:M&M-EQ,NSE:A-EQ", "", 'both \' and "'])
def test_values_survive_a_write_and_a_read(tmp_path, value):
    path = tmp_path / ".env"
    envfile.update_env(path, {"K": value})
    assert envfile.read_env(path) == {"K": value}


def test_a_value_with_a_line_break_is_refused_and_the_file_is_left_alone(tmp_path):
    path = tmp_path / ".env"
    write(path, "A=1\n")
    with pytest.raises(ValueError):
        envfile.update_env(path, {"B": "x\nINJECTED=1"})
    assert read(path) == "A=1\n"


def test_a_bad_setting_name_is_refused(tmp_path):
    with pytest.raises(ValueError):
        envfile.update_env(tmp_path / ".env", {"bad name": "1"})


def test_no_temporary_files_are_left_behind(tmp_path):
    path = tmp_path / ".env"
    envfile.update_env(path, {"A": "1"})
    envfile.update_env(path, {"A": "2"})
    assert sorted(p.name for p in tmp_path.iterdir()) == [".env"]


def test_a_failed_write_leaves_the_original_file_intact(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    write(path, "A=1\n")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(envfile.os, "replace", boom)
    with pytest.raises(OSError):
        envfile.update_env(path, {"A": "2"})
    assert read(path) == "A=1\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == [".env"]


@pytest.mark.skipif(os.name == "nt", reason="file permissions work differently on Windows")
def test_the_file_is_readable_by_its_owner_only(tmp_path):
    path = tmp_path / ".env"
    envfile.update_env(path, {"A": "1"})
    assert (path.stat().st_mode & 0o077) == 0


def test_read_env_understands_quotes_comments_and_export(tmp_path):
    path = tmp_path / ".env"
    write(path, "# c\nA=1\nexport B=\"two words\"\nC=three  # trailing note\nD='four'\n\nnot a setting\n")
    assert envfile.read_env(path) == {"A": "1", "B": "two words", "C": "three", "D": "four"}


def test_read_env_of_a_missing_file_is_empty(tmp_path):
    assert envfile.read_env(tmp_path / "nope") == {}