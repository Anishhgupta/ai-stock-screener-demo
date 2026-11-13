import sys
import types

import fyers_login


def test_the_auth_code_is_found_in_a_pasted_redirect_url():
    url = "https://www.google.com/?s=ok&code=200&auth_code=eyJ0eXAiOiJKV1Qi.abc-def_123&state=None"
    assert fyers_login._extract_auth_code(url) == "eyJ0eXAiOiJKV1Qi.abc-def_123"


def test_a_bare_code_is_accepted_and_junk_is_not():
    assert fyers_login._extract_auth_code("A" * 30) == "A" * 30
    assert fyers_login._extract_auth_code("hello") is None
    assert fyers_login._extract_auth_code("https://x.example/?s=ok") is None
    assert fyers_login._extract_auth_code("https://x.example/?auth_code=") is None


def test_save_token_replaces_only_the_token(tmp_path):
    path = tmp_path / ".env"
    path.write_text("FYERS_CLIENT_ID=ABC\nFYERS_ACCESS_TOKEN=old\n# keep me\n", encoding="utf-8")
    assert fyers_login.save_token("eyJ.new-token_1", path) is True
    assert path.read_text(encoding="utf-8") == "FYERS_CLIENT_ID=ABC\nFYERS_ACCESS_TOKEN=eyJ.new-token_1\n# keep me\n"


def test_save_token_says_no_when_it_cannot_write(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert fyers_login.save_token("t", blocker / ".env") is False


class FakeSession:
    def __init__(self, **kwargs):
        pass

    def generate_authcode(self):
        return "https://login.example/auth"

    def set_token(self, code):
        pass

    def generate_token(self):
        return {"access_token": "SECRET-TOKEN-VALUE-123"}


def run_login(monkeypatch, target):
    fake = types.ModuleType("fyers_apiv3")
    fake.fyersModel = types.SimpleNamespace(SessionModel=FakeSession)
    monkeypatch.setitem(sys.modules, "fyers_apiv3", fake)
    for name in ("FYERS_CLIENT_ID", "FYERS_SECRET_ID", "FYERS_REDIRECT_URI"):
        monkeypatch.setenv(name, "x")
    monkeypatch.setattr("builtins.input", lambda prompt="": "https://x.example/?auth_code=" + "a" * 30)
    monkeypatch.setattr(fyers_login, "ENV_PATH", target)
    fyers_login.main()


def test_the_login_saves_the_token_and_does_not_print_it(monkeypatch, tmp_path, capsys):
    target = tmp_path / ".env"
    run_login(monkeypatch, target)
    assert "SECRET-TOKEN-VALUE-123" not in capsys.readouterr().out
    assert target.read_text(encoding="utf-8") == "FYERS_ACCESS_TOKEN=SECRET-TOKEN-VALUE-123\n"


def test_the_login_shows_the_token_only_when_it_cannot_save_it(monkeypatch, tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    run_login(monkeypatch, blocker / ".env")
    assert "SECRET-TOKEN-VALUE-123" in capsys.readouterr().out