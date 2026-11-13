import envfile
import setup_wizard as wiz


class Script:
    """A scripted stand-in for the person at the keyboard."""

    def __init__(self, texts=(), secrets=(), yes=()):
        self.texts, self.secrets, self.yes = list(texts), list(secrets), list(yes)
        self.said, self.asked = [], []

    def say(self, text=""):
        self.said.append(str(text))

    def ask(self, question, default=""):
        self.asked.append(question)
        return self.texts.pop(0) or default

    def ask_secret(self, question):
        self.asked.append(question)
        return self.secrets.pop(0)

    def confirm(self, question, default=False):
        self.asked.append(question)
        answer = self.yes.pop(0)
        return default if answer is None else answer

    def everything(self):
        return "\n".join(self.said + self.asked)


class Interrupted(Script):
    def ask(self, question, default=""):
        raise KeyboardInterrupt


def write(path, text):
    path.write_text(text, encoding="utf-8")


def run(path, io, *extra):
    return wiz.main(["--env", str(path), *extra], io=io)


def test_a_mock_setup_needs_no_credentials(tmp_path):
    path = tmp_path / ".env"
    io = Script(texts=["1", ""], yes=[False, False, True])
    assert run(path, io) == 0
    assert envfile.read_env(path) == {"BROKER": "mock", "SCREENER_MIN_DEPTH_QTY": "2000"}
    assert io.secrets == []


def test_a_fyers_setup_saves_the_answers_and_never_shows_the_secret(tmp_path):
    path = tmp_path / ".env"
    io = Script(texts=["2", "ABC12345-100", "https://www.google.com", "5,000"],
                secrets=["TopSecret-value-123"], yes=[False, False, True])
    assert run(path, io) == 0
    saved = envfile.read_env(path)
    assert saved == {"BROKER": "fyers", "FYERS_CLIENT_ID": "ABC12345-100",
                     "FYERS_SECRET_ID": "TopSecret-value-123",
                     "FYERS_REDIRECT_URI": "https://www.google.com", "SCREENER_MIN_DEPTH_QTY": "5000"}
    assert "TopSecret" not in io.everything()
    assert any("python fyers_login.py" in line for line in io.said)


def test_an_angel_one_setup_hides_all_three_secrets(tmp_path):
    path = tmp_path / ".env"
    io = Script(texts=["3", "A1234567", ""], secrets=["key-SECRET-1", "pass-SECRET-2", "totp-SECRET-3"],
                yes=[False, False, True])
    assert run(path, io) == 0
    saved = envfile.read_env(path)
    assert saved["ANGEL_API_KEY"] == "key-SECRET-1" and saved["ANGEL_PASSWORD"] == "pass-SECRET-2"
    assert saved["ANGEL_TOTP_SECRET"] == "totp-SECRET-3" and saved["ANGEL_CLIENT_ID"] == "A1234567"
    for value in ("key-SECRET-1", "pass-SECRET-2", "totp-SECRET-3"):
        assert value not in io.everything()
    assert not any("fyers_login" in line for line in io.said)


def test_pressing_enter_keeps_a_saved_secret_and_other_lines_are_untouched(tmp_path):
    path = tmp_path / ".env"
    write(path, "# my notes\nBROKER=fyers\nFYERS_CLIENT_ID=OLD-100\nFYERS_SECRET_ID=oldsecret\n"
                "FYERS_REDIRECT_URI=https://x.example\nSCREENER_MIN_DEPTH_QTY=2000\nOTHER=keep\n")
    io = Script(texts=["", "NEW-100", "", ""], secrets=[""], yes=[False, False, True])
    assert run(path, io) == 0
    assert path.read_text(encoding="utf-8") == (
        "# my notes\nBROKER=fyers\nFYERS_CLIENT_ID=NEW-100\nFYERS_SECRET_ID=oldsecret\n"
        "FYERS_REDIRECT_URI=https://x.example\nSCREENER_MIN_DEPTH_QTY=2000\nOTHER=keep\n")


def test_when_nothing_changes_nothing_is_written(tmp_path):
    path = tmp_path / ".env"
    text = "BROKER=fyers\nFYERS_CLIENT_ID=A\nFYERS_SECRET_ID=s\nFYERS_REDIRECT_URI=https://x.example\nSCREENER_MIN_DEPTH_QTY=2000\n"
    write(path, text)
    io = Script(texts=["", "", "", ""], secrets=[""], yes=[False, False])
    assert run(path, io) == 0
    assert path.read_text(encoding="utf-8") == text
    assert any("Nothing to change" in line for line in io.said)


def test_a_bad_setting_is_explained_and_asked_again(tmp_path):
    path = tmp_path / ".env"
    io = Script(texts=["1", "", "60000", "abc", "0.8", "1.6", "30", "5"], yes=[True, False, True])
    assert run(path, io) == 0
    saved = envfile.read_env(path)
    assert saved["SCREENER_CAPITAL_PER_TRADE"] == "60000" and saved["SCREENER_STOP_LOSS_PERCENT"] == "0.8"
    assert saved["SCREENER_TARGET_PERCENT"] == "1.6" and saved["SCREENER_MAX_HOLD_MINUTES"] == "30"
    assert saved["SCREENER_MAX_POSITIONS"] == "5"
    assert any("SCREENER_STOP_LOSS_PERCENT" in line and "number" in line for line in io.said)


def test_settings_left_at_their_defaults_are_not_written(tmp_path):
    path = tmp_path / ".env"
    io = Script(texts=["1", "", "", "", "", "", ""], yes=[True, False, True])
    assert run(path, io) == 0
    assert envfile.read_env(path) == {"BROKER": "mock", "SCREENER_MIN_DEPTH_QTY": "2000"}


def test_three_bad_answers_keep_the_current_value(tmp_path):
    path = tmp_path / ".env"
    io = Script(texts=["1", "lots", "many", "few"], yes=[False, False, True])
    assert run(path, io) == 0
    assert envfile.read_env(path)["SCREENER_MIN_DEPTH_QTY"] == "2000"


def test_your_own_watchlist_is_tidied_and_a_bad_one_is_asked_again(tmp_path):
    path = tmp_path / ".env"
    io = Script(texts=["1", "", "bogus", "nse:sbin-eq, NSE:PNB-EQ,nse:sbin-eq"], yes=[False, True, True])
    assert run(path, io) == 0
    assert envfile.read_env(path)["SCREENER_WATCHLIST"] == "NSE:SBIN-EQ,NSE:PNB-EQ"
    assert any("bogus" in line.lower() for line in io.said)


def test_a_saved_custom_watchlist_can_be_cleared(tmp_path):
    path = tmp_path / ".env"
    write(path, "BROKER=mock\nSCREENER_MIN_DEPTH_QTY=2000\nSCREENER_WATCHLIST=NSE:SBIN-EQ\n")
    io = Script(texts=["", ""], yes=[False, False, True, True])
    assert run(path, io) == 0
    assert envfile.read_env(path)["SCREENER_WATCHLIST"] == ""


def test_saying_no_at_the_end_writes_nothing(tmp_path):
    path = tmp_path / ".env"
    io = Script(texts=["1", ""], yes=[False, False, False])
    assert run(path, io) == 1
    assert not path.exists()


def test_cancelling_with_ctrl_c_writes_nothing(tmp_path):
    path = tmp_path / ".env"
    assert run(path, Interrupted()) == 1
    assert not path.exists()


def test_the_broker_can_be_chosen_by_number_or_name_and_junk_is_refused():
    assert wiz.choose_broker(Script(texts=["3"]), "mock") == "angelone"
    assert wiz.choose_broker(Script(texts=["fyers"]), "mock") == "fyers"
    io = Script(texts=["9", "1"])
    assert wiz.choose_broker(io, "fyers") == "mock"
    assert any("1, 2 or 3" in line for line in io.said)


def test_a_warning_appears_when_git_would_not_ignore_the_env_file(tmp_path):
    path = tmp_path / ".env"
    io = Script(texts=["1", ""], yes=[False, False, True])
    run(path, io)
    assert any("WARNING" in line and ".gitignore" in line for line in io.said)

    other = tmp_path / "safe"
    other.mkdir()
    (other / ".gitignore").write_text("*.db\n.env\n", encoding="utf-8")
    io = Script(texts=["1", ""], yes=[False, False, True])
    run(other / ".env", io)
    assert not any("WARNING" in line for line in io.said)


def test_check_reports_what_is_missing_without_showing_values(tmp_path):
    path = tmp_path / ".env"
    write(path, "BROKER=fyers\nFYERS_CLIENT_ID=ABC-100\nFYERS_SECRET_ID=hunter2-secret\nFYERS_REDIRECT_URI=https://x.example\n")
    io = Script()
    assert run(path, io, "--check") == 1
    text = io.everything()
    assert "FYERS_ACCESS_TOKEN: MISSING" in text and "fyers_login.py" in text
    assert "hunter2" not in text and "ABC-100" not in text


def test_check_passes_when_everything_is_set_and_valid(tmp_path):
    path = tmp_path / ".env"
    write(path, "BROKER=fyers\nFYERS_CLIENT_ID=A\nFYERS_SECRET_ID=s\nFYERS_REDIRECT_URI=https://x.example\n"
                "FYERS_ACCESS_TOKEN=tok-SECRET\nSCREENER_MIN_DEPTH_QTY=2000\nSCREENER_STOP_LOSS_PERCENT=0.8\n")
    io = Script()
    assert run(path, io, "--check") == 0
    assert "All good." in io.everything() and "tok-SECRET" not in io.everything()


def test_check_flags_an_invalid_setting_and_a_missing_file(tmp_path):
    path = tmp_path / ".env"
    write(path, "BROKER=mock\nSCREENER_STOP_LOSS_PERCENT=abc\nSCREENER_MIN_DEPTH_QTY=lots\n")
    io = Script()
    assert run(path, io, "--check") == 1
    assert "SCREENER_STOP_LOSS_PERCENT" in io.everything() and "SCREENER_MIN_DEPTH_QTY" in io.everything()
    io = Script()
    assert run(tmp_path / "missing.env", io, "--check") == 1
    assert "No .env file" in io.everything()