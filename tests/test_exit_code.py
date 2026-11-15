"""The process leaves cleanly and promptly when a session ends, and reports whether it crashed."""
import ast
import inspect

import main


class FakeApp:
    def __init__(self, outcome=None):
        self.outcome, self.calls = outcome, []

    def run(self, duration):
        self.calls.append(duration)
        if isinstance(self.outcome, BaseException):
            raise self.outcome


def test_a_normal_end_is_exit_code_zero():
    app = FakeApp()
    assert main.run_to_exit_code(app, 5) == 0 and app.calls == [5]


def test_a_graceful_stop_through_sys_exit_keeps_its_code():
    assert main.run_to_exit_code(FakeApp(SystemExit(0)), None) == 0
    assert main.run_to_exit_code(FakeApp(SystemExit(None)), None) == 0
    assert main.run_to_exit_code(FakeApp(SystemExit(2)), None) == 2


def test_a_text_message_exit_is_a_failure_and_the_message_is_shown(capsys):
    assert main.run_to_exit_code(FakeApp(SystemExit("bad news")), None) == 1
    assert "bad news" in capsys.readouterr().err


def test_a_crash_is_exit_code_one_with_its_traceback(capsys):
    assert main.run_to_exit_code(FakeApp(RuntimeError("boom")), None) == 1
    err = capsys.readouterr().err
    assert "RuntimeError" in err and "boom" in err


def test_ctrl_c_before_the_handler_exists_is_130():
    assert main.run_to_exit_code(FakeApp(KeyboardInterrupt()), None) == 130


def test_the_hard_exit_is_only_ever_called_from_the_command_line_entry_point():
    tree = ast.parse(inspect.getsource(main))
    blocks = [n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.dump(n.test)]
    assert len(blocks) == 1
    inside = {id(c) for c in ast.walk(blocks[0]) if isinstance(c, ast.Call)}
    exits = [c for c in ast.walk(tree) if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "_exit"]
    assert exits and all(id(c) in inside for c in exits)      # calling it elsewhere would kill pytest