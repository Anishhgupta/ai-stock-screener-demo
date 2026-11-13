"""
Read and update a .env file without disturbing it.

Keeps comments, blank lines, other keys and the file's line endings; replaces the
value of a key that is already there; appends keys that are not; and writes through
a temporary file, so an interrupted write cannot leave a half-written .env.
Values are never printed or logged here.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

_KEY_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")
_KEY_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PLAIN = re.compile(r"^[A-Za-z0-9_./:@+,%=&-]*$")


def format_value(value: str) -> str:
    """A value as it should appear after the '=': bare if safe, otherwise quoted."""
    if any(ch in value for ch in "\r\n\x00"):
        raise ValueError("A value cannot contain a line break.")
    if _PLAIN.match(value):
        return value
    if "'" not in value:
        return f"'{value}'"          # single quotes: no escapes and no $ expansion
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _unquote(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        inner = raw[1:-1]
        if raw[0] == '"':
            inner = inner.replace('\\"', '"').replace("\\\\", "\\")
        return inner
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()


def read_env(path) -> dict:
    """The KEY=value pairs in a .env file (empty if the file does not exist)."""
    p = Path(path)
    if not p.exists():
        return {}
    values = {}
    with open(p, encoding="utf-8", newline="") as handle:
        for line in re.split(r"\r\n|\n", handle.read()):
            match = _KEY_LINE.match(line)
            if match:
                values[match.group(1)] = _unquote(match.group(2))
    return values


def update_env(path, updates: dict) -> None:
    """Set these keys in the .env file, leaving everything else as it was."""
    path = Path(path)
    new_lines = {}
    for key, value in updates.items():
        if not _KEY_NAME.match(key):
            raise ValueError(f"{key!r} is not a valid setting name.")
        new_lines[key] = f"{key}={format_value(str(value))}"

    text = ""
    if path.exists():
        with open(path, encoding="utf-8", newline="") as handle:
            text = handle.read()
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = re.split(r"\r\n|\n", text) if text else []
    if lines and lines[-1] == "":
        lines.pop()                    # the file ended with a newline

    seen = set()
    for i, line in enumerate(lines):
        match = _KEY_LINE.match(line)
        if match and match.group(1) in new_lines:
            lines[i] = new_lines[match.group(1)]
            seen.add(match.group(1))
    for key in new_lines:
        if key not in seen:
            lines.append(new_lines[key])
    output = newline.join(lines) + newline

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".env.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(output)
        try:
            os.chmod(temp, 0o600)      # owner only where the system supports it
        except OSError:
            pass
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise