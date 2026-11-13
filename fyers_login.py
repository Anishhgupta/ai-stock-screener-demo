"""
Fyers daily login helper.

Unlike Angel One's TOTP (which can regenerate itself automatically),
Fyers uses an OAuth-style flow that requires a NEW access token to be
generated once per trading day via a manual browser login step. Run
this script each morning before `python main.py --broker fyers`.

Setup (one-time): run `python setup_wizard.py`, or set these in your .env:
         FYERS_CLIENT_ID=<your App ID, e.g. ABC12345-100>
         FYERS_SECRET_ID=<your Secret ID>
         FYERS_REDIRECT_URI=<the exact Redirect URL you set on the app,
                              e.g. https://www.google.com>
    after creating an app at https://myapi.fyers.in/dashboard -> Create App.

Usage:
    python fyers_login.py

This will:
    1. Print a login URL.
    2. You open it, log into Fyers, and approve the app.
    3. Fyers redirects your browser to FYERS_REDIRECT_URI with
       "...&auth_code=XXXXX&..." in the URL -- copy that auth_code value.
    4. Paste it back into this script when prompted.
    5. It exchanges the auth_code for an access token and saves it into
       .env as FYERS_ACCESS_TOKEN (without printing it). If it cannot
       write .env, it prints the token for you to paste instead.

The resulting access token is valid for the rest of that trading day
(Fyers tokens expire daily) -- re-run this script each morning.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
load_dotenv()

from brokers.base import get_env
from envfile import update_env

ENV_PATH = Path(__file__).resolve().parent / ".env"


def save_token(token: str, path=None) -> bool:
    """Write today's token into .env. False if that was not possible (the caller then shows it instead)."""
    target = ENV_PATH if path is None else path
    try:
        update_env(target, {"FYERS_ACCESS_TOKEN": token})
        return True
    except (OSError, ValueError):
        return False


def main():
    client_id = get_env("FYERS_CLIENT_ID")
    secret_id = get_env("FYERS_SECRET_ID")
    redirect_uri = get_env("FYERS_REDIRECT_URI")

    from fyers_apiv3 import fyersModel

    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_id,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code",
    )

    auth_url = session.generate_authcode()
    print("\n" + "=" * 70)
    print("STEP 1: Open this URL in your browser and log in to Fyers:")
    print("=" * 70)
    print(auth_url)
    print(
        "\nAfter logging in and approving the app, Fyers will redirect you "
        "to your Redirect URL with something like:\n"
        f"  {redirect_uri}?s=ok&code=200&auth_code=XXXXXXXX...\n"
        "Copy the FULL resulting URL from your browser's address bar "
        "(even if the page itself shows an error/blank page -- that's "
        "normal, only the URL matters)."
    )

    pasted = input("\nSTEP 2: Paste the full redirected URL here: ").strip()

    auth_code = _extract_auth_code(pasted)
    if not auth_code:
        print(
            "\nCouldn't find an auth_code in what you pasted. Make sure you "
            "copied the FULL URL from the address bar after being "
            "redirected, not just part of it."
        )
        return

    session.set_token(auth_code)
    response = session.generate_token()

    if "access_token" not in response:
        print("\nToken generation FAILED. Fyers response:")
        print(response)
        print(
            "\nCommon causes: auth_code already used/expired (they're "
            "single-use and short-lived -- restart this script and get a "
            "fresh one), or FYERS_SECRET_ID is wrong."
        )
        return

    access_token = response["access_token"]
    print("\n" + "=" * 70)
    if save_token(access_token):
        print("SUCCESS. Today's access token was saved into .env (it is not shown here).")
        print("=" * 70)
        print(
            "\nContainers read .env only when they are created, so recreate them:\n"
            "  docker compose --profile live down\n"
            "  docker compose --profile live up -d\n"
            "\nThis token is valid for the rest of today's trading session "
            "only -- re-run this script tomorrow morning before your next "
            "live run."
        )
        return
    print("SUCCESS, but the token could not be saved into .env. Your access token for today:")
    print("=" * 70)
    print(access_token)
    print(
        "\nPaste this into .env as:\n"
        f"  FYERS_ACCESS_TOKEN={access_token}\n"
        "\nThis token is valid for the rest of today's trading session "
        "only -- re-run this script tomorrow morning before your next "
        "live run."
    )


def _extract_auth_code(url_or_code: str) -> str | None:
    # Accept either a full redirected URL or a bare auth_code, in case
    # the person pastes just the code itself.
    if "auth_code=" in url_or_code:
        parsed = urlparse(url_or_code)
        qs = parse_qs(parsed.query)
        vals = qs.get("auth_code")
        return vals[0] if vals else None
    match = re.match(r"^[A-Za-z0-9._-]{20,}$", url_or_code)
    return url_or_code if match else None


if __name__ == "__main__":
    main()