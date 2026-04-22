"""Upstox OAuth2 authorization-code flow.

Usage:
    python scripts/auth.py

Opens the Upstox login page in a browser. After you log in and authorize,
Upstox redirects to the local callback URL; this module captures the code
and exchanges it for an access token, then writes the token back to .env.

Access tokens expire daily at 03:30 IST; re-run this each morning.
"""
from __future__ import annotations

import http.server
import secrets
import socketserver
import threading
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import httpx

from nse_bot.config import ROOT, load_config

AUTH_URL = "https://api.upstox.com/v2/login/authorization/dialog"
TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"


@dataclass
class _Capture:
    code: str | None = None
    state: str | None = None
    error: str | None = None


def _make_handler(capture: _Capture, expected_state: str):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            capture.state = (qs.get("state") or [None])[0]
            capture.code = (qs.get("code") or [None])[0]
            capture.error = (qs.get("error") or [None])[0]

            if capture.state != expected_state:
                capture.error = "state_mismatch"

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            body = (
                "<h2>Upstox auth received.</h2>"
                "<p>You can close this tab and return to the terminal.</p>"
                if capture.code and not capture.error
                else f"<h2>Auth failed: {capture.error or 'unknown'}</h2>"
            )
            self.wfile.write(body.encode("utf-8"))

        def log_message(self, *args, **kwargs):  # silence default logging
            return

    return Handler


def _run_callback_server(redirect_uri: str, capture: _Capture, expected_state: str) -> None:
    parsed = urllib.parse.urlparse(redirect_uri)
    port = parsed.port or 5555
    host = parsed.hostname or "localhost"
    handler = _make_handler(capture, expected_state)
    with socketserver.TCPServer((host, port), handler) as httpd:
        httpd.timeout = 300
        while capture.code is None and capture.error is None:
            httpd.handle_request()


def run_oauth_flow() -> str:
    cfg = load_config()
    if not cfg.api_key or not cfg.api_secret:
        raise SystemExit(
            "Missing UPSTOX_API_KEY / UPSTOX_API_SECRET. "
            "Copy .env.example to .env and fill in your credentials."
        )

    state = secrets.token_urlsafe(16)
    params = {
        "client_id": cfg.api_key,
        "redirect_uri": cfg.redirect_uri,
        "response_type": "code",
        "state": state,
    }
    url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"
    print(f"Opening browser for Upstox login...\n  {url}")
    capture = _Capture()
    server_thread = threading.Thread(
        target=_run_callback_server,
        args=(cfg.redirect_uri, capture, state),
        daemon=True,
    )
    server_thread.start()
    webbrowser.open(url)
    server_thread.join(timeout=300)

    if capture.error or not capture.code:
        raise SystemExit(f"Auth failed: {capture.error or 'no code received'}")

    token = _exchange_code(cfg.api_key, cfg.api_secret, cfg.redirect_uri, capture.code)
    _write_token_to_env(token)
    print("Access token saved to .env (UPSTOX_ACCESS_TOKEN).")
    return token


def _exchange_code(api_key: str, api_secret: str, redirect_uri: str, code: str) -> str:
    r = httpx.post(
        TOKEN_URL,
        data={
            "code": code,
            "client_id": api_key,
            "client_secret": api_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        headers={"Accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    token = data.get("access_token")
    if not token:
        raise SystemExit(f"Token exchange response missing access_token: {data}")
    return token


def _write_token_to_env(token: str) -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        env_path.write_text(f"UPSTOX_ACCESS_TOKEN={token}\n", encoding="utf-8")
        return

    lines = env_path.read_text(encoding="utf-8").splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith("UPSTOX_ACCESS_TOKEN="):
            lines[i] = f"UPSTOX_ACCESS_TOKEN={token}"
            replaced = True
            break
    if not replaced:
        lines.append(f"UPSTOX_ACCESS_TOKEN={token}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
