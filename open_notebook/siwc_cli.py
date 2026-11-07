"""Local SIWC login/model registration. Tokens never enter API responses."""

import argparse
import asyncio
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from urllib.parse import parse_qs, urlparse

from open_notebook.ai.siwc_auth import SIWCSession
from open_notebook.exceptions import OpenNotebookError


def login(session, args):
    pending = None
    authorize_url = ""
    outcome = {}
    request_lock = Lock()

    class Callback(BaseHTTPRequestHandler):
        timeout = 10

        def log_message(self, *_):
            pass  # Authorization codes and state must never enter access logs.

        def do_GET(self):
            # Browsers pre-open idle sockets; serialize only complete requests.
            with request_lock:
                self.handle_get()

        def handle_get(self):
            nonlocal pending, authorize_url
            parsed = urlparse(self.path)
            if self.headers.get("Host") != f"127.0.0.1:{args.port}":
                self.send_error(400)
                return
            if parsed.path == "/start":
                if pending is not None and pending["used"]:
                    self.send_error(409, "Sign-in already processed. Restart login.")
                    return
                if pending is None or time.time() >= pending["expires"]:
                    pending, authorize_url = session.begin(
                        f"http://127.0.0.1:{args.port}/auth/callback",
                        args.account,
                        args.new_account,
                    )
                self.send_response(302)
                self.send_header("Location", authorize_url)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                return
            if parsed.path != "/auth/callback":
                self.send_error(404)
                return
            if pending is None:
                self.send_error(400)
                return
            try:
                query = parse_qs(parsed.query, max_num_fields=20)
                if any(len(value) != 1 for value in query.values()):
                    raise ValueError("Duplicate callback parameters")
                account = session.complete(
                    pending,
                    {key: value[0] for key, value in query.items()},
                    args.allow_background,
                )
                outcome["account"] = account
                message = "ChatGPT connected. You may close this window."
                status = 200
            except (OpenNotebookError, ValueError):
                if pending["used"]:
                    outcome["failed"] = True
                message = (
                    "ChatGPT connection failed. Restart the login command."
                    if pending["used"]
                    else "Invalid or expired sign-in attempt. Open /start again and finish that sign-in."
                )
                status = 400
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(message.encode())

    # Docker publishes this listener only on Windows loopback, never on the LAN.
    with ThreadingHTTPServer((args.bind, args.port), Callback) as server:
        server.timeout = 1
        print(
            f"Open http://127.0.0.1:{args.port}/start to Continue with ChatGPT.",
            flush=True,
        )
        print(
            "Document excerpts and questions will be sent to OpenAI. Plan limits and OpenAI terms apply.",
            flush=True,
        )
        print(
            "Background requests: "
            + ("explicitly allowed" if args.allow_background else "disabled"),
            flush=True,
        )
        while not outcome and (pending is None or time.time() < pending["expires"]):
            server.handle_request()
    if not outcome.get("account"):
        raise ValueError(
            "SIWC login did not complete. No connected session was replaced."
        )
    print("ChatGPT connected. Run the models command to discover available models.")


async def register(session, account):
    from open_notebook.ai.models import Model
    from open_notebook.domain.credential import Credential

    catalog = await asyncio.to_thread(session.models, account)
    credentials = await Credential.get_by_provider("chatgpt_siwc")
    credential = next(
        (c for c in credentials if (c.config or {}).get("siwc_client_id") == account),
        None,
    )
    if credential is None:
        credential = Credential(
            name="ChatGPT SIWC " + account[-8:],
            provider="chatgpt_siwc",
            modalities=["language"],
            config={"siwc_client_id": account},
        )
        await credential.save()
    if not credential.id:
        raise ValueError("SIWC registration metadata did not receive a database ID.")
    existing = await Model.get_by_credential(credential.id)
    names = {model.name for model in existing}
    for item in catalog:
        if item["slug"] not in names:
            await Model(
                name=item["slug"],
                provider="chatgpt_siwc",
                type="language",
                credential=credential.id,
            ).save()
    print(json.dumps({"registered": len(catalog), "defaults_changed": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["login", "status", "models", "register", "logout"]
    )
    parser.add_argument("--account")
    parser.add_argument("--new-account", action="store_true")
    parser.add_argument(
        "--allow-background",
        action="store_true",
        help="Explicit consent to document transformations and other background requests",
    )
    parser.add_argument("--port", type=int, default=18557)
    parser.add_argument("--bind", choices=["127.0.0.1", "0.0.0.0"], default="127.0.0.1")
    args = parser.parse_args()
    session = SIWCSession()
    if args.command == "login":
        login(session, args)
        return
    with session.lock():
        data = session.read()
    account = args.account or data["active"]
    if args.command == "status":
        print(
            json.dumps(
                {
                    "active": data["active"],
                    "accounts": [
                        {
                            "client_id": key,
                            "connected": bool(value.get("access_token")),
                            "allow_background": value.get("allow_background", False),
                        }
                        for key, value in data["accounts"].items()
                    ],
                }
            )
        )
    elif not account:
        raise ValueError("Sign in with ChatGPT first.")
    elif args.command == "models":
        print(
            json.dumps(
                [
                    {
                        "slug": m["slug"],
                        "display_name": m.get("display_name", m["slug"]),
                    }
                    for m in session.models(account)
                ]
            )
        )
    elif args.command == "register":
        asyncio.run(register(session, account))
    elif args.command == "logout":
        confirmed = session.logout(account)
        print(
            "Signed out. Remote revocation "
            + (
                "confirmed."
                if confirmed
                else "was not confirmed; disconnect this app in ChatGPT Settings."
            )
        )


if __name__ == "__main__":
    try:
        main()
    except (OpenNotebookError, ValueError):
        # Deliberately avoid tracebacks containing pending OAuth values or tokens.
        print(
            "SIWC command failed. Check the connection and repeat the official login if required.",
            flush=True,
        )
        raise SystemExit(1) from None
