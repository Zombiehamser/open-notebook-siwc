"""Offline SIWC contract checks. All accounts, tokens and documents are synthetic."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from langchain_core.messages import HumanMessage, SystemMessage

from open_notebook.ai.siwc import (
    SIWCChatModel,
    SIWCLanguageModel,
    events,
    request_input,
)
from open_notebook.ai.siwc_auth import (
    ISSUER,
    RESOURCE,
    SCOPES,
    PlanLimitReached,
    SignInRequired,
    SIWCConfigurationError,
    SIWCSession,
)
from open_notebook.exceptions import IncompleteGenerationError, NetworkError


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setenv("OPEN_NOTEBOOK_SIWC_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.setenv("OPEN_NOTEBOOK_SIWC_PROXY", "http://127.0.0.1:9")
    monkeypatch.delenv("OPEN_NOTEBOOK_PROCESS_ROLE", raising=False)
    session = SIWCSession()
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    key.update(kid="synthetic-key", alg="RS256", use="sig")
    state: dict[str, Any] = {
        "requests": [],
        "refreshes": 0,
        "nonce": "",
        "scopes": SCOPES,
        "subject": "synthetic-owner",
        "failed": False,
        "events": [
            {"type": "response.output_text.delta", "delta": "Маяк находится в Туле."},
            {"type": "response.completed", "response": {"status": "completed"}},
        ],
    }

    def handler(request):
        state["requests"].append(request)
        if request.url.path == "/.well-known/openid-configuration":
            return httpx.Response(
                200,
                json={
                    "issuer": ISSUER,
                    "jwks_uri": ISSUER + "/.well-known/jwks.json",
                    "revocation_endpoint": ISSUER + "/revoke",
                },
            )
        if request.url.path == "/.well-known/jwks.json":
            return httpx.Response(200, json={"keys": [key]})
        if request.url.path.endswith("/oauth/token"):
            form = parse_qs(request.content.decode())
            if form["grant_type"] == ["refresh_token"]:
                state["refreshes"] += 1
            if state["failed"]:
                return httpx.Response(
                    400, json={"error": "invalid_grant", "secret": "never-echo-this"}
                )
            now = int(time.time())
            token = jwt.encode(
                {
                    "iss": ISSUER,
                    "aud": "oaiapp_synthetic",
                    "sub": state["subject"],
                    "iat": now,
                    "exp": now + 3600,
                    "nonce": state["nonce"],
                },
                private,
                algorithm="RS256",
                headers={"kid": "synthetic-key"},
            )
            return httpx.Response(
                200,
                json={
                    "access_token": "synthetic-access",
                    "refresh_token": "synthetic-rotated",
                    "id_token": token,
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": state["scopes"],
                },
            )
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "slug": "visible-model",
                            "display_name": "Visible",
                            "visibility": "list",
                        },
                        {"slug": "hidden-model", "visibility": "hidden"},
                    ]
                },
            )
        if request.url.path == "/v1/responses":
            return httpx.Response(
                200,
                text="".join(
                    "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                    for event in state["events"]
                ),
            )
        if request.url.path == "/revoke":
            return httpx.Response(200)
        raise AssertionError("Unexpected network destination")

    monkeypatch.setattr(
        session,
        "http",
        lambda: httpx.Client(transport=httpx.MockTransport(handler), trust_env=False),
    )
    monkeypatch.setattr("open_notebook.ai.siwc.SIWCSession", lambda: session)

    def connect(background=False):
        pending, url = session.begin("http://127.0.0.1:18557/auth/callback")
        state["nonce"] = pending["nonce"]
        account = session.complete(
            pending,
            {
                "state": pending["state"],
                "code": "synthetic-code",
                "client_id": "oaiapp_synthetic",
            },
            background,
        )
        return account, pending, url

    state["connect"] = connect
    return session, state


def test_login_persists_host_and_does_not_use_shared_client(fake):
    session, state = fake
    account, pending, url = state["connect"]()
    query = parse_qs(urlparse(url).query)
    assert query["client_id"] == ["dynamic_agent_client"]
    assert query["resource"] == [RESOURCE]
    assert query["code_challenge_method"] == ["S256"]
    with session.lock():
        saved = session.read()
    assert saved["host_id"] == query["ext_agent_host_id"][0]
    assert saved["accounts"][account]["subject"] == "synthetic-owner"
    assert saved["accounts"][account]["allow_background"] is False
    assert session.path.stat().st_mode & 0o077 == 0
    second, returning = session.begin(pending["redirect_uri"])
    assert parse_qs(urlparse(returning).query)["client_id"] == [account]
    assert "agent_name_hint" not in parse_qs(urlparse(returning).query)
    assert "id_token_hint" not in returning
    assert second["state"] != pending["state"]


@pytest.mark.parametrize(
    "change", ["state", "expired", "client_id", "nonce", "scope", "denied"]
)
def test_invalid_login_is_rejected_before_any_identity_replacement(fake, change):
    session, state = fake
    pending, _ = session.begin("http://127.0.0.1:18557/auth/callback")
    state["nonce"] = pending["nonce"]
    callback = {
        "state": pending["state"],
        "code": "synthetic-code",
        "client_id": "oaiapp_synthetic",
    }
    if change == "state":
        callback["state"] = "wrong"
    if change == "expired":
        pending["expires"] = 0
    if change == "client_id":
        callback["client_id"] = "dynamic_agent_client"
    if change == "nonce":
        state["nonce"] = "wrong"
    if change == "scope":
        state["scopes"] = "openid profile email"
    if change == "denied":
        callback["error"] = "access_denied"
    with pytest.raises(SignInRequired):
        session.complete(pending, callback)
    with session.lock():
        assert not session.read()["accounts"]
    if change in ("state", "expired", "client_id", "denied"):
        assert not state["requests"]


def test_callback_is_one_use_and_returning_identity_must_match(fake):
    session, state = fake
    account, pending, _ = state["connect"]()
    with pytest.raises(SignInRequired):
        session.complete(pending, {"state": pending["state"]})
    pending, _ = session.begin("http://127.0.0.1:18557/auth/callback")
    state.update(nonce=pending["nonce"], subject="someone-else")
    with pytest.raises(SignInRequired):
        session.complete(pending, {"state": pending["state"], "code": "code"})
    assert session.read()["accounts"][account]["subject"] == "synthetic-owner"


def test_refresh_rotation_is_serialized_across_instances(fake, monkeypatch):
    session, state = fake
    account, _, _ = state["connect"]()
    with session.lock():
        data = session.read()
        data["accounts"][account]["expires_at"] = 0
        session.save(data)
    second = SIWCSession()
    monkeypatch.setattr(second, "http", session.http)
    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(lambda s: s.access_token(account), [session, second]))
    assert values == ["synthetic-access", "synthetic-access"]
    assert state["refreshes"] == 1
    assert session.read()["accounts"][account]["refresh_token"] == "synthetic-rotated"


def test_failed_refresh_clears_tokens_without_leaking_body(fake):
    session, state = fake
    account, _, _ = state["connect"]()
    with session.lock():
        data = session.read()
        data["accounts"][account]["expires_at"] = 0
        session.save(data)
    state["failed"] = True
    with pytest.raises(SignInRequired) as error:
        session.access_token(account)
    assert "never-echo-this" not in str(error.value)
    assert "access_token" not in session.read()["accounts"][account]


def test_worker_requires_explicit_background_consent(fake, monkeypatch):
    session, state = fake
    account, _, _ = state["connect"]()
    monkeypatch.setenv("OPEN_NOTEBOOK_PROCESS_ROLE", "worker")
    with pytest.raises(SIWCConfigurationError):
        session.access_token(account)
    monkeypatch.delenv("OPEN_NOTEBOOK_PROCESS_ROLE")
    state["connect"](background=True)
    monkeypatch.setenv("OPEN_NOTEBOOK_PROCESS_ROLE", "worker")
    assert session.access_token(account) == "synthetic-access"


def test_catalog_filters_visibility_preserves_order_and_logout(fake):
    session, state = fake
    account, _, _ = state["connect"]()
    assert [m["slug"] for m in session.models(account)] == ["visible-model"]
    assert session.logout(account)
    assert session.read()["accounts"][account]["client_id"] == account
    assert "access_token" not in session.read()["accounts"][account]


def test_no_proxy_means_no_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("OPEN_NOTEBOOK_SIWC_AUTH_DIR", str(tmp_path))
    monkeypatch.delenv("OPEN_NOTEBOOK_SIWC_PROXY", raising=False)
    with pytest.raises(SIWCConfigurationError):
        SIWCSession()


def test_closed_proxy_never_falls_back_to_direct_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("OPEN_NOTEBOOK_SIWC_AUTH_DIR", str(tmp_path))
    monkeypatch.setenv("OPEN_NOTEBOOK_SIWC_PROXY", "http://127.0.0.1:1")
    session = SIWCSession()
    with session.lock():
        data = session.read()
        data["accounts"]["oaiapp_synthetic"] = {
            "access_token": "synthetic",
            "expires_at": time.time() + 3600,
        }
        session.save(data)
    with pytest.raises(NetworkError):
        session.models("oaiapp_synthetic")


def test_responses_contract_and_esperanto_language_model(fake):
    _, state = fake
    account, _, _ = state["connect"]()
    model = SIWCLanguageModel(
        model_name="visible-model",
        account_id=account,
        config={"structured": {"type": "json"}},
    )
    response = model.chat_complete(
        [
            {"role": "system", "content": "Ответь по документу."},
            {"role": "user", "content": "Проект Маяк находится в Туле. Где проект?"},
        ]
    )
    assert "Туле" in response.content
    request = next(r for r in state["requests"] if r.url.path == "/v1/responses")
    payload = json.loads(request.content)
    assert payload["store"] is False and payload["stream"] is True
    assert payload["input"][0]["role"] == "developer"
    assert payload["text"]["format"]["type"] == "json_object"
    assert not set(payload) & {
        "temperature",
        "max_output_tokens",
        "previous_response_id",
        "top_p",
    }
    assert request.headers["authorization"] == "Bearer synthetic-access"


@pytest.mark.asyncio
async def test_async_langchain_and_esperanto_streams(fake):
    _, state = fake
    account, _, _ = state["connect"]()
    model = SIWCLanguageModel(model_name="visible-model", account_id=account)
    assert (
        "Туле"
        in (
            await model.to_langchain().ainvoke([HumanMessage(content="Где Маяк?")])
        ).content
    )
    chunks = [chunk async for chunk in model.to_langchain().astream("Где Маяк?")]
    assert "Туле" in "".join(chunk.content for chunk in chunks)
    stream = await model.achat_complete(
        [{"role": "user", "content": "Где Маяк?"}], stream=True
    )
    chunks = [chunk async for chunk in stream]
    assert chunks[-1].choices[0].finish_reason == "stop"


@pytest.mark.parametrize(
    "tail", [[], [{"type": "response.incomplete"}], [{"type": "response.completed"}]]
)
def test_empty_or_incomplete_inference_is_never_success(fake, tail):
    _, state = fake
    account, _, _ = state["connect"]()
    state["events"] = tail
    with pytest.raises(IncompleteGenerationError):
        SIWCChatModel(model_name="visible-model", account_id=account).invoke("Test")


def test_quota_error_after_partial_output_is_permanent(fake):
    _, state = fake
    account, _, _ = state["connect"]()
    state["events"] = [
        {"type": "response.output_text.delta", "delta": "partial"},
        {
            "type": "response.failed",
            "response": {
                "error": {"code": "subscription_sharing_usage_limit_exceeded"}
            },
        },
    ]
    with pytest.raises(PlanLimitReached) as error:
        SIWCChatModel(model_name="visible-model", account_id=account).invoke("Test")
    assert isinstance(error.value, ValueError)


def test_sse_frames_and_unsupported_inputs():
    assert list(
        events([": ping", 'data: {"type":', 'data: "response.completed"}', ""])
    ) == [{"type": "response.completed"}]
    with pytest.raises(IncompleteGenerationError):
        list(events(["data: {} "]))
    with pytest.raises(SIWCConfigurationError):
        request_input([{"role": "tool", "content": "value"}])
    assert request_input([SystemMessage(content="policy")])[0]["role"] == "developer"


@pytest.mark.asyncio
async def test_model_manager_uses_only_linked_siwc_account(fake, monkeypatch):
    from open_notebook.ai.models import Model, ModelManager
    from open_notebook.domain.credential import Credential

    credential = Credential(
        name="Synthetic",
        provider="chatgpt_siwc",
        config={"siwc_client_id": "oaiapp_synthetic"},
    )
    model = Model(
        name="visible-model",
        provider="chatgpt_siwc",
        type="language",
        credential="credential:synthetic",
    )
    monkeypatch.setattr(Model, "get", AsyncMock(return_value=model))
    monkeypatch.setattr(Model, "get_credential_obj", AsyncMock(return_value=credential))
    provider = await ModelManager().get_model("model:synthetic")
    assert isinstance(provider, SIWCLanguageModel)
    assert provider.account_id == "oaiapp_synthetic"
    monkeypatch.setattr(Model, "get_credential_obj", AsyncMock(return_value=None))
    with pytest.raises(SIWCConfigurationError):
        await ModelManager().get_model("model:synthetic")


def test_siwc_storage_configuration_is_not_an_api_key_migration(fake):
    from api.credentials_service import check_env_configured

    assert check_env_configured("chatgpt_siwc") is False


def test_refresh_diagnostics_do_not_capture_token_bearing_frames(fake):
    import io

    from loguru import logger

    session, state = fake
    account, _, _ = state["connect"]()
    with session.lock():
        data = session.read()
        data["accounts"][account]["expires_at"] = 0
        session.save(data)
    state["scopes"] = "openid email"
    diagnostic = io.StringIO()
    sink = logger.add(diagnostic, diagnose=True, backtrace=False)
    try:
        try:
            session.access_token(account)
        except SignInRequired as error:
            logger.opt(exception=error).error("Synthetic SIWC refresh failure")
        else:
            pytest.fail("Expected rejected plan permission")
    finally:
        logger.remove(sink)
    assert "synthetic-access" not in diagnostic.getvalue()
    assert "synthetic-rotated" not in diagnostic.getvalue()


def test_login_attempt_starts_only_when_owner_opens_start(monkeypatch):
    import io
    from types import SimpleNamespace
    from unittest.mock import Mock

    from open_notebook import siwc_cli

    session = Mock()
    pending = {"state": "synthetic-state", "expires": time.time() + 600, "used": False}
    session.begin.return_value = (pending, "https://auth.openai.com/synthetic")
    session.complete.side_effect = [
        SignInRequired("Invalid or expired SIWC sign-in attempt."),
        "synthetic-account",
    ]
    args = SimpleNamespace(
        port=18557,
        bind="127.0.0.1",
        account=None,
        new_account=False,
        allow_background=False,
    )

    class Server:
        def __init__(self, address, handler):
            session.begin.assert_not_called()
            self.handler = handler
            self.count = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def handle_request(self):
            handler = object.__new__(self.handler)
            handler.headers = {"Host": "127.0.0.1:18557"}
            handler.send_response = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()
            handler.wfile = io.BytesIO()
            handler.path = [
                "/start",
                "/start",
                "/auth/callback?state=wrong-state&code=synthetic-code",
                "/auth/callback?state=synthetic-state&code=synthetic-code",
            ][self.count]
            self.count += 1
            handler.do_GET()
            if self.count == 3:
                handler.send_response.assert_called_once_with(400)
                assert (
                    "Invalid or expired sign-in attempt"
                    in handler.wfile.getvalue().decode()
                )
                assert "synthetic-code" not in handler.wfile.getvalue().decode()

    monkeypatch.setattr(siwc_cli, "ThreadingHTTPServer", Server)
    siwc_cli.login(session, args)
    session.begin.assert_called_once()
    assert session.complete.call_count == 2
    session.complete.assert_called_with(
        pending, {"state": "synthetic-state", "code": "synthetic-code"}, False
    )


def test_login_serves_requests_while_browser_preconnection_is_idle(monkeypatch):
    import socket
    from contextlib import closing
    from http.client import HTTPConnection
    from threading import Event
    from types import SimpleNamespace
    from unittest.mock import Mock

    from open_notebook import siwc_cli

    ready = Event()
    accepted = Event()
    session = Mock()
    pending = {"expires": time.time() + 600, "used": False}
    session.begin.return_value = (pending, "https://auth.openai.com/synthetic")
    session.complete.return_value = "synthetic-account"
    args = SimpleNamespace(
        port=0,
        bind="127.0.0.1",
        account=None,
        new_account=False,
        allow_background=False,
    )

    class Server(siwc_cli.ThreadingHTTPServer):
        def __init__(self, address, handler):
            super().__init__(address, handler)
            args.port = self.server_address[1]
            ready.set()

        def get_request(self):
            connection = super().get_request()
            accepted.set()
            return connection

    monkeypatch.setattr(siwc_cli, "ThreadingHTTPServer", Server)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(siwc_cli.login, session, args)
        assert ready.wait(5)
        address = ("127.0.0.1", args.port)
        idle = socket.create_connection(address, timeout=2)
        try:
            # Ensure the idle socket is accepted before opening a real request.
            assert accepted.wait(5)
            with closing(HTTPConnection(*address, timeout=2)) as client:
                client.request("GET", "/start")
                response = client.getresponse()
                assert response.status == 302
                assert (
                    response.getheader("Location")
                    == "https://auth.openai.com/synthetic"
                )
        finally:
            idle.close()
            # Also terminate the login loop if the assertion above fails.
            with closing(HTTPConnection(*address, timeout=3)) as client:
                client.request("GET", "/auth/callback?state=test&code=synthetic-code")
                assert client.getresponse().status == 200
            task.result(timeout=5)
    session.begin.assert_called_once()
    session.complete.assert_called_once_with(
        pending, {"state": "test", "code": "synthetic-code"}, False
    )
