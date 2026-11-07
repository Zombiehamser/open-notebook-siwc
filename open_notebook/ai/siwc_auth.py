"""Official SIWC public-client flow. No browser cookies or shared OAuth client."""

import base64
import hashlib
import json
import os
import re
import secrets
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from urllib.parse import urlencode, urlparse

import httpx
import jwt

from open_notebook.exceptions import (
    AuthenticationError,
    ConfigurationError,
    ExternalServiceError,
    NetworkError,
    OpenNotebookError,
    RateLimitError,
)

ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
AUTHORIZE = ISSUER + "/api/accounts/authorize"
TOKEN = ISSUER + "/api/accounts/oauth/token"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
APP_NAME = "Open Notebook SIWC Community Fork"


def safe_state_errors(method):
    """Do not retain token-bearing stack frames in application diagnostics."""

    @wraps(method)
    def call(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except OpenNotebookError as error:
            raise type(error)(str(error)) from None
        except (OSError, KeyError, TypeError, ValueError):
            raise SIWCConfigurationError(
                "SIWC could not safely process its local state or service response."
            ) from None

    return call


class SignInRequired(AuthenticationError, ValueError):
    """Permanent for a background job; the owner must sign in again."""


class PlanLimitReached(RateLimitError, ValueError):
    """Do not automatically repeat a subscription request after a quota error."""


class SIWCConfigurationError(ConfigurationError, ValueError):
    """Do not retry an unconfigured or unconsented background request."""


def check_response(response):
    # Never expose an OAuth body, request URL or credentials in an exception.
    if response.status_code < 300:
        return
    if response.status_code == 429:
        raise PlanLimitReached(
            "ChatGPT plan limit reached. Review ChatGPT Settings → Usage."
        )
    if response.status_code in (400, 401, 403):
        raise SignInRequired(
            "ChatGPT authorization or permission was rejected. Reconnect using SIWC."
        )
    raise ExternalServiceError(f"SIWC service returned HTTP {response.status_code}.")


class SIWCSession:
    def __init__(self):
        directory = os.environ.get("OPEN_NOTEBOOK_SIWC_AUTH_DIR", "")
        proxy = os.environ.get("OPEN_NOTEBOOK_SIWC_PROXY", "")
        if not directory or not Path(directory).is_absolute():
            raise SIWCConfigurationError(
                "Set a separate absolute OPEN_NOTEBOOK_SIWC_AUTH_DIR."
            )
        if (
            urlparse(proxy).scheme not in ("http", "https", "socks5", "socks5h")
            or not urlparse(proxy).hostname
        ):
            raise SIWCConfigurationError(
                "An explicit OPEN_NOTEBOOK_SIWC_PROXY is required; direct fallback is disabled."
            )
        self.directory = Path(directory)
        self.proxy = proxy
        if self.directory.is_symlink():
            raise SIWCConfigurationError("SIWC storage must not be a symlink.")
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.directory.chmod(0o700)
        self.path = self.directory / "sessions.json"

    def http(self):
        return httpx.Client(
            proxy=self.proxy, trust_env=False, follow_redirects=False, timeout=180
        )

    @contextmanager
    def lock(self, name="sessions"):
        # Both API and worker share this lock across processes and rotating tokens.
        path = self.directory / (name + ".lock")
        if path.is_symlink():
            raise SIWCConfigurationError("Invalid SIWC lock file.")
        with os.fdopen(os.open(path, os.O_CREAT | os.O_RDWR, 0o600), "r+b") as handle:
            if sys.platform == "win32":
                import msvcrt

                handle.write(b"0")
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if sys.platform == "win32":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    @safe_state_errors
    def read(self):
        if self.path.is_symlink():
            raise SIWCConfigurationError("Invalid SIWC storage file.")
        if not self.path.exists():
            return {
                "host_id": "urn:uuid:" + str(uuid.uuid4()),
                "active": None,
                "accounts": {},
            }
        if os.name != "nt" and self.path.stat().st_mode & 0o077:
            raise SIWCConfigurationError(
                "SIWC credential file must have owner-only permissions."
            )
        return json.loads(self.path.read_text(encoding="utf-8"))

    @safe_state_errors
    def save(self, data):
        fd, temporary = tempfile.mkstemp(prefix=".sessions-", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def begin(self, redirect_uri, account=None, new_account=False):
        parsed = urlparse(redirect_uri)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.path != "/auth/callback"
            or parsed.query
            or parsed.fragment
            or parsed.username
        ):
            raise SIWCConfigurationError(
                "Use an HTTP 127.0.0.1 loopback /auth/callback URI."
            )
        with self.lock():
            data = self.read()
            selected = None if new_account else (account or data["active"])
            if selected and selected not in data["accounts"]:
                raise SignInRequired("Unknown saved ChatGPT registration.")
            self.save(data)  # Persist host identity before the first sign-in.
        verifier = secrets.token_urlsafe(48)
        pending = {
            "state": secrets.token_urlsafe(32),
            "nonce": secrets.token_urlsafe(32),
            "verifier": verifier,
            "redirect_uri": redirect_uri,
            "client_id": selected,
            "expires": time.time() + 600,
            "used": False,
        }
        query = {
            "client_id": selected or "dynamic_agent_client",
            "ext_agent_host_id": data["host_id"],
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": SCOPES,
            "resource": RESOURCE,
            "state": pending["state"],
            "nonce": pending["nonce"],
            "code_challenge_method": "S256",
            "code_challenge": base64.urlsafe_b64encode(
                hashlib.sha256(verifier.encode()).digest()
            )
            .decode()
            .rstrip("="),
        }
        if not selected:
            query["agent_name_hint"] = APP_NAME
        # Omit optional ID-token/login hints so no account token enters browser URLs.
        return pending, AUTHORIZE + "?" + urlencode(query)

    def validate_identity(self, client, token, client_id, nonce=None, subject=None):
        discovery_response = client.get(ISSUER + "/.well-known/openid-configuration")
        check_response(discovery_response)
        discovery = discovery_response.json()
        jwks_url = discovery.get("jwks_uri", "")
        parsed = urlparse(jwks_url)
        if (
            discovery.get("issuer") != ISSUER
            or parsed.scheme != "https"
            or parsed.hostname != "auth.openai.com"
            or parsed.port not in (None, 443)
            or parsed.username
        ):
            raise SignInRequired("Unexpected SIWC identity configuration.")
        response = client.get(jwks_url)
        check_response(response)
        header = jwt.get_unverified_header(token)
        if header.get("alg") != "RS256":
            raise SignInRequired("Unsupported SIWC ID-token signature algorithm.")
        keys = [
            key
            for key in response.json()["keys"]
            if key.get("kid") == header.get("kid")
        ]
        if len(keys) != 1:
            raise SignInRequired("SIWC ID-token signing key was not found.")
        claims = jwt.decode(
            token,
            jwt.PyJWK.from_dict(keys[0]).key,
            algorithms=["RS256"],
            audience=client_id,
            issuer=ISSUER,
            options={"require": ["exp", "iat", "sub", "iss", "aud"]},
        )
        if nonce is not None and not secrets.compare_digest(
            str(claims.get("nonce", "")), nonce
        ):
            raise SignInRequired("SIWC nonce mismatch.")
        if subject is not None and claims["sub"] != subject:
            raise SignInRequired("SIWC returned a different ChatGPT account.")
        return claims

    @safe_state_errors
    def complete(self, pending, callback, allow_background=False):
        if (
            pending["used"]
            or time.time() > pending["expires"]
            or not secrets.compare_digest(callback.get("state", ""), pending["state"])
        ):
            raise SignInRequired("Invalid or expired SIWC sign-in attempt.")
        pending["used"] = True
        if callback.get("error"):
            raise SignInRequired("ChatGPT sign-in or plan consent was declined.")
        client_id = callback.get("client_id") or pending["client_id"]
        if (
            not client_id
            or client_id == "dynamic_agent_client"
            or not re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", client_id)
            or (pending["client_id"] and client_id != pending["client_id"])
        ):
            raise SignInRequired(
                "SIWC registration did not return the expected issued client ID."
            )
        if not callback.get("code"):
            raise SignInRequired("SIWC authorization code is missing.")
        try:
            with self.http() as client, self.lock():
                data = self.read()
                previous = data["accounts"].get(client_id)
                response = client.post(
                    TOKEN,
                    data={
                        "grant_type": "authorization_code",
                        "client_id": client_id,
                        "code": callback["code"],
                        "code_verifier": pending["verifier"],
                        "redirect_uri": pending["redirect_uri"],
                        "resource": RESOURCE,
                    },
                )
                check_response(response)
                tokens = response.json()
                claims = self.validate_identity(
                    client,
                    tokens["id_token"],
                    client_id,
                    pending["nonce"],
                    previous["subject"] if previous else None,
                )
                record = {
                    "subject": claims["sub"],
                    "email": claims.get("email", ""),
                    "issuer": ISSUER,
                    "client_id": client_id,
                    "allow_background": bool(allow_background),
                }
                self.update_tokens(record, tokens)
                data["accounts"][client_id] = record
                data["active"] = client_id
                self.save(data)
                return client_id
        except httpx.HTTPError:
            raise NetworkError(
                "SIWC network request failed through the configured proxy."
            ) from None
        except (jwt.PyJWTError, KeyError, TypeError, json.JSONDecodeError):
            raise SignInRequired(
                "SIWC returned an invalid identity or token response."
            ) from None

    @staticmethod
    def update_tokens(record, tokens):
        scopes = tokens.get("scope", "").split()
        if not {"chatgpt.tokens.use.direct", "resource.invoke"}.issubset(scopes):
            raise SignInRequired("ChatGPT plan-use permission was not granted.")
        if (
            str(tokens.get("token_type", "")).lower() != "bearer"
            or not tokens.get("access_token")
            or not tokens.get("refresh_token")
        ):
            raise SignInRequired("SIWC did not return renewable bearer credentials.")
        lifetime = float(tokens["expires_in"])
        if not 0 < lifetime <= 86400:
            raise SignInRequired("Invalid SIWC token lifetime.")
        record.update({key: tokens[key] for key in ("access_token", "refresh_token")})
        if tokens.get("id_token"):
            record["id_token"] = tokens["id_token"]
        record.update(scopes=scopes, expires_at=time.time() + lifetime)

    @safe_state_errors
    def access_token(self, account):
        with self.lock():
            data = self.read()
            record = data["accounts"].get(account, {})
            if not record.get("access_token"):
                raise SignInRequired(
                    "Connect this ChatGPT account using the SIWC login command."
                )
            if os.environ.get(
                "OPEN_NOTEBOOK_PROCESS_ROLE"
            ) == "worker" and not record.get("allow_background"):
                raise SIWCConfigurationError(
                    "Background ChatGPT requests need explicit consent at SIWC login."
                )
            if record["expires_at"] > time.time() + 60:
                return record["access_token"]
            try:
                with self.http() as client:
                    response = client.post(
                        TOKEN,
                        data={
                            "grant_type": "refresh_token",
                            "client_id": account,
                            "refresh_token": record["refresh_token"],
                            "resource": RESOURCE,
                        },
                    )
                    if response.status_code in (400, 401, 403):
                        for key in ("access_token", "refresh_token", "id_token"):
                            record.pop(key, None)
                        self.save(data)
                    check_response(response)
                    tokens = response.json()
                    if tokens.get("id_token"):
                        self.validate_identity(
                            client,
                            tokens["id_token"],
                            account,
                            subject=record["subject"],
                        )
                    self.update_tokens(record, tokens)
                    self.save(data)
                    return record["access_token"]
            except httpx.HTTPError:
                raise NetworkError(
                    "SIWC refresh failed through the configured proxy."
                ) from None
            except (jwt.PyJWTError, KeyError, TypeError, json.JSONDecodeError):
                raise SignInRequired(
                    "SIWC refresh returned invalid credentials."
                ) from None

    @safe_state_errors
    def models(self, account):
        token = self.access_token(account)
        try:
            with self.http() as client:
                response = client.get(
                    RESOURCE + "/models", headers={"Authorization": "Bearer " + token}
                )
                check_response(response)
                return [
                    m
                    for m in response.json()["models"]
                    if m.get("visibility") == "list"
                ]
        except httpx.HTTPError:
            raise NetworkError(
                "SIWC model discovery failed through the configured proxy."
            ) from None

    def logout(self, account):
        confirmed = False
        with self.lock():
            data = self.read()
            record = data["accounts"].get(account, {})
            try:
                if record.get("refresh_token"):
                    with self.http() as client:
                        response = client.get(
                            ISSUER + "/.well-known/openid-configuration"
                        )
                        check_response(response)
                        endpoint = response.json()["revocation_endpoint"]
                        parsed = urlparse(endpoint)
                        if (
                            parsed.scheme != "https"
                            or parsed.hostname != "auth.openai.com"
                            or parsed.port not in (None, 443)
                            or parsed.username
                        ):
                            raise SignInRequired("Unexpected SIWC revocation endpoint.")
                        response = client.post(
                            endpoint,
                            data={
                                "token": record["refresh_token"],
                                "token_type_hint": "refresh_token",
                                "client_id": account,
                            },
                        )
                        confirmed = response.status_code == 200
                else:
                    confirmed = True
            except (httpx.HTTPError, KeyError, ValueError, ExternalServiceError):
                confirmed = False
            finally:
                for key in ("access_token", "refresh_token", "id_token"):
                    record.pop(key, None)
                self.save(data)
        return confirmed
