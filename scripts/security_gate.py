"""Small single-user ASGI password gate for a temporarily public Kaggle tunnel."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from http.cookies import SimpleCookie
from urllib.parse import parse_qs


COOKIE_NAME = "rvc_session"
SESSION_SECONDS = 8 * 60 * 60
MAX_LOGIN_BODY = 16 * 1024


class PasswordGate:
    def __init__(self, app, password: str):
        if len(password) < 20:
            raise ValueError("RVC_ACCESS_PASSWORD must be at least 20 characters.")
        self.app = app
        self.password = password.encode("utf-8")
        self.signing_key = secrets.token_bytes(32)

    def _cookie(self, expires: int) -> str:
        payload = f"{expires}.{secrets.token_urlsafe(12)}"
        signature = hmac.new(self.signing_key, payload.encode(), hashlib.sha256).hexdigest()
        return f"{payload}.{signature}"

    def _valid_cookie(self, headers) -> bool:
        raw = dict(headers).get(b"cookie", b"").decode("latin1", "ignore")
        jar = SimpleCookie()
        try:
            jar.load(raw)
            value = jar[COOKIE_NAME].value
            expires_text, nonce, signature = value.split(".", 2)
            expires = int(expires_text)
        except (KeyError, ValueError, TypeError):
            return False
        if expires <= int(time.time()) or expires > int(time.time()) + SESSION_SECONDS + 60:
            return False
        payload = f"{expires}.{nonce}"
        expected = hmac.new(self.signing_key, payload.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, expected)

    @staticmethod
    async def _reply(send, status: int, body: bytes, headers=()):
        base = [(b"content-type", b"text/html; charset=utf-8"), (b"cache-control", b"no-store")]
        await send({"type": "http.response.start", "status": status, "headers": base + list(headers)})
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    def _secure_cookie(scope) -> bool:
        forwarded = dict(scope.get("headers", [])).get(b"x-forwarded-proto", b"").decode("latin1")
        return forwarded.split(",", 1)[0].strip().lower() == "https" or scope.get("scheme") == "https"

    async def __call__(self, scope, receive, send):
        kind = scope.get("type")
        if kind == "lifespan":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")
        if kind == "http" and path == "/login":
            method = scope.get("method", "GET").upper()
            if method == "GET":
                page = """<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RVC — вход</title><style>body{font:16px system-ui;background:#111827;color:#f9fafb;display:grid;min-height:100vh;place-items:center;margin:0}.card{width:min(390px,calc(100% - 40px));padding:28px;border:1px solid #374151;border-radius:18px;background:#1f2937}input,button{box-sizing:border-box;width:100%;padding:13px;margin-top:12px;border-radius:9px;border:1px solid #4b5563;font:inherit}button{background:#7c3aed;color:white;border:0;cursor:pointer}p{color:#cbd5e1;line-height:1.5}</style><main class="card"><h1>Вход в RVC</h1><p>Введите пароль доступа, заданный в Kaggle Secrets.</p><form method="post" action="/login"><input name="password" type="password" autocomplete="current-password" required autofocus><button type="submit">Войти</button></form></main></html>""".encode("utf-8")
                await self._reply(send, 200, page, [(b"content-security-policy", b"default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")])
                return
            if method != "POST":
                await self._reply(send, 405, b"Method not allowed")
                return

            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > MAX_LOGIN_BODY:
                    await self._reply(send, 413, b"Request too large")
                    return
                if not message.get("more_body", False):
                    break
            try:
                fields = parse_qs(body.decode("utf-8"), max_num_fields=8)
                supplied = fields.get("password", [""])[0].encode("utf-8")
            except (UnicodeDecodeError, ValueError):
                supplied = b""
            if not hmac.compare_digest(supplied, self.password):
                await self._reply(send, 401, "<!doctype html><meta charset=utf-8><p>Неверный пароль. Вернитесь назад и попробуйте ещё раз.</p>".encode("utf-8"))
                return

            expires = int(time.time()) + SESSION_SECONDS
            cookie = f"{COOKIE_NAME}={self._cookie(expires)}; Path=/; HttpOnly; SameSite=Lax; Max-Age={SESSION_SECONDS}"
            if self._secure_cookie(scope):
                cookie += "; Secure"
            await send({"type": "http.response.start", "status": 303, "headers": [(b"location", b"/"), (b"cache-control", b"no-store"), (b"set-cookie", cookie.encode("ascii"))]})
            await send({"type": "http.response.body", "body": b""})
            return

        if kind == "http":
            if self._valid_cookie(scope.get("headers", [])):
                await self.app(scope, receive, send)
                return
            await send({"type": "http.response.start", "status": 303, "headers": [(b"location", b"/login"), (b"cache-control", b"no-store")]})
            await send({"type": "http.response.body", "body": b""})
            return

        if kind == "websocket":
            if self._valid_cookie(scope.get("headers", [])):
                await self.app(scope, receive, send)
            else:
                await send({"type": "websocket.close", "code": 4401, "reason": "Login required"})
            return

        await self.app(scope, receive, send)

