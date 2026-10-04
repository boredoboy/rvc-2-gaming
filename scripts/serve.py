from __future__ import annotations

import os

import uvicorn
from backend.main import app
from security_gate import PasswordGate


password = os.environ.get("RVC_ACCESS_PASSWORD", "")
uvicorn.run(
    PasswordGate(app, password),
    host="127.0.0.1",
    port=8000,
    access_log=False,
    proxy_headers=True,
    forwarded_allow_ips="127.0.0.1",
)

