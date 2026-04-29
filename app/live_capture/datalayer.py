"""
ctrlX CORE Data Layer client using httpx (sync).

Drop-in replacement for the requests-based final_datalayer_api.CtrlXDataLayer,
using httpx which is already in requirements-snap.txt.
"""

import logging
import urllib.parse
from datetime import datetime, timedelta

import httpx

logger = logging.getLogger("live_capture.datalayer")


class CtrlXDataLayer:
    """
    Synchronous ctrlX CORE Data Layer client.
    Uses httpx instead of requests so it works in the snap environment.

    API endpoints:
        Auth:   POST https://<ip>/identity-manager/api/v2/auth/token
        Read:   GET  https://<ip>/automation/api/v2/nodes/<path>
        Write:  POST https://<ip>/automation/api/v1/<encoded-path>?format=json
    """

    def __init__(self, ip: str, username: str, password: str):
        self.ip = ip
        self.username = username
        self.password = password
        self._base = f"https://{ip}"
        self._token: str | None = None
        self._token_expiry: datetime | None = None
        # SSL verification disabled — ctrlX uses a self-signed certificate
        self._client = httpx.Client(verify=False, timeout=5.0)

    # ── Authentication ────────────────────────────────────────────────────────

    def get_token(self) -> str | None:
        url = f"{self._base}/identity-manager/api/v2/auth/token"
        try:
            resp = self._client.post(url, json={"name": self.username, "password": self.password})
            if resp.status_code in (200, 201):
                data = resp.json()
                self._token = data.get("access_token")
                expires_in = data.get("expires_in", 3600)
                self._token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
                return self._token
            logger.warning("Auth failed: HTTP %s — %s", resp.status_code, resp.text[:200])
            return None
        except Exception:
            logger.exception("Authentication error")
            return None

    def _ensure_token(self):
        if not self._token or (self._token_expiry and datetime.now() >= self._token_expiry):
            self.get_token()

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token}"}

    # ── Read / Write ──────────────────────────────────────────────────────────

    def read_node(self, path: str) -> dict:
        """
        Read a Data Layer node.
        Returns a dict with at least a 'value' key on success.
        """
        self._ensure_token()
        clean = path.lstrip("/")
        url = f"{self._base}/automation/api/v2/nodes/{clean}"
        resp = self._client.get(url, headers=self._auth_headers())
        if resp.status_code == 200:
            return resp.json()
        raise Exception(f"read_node '{path}' failed ({resp.status_code}): {resp.text[:200]}")

    def write_node(self, path: str, value=None, value_type: str | None = None):
        """
        Write to a Data Layer node.

        Args:
            path:       Node path (leading slash optional)
            value:      Value to write — None for pure command triggers
            value_type: Optional ctrlX type string (e.g. "bool8", "float64")
        """
        self._ensure_token()
        if not path.startswith("/"):
            path = "/" + path
        encoded = urllib.parse.quote(path, safe="")
        url = f"{self._base}/automation/api/v1/{encoded}?format=json"

        if value is None:
            payload: dict = {}
        elif value_type:
            payload = {"value": value, "type": value_type}
        elif isinstance(value, dict):
            payload = value
        else:
            payload = {"value": value}

        resp = self._client.post(
            url,
            headers={**self._auth_headers(), "Content-Type": "application/json"},
            json=payload,
        )
        if resp.status_code not in (200, 201, 204):
            raise Exception(f"write_node '{path}' failed ({resp.status_code}): {resp.text[:200]}")

    def close(self):
        self._client.close()
