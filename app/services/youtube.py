"""
YouTube Data API v3 — one-click "Login with Google" + multi-account Shorts upload.

For the app maintainer (one-time setup):
  1. GCP Console → Enable "YouTube Data API v3"
  2. Credentials → Create OAuth 2.0 Client ID → Desktop application
     (Google may not issue a Client Secret for Desktop apps — that's normal.
      Use PKCE flow: only Client ID is required.)
  3. Add  http://localhost:8599  to "Authorized redirect URIs"
  4. Paste Client ID (and Secret if issued) into Basic Settings in the app UI
     (saved to config.toml — git-ignored)

End users: just click "Login with Google" in the app — no GCP, no JSON files.
"""
import base64
import datetime
import hashlib
import json
import os
import re
import secrets
import threading
from typing import Optional
from urllib.parse import parse_qs, urlparse

_lock = threading.Lock()

from loguru import logger

from app.config import config
from app.utils import utils

# Credentials are read from config.toml at call time (never hardcoded).
# Add to config.toml:
#   [app]
#   youtube_client_id     = "xxxx.apps.googleusercontent.com"
#   youtube_client_secret = "xxxx"   # optional — omit if Google didn't issue one

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]
_STORAGE   = utils.storage_dir()
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
_CALLBACK_PORT = 8599
_REDIRECT_URI  = f"http://localhost:{_CALLBACK_PORT}"

# Result written by the OAuth callback thread; read by the UI on next rerun
_oauth_result: dict = {}
# PKCE verifier stored here so the callback thread can read it
_pkce_verifier: str = ""
# True while a callback server is listening on _CALLBACK_PORT
_server_running: bool = False


# ---------------------------------------------------------------------------
# Setup check
# ---------------------------------------------------------------------------

def _client_id() -> str:
    return config.app.get("youtube_client_id", "").strip()


def _client_secret() -> str:
    return config.app.get("youtube_client_secret", "").strip()


def is_app_configured() -> bool:
    """True when at least a Client ID is present in config (secret is optional for PKCE)."""
    return bool(_client_id())


# ---------------------------------------------------------------------------
# PKCE helpers (RFC 7636) — used when no client_secret is available
# ---------------------------------------------------------------------------

def _pkce_pair() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) for PKCE S256."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


# ---------------------------------------------------------------------------
# Token file helpers
# ---------------------------------------------------------------------------

def _token_file(nickname: str) -> str:
    safe = re.sub(r"[^\w\-]", "_", nickname)
    return os.path.join(_STORAGE, f"youtube_token_{safe}.json")


def list_accounts() -> list[str]:
    try:
        return sorted(
            re.match(r"^youtube_token_(.+)\.json$", f).group(1)
            for f in os.listdir(_STORAGE)
            if re.match(r"^youtube_token_(.+)\.json$", f)
        )
    except Exception:
        return []


def delete_account(nickname: str) -> bool:
    path = _token_file(nickname)
    if os.path.exists(path):
        os.remove(path)
        logger.info(f"YouTube account removed: {nickname}")
        return True
    return False


# ---------------------------------------------------------------------------
# Credential I/O
# ---------------------------------------------------------------------------

def _save_token(nickname: str, data: dict) -> None:
    os.makedirs(_STORAGE, exist_ok=True)
    payload = {
        "access_token":  data.get("access_token"),
        "refresh_token": data.get("refresh_token"),
        "token_uri":     _TOKEN_URL,
        "client_id":     _client_id(),
        "client_secret": _client_secret() or None,  # None when PKCE (no secret issued)
        "scopes":        SCOPES,
    }
    with open(_token_file(nickname), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"YouTube token saved: {nickname}")


def _refresh_token_manual(data: dict, path: str) -> bool:
    """Refresh access token without client_secret (PKCE / public client)."""
    try:
        import requests as _req
        body = {
            "client_id":     data.get("client_id", _client_id()),
            "refresh_token": data["refresh_token"],
            "grant_type":    "refresh_token",
        }
        secret = data.get("client_secret") or _client_secret()
        if secret:
            body["client_secret"] = secret
        resp = _req.post(_TOKEN_URL, data=body, timeout=15)
        new_data = resp.json()
        if "access_token" in new_data:
            data["access_token"] = new_data["access_token"]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        logger.warning(f"Token refresh failed: {new_data}")
        return False
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        return False


def _load_credentials(nickname: str):
    try:
        from google.oauth2.credentials import Credentials
    except ImportError:
        logger.warning("google-auth not installed")
        return None

    path = _token_file(nickname)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        secret = data.get("client_secret") or _client_secret() or None

        creds = Credentials(
            token=data.get("access_token"),
            refresh_token=data.get("refresh_token"),
            token_uri=data.get("token_uri", _TOKEN_URL),
            client_id=data.get("client_id", _client_id()),
            client_secret=secret,
            scopes=data.get("scopes", SCOPES),
        )
        if creds.expired and creds.refresh_token:
            if secret:
                from google.auth.transport.requests import Request
                creds.refresh(Request())
                data["access_token"] = creds.token
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            else:
                # PKCE public client — refresh without secret
                if not _refresh_token_manual(data, path):
                    return None
                creds = Credentials(
                    token=data.get("access_token"),
                    refresh_token=data.get("refresh_token"),
                    token_uri=data.get("token_uri", _TOKEN_URL),
                    client_id=data.get("client_id", _client_id()),
                    client_secret=None,
                    scopes=data.get("scopes", SCOPES),
                )
        return creds
    except Exception as e:
        logger.error(f"Failed to load YouTube credentials ({nickname}): {e}")
        return None


def is_authenticated(nickname: str) -> bool:
    return _load_credentials(nickname) is not None


# ---------------------------------------------------------------------------
# OAuth2 — local-server "Login with Google" flow
# ---------------------------------------------------------------------------

def start_login(nickname: str) -> str:
    """
    Start a one-shot local HTTP server on port 8599 to receive the OAuth2 callback,
    and return the Google authorization URL for the UI to display as a link.

    Supports both:
      - Traditional flow (client_secret present)
      - PKCE flow (no client_secret — Desktop app type in GCP)

    The result is written to _oauth_result when the callback arrives:
        {"success": True, "nickname": nickname}   — token saved
        {"error": "<message>"}                     — something went wrong

    Returns the auth URL string on success, or "" on configuration error.
    """
    global _oauth_result, _pkce_verifier, _server_running
    with _lock:
        if _server_running:
            logger.warning("YouTube OAuth server already running — ignoring duplicate start_login() call")
            return ""
        _oauth_result = {}

    if not is_app_configured():
        with _lock:
            _oauth_result = {"error": "YouTube Client ID not configured. Enter it in Basic Settings."}
        return ""

    import urllib.parse as _up

    use_pkce = not _client_secret()
    verifier, challenge = _pkce_pair()
    if use_pkce:
        _pkce_verifier = verifier
        logger.info("YouTube OAuth2: using PKCE flow (no client_secret)")
    else:
        _pkce_verifier = ""
        logger.info("YouTube OAuth2: using confidential client flow")

    params = {
        "client_id":     _client_id(),
        "redirect_uri":  _REDIRECT_URI,
        "response_type": "code",
        "scope":         " ".join(SCOPES),
        "access_type":   "offline",
        "prompt":        "consent",
    }
    if use_pkce:
        params["code_challenge"]        = challenge
        params["code_challenge_method"] = "S256"
    auth_url = _AUTH_URL + "?" + _up.urlencode(params)

    def _serve() -> None:
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import urllib.parse as _up2

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                qs = _up2.parse_qs(_up2.urlparse(self.path).query)
                code  = (qs.get("code")  or [None])[0]
                error = (qs.get("error") or [None])[0]

                def _html_ok():
                    return (
                        "<html><body style='font-family:sans-serif;text-align:center;padding:60px'>"
                        "<h2>✅ YouTubeの認証が完了しました</h2>"
                        "<p>このタブを閉じてアプリに戻ってください。</p>"
                        "</body></html>"
                    ).encode()

                def _html_ng(msg: str):
                    return (
                        f"<html><body style='font-family:sans-serif;text-align:center;padding:60px'>"
                        f"<h2>❌ 認証に失敗しました</h2><pre style='text-align:left;display:inline-block'>{msg}</pre>"
                        f"</body></html>"
                    ).encode()

                if code:
                    try:
                        import requests as _req
                        token_body: dict = {
                            "code":         code,
                            "client_id":    _client_id(),
                            "redirect_uri": _REDIRECT_URI,
                            "grant_type":   "authorization_code",
                        }
                        if _pkce_verifier:
                            token_body["code_verifier"] = _pkce_verifier
                        secret = _client_secret()
                        if secret:
                            token_body["client_secret"] = secret
                        logger.info(f"YouTube token exchange: redirect_uri={_REDIRECT_URI} pkce={'yes' if _pkce_verifier else 'no'}")
                        resp = _req.post(_TOKEN_URL, data=token_body, timeout=15)
                        token_data = resp.json()
                        logger.info(f"YouTube token response: {token_data}")
                        if "access_token" in token_data:
                            _save_token(nickname, token_data)
                            with _lock:
                                _oauth_result["success"] = True
                                _oauth_result["nickname"] = nickname
                            body = _html_ok()
                        else:
                            err = token_data.get("error_description") or token_data.get("error") or str(token_data)
                            logger.error(f"YouTube token exchange failed: {err}")
                            with _lock:
                                _oauth_result["error"] = err
                            body = _html_ng(err)
                    except Exception as ex:
                        logger.error(f"YouTube token exchange exception: {ex}")
                        with _lock:
                            _oauth_result["error"] = str(ex)
                        body = _html_ng(str(ex))
                else:
                    with _lock:
                        _oauth_result["error"] = error or "cancelled"
                    body = _html_ng(error or "cancelled")

                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        global _server_running
        with _lock:
            _server_running = True
        try:
            # Bind to 0.0.0.0 so requests from the host browser (via Docker port mapping) reach us
            server = HTTPServer(("0.0.0.0", _CALLBACK_PORT), _Handler)
            server.timeout = 300
            server.handle_request()
        except Exception as ex:
            with _lock:
                _oauth_result["error"] = str(ex)
        finally:
            with _lock:
                _server_running = False

    threading.Thread(target=_serve, daemon=True).start()
    return auth_url


def oauth_result() -> dict:
    """Return the current OAuth result (may be empty if still in progress)."""
    with _lock:
        return dict(_oauth_result)


# ---------------------------------------------------------------------------
# Channel info
# ---------------------------------------------------------------------------

def get_channel_name(nickname: str) -> str:
    try:
        from googleapiclient.discovery import build
    except ImportError:
        return ""
    creds = _load_credentials(nickname)
    if not creds:
        return ""
    try:
        yt = build("youtube", "v3", credentials=creds)
        resp = yt.channels().list(part="snippet", mine=True).execute()
        items = resp.get("items", [])
        return items[0]["snippet"]["title"] if items else ""
    except Exception as e:
        logger.warning(f"Failed to fetch channel name ({nickname}): {e}")
        return ""


# ---------------------------------------------------------------------------
# Video upload
# ---------------------------------------------------------------------------

def upload_video(
    video_path: str,
    title: str,
    description: str,
    tags: list[str],
    account: str,
    privacy: str = "private",
    publish_at: Optional[datetime.datetime] = None,
) -> dict:
    """
    Upload a video to the YouTube channel associated with *account*.

    Returns {"success": True, "video_id": str, "url": str}
         or {"success": False, "error": str}
    """
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        return {"success": False, "error": "google-api-python-client not installed"}

    creds = _load_credentials(account)
    if not creds:
        return {"success": False, "error": f"Account '{account}' not authenticated"}
    if not os.path.exists(video_path):
        return {"success": False, "error": f"Video file not found: {video_path}"}

    try:
        yt = build("youtube", "v3", credentials=creds)

        body = {
            "snippet": {
                "title":       title[:100],
                "description": description[:5000],
                "tags":        [t.lstrip("#") for t in (tags or [])][:500],
                "categoryId":  "22",
            },
            "status": {
                "privacyStatus":           "private" if publish_at else privacy,
                "selfDeclaredMadeForKids": False,
            },
        }
        if publish_at:
            body["status"]["publishAt"] = publish_at.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        media = MediaFileUpload(
            video_path, mimetype="video/*", resumable=True, chunksize=10 * 1024 * 1024
        )
        req = yt.videos().insert(part="snippet,status", body=body, media_body=media)

        logger.info(f"Uploading to YouTube Shorts [{account}]: {title!r}")
        response = None
        while response is None:
            st, response = req.next_chunk()
            if st:
                logger.info(f"YouTube upload [{account}]: {int(st.progress() * 100)}%")

        video_id = response["id"]
        url = f"https://www.youtube.com/shorts/{video_id}"
        logger.success(f"YouTube upload complete [{account}]: {url}")
        return {"success": True, "video_id": video_id, "url": url}

    except Exception as e:
        logger.error(f"YouTube upload failed [{account}]: {e}")
        return {"success": False, "error": str(e)}
