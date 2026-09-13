"""Minimal Telegram text delivery adapter with credential-safe errors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class TelegramSendResult:
    """Provider confirmation without retaining credentials or destination text."""

    provider_message_id: int | None


def send_telegram_text(
    text: str,
    *,
    credential: str,
    target: str,
    timeout: float = 15.0,
) -> TelegramSendResult:
    """Send one plain-text message. Errors never include the credential."""

    message = str(text)
    secret = str(credential).strip()
    destination = str(target).strip()
    if not message:
        raise ValueError("text must be nonempty")
    if not secret or not destination:
        raise ValueError("Telegram credential and target must be nonempty")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a finite number greater than zero")
    timeout_value = float(timeout)
    if not math.isfinite(timeout_value) or timeout_value <= 0:
        raise ValueError("timeout must be a finite number greater than zero")

    endpoint = "https://api.telegram.org/" + "bot" + secret + "/sendMessage"
    try:
        response = requests.post(
            endpoint,
            json={
                "chat_id": destination,
                "text": message,
                "disable_web_page_preview": True,
            },
            timeout=timeout_value,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Telegram send failed: {type(exc).__name__}") from exc
    if response.status_code != 200:
        raise RuntimeError(f"Telegram send failed with HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Telegram send returned invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise RuntimeError("Telegram API did not confirm message delivery")
    result = payload.get("result")
    message_id = result.get("message_id") if isinstance(result, dict) else None
    if message_id is not None and (isinstance(message_id, bool) or not isinstance(message_id, int)):
        message_id = None
    return TelegramSendResult(provider_message_id=message_id)
