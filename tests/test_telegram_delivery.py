from unittest.mock import Mock, patch

import pytest
import requests

from tradingagents.telegram_delivery import send_telegram_text


def _response(status=200, payload=None):
    response = Mock()
    response.status_code = status
    response.json.return_value = {"ok": True, "result": {"message_id": 42}} if payload is None else payload
    return response


def test_send_telegram_text_posts_plain_text_without_leaking_credential():
    with patch(
        "tradingagents.telegram_delivery.requests.post",
        return_value=_response(),
    ) as post:
        result = send_telegram_text(
            "hello",
            credential="dummy-credential",
            target="12345",
            timeout=7,
        )

    assert result.provider_message_id == 42
    args, kwargs = post.call_args
    assert args[0].endswith("/sendMessage")
    assert "dummy-credential" in args[0]
    assert kwargs["json"] == {
        "chat_id": "12345",
        "text": "hello",
        "disable_web_page_preview": True,
    }
    assert kwargs["timeout"] == 7


def test_http_error_does_not_echo_response_or_credential():
    response = _response(status=401)
    response.text = "credential=dummy-credential"
    with (
        patch("tradingagents.telegram_delivery.requests.post", return_value=response),
        pytest.raises(RuntimeError) as exc,
    ):
        send_telegram_text("hello", credential="dummy-credential", target="12345")
    message = str(exc.value)
    assert "HTTP 401" in message
    assert "dummy-credential" not in message


def test_request_exception_is_sanitized():
    with (
        patch(
            "tradingagents.telegram_delivery.requests.post",
            side_effect=requests.Timeout("dummy-credential"),
        ),
        pytest.raises(RuntimeError) as exc,
    ):
        send_telegram_text("hello", credential="dummy-credential", target="12345")
    assert "Timeout" in str(exc.value)
    assert "dummy-credential" not in str(exc.value)


def test_invalid_json_and_unconfirmed_provider_fail_closed():
    invalid_json = _response()
    invalid_json.json.side_effect = ValueError("bad json")
    with (
        patch("tradingagents.telegram_delivery.requests.post", return_value=invalid_json),
        pytest.raises(RuntimeError, match="invalid JSON"),
    ):
        send_telegram_text("hello", credential="dummy", target="123")

    with (
        patch(
            "tradingagents.telegram_delivery.requests.post",
            return_value=_response(payload={"ok": False}),
        ),
        pytest.raises(RuntimeError, match="did not confirm"),
    ):
        send_telegram_text("hello", credential="dummy", target="123")


def test_missing_message_id_is_allowed_after_provider_ok():
    with patch(
        "tradingagents.telegram_delivery.requests.post",
        return_value=_response(payload={"ok": True, "result": {}}),
    ):
        result = send_telegram_text("hello", credential="dummy", target="123")
    assert result.provider_message_id is None


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, "15"])
def test_invalid_timeout_rejected_before_network(timeout):
    with (
        patch("tradingagents.telegram_delivery.requests.post") as post,
        pytest.raises(ValueError, match="timeout"),
    ):
        send_telegram_text("hello", credential="dummy", target="123", timeout=timeout)
    post.assert_not_called()
