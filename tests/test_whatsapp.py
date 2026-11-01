"""WhatsApp sender against a fake HTTP server (httpx.MockTransport): no network, no Meta account needed."""

import json

import httpx
import pytest

from shifa.config import get_settings
from shifa.messaging.whatsapp import WhatsAppSender


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()
    for k, v in {"whatsapp_token": "TEST", "whatsapp_phone_number_id": "12345", "whatsapp_mode": "text"}.items():
        monkeypatch.setattr(s, k, v)
    return s


def sender_with(handler):
    return WhatsAppSender(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_text_message_payload_and_success(settings):
    seen = {}

    def handler(request):
        seen["url"], seen["auth"], seen["body"] = str(request.url), request.headers["authorization"], json.loads(request.content)
        return httpx.Response(200, json={"messages": [{"id": "wamid.ABC"}]})

    result = sender_with(handler).send("0300-1234567", "Salam")
    assert result.ok and result.external_id == "wamid.ABC"
    assert seen["url"] == "https://graph.facebook.com/v23.0/12345/messages"
    assert seen["auth"] == "Bearer TEST"
    assert seen["body"]["to"] == "923001234567" and seen["body"]["text"]["body"] == "Salam"


def test_template_mode_puts_message_in_body_variable(settings, monkeypatch):
    monkeypatch.setattr(settings, "whatsapp_mode", "template")
    body = sender_with(lambda r: httpx.Response(200, json={}))._payload("03001234567", "Reminder text")
    assert body["type"] == "template" and body["template"]["name"] == "appointment_update"
    assert body["template"]["components"][0]["parameters"][0]["text"] == "Reminder text"


@pytest.mark.parametrize("status,code,retryable", [
    (429, 130429, True),   # rate limited: try again later
    (503, None, True),     # Meta outage
    (400, 131047, False),  # outside the 24 h window: retrying can't help
    (401, 190, False),     # expired token
    (400, 131030, False),  # recipient not on the test list
])
def test_error_classification(settings, status, code, retryable):
    def handler(request):
        return httpx.Response(status, json={"error": {"code": code, "message": "nope"}})

    result = sender_with(handler).send("03001234567", "x")
    assert not result.ok and result.retryable is retryable


def test_network_error_is_retryable(settings):
    def handler(request):
        raise httpx.ConnectError("no internet")

    assert sender_with(handler).send("03001234567", "x").retryable


def test_missing_credentials_fail_loudly(monkeypatch):
    monkeypatch.setattr(get_settings(), "whatsapp_token", "")
    with pytest.raises(ValueError, match="SHIFA_WHATSAPP_TOKEN"):
        WhatsAppSender()
