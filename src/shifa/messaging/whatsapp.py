"""WhatsApp Cloud API sender (Meta). Free test number for development; up to a few verified recipients.

Two modes, because of WhatsApp's 24-hour rule:
- "text": free-form messages. Only delivered within 24 h after the patient last messaged the clinic number
  (the "customer service window"). For testing: send "hi" to the test number from your WhatsApp first.
- "template": a pre-approved message template. Required for business-initiated messages such as a reminder the
  evening before. The template takes the message as its body variable {{1}}.

Errors are classified so the dispatcher knows whether retrying can help.
"""

import logging

import httpx

from shifa.config import get_settings
from shifa.messaging.dispatch import SendResult, to_e164

log = logging.getLogger("shifa.whatsapp")
API = "https://graph.facebook.com/{version}/{phone_number_id}/messages"
# Meta error codes that retrying won't fix (https://developers.facebook.com/docs/whatsapp/cloud-api/support/error-codes)
PERMANENT = {
    190: "access token invalid or expired (temporary tokens last 24 h)",
    131030: "recipient not in the allowed test list: add it in API Setup",
    131026: "message undeliverable (not a WhatsApp number?)",
    131047: "outside the 24-hour window: use template mode",
    132001: "template does not exist or is not approved yet",
    100: "invalid parameter",
}


class WhatsAppSender:
    status_on_success = "sent"

    def __init__(self, client: httpx.Client | None = None):
        s = get_settings()
        if not (s.whatsapp_token and s.whatsapp_phone_number_id):
            raise ValueError("Set SHIFA_WHATSAPP_TOKEN and SHIFA_WHATSAPP_PHONE_NUMBER_ID in .env")
        self.settings = s
        self.url = API.format(version=s.whatsapp_api_version, phone_number_id=s.whatsapp_phone_number_id)
        self.client = client or httpx.Client(timeout=20)

    def _payload(self, phone: str, body: str) -> dict:
        base = {"messaging_product": "whatsapp", "recipient_type": "individual", "to": to_e164(phone)}
        if self.settings.whatsapp_mode == "template":
            return base | {"type": "template", "template": {
                "name": self.settings.whatsapp_template_name,
                "language": {"code": self.settings.whatsapp_template_lang},
                "components": [{"type": "body", "parameters": [{"type": "text", "text": body}]}]}}
        return base | {"type": "text", "text": {"preview_url": False, "body": body}}

    def send(self, phone: str, body: str) -> SendResult:
        try:
            r = self.client.post(self.url, json=self._payload(phone, body),
                                 headers={"Authorization": f"Bearer {self.settings.whatsapp_token}"})
        except httpx.HTTPError as e:  # network trouble: worth retrying
            return SendResult(ok=False, error=f"network: {e}", retryable=True)
        if r.status_code == 200:
            return SendResult(ok=True, external_id=r.json().get("messages", [{}])[0].get("id", ""))
        err = (r.json().get("error", {}) if r.headers.get("content-type", "").startswith("application/json") else {})
        code = err.get("code")
        hint = PERMANENT.get(code, "")
        message = f"HTTP {r.status_code} code {code}: {err.get('message', r.text[:200])}" + (f" ({hint})" if hint else "")
        # 429 and 5xx are transient; known codes and other 4xx are not.
        retryable = r.status_code == 429 or r.status_code >= 500
        return SendResult(ok=False, error=message, retryable=retryable and code not in PERMANENT)
