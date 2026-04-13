"""
Cal.com v2 REST integration — public endpoints (no auth required for guest bookings).

For personal booking pages:
  - GET  /v2/slots   → public, needs only eventTypeId
  - POST /v2/bookings → public guest booking, needs only eventTypeId

Set CALCOM_EVENT_TYPE_ID in .env (the numeric ID from your Cal.com event URL).
Optionally set CALCOM_API_KEY for authenticated requests (personal API key from cal.com/settings/developer).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from server.config import settings

_V2_BASE = "https://api.cal.com/v2"
_SLOTS_VERSION    = "2024-09-04"
_BOOKINGS_VERSION = "2026-02-25"


def _is_configured() -> bool:
    return bool(settings.calcom_event_type_id)


def _headers(extra: dict | None = None) -> dict:
    """Build request headers. Uses personal API key if set, otherwise anonymous."""
    h: dict = {"Content-Type": "application/json"}
    if settings.calcom_api_key:
        h["Authorization"] = f"Bearer {settings.calcom_api_key}"
    if extra:
        h.update(extra)
    return h


# ── Public API ───────────────────────────────────────────────────────────────

async def fetch_slots_in_window(
    date: str,
    start_time: str,
    end_time: str,
    timezone_name: str = "UTC",
) -> list[dict[str, Any]]:
    """
    Return Pooja's available slots inside the recruiter's window.

    Args:
        date:          "YYYY-MM-DD"
        start_time:    "HH:MM"  (24-hour, in recruiter's timezone)
        end_time:      "HH:MM"
        timezone_name: IANA timezone string

    Returns:
        Sorted list of {"start": ISO, "end": ISO} dicts.
    """
    if not _is_configured():
        return []

    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("UTC")

    try:
        start_dt = datetime.strptime(f"{date}T{start_time}", "%Y-%m-%dT%H:%M").replace(tzinfo=tz)
        end_dt   = datetime.strptime(f"{date}T{end_time}",   "%Y-%m-%dT%H:%M").replace(tzinfo=tz)
    except ValueError:
        return []

    params = {
        "eventTypeId": str(settings.calcom_event_type_id),
        "start":       start_dt.isoformat(),
        "end":         end_dt.isoformat(),
        "timeZone":    timezone_name,
        "format":      "range",
    }
    hdrs = _headers({"cal-api-version": _SLOTS_VERSION})
    # slots endpoint is public — remove auth header to avoid 401 on free accounts
    hdrs.pop("Authorization", None)

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{_V2_BASE}/slots", params=params, headers=hdrs)
        if r.status_code != 200:
            return []
        data = r.json()

    slots: list[dict[str, Any]] = []
    inner = data.get("data", {})
    if isinstance(inner, dict):
        for day_slots in inner.values():
            for slot in day_slots or []:
                if isinstance(slot, dict) and slot.get("start"):
                    slots.append({"start": slot["start"], "end": slot.get("end")})
                elif isinstance(slot, str):
                    slots.append({"start": slot, "end": None})

    slots.sort(key=lambda s: s["start"])
    return slots


async def create_booking(
    *,
    start_iso: str,
    attendee_email: str,
    attendee_name: str,
    timezone_name: str,
) -> dict[str, Any]:
    """
    Book a slot. Returns {"ok", "uid", "start", "end", "location", "status"}.
    location is the Google Meet / video conferencing link when Cal.com generates one.
    """
    if not _is_configured():
        return {"ok": False, "error": "CALCOM_EVENT_TYPE_ID not set in .env."}

    payload: dict[str, Any] = {
        "start": start_iso,
        "eventTypeId": settings.calcom_event_type_id,
        "attendee": {
            "name":     attendee_name,
            "email":    attendee_email,
            "timeZone": timezone_name,
        },
    }

    hdrs = _headers({"cal-api-version": _BOOKINGS_VERSION})

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{_V2_BASE}/bookings", json=payload, headers=hdrs)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}

    if r.status_code in (200, 201):
        booking = body.get("data", {})
        return {
            "ok":       True,
            "uid":      booking.get("uid"),
            "start":    booking.get("start"),
            "end":      booking.get("end"),
            "location": booking.get("location"),   # video conferencing URL
            "status":   booking.get("status"),
        }
    return {"ok": False, "http_status": r.status_code, "detail": body}
