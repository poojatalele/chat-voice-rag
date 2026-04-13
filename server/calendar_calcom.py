"""
Cal.com REST integration (free tier).
Requires CALCOM_API_KEY, CALCOM_USERNAME, CALCOM_EVENT_TYPE_ID in env.
Docs: https://developer.cal.com/api
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from server.config import settings


async def fetch_slots(
    days_ahead: int = 14,
    timezone_name: str = "UTC",
) -> list[dict[str, Any]]:
    """Return list of {start, end} ISO strings from Cal.com slots API."""
    if not settings.calcom_api_key or not settings.calcom_username or not settings.calcom_event_type_id:
        return []

    start = datetime.now(timezone.utc)
    end = start + timedelta(days=days_ahead)
    params = {
        "username": settings.calcom_username,
        "eventTypeId": settings.calcom_event_type_id,
        "startTime": start.isoformat().replace("+00:00", "Z"),
        "endTime": end.isoformat().replace("+00:00", "Z"),
        "timeZone": timezone_name,
    }
    headers = {"Authorization": f"Bearer {settings.calcom_api_key}"}
    url = f"{settings.calcom_base_url.rstrip('/')}/slots"
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, params=params, headers=headers)
        if r.status_code != 200:
            return []
        data = r.json()
    # Response shape varies; normalize common shapes
    slots: list[dict[str, Any]] = []
    if isinstance(data, dict):
        inner = data.get("slots") or data.get("data") or {}
        if isinstance(inner, dict):
            for day, times in inner.items():
                if isinstance(times, list):
                    for t in times:
                        if isinstance(t, str):
                            slots.append({"start": t, "end": None})
                        elif isinstance(t, dict):
                            slots.append(
                                {
                                    "start": t.get("time") or t.get("start"),
                                    "end": t.get("end"),
                                }
                            )
    return slots[:20]


async def create_booking(
    *,
    start_iso: str,
    attendee_email: str,
    attendee_name: str,
    timezone_name: str,
    notes: str = "",
) -> dict[str, Any]:
    if not settings.calcom_api_key:
        return {"ok": False, "error": "CALCOM_API_KEY not configured"}

    payload = {
        "eventTypeId": settings.calcom_event_type_id,
        "start": start_iso,
        "responses": {
            "name": attendee_name,
            "email": attendee_email,
            "notes": notes,
        },
        "timeZone": timezone_name,
        "language": "en",
        "metadata": {},
    }
    headers = {
        "Authorization": f"Bearer {settings.calcom_api_key}",
        "Content-Type": "application/json",
    }
    url = f"{settings.calcom_base_url.rstrip('/')}/bookings"
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return {"ok": r.status_code in (200, 201), "status": r.status_code, "body": body}
