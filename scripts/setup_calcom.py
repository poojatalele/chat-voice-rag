#!/usr/bin/env python3
"""
One-time Cal.com OAuth setup.

Run once to:
  1. Create a managed user (Pooja) under your OAuth client
  2. Create a 30-min "Interview" event type for that user
  3. Print the env vars to paste into .env

Usage:
    python scripts/setup_calcom.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from server.config import settings

V2 = "https://api.cal.com/v2"


def die(msg: str, resp: httpx.Response | None = None) -> None:
    print(f"\n[ERROR] {msg}")
    if resp is not None:
        print(f"  Status: {resp.status_code}")
        try:
            print(f"  Body:   {json.dumps(resp.json(), indent=2)}")
        except Exception:
            print(f"  Body:   {resp.text}")
    sys.exit(1)


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val or default


def main() -> None:
    print("=" * 60)
    print("Cal.com OAuth Setup")
    print("=" * 60)

    client_id     = settings.calcom_client_id     or ask("Cal.com OAuth Client ID")
    client_secret = settings.calcom_client_secret or ask("Cal.com OAuth Client Secret")

    if not client_id or not client_secret:
        die("CALCOM_CLIENT_ID and CALCOM_CLIENT_SECRET are required in .env or entered above.")

    platform_headers = {"x-cal-secret-key": client_secret}

    # ── Step 1: Create managed user ──────────────────────────────────────────
    print("\n[1/3] Creating managed user (Pooja)…")

    email = ask("Pooja's email (for Cal.com managed user)", "pooja@example.com")
    name  = ask("Pooja's display name", "Pooja Talele")

    with httpx.Client(timeout=30) as client:
        r = client.post(
            f"{V2}/oauth-clients/{client_id}/users",
            headers=platform_headers,
            json={
                "email":      email,
                "name":       name,
                "timeZone":   "Asia/Kolkata",
                "timeFormat": 12,
                "weekStart":  "Monday",
            },
        )

    if r.status_code not in (200, 201):
        die("Failed to create managed user", r)

    user_data     = r.json()["data"]
    user_id       = str(user_data["id"])
    access_token  = user_data["accessToken"]
    refresh_token = user_data["refreshToken"]
    print(f"  Managed user created. ID: {user_id}")

    # ── Step 2: Create event type ─────────────────────────────────────────────
    print("\n[2/3] Creating 'Interview with Pooja' event type (30 min)…")

    user_headers = {
        "Authorization":   f"Bearer {access_token}",
        "cal-api-version": "2024-06-14",
        "Content-Type":    "application/json",
    }

    with httpx.Client(timeout=30) as client:
        r = client.post(
            f"{V2}/event-types",
            headers=user_headers,
            json={
                "title":           "Interview with Pooja",
                "slug":            "interview",
                "lengthInMinutes": 30,
                "description":     "30-minute interview / screening call with Pooja Talele.",
            },
        )

    if r.status_code not in (200, 201):
        die("Failed to create event type", r)

    event_data     = r.json()["data"]
    event_type_id  = event_data["id"]
    booking_url    = event_data.get("bookingUrl", "")
    print(f"  Event type created. ID: {event_type_id}")
    if booking_url:
        print(f"  Booking URL: {booking_url}")

    # ── Step 3: Print env vars ────────────────────────────────────────────────
    print("\n[3/3] Done! Add these to your .env and Render environment:\n")
    print("─" * 60)
    print(f"CALCOM_CLIENT_ID={client_id}")
    print(f"CALCOM_CLIENT_SECRET={client_secret}")
    print(f"CALCOM_USER_ID={user_id}")
    print(f"CALCOM_ACCESS_TOKEN={access_token}")
    print(f"CALCOM_REFRESH_TOKEN={refresh_token}")
    print(f"CALCOM_EVENT_TYPE_ID={event_type_id}")
    print("─" * 60)
    print("""
Note:
  - Access tokens expire in 60 min; the app refreshes them automatically.
  - If you restart the server, the access token in .env is used as the
    initial token and refreshed on first expiry — no action needed.
  - Do NOT re-run this script (it will create a duplicate managed user).
    If you need to re-run, delete the managed user from Cal.com first.
""")


if __name__ == "__main__":
    main()
