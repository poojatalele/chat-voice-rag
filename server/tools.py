from __future__ import annotations

import json
from server import calendar_calcom

# Tool definitions for Groq function calling
GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_availability",
            "description": "Fetch Pooja's available interview slots from her Cal.com calendar for the next 14 days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone name e.g. 'America/New_York'. Defaults to UTC.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_booking",
            "description": "Book a specific interview slot with Pooja on Cal.com.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_iso": {
                        "type": "string",
                        "description": "ISO 8601 datetime of the slot to book (from get_availability results).",
                    },
                    "attendee_name": {
                        "type": "string",
                        "description": "Full name of the person booking the interview.",
                    },
                    "attendee_email": {
                        "type": "string",
                        "description": "Email address of the person booking.",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone name.",
                    },
                },
                "required": ["start_iso", "attendee_name", "attendee_email"],
            },
        },
    },
]

SCHEDULING_KEYWORDS = {
    "schedule", "availability", "available", "book", "booking",
    "interview", "meeting", "slot", "slots", "calendar", "free",
    "when", "time", "reserve", "appointment",
}


def has_scheduling_intent(text: str) -> bool:
    words = set(text.lower().split())
    return bool(words & SCHEDULING_KEYWORDS)


async def execute_tool(name: str, args: dict) -> tuple[str, list | None]:
    """Execute a tool call. Returns (json_result, slots_list_or_None)."""
    if name == "get_availability":
        slots = await calendar_calcom.fetch_slots(
            days_ahead=14,
            timezone_name=args.get("timezone", "UTC"),
        )
        if not slots:
            result = {
                "slots": [],
                "message": "No slots found. Cal.com calendar is not configured on this server.",
            }
            return json.dumps(result), []
        return json.dumps({"slots": slots}), slots

    if name == "create_booking":
        result = await calendar_calcom.create_booking(
            start_iso=args["start_iso"],
            attendee_name=args["attendee_name"],
            attendee_email=args["attendee_email"],
            timezone_name=args.get("timezone", "UTC"),
        )
        return json.dumps(result), None

    return json.dumps({"error": f"Unknown tool: {name}"}), None
