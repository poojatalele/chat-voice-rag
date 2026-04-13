import { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

// ── Types ────────────────────────────────────────────────────────────────────

type Role = "user" | "assistant";
type Slot = { start: string; end?: string | null };
type Citation = {
  source: string;
  repo_name?: string | null;
  file_path: string;
  section: string;
  chunk_index?: number;
  score: number;
  snippet: string;
};
type Msg = {
  id: string;
  role: Role;
  content: string;
  citations?: Citation[];
  abstained?: boolean;
  streaming?: boolean;
  slots?: Slot[];
};
type BookStep = "window" | "slots" | "details" | "confirmed";
type Confirmation = { start: string; end?: string; location?: string; uid: string };

// ── Constants ────────────────────────────────────────────────────────────────

const apiBase = () => (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

/** 30-minute interval options for the time dropdowns */
const TIME_OPTIONS: { label: string; value: string }[] = (() => {
  const opts: { label: string; value: string }[] = [];
  for (let h = 0; h < 24; h++) {
    for (const m of [0, 30]) {
      const hh = String(h).padStart(2, "0");
      const mm = m === 0 ? "00" : "30";
      const ampm = h < 12 ? "AM" : "PM";
      const h12 = h === 0 ? 12 : h > 12 ? h - 12 : h;
      opts.push({ label: `${h12}:${mm} ${ampm}`, value: `${hh}:${mm}` });
    }
  }
  return opts;
})();

const SUGGESTED = [
  "Why are you the right fit for this role?",
  "Tell me about your Blood Report Analysis project — why CrewAI?",
  "Walk me through your resume and key achievements.",
  "What is your experience with RAG systems and LLMs?",
  "Tell me about the Reddit automation project and its tradeoffs.",
  "What are your core technical skills and tools?",
];

const TIMEZONES = [
  { label: "UTC (UTC+0)", value: "UTC" },
  { label: "New York (EST/EDT)", value: "America/New_York" },
  { label: "Chicago (CST/CDT)", value: "America/Chicago" },
  { label: "Denver (MST/MDT)", value: "America/Denver" },
  { label: "Los Angeles (PST/PDT)", value: "America/Los_Angeles" },
  { label: "London (GMT/BST)", value: "Europe/London" },
  { label: "Paris / Berlin (CET)", value: "Europe/Paris" },
  { label: "Dubai (GST)", value: "Asia/Dubai" },
  { label: "India (IST)", value: "Asia/Kolkata" },
  { label: "Singapore (SGT)", value: "Asia/Singapore" },
  { label: "Tokyo (JST)", value: "Asia/Tokyo" },
  { label: "Sydney (AEST)", value: "Australia/Sydney" },
];

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatDateTime(iso: string, tz: string) {
  try {
    return new Date(iso).toLocaleString("en-US", {
      timeZone: tz,
      weekday: "short", month: "short", day: "numeric",
      hour: "2-digit", minute: "2-digit", timeZoneName: "short",
    });
  } catch {
    return iso;
  }
}

function autoTz() {
  return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
}

function todayStr() {
  return new Date().toISOString().slice(0, 10);
}

// ── SSE streaming ────────────────────────────────────────────────────────────

async function streamChat(
  messages: { role: string; content: string }[],
  onToken: (t: string) => void,
  onSlots: (slots: Slot[]) => void,
  onDone: (p: { abstained: boolean; confidence: number; citations: Citation[] }) => void,
) {
  const res = await fetch(`${apiBase()}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });
  if (!res.ok) throw new Error(`Chat failed: ${res.status}`);
  const reader = res.body!.getReader();
  const dec = new TextDecoder();
  let carry = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    carry += dec.decode(value, { stream: true });
    const blocks = carry.split("\n\n");
    carry = blocks.pop() ?? "";
    for (const block of blocks) {
      let ev = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) ev = line.slice(6).trim();
        if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      try {
        if (ev === "token") { const j = JSON.parse(data); if (j.t) onToken(j.t); }
        if (ev === "slots") { const j = JSON.parse(data); if (j.slots) onSlots(j.slots); }
        if (ev === "done")  { onDone(JSON.parse(data)); }
      } catch { /* ignore */ }
    }
  }
}

// ── BookingModal ─────────────────────────────────────────────────────────────

function BookingModal({ onClose }: { onClose: () => void }) {
  const userTz = autoTz();
  const defaultTz = TIMEZONES.find((t) => t.value === userTz) ? userTz : "UTC";

  const [step, setStep]           = useState<BookStep>("window");
  const [date, setDate]           = useState(todayStr());
  const [startTime, setStartTime] = useState("08:00");
  const [endTime, setEndTime]     = useState("20:00");
  const [timezone, setTimezone]   = useState(defaultTz);
  const [slots, setSlots]         = useState<Slot[]>([]);
  const [selectedSlot, setSelectedSlot] = useState<Slot | null>(null);
  const [name, setName]           = useState("");
  const [email, setEmail]         = useState("");
  const [confirmation, setConfirmation] = useState<Confirmation | null>(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState<string | null>(null);

  // Step 1 → 2: fetch available slots
  const findSlots = async () => {
    setError(null);
    setLoading(true);
    try {
      const r = await fetch(`${apiBase()}/api/availability`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ date, start: startTime, end: endTime, timezone }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || "Failed to fetch slots");
      const found: Slot[] = j.slots || [];
      setSlots(found);
      setStep("slots");
      if (!found.length) setError(
        `No available slots on ${date} between ${startTime}–${endTime} (${timezone}). ` +
        `Pooja's calendar is in IST (UTC+5:30) — try a wider window or switch your timezone to IST.`
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  // Step 2 → 3: user picks a slot
  const pickSlot = (slot: Slot) => {
    setSelectedSlot(slot);
    setError(null);
    setStep("details");
  };

  // Step 3 → 4: confirm booking
  const confirmBooking = async () => {
    if (!name.trim() || !email.trim()) { setError("Name and email are required."); return; }
    if (!selectedSlot?.start) return;
    setError(null);
    setLoading(true);
    try {
      const r = await fetch(`${apiBase()}/api/book`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start_iso: selectedSlot.start,
          attendee_name: name.trim(),
          attendee_email: email.trim(),
          timezone,
        }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(JSON.stringify(j.detail || j));
      setConfirmation({ start: j.start, end: j.end, location: j.location, uid: j.uid });
      setStep("confirmed");
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    // Backdrop
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b">
          <div>
            <h2 className="text-base font-semibold text-slate-900">Book an Interview with Pooja</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              {step === "window"    && "Step 1 of 3 — Choose your availability window"}
              {step === "slots"     && "Step 2 of 3 — Pick a slot"}
              {step === "details"   && "Step 3 of 3 — Your details"}
              {step === "confirmed" && "All done!"}
            </p>
          </div>
          <button type="button" onClick={onClose} className="text-slate-400 hover:text-slate-600 text-xl leading-none">×</button>
        </div>

        <div className="p-5 space-y-4">

          {/* ── Step 1: Window ──────────────────────────────────────────── */}
          {step === "window" && (
            <>
              <div className="space-y-3">
                <label className="block">
                  <span className="text-xs font-medium text-slate-600">Date</span>
                  <input
                    type="date"
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-forest/40"
                    value={date}
                    min={todayStr()}
                    onChange={(e) => setDate(e.target.value)}
                  />
                </label>

                <div className="grid grid-cols-2 gap-3">
                  <label className="block">
                    <span className="text-xs font-medium text-slate-600">From</span>
                    <select
                      className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-forest/40 bg-white"
                      value={startTime}
                      onChange={(e) => {
                        setStartTime(e.target.value);
                        // auto-advance end if it's not after start
                        if (e.target.value >= endTime) {
                          const idx = TIME_OPTIONS.findIndex((o) => o.value === e.target.value);
                          setEndTime(TIME_OPTIONS[Math.min(idx + 4, TIME_OPTIONS.length - 1)].value);
                        }
                      }}
                    >
                      {TIME_OPTIONS.map((o) => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                  </label>
                  <label className="block">
                    <span className="text-xs font-medium text-slate-600">Until</span>
                    <select
                      className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-forest/40 bg-white"
                      value={endTime}
                      onChange={(e) => setEndTime(e.target.value)}
                    >
                      {TIME_OPTIONS.filter((o) => o.value > startTime).map((o) => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                  </label>
                </div>
                <p className="text-[11px] text-slate-400 -mt-1">
                  Times are in your selected timezone. Pooja is in IST (UTC+5:30).
                </p>

                <label className="block">
                  <span className="text-xs font-medium text-slate-600">Your timezone</span>
                  <select
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-forest/40 bg-white"
                    value={timezone}
                    onChange={(e) => setTimezone(e.target.value)}
                  >
                    {/* Prepend auto-detected if not in list */}
                    {!TIMEZONES.find((t) => t.value === userTz) && (
                      <option value={userTz}>{userTz} (detected)</option>
                    )}
                    {TIMEZONES.map((tz) => (
                      <option key={tz.value} value={tz.value}>{tz.label}</option>
                    ))}
                  </select>
                </label>
              </div>

              {error && <p className="text-xs text-red-500 bg-red-50 rounded-lg px-3 py-2">{error}</p>}

              <button
                type="button"
                onClick={() => void findSlots()}
                disabled={loading || !date}
                className="w-full rounded-xl bg-forest text-white py-2.5 text-sm font-semibold disabled:opacity-40 hover:bg-forest/90 transition-colors"
              >
                {loading ? "Finding slots…" : "Find available slots"}
              </button>
            </>
          )}

          {/* ── Step 2: Slot picker ─────────────────────────────────────── */}
          {step === "slots" && (
            <>
              {slots.length > 0 ? (
                <div className="space-y-2">
                  <p className="text-xs text-slate-500">Available on {date} in {timezone.split("/").pop()}:</p>
                  <div className="grid grid-cols-2 gap-2 max-h-64 overflow-y-auto pr-1">
                    {slots.map((s, i) => (
                      <button
                        key={i}
                        type="button"
                        onClick={() => pickSlot(s)}
                        className="rounded-xl border border-slate-200 bg-slate-50 hover:border-forest hover:bg-forest/5 px-3 py-2.5 text-sm font-medium text-slate-800 transition-colors text-left"
                      >
                        {s.start ? new Date(s.start).toLocaleTimeString("en-US", {
                          timeZone: timezone, hour: "2-digit", minute: "2-digit",
                        }) : "—"}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="rounded-xl bg-amber-50 border border-amber-200 px-4 py-3 text-sm text-amber-800">
                  {error || "No matching slots found. Try a different day or wider window."}
                </div>
              )}

              <button
                type="button"
                onClick={() => { setStep("window"); setError(null); }}
                className="w-full rounded-xl border border-slate-200 py-2 text-sm text-slate-600 hover:bg-slate-50 transition-colors"
              >
                Back — change window
              </button>
            </>
          )}

          {/* ── Step 3: Details ─────────────────────────────────────────── */}
          {step === "details" && selectedSlot && (
            <>
              <div className="rounded-xl bg-forest/5 border border-forest/20 px-4 py-3">
                <p className="text-xs text-forest font-semibold uppercase tracking-wide mb-0.5">Selected slot</p>
                <p className="text-sm font-medium text-slate-800">{formatDateTime(selectedSlot.start!, timezone)}</p>
              </div>

              <div className="space-y-3">
                <label className="block">
                  <span className="text-xs font-medium text-slate-600">Your name</span>
                  <input
                    type="text"
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-forest/40"
                    placeholder="Jane Smith"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                  />
                </label>
                <label className="block">
                  <span className="text-xs font-medium text-slate-600">Your email</span>
                  <input
                    type="email"
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-forest/40"
                    placeholder="jane@company.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                  />
                </label>
              </div>

              {error && <p className="text-xs text-red-500 bg-red-50 rounded-lg px-3 py-2">{error}</p>}

              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => { setStep("slots"); setError(null); }}
                  className="flex-1 rounded-xl border border-slate-200 py-2 text-sm text-slate-600 hover:bg-slate-50 transition-colors"
                >
                  Back
                </button>
                <button
                  type="button"
                  onClick={() => void confirmBooking()}
                  disabled={loading}
                  className="flex-1 rounded-xl bg-forest text-white py-2 text-sm font-semibold disabled:opacity-40 hover:bg-forest/90 transition-colors"
                >
                  {loading ? "Booking…" : "Confirm booking"}
                </button>
              </div>
            </>
          )}

          {/* ── Step 4: Confirmation ─────────────────────────────────────── */}
          {step === "confirmed" && confirmation && (
            <div className="space-y-4 text-center py-2">
              <div className="text-4xl">🎉</div>
              <div>
                <h3 className="text-base font-semibold text-slate-900">You're booked!</h3>
                <p className="text-sm text-slate-500 mt-1">Calendar invite sent to <span className="font-medium text-slate-700">{email}</span></p>
              </div>

              <div className="rounded-xl bg-slate-50 border border-slate-200 px-4 py-3 text-left space-y-2">
                <div>
                  <p className="text-[11px] text-slate-400 uppercase tracking-wide font-semibold">Date & time</p>
                  <p className="text-sm text-slate-800 font-medium">{formatDateTime(confirmation.start, timezone)}</p>
                </div>
                {confirmation.location && (
                  <div>
                    <p className="text-[11px] text-slate-400 uppercase tracking-wide font-semibold">Google Meet</p>
                    <a
                      href={confirmation.location}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-forest font-medium hover:underline break-all"
                    >
                      {confirmation.location}
                    </a>
                  </div>
                )}
              </div>

              <button
                type="button"
                onClick={onClose}
                className="w-full rounded-xl bg-slate-900 text-white py-2.5 text-sm font-semibold hover:bg-slate-700 transition-colors"
              >
                Close
              </button>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}

// ── SlotCards (inline in chat) ────────────────────────────────────────────────

function SlotCards({ slots, msgId }: { slots: Slot[]; msgId: string }) {
  const [name, setName]   = useState("");
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState<Record<number, string>>({});
  const tz = autoTz();

  const book = async (slot: Slot, i: number) => {
    if (!name.trim() || !email.trim()) {
      setStatus((s) => ({ ...s, [i]: "Enter your name and email first." }));
      return;
    }
    setStatus((s) => ({ ...s, [i]: "Booking…" }));
    try {
      const r = await fetch(`${apiBase()}/api/book`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ start_iso: slot.start, attendee_name: name, attendee_email: email, timezone: tz }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(JSON.stringify(j));
      setStatus((s) => ({ ...s, [i]: j.location ? `Confirmed! Meet: ${j.location}` : "Confirmed! Check your email." }));
    } catch (e) {
      setStatus((s) => ({ ...s, [i]: `Failed: ${e}` }));
    }
  };

  return (
    <div className="mt-3 space-y-2 border-t border-slate-100 pt-3">
      <div className="flex gap-2">
        <input className="flex-1 rounded-lg border border-slate-200 px-2 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-forest/40" placeholder="Your name" value={name} onChange={(e) => setName(e.target.value)} />
        <input className="flex-1 rounded-lg border border-slate-200 px-2 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-forest/40" placeholder="Your email" value={email} onChange={(e) => setEmail(e.target.value)} />
      </div>
      <div className="grid grid-cols-2 gap-2 max-h-48 overflow-y-auto">
        {slots.map((s, i) => (
          <div key={`${msgId}-${i}`} className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 flex flex-col gap-1.5">
            <span className="text-xs font-medium text-slate-700">{s.start ? new Date(s.start).toLocaleTimeString("en-US", { timeZone: tz, hour: "2-digit", minute: "2-digit", weekday: "short" }) : "—"}</span>
            {status[i] ? (
              <span className={`text-[11px] leading-tight ${status[i].startsWith("Confirmed") ? "text-emerald-600" : "text-red-500"}`}>{status[i]}</span>
            ) : (
              <button type="button" onClick={() => void book(s, i)} className="rounded-lg bg-forest text-white px-2 py-1 text-[11px] font-semibold hover:bg-forest/80 transition-colors">Book</button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput]       = useState("");
  const [busy, setBusy]         = useState(false);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [showBooking, setShowBooking] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const scrollDown = () => bottomRef.current?.scrollIntoView({ behavior: "smooth" });

  const send = async (text?: string) => {
    const msg = (text ?? input).trim();
    if (!msg || busy) return;
    setInput("");
    const userMsg: Msg = { id: crypto.randomUUID(), role: "user", content: msg };
    const asstId = crypto.randomUUID();
    setMessages((m) => [...m, userMsg, { id: asstId, role: "assistant", content: "", streaming: true }]);
    setBusy(true);
    scrollDown();
    const history = [...messages, userMsg].map((x) => ({ role: x.role, content: x.content }));
    try {
      await streamChat(
        history,
        (t) => {
          setMessages((m) => m.map((r) => r.id === asstId ? { ...r, content: r.content + t } : r));
          scrollDown();
        },
        (slots) => setMessages((m) => m.map((r) => r.id === asstId ? { ...r, slots } : r)),
        (done) => setMessages((m) => m.map((r) => r.id === asstId ? { ...r, streaming: false, citations: done.citations, abstained: done.abstained } : r)),
      );
    } catch (e) {
      setMessages((m) => m.map((r) => r.id === asstId ? { ...r, streaming: false, content: `Error: ${e}` } : r));
    } finally {
      setBusy(false);
      scrollDown();
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      {showBooking && <BookingModal onClose={() => setShowBooking(false)} />}

      {/* Header */}
      <header className="border-b bg-white px-5 py-3 flex flex-wrap items-center justify-between gap-2 shadow-sm sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-full bg-forest flex items-center justify-center text-white font-bold text-sm select-none">P</div>
          <div>
            <h1 className="text-base font-semibold text-slate-900 leading-tight">Pooja Talele · AI Representative</h1>
            <p className="text-[11px] text-slate-500">RAG-grounded · Resume + GitHub · No hallucination</p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => setShowBooking(true)}
          className="rounded-xl bg-forest text-white px-4 py-2 text-sm font-semibold hover:bg-forest/90 transition-colors shadow-sm"
        >
          Book a Call
        </button>
      </header>

      {/* Chat */}
      <main className="flex-1 overflow-y-auto p-4 space-y-4 max-w-3xl mx-auto w-full pb-28">
        {messages.length === 0 && (
          <div className="space-y-4 pt-4">
            <div className="rounded-xl border border-dashed border-slate-300 bg-white p-6 text-center text-slate-500 text-sm">
              Ask anything about Pooja's background, projects, or fit.
              <br />
              <span className="text-xs text-slate-400">All answers grounded in her actual resume. Use "Book a Call" to schedule.</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {SUGGESTED.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => void send(q)}
                  disabled={busy}
                  className="text-left rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700 hover:border-forest hover:bg-forest/5 transition-colors shadow-sm disabled:opacity-40"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m) => (
          <div key={m.id} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-[85%] rounded-2xl px-4 py-3 shadow-sm ${
              m.role === "user"
                ? "bg-forest text-white rounded-br-sm"
                : "bg-white border border-slate-200 text-slate-800 rounded-bl-sm"
            }`}>
              <div className="text-[11px] font-medium opacity-60 mb-1">{m.role === "user" ? "You" : "Pooja's AI"}</div>

              {m.role === "assistant" ? (
                <div className="prose prose-sm max-w-none prose-p:my-1">
                  <ReactMarkdown>{m.content || (m.streaming ? "…" : "")}</ReactMarkdown>
                </div>
              ) : (
                <p className="whitespace-pre-wrap text-sm">{m.content}</p>
              )}

              {/* Inline slot cards from tool calling */}
              {m.slots && m.slots.length > 0 && <SlotCards slots={m.slots} msgId={m.id} />}

              {/* Citations */}
              {m.role === "assistant" && !m.streaming && (
                <div className="mt-2 pt-2 border-t border-slate-100">
                  <button
                    type="button"
                    className="text-[11px] font-medium text-slate-400 hover:text-forest transition-colors"
                    onClick={() => setExpanded((e) => ({ ...e, [m.id]: !e[m.id] }))}
                  >
                    {expanded[m.id] ? "Hide sources" : "Sources"}
                    {m.abstained ? " (no match)" : m.citations?.length ? ` (${m.citations.length})` : ""}
                  </button>
                  {expanded[m.id] && (
                    <ul className="mt-2 space-y-2 text-xs text-slate-600">
                      {m.abstained || !m.citations?.length ? (
                        <li className="text-slate-400 italic">No citations — not grounded in retrieved materials.</li>
                      ) : m.citations.map((c, i) => (
                        <li key={i} className="rounded-lg bg-slate-50 p-2 border border-slate-100">
                          <div className="font-mono text-[10px] text-slate-400">
                            {c.source}{c.repo_name ? ` · ${c.repo_name}` : ""} · {c.file_path} · score {c.score}
                          </div>
                          <div className="mt-1 text-slate-600 leading-relaxed">{c.snippet}</div>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </main>

      {/* Floating input */}
      <div className="fixed bottom-0 left-0 right-0 bg-white border-t shadow-lg p-3">
        <div className="max-w-3xl mx-auto flex gap-2 items-end">
          <textarea
            className="flex-1 min-h-[44px] max-h-36 rounded-xl border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-forest/40 resize-none"
            placeholder="Ask about background, projects, or availability…"
            value={input}
            disabled={busy}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void send(); } }}
          />
          <button
            type="button"
            onClick={() => void send()}
            disabled={busy || !input.trim()}
            className="rounded-xl bg-forest px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-40 hover:bg-forest/90 transition-colors"
          >
            {busy ? "…" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
