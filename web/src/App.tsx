import { useCallback, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

type Role = "user" | "assistant";

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
};

const apiBase = () => (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

async function streamChat(
  messages: { role: string; content: string }[],
  onToken: (t: string) => void,
  onDone: (payload: { abstained: boolean; confidence: number; citations: Citation[] }) => void
) {
  const res = await fetch(`${apiBase()}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });
  if (!res.ok) throw new Error(`Chat failed: ${res.status}`);
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No body");
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
      const lines = block.split("\n");
      let data = "";
      for (const line of lines) {
        if (line.startsWith("event:")) ev = line.slice(6).trim();
        if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (ev === "token" && data) {
        try {
          const j = JSON.parse(data);
          if (j.t) onToken(j.t);
        } catch {
          /* ignore */
        }
      }
      if (ev === "done" && data) {
        const j = JSON.parse(data);
        onDone({
          abstained: !!j.abstained,
          confidence: j.confidence ?? 0,
          citations: j.citations || [],
        });
      }
    }
  }
}

export default function App() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [tz, setTz] = useState(Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC");
  const [slots, setSlots] = useState<{ start?: string; end?: string | null }[]>([]);
  const [slotLoading, setSlotLoading] = useState(false);
  const [bookEmail, setBookEmail] = useState("");
  const [bookName, setBookName] = useState("");
  const [bookStatus, setBookStatus] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const scrollDown = () => bottomRef.current?.scrollIntoView({ behavior: "smooth" });

  const loadSlots = useCallback(async () => {
    setSlotLoading(true);
    setBookStatus(null);
    try {
      const r = await fetch(`${apiBase()}/api/calendar/availability`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ timezone: tz, days_ahead: 14 }),
      });
      const j = await r.json();
      setSlots(j.slots || []);
      if (!j.slots?.length) setBookStatus("No slots returned — configure Cal.com env on API.");
    } catch (e) {
      setBookStatus(String(e));
    } finally {
      setSlotLoading(false);
    }
  }, [tz]);

  const bookSlot = async (start_iso: string) => {
    if (!bookEmail || !bookName) {
      setBookStatus("Enter name and email first.");
      return;
    }
    setBookStatus("Booking…");
    try {
      const r = await fetch(`${apiBase()}/api/calendar/book`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          start_iso,
          attendee_email: bookEmail,
          attendee_name: bookName,
          timezone: tz,
          notes: "Booked via AI Persona web",
        }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(JSON.stringify(j));
      setBookStatus("Booking confirmed. Check email / Cal.com dashboard.");
    } catch (e) {
      setBookStatus(`Booking failed: ${e}`);
    }
  };

  const send = async () => {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    const userMsg: Msg = { id: crypto.randomUUID(), role: "user", content: text };
    const asstId = crypto.randomUUID();
    setMessages((m) => [...m, userMsg, { id: asstId, role: "assistant", content: "", streaming: true }]);
    setBusy(true);
    scrollDown();
    const history = [...messages, userMsg].map((x) => ({ role: x.role, content: x.content }));
    try {
      await streamChat(
        history,
        (t) => {
          setMessages((m) =>
            m.map((row) => (row.id === asstId ? { ...row, content: row.content + t } : row))
          );
          scrollDown();
        },
        (done) => {
          setMessages((m) =>
            m.map((row) =>
              row.id === asstId
                ? {
                    ...row,
                    streaming: false,
                    citations: done.citations,
                    abstained: done.abstained,
                  }
                : row
            )
          );
        }
      );
    } catch (e) {
      setMessages((m) =>
        m.map((row) =>
          row.id === asstId
            ? { ...row, streaming: false, content: `Error: ${e}` }
            : row
        )
      );
    } finally {
      setBusy(false);
      scrollDown();
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b bg-white px-4 py-3 flex flex-wrap items-center justify-between gap-2 shadow-sm">
        <div className="flex items-center gap-2">
          <h1 className="text-lg font-semibold text-forest">AI Representative</h1>
          <span className="text-xs rounded-full bg-forest-light text-forest-dark px-2 py-0.5 font-medium">
            Knowledge: Resume + GitHub (RAG)
          </span>
        </div>
        <p className="text-xs text-slate-500 max-w-md text-right">
          Grounded answers only. API: <code className="bg-slate-100 px-1 rounded">{apiBase() || "(set VITE_API_BASE_URL)"}</code>
        </p>
      </header>

      <div className="flex flex-1 flex-col lg:flex-row max-w-[1600px] mx-auto w-full">
        <main className="flex-1 flex flex-col border-r border-slate-200 bg-slate-50/80 min-h-[50vh]">
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 && (
              <div className="rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center text-slate-500">
                Ask about experience, projects, or fit. Citations appear under assistant replies when grounded.
              </div>
            )}
            {messages.map((m) => (
              <div key={m.id} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 shadow-sm ${
                    m.role === "user"
                      ? "bg-forest text-white rounded-br-md"
                      : "bg-white border border-slate-200 text-slate-800 rounded-bl-md"
                  }`}
                >
                  <div className="text-xs opacity-70 mb-1">{m.role === "user" ? "You" : "Assistant"}</div>
                  {m.role === "assistant" ? (
                    <div className="prose prose-sm max-w-none prose-p:my-1">
                      <ReactMarkdown>{m.content || (m.streaming ? "…" : "")}</ReactMarkdown>
                    </div>
                  ) : (
                    <p className="whitespace-pre-wrap">{m.content}</p>
                  )}
                  {m.role === "assistant" && !m.streaming && (
                    <div className="mt-3 border-t border-slate-100 pt-2">
                      <button
                        type="button"
                        className="text-xs font-medium text-forest hover:underline"
                        onClick={() => setExpanded((e) => ({ ...e, [m.id]: !e[m.id] }))}
                      >
                        {expanded[m.id] ? "Hide sources" : "Sources"}
                        {m.abstained ? " (abstained)" : m.citations?.length ? ` (${m.citations.length})` : ""}
                      </button>
                      {expanded[m.id] && (
                        <ul className="mt-2 space-y-2 text-xs text-slate-600">
                          {m.abstained || !m.citations?.length ? (
                            <li>No citations — answer not grounded in retrieved materials.</li>
                          ) : (
                            m.citations.map((c, i) => (
                              <li key={i} className="rounded-lg bg-slate-50 p-2 border border-slate-100">
                                <div className="font-mono text-[11px] text-slate-500">
                                  {c.source} {c.repo_name ? `· ${c.repo_name}` : ""} · {c.file_path} · score{" "}
                                  {c.score}
                                </div>
                                <div className="mt-1 text-slate-700">{c.snippet}</div>
                              </li>
                            ))
                          )}
                        </ul>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>

          <div className="border-t bg-white p-3 space-y-2">
            <div className="flex gap-2">
              <textarea
                className="flex-1 min-h-[48px] max-h-40 rounded-xl border border-slate-200 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-forest/40"
                placeholder="Ask about background, GitHub projects, or scheduling…"
                value={input}
                disabled={busy}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void send();
                  }
                }}
              />
              <button
                type="button"
                onClick={() => void send()}
                disabled={busy}
                className="self-end rounded-xl bg-forest px-5 py-2 text-sm font-semibold text-white disabled:opacity-50"
              >
                Send
              </button>
            </div>
          </div>
        </main>

        <aside className="w-full lg:w-[380px] bg-white p-4 space-y-4 border-t lg:border-t-0 lg:border-l border-slate-200">
          <h2 className="text-sm font-semibold text-slate-800">Schedule interview</h2>
          <label className="block text-xs text-slate-500">
            IANA timezone
            <input
              className="mt-1 w-full rounded-lg border border-slate-200 px-2 py-1 text-sm"
              value={tz}
              onChange={(e) => setTz(e.target.value)}
            />
          </label>
          <button
            type="button"
            onClick={() => void loadSlots()}
            disabled={slotLoading}
            className="w-full rounded-lg bg-slate-800 text-white py-2 text-sm font-medium disabled:opacity-50"
          >
            {slotLoading ? "Loading…" : "Load availability"}
          </button>
          <div className="space-y-2">
            <input
              className="w-full rounded-lg border border-slate-200 px-2 py-1 text-sm"
              placeholder="Your name"
              value={bookName}
              onChange={(e) => setBookName(e.target.value)}
            />
            <input
              className="w-full rounded-lg border border-slate-200 px-2 py-1 text-sm"
              placeholder="Your email"
              value={bookEmail}
              onChange={(e) => setBookEmail(e.target.value)}
            />
          </div>
          <div className="max-h-64 overflow-y-auto space-y-2">
            {slots.map((s, i) => (
              <div
                key={i}
                className="rounded-lg border border-slate-200 p-2 text-xs flex justify-between items-center gap-2"
              >
                <span className="font-mono break-all">{s.start || JSON.stringify(s)}</span>
                {s.start && (
                  <button
                    type="button"
                    className="shrink-0 rounded bg-forest text-white px-2 py-1 text-[11px]"
                    onClick={() => void bookSlot(s.start!)}
                  >
                    Book
                  </button>
                )}
              </div>
            ))}
          </div>
          {bookStatus && <p className="text-xs text-slate-600">{bookStatus}</p>}
        </aside>
      </div>
    </div>
  );
}
