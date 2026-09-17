"""Token-speed and latency benchmark for the LLM translation endpoint.

Targets an OpenAI-compatible /v1 endpoint (vLLM / llama.cpp / ollama shim)
exactly like the live `engine/llm.py` LlmStage: streamed SSE, sentence-sized
chunks, one request per speech window. In the live pipeline the gate releases
text to the LLM once `gate.release_words` (5) source words accumulate or the
speaker pauses (`gate.gap_reset_s`), so each request approximates a short
speech window rather than one request per ASR sentence utterance.

Usage:
    .venv/bin/python tools/bench_llm.py [max_windows] [seconds_per_window] [runs] [target_language]

Metrics per utterance (then aggregated):
    ttft       time to first token
    first_chunk  time to first sentence-sized ASSISTANT_CHUNK
    tok/s      streamed tokens / streaming time
    total      streamed tokens + wall time for the utterance
"""
from __future__ import annotations

import asyncio
import json
import logging
import statistics
import sys
import time
from datetime import datetime, timezone

import httpx

logging.disable(logging.CRITICAL)
sys.path.insert(0, ".")
from engine.llm import _token_text  # noqa: E402

BASE_URL = "http://xavier:8081/v1"
MODEL = "Qwen/Qwen3-8B-AWQ"
TEMPERATURE = 0.3
MAX_TOKENS = 256
TIMEOUT = 120
# Input side: how long of a speech window each translation request covers
# (the live gate releases after ~release_words words or a pause; this is the
# same idea as a character-budget window for the benchmark).
WINDOW_SECONDS = 5.0
# Prior source window and its translation, fed to the model so it can
# continue the translation seamlessly.
CONTEXT_WINDOWS = 1
# English conversational speech ~150 wpm -> ~15 chars/s, incl. spaces.
SPEECH_CHARS_PER_SEC = 15.0
# Output side: sentence-chunk size used when flushing streamed tokens,
# matching the live pipeline's _CHUNK_MAX_CHARS.
MAX_CHUNK_CHARS = 180

_SENTENCE_END = ".!?\n"

SYSTEM = (
    "Bạn là một phiên dịch viên cabin. Dịch tiếp câu sau sao cho tự nhiên "
    "và khớp với ngữ điệu, chỉ viết phần bổ sung thêm, không markdown."
)

# Full source transcript (headers/speaker labels already stripped, matching
# what the ASR stage would actually emit for translation).
SOURCE = {
    "title": "UN Security Council 9/11 anniversary meeting",
    "text": (
        "Colleagues, if you don't mind, we're going to wait. Colleagues, we're going to wait "
        "five minutes, if you don't mind. Five minutes, can we wait? Excellence, colleagues, "
        "ladies and gentlemen, 25 years ago, on the 11th of September 2001, Pardon me. The "
        "10,221st meeting of the Security Council is called to order. Twenty-five years ago, "
        "on the 11th of September 2001, one of the deadliest terrorist attacks struck the "
        "United States. Thousands of people were killed. And here in New York, the Twin Towers "
        "were destroyed. and more than 3,000 people of 19 nationalities lost their lives. At "
        "the outset of this meeting, I invite all Security Council members and those present "
        "in the Chamber to stand and observe a minute of silence to honour with the city of "
        "New York and the United States, and to honour all victims of terrorism and their "
        "families, and to honour all victims of terrorism. Thank you. The provisional agenda "
        "for this meeting is threats to international peace and security caused by terrorist "
        "acts. The agenda is adopted. In accordance with rule 39 of the Council's provisional "
        "rules of procedure, I invite the following briefers to participate in this meeting: "
        "Ms. Natalia German, Executive Director, Counter-Terrorism Committee Executive "
        "Directorate; Mr. Ehab Omaish, Director for Policy and Coordination, United Nations "
        "Office of Counter-Terrorism; And Ms. Noorin Chowdhury-Fink, Executive Director, the "
        "Global Internet Forum to Counter Terrorism. It is so decided. The Security Council "
        "will now begin its consideration of item 2 of the agenda. The Council has the text of "
        "the presidential statement. I now give the floor to Ms. Natalia German. "
        "Mr. President, Excellencies, Distinguished members of the Council, At the outset, I "
        "would like to reiterate our solidarity with the survivors and the families of those "
        "who perished in the 9/11 attacks as well as our sympathies for all victims of "
        "terrorist attacks worldwide. In 2001, the Security Council's response was swift and "
        "united. Acting under Chapter VII of the United Nations Charter, the Council adopted a "
        "series of obligations in Resolution 1373. They are as relevant today as they were 25 "
        "years ago. At their heart is solidarity and international cooperation in acting to "
        "prevent terrorism, bring terrorists to justice, and to ensure they cannot find safe "
        "haven. Both before 9/11 and since, the Security Council has unanimously condemned "
        "terrorism in all its forms and manifestations. It continues to provide a strong and "
        "collective voice against all acts, methods, and practices of terrorism as criminal "
        "and unjustifiable, regardless of their motivation. Since the adoption of Resolution "
        "1373, the Council has passed more than 20 additional resolutions on "
        "counter-terrorism. They include landmark decisions addressing terrorist incitement "
        "and recruitment, countering violent extremism conducive to terrorism, taking action "
        "to impede terrorist travel and strengthen borders, and measures to prevent financial "
        "flows to terrorists. At the same time, the Council has been consistent in reminding "
        "States that measures taken to counter terrorism must comply with their obligations "
        "under international law, including human rights, humanitarian and refugee law. At "
        "the institutional level, the Council created the Counter-Terrorism Committee to "
        "assist Member States to achieve the full implementation of resolution 1373 and its "
        "successors. To support the Committee, the Council established the "
        "Counter-Terrorism Committee Executive Directorate, or CTED. CTED provides neutral, "
        "independent and expert assessments of States' counter-terrorism measures on behalf "
        "of the Counter-Terrorism Committee. Our assessment reports contain recommendations "
        "to States as well as suggestions for technical assistance delivery by our global "
        "Counter-Terrorism Compact partners. And since 2001, Member States have made real "
        "progress in implementing the resolutions of the Security Council, and yet much "
        "remains to be done. As terrorism evolves, States' responses must adapt. The work of "
        "this Council and its Counter-Terrorism Committee remains critical in supporting "
        "Member States to stay abreast of the threat and to update their legislation, "
        "institutions, methodologies and practices. Excellencies, I wish to make a few "
        "observations on how the terrorism threat has evolved in recent years. terrorism has "
        "become more fragmented and diffuse. Terrorist groups adapt their structures, "
        "decentralize operations, diversify financial streams, and take advantage of "
        "instability to sustain their influence. They continue to exploit fragile governance, "
        "competition over resources, and local grievances to expand their reach. They "
        "skillfully tailor their messages and recruitment strategies to the specific "
        "vulnerabilities of men and women. Terrorist groups and organized criminal networks "
        "are also using maritime routes and critical infrastructure to facilitate the "
        "movement of individuals, weapons, and illicit goods. At the same time, CTED has "
        "identified a growing role of decentralized cells, loosely connected networks, and "
        "lone actors whose pathways to violence are often less visible and more difficult to "
        "disrupt. A second observation is that technology is having an accelerating impact on "
        "the threat environment. Terrorist groups are using new technologies not only for "
        "propaganda and recruitment, but also for attack planning, intelligence gathering, "
        "and financing. In addition, unmanned aircraft systems have become more accessible, "
        "affordable, and effective. Terrorist actors are employing these systems for "
        "surveillance, reconnaissance, and attacks. A third observation demands particular "
        "attention. The recruitment and exploitation of children and youth is emerging as a "
        "pressing challenge facing the counter-terrorist community. Terrorist and violent "
        "extremist actors are exploiting social media platforms, gaming environments, and "
        "other virtual spaces to identify vulnerabilities and recruit individuals, especially "
        "young people. Advances in artificial intelligence and automated content generation "
        "are further increasing the speed, scale and precision. A fourth observation concerns "
        "terrorist financing. Financing networks are becoming more decentralized and "
        "difficult to detect. Traditional methods such as cash couriers and hawala-like "
        "systems remain widely used. And at the same time, terrorist actors are increasingly "
        "exploiting online payment services, crowdfunding platforms, mobile money systems, "
        "and virtual assets. Many groups continue to generate revenue through criminal "
        "activity, including extortion, kidnapping for ransom, trafficking, and the "
        "exploitation of natural resources. The nexus between terrorism and organized crime "
        "remains a prominent feature of several contemporary conflict environments. These "
        "trends have been highlighted in recent reports, including the Financial Action Task "
        "Force 2025 Comprehensive Update on Terrorism Financing. CETA was very pleased to "
        "co-lead that report with the French Treasury. It underscores the need to continue "
        "enhancing the understanding of terrorism financing risks associated with new and "
        "emerging financial technologies and fundraising methods as the first and critical "
        "step for developing appropriate responses. Mr. President, on this day, when we "
        "remember the victims of 9/11 and of all terrorist attacks, We owe them our continued "
        "commitment to prevention and preparedness, and to international cooperation and "
        "solidarity in the fight against terrorism, firmly grounded in human rights and the "
        "rule of law. In the 25 years since 9/11, the Security Council has maintained a "
        "strong sense of unity and common purpose in the fight against terrorism, and I am "
        "confident that the Council and its Counter-Terrorism Committee will continue to "
        "demonstrate their resolve to anticipate and respond to the evolving terrorist "
        "threat. For its part, CTED remains committed to deliver on its mandate to assist "
        "Member States to implement their obligations under the relevant Security Council "
        "resolutions and advance our common objective of a world free from terrorism. I thank "
        "you. "
        "I thank Ms. Kerman for her briefing. I now give the floor to Mr. Ayad Omesh. "
        "Thank you, Mr. President. Twenty-five years ago, A cowardly terrorist act cut short "
        "the lives of almost 3,000 people here in the United States. It left many more "
        "injured and devastated countless families and communities around the world, claiming "
        "victims from 90 countries. The unimaginable horror of that day terrorized the whole "
        "human family. In minutes, it drastically changed the course of history, making the "
        "world less safe for all people. Today, we honor the memory of those who lost their "
        "lives, and we stand with the survivors. We extend our deepest sympathies to the "
        "victims, their loved ones, the first responders, the people of the United States, "
        "and all communities affected, including here in New York City, home to the United "
        "Nations. As we mark this solemn anniversary, we remember all victims of terrorism "
        "around the world. Their experiences remind us of the profound human cost of "
        "terrorism and our collective responsibility to prevent such suffering. The voices "
        "and stories of survivors of loss, fear, resolve, and hope must remain at the heart "
        "of our efforts. Today is a moment of remembrance, but also an opportunity to take "
        "stock. A quarter century of sustained international mobilization has delivered "
        "significant progress in countering terrorism, thanks to the impetus provided by the "
        "Security Council. These efforts are still needed. The United Nations remain an "
        "indispensable forum. Here, in the very city that suffered a devastating terrorist "
        "attack, Member States can come together to prevent and counter terrorism in all of "
        "its forms and manifestations. The attack of September 11th exposed the profound "
        "danger posed by a single terrorist organization operating across borders. Today, "
        "the threat is more diffuse, more decentralized, and increasingly enabled by new "
        "technologies. New terrorist tactics are attracting a diverse range of individuals "
        "and, worryingly, increasingly young people. Terrorist groups exploit instability "
        "and organized crime networks, leverage online platforms for incitement and "
        "recruitment, and misuse emerging technologies, including artificial intelligence. "
        "However, opportunities to use new technologies to counter terrorism in full respect "
        "of international law have also developed. Over the past 25 years, Member States "
        "have built a comprehensive international counterterrorism framework. The United "
        "Nations and this Council have played a central role by focusing national efforts on "
        "gaps and new trends, fostering cooperation, and enabling capacity building that "
        "responds to local needs. Three lessons stand out. First, security measures remain "
        "indispensable, but they must be accompanied by proactive and sustained prevention "
        "efforts to address the conditions terrorists exploit to radicalize, incite, and "
        "recruit people to violence. Second, international cooperation and whole of society "
        "partnerships are essential. Communities, civil society, women, youth, and victims "
        "of terrorism all have a role to play in building resilience against violent "
        "extremist narratives and affirming that acts of terrorism can never be justified. "
        "Third, effective counterterrorism and respect for international law are mutually "
        "reinforcing. Measures that undermine the rule of law and human rights ultimately "
        "weaken our collective efforts. Mr. President, much has changed since September "
        "2001. Terrorist groups have lost leaders and territory. International cooperation "
        "has deepened. Member states have strengthened legal frameworks, institutional "
        "capacity, and mechanisms for disrupting terrorist networks. These are significant "
        "achievements, but they are not grounds for complacency. Terrorist actors have "
        "adapted their tactics, and our collective response must keep pace. The tragedy of "
        "September 11th has changed countless lives and reshaped the international security "
        "landscape. Let us carry forward the lessons of the past 25 years with determination, "
        "principled action, and solidarity. The continued leadership of the Security Council "
        "is critical in this regard. The United Nations Office of Counter-Terrorism remains "
        "committed to supporting this Council and all Member States, working with our "
        "partners across the United Nations system to translate that resolve into collective "
        "action. We are committed to building a future free from terrorism. Thank you, Mr. "
        "President."
    ),
}


def _split_sentences(text: str):
    """Split into sentence-sized units, preserving the boundary punctuation."""
    buf = ""
    for ch in text:
        buf += ch
        if ch in _SENTENCE_END:
            s = buf.strip()
            if s:
                yield s
            buf = ""
    s = buf.strip()
    if s:
        yield s


def split_windows(text: str, seconds: float = WINDOW_SECONDS,
                  chars_per_sec: float = SPEECH_CHARS_PER_SEC) -> list[str]:
    """Split the transcript into rolling speech windows like the live gate:
    each window is the text spoken in `seconds` of speech (char budget =
    seconds * chars_per_sec), cut at the nearest sentence boundary. A single
    sentence longer than a window is split mid-sentence at the budget."""
    budget = int(seconds * chars_per_sec)
    out = []
    group = []
    size = 0
    for sent in _split_sentences(text):
        if not sent:
            continue
        # long single sentence split at window budget
        while len(sent) > budget:
            if group:
                out.append(" ".join(group))
                group = []
                size = 0
            cut = sent.rfind(" ", 0, budget)
            if cut < budget // 2:
                cut = budget
            out.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if not sent:
            continue
        if group and size + len(sent) > budget:
            out.append(" ".join(group))
            group = []
            size = 0
        group.append(sent)
        size += len(sent) + 1
    if group:
        out.append(" ".join(group))
    return out


def _ends_sentence(buf: str, max_chars: int) -> bool:
    return buf.rstrip().endswith(tuple(_SENTENCE_END)) or len(buf) >= max_chars


async def bench_utterance(client: httpx.AsyncClient, messages: list[dict], target_language: str) -> dict:
    payload = {
        "model": MODEL,
        "stream": True,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM + f" Ngôn ngữ cần dịch đến: {target_language}."},
            *messages,
        ],
    }
    t_start = time.perf_counter()
    ttft = None
    first_chunk = None
    buf = ""
    out = []
    n_tokens = 0
    streaming_t0 = None
    async with client.stream("POST", "/chat/completions", json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.aiter_lines():
            if not raw.startswith("data: "):
                continue
            data = raw[len("data: "):]
            if data.strip() == "[DONE]":
                break
            tok = _token_text(data)
            if not tok:
                continue
            if streaming_t0 is None:
                streaming_t0 = time.perf_counter()
            if ttft is None:
                ttft = (time.perf_counter() - t_start) * 1000
            n_tokens += 1
            buf += tok
            out.append(tok)
            if first_chunk is None and _ends_sentence(buf, MAX_CHUNK_CHARS):
                first_chunk = (time.perf_counter() - t_start) * 1000
                buf = ""
    wall = (time.perf_counter() - t_start) * 1000
    if first_chunk is None and buf:
        first_chunk = wall
    streaming_ms = (time.perf_counter() - t_start) * 1000 if streaming_t0 is None else \
        (time.perf_counter() - streaming_t0) * 1000
    tok_s = (n_tokens / streaming_ms * 1000) if streaming_ms > 0 else float("nan")
    return {
        "ttft": ttft,
        "first_chunk": first_chunk,
        "n_tokens": n_tokens,
        "wall": wall,
        "tok_s": tok_s,
        "out": "".join(out),
    }


def _pct(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    return s[f] + (s[c] - s[f]) * (k - f)


def _summarize(vals):
    if not vals:
        return (float("nan"),) * 4
    return (statistics.mean(vals), _pct(vals, 50), _pct(vals, 95), max(vals))


def _fmt_row(label, vals, unit="ms"):
    m, p50, p95, mx = _summarize(vals)
    print("  %-18s %10.1f %10.1f %10.1f %10.1f %s" % (label, m, p50, p95, mx, unit))


async def main():
    max_windows = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    seconds_per_window = float(sys.argv[2]) if len(sys.argv) > 2 else WINDOW_SECONDS
    n_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    target_language = sys.argv[4] if len(sys.argv) > 4 else "tiếng Việt"

    windows = split_windows(SOURCE["text"], seconds_per_window)
    if max_windows and max_windows < len(windows):
        windows = windows[:max_windows]

    system = SYSTEM + f" Ngôn ngữ cần dịch đến: {target_language}."

    def build_messages(prev, u):
        msgs = [{"role": "system", "content": system}]
        if prev is not None:
            msgs.append({"role": "user", "content": prev[0]})
            if prev[1]:
                msgs.append({"role": "assistant", "content": prev[1]})
        msgs.append({"role": "user", "content": u})
        return msgs

    log_path = "bench_llm.log"
    logf = open(log_path, "a", encoding="utf-8")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    logf.write("\n===== run %s | %s -> %s | model=%s | windows=%d x %.1fs | ctx=%d =====\n"
               % (stamp, "en", target_language, MODEL, len(windows), seconds_per_window,
                  CONTEXT_WINDOWS))
    logf.flush()

    print("source : %s (%d chars)" % (SOURCE["title"], len(SOURCE["text"])))
    print("windows: %d x %.1fs speech (~%d chars each), context = last window + its translation"
          % (len(windows), seconds_per_window, int(seconds_per_window * SPEECH_CHARS_PER_SEC)))
    print("endpoint: %s  model: %s  target: %s" % (BASE_URL, MODEL, target_language))

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
        for run in range(n_runs):
            print("\n=== run %d ===" % run)
            per = []
            prev = None  # (source, translation) of the last window
            for i, u in enumerate(windows):
                messages = build_messages(prev, u)
                p = await bench_utterance(client, messages, target_language)
                per.append(p)
                print("  #%02d prompt:" % i)
                for m in messages:
                    print("    <%s> %s" % (m["role"], m["content"]))
                print("  #%02d out      : %s" % (i, p["out"].strip() or "(empty)"))
                print("  [ttft %.0fms | 1st %.0fms | %d tok | %.1f tok/s | %.0fms]"
                      % (p["ttft"] or 0, p["first_chunk"] or 0, p["n_tokens"], p["tok_s"], p["wall"]))
                logf.write("[%02d] prompt:\n" % i)
                for m in messages:
                    logf.write("[%02d]   <%s> %s\n" % (i, m["role"], m["content"]))
                logf.write("[%02d] out: %s\n" % (i, p["out"].strip()))
                logf.flush()
                prev = (u, p["out"])
            n_tok_total = sum(p["n_tokens"] for p in per)
            wall_total = sum(p["wall"] for p in per) / 1000.0
            tok_total_s = n_tok_total / wall_total if wall_total > 0 else float("nan")

            print("  aggregates (mean/p50/p95/max):")
            _fmt_row("TTFT", [p["ttft"] for p in per if p["ttft"] is not None])
            _fmt_row("first chunk", [p["first_chunk"] for p in per if p["first_chunk"] is not None])
            _fmt_row("tokens/sec", [p["tok_s"] for p in per], unit="tok/s")
            _fmt_row("window wall", [p["wall"] for p in per])

            print("  totals: %d tokens in %.2fs -> %.1f tok/s overall, %.1f ms/window"
                  % (n_tok_total, wall_total, tok_total_s, wall_total / len(per) * 1000))
    logf.close()


if __name__ == "__main__":
    asyncio.run(main())
