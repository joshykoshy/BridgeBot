"""Gradio Blocks UI (v1 used Gradio too): Ask, Brief me, Evals.

The UI calls the same Python service functions as the API and CLI (ask(), the agent graph,
the eval runner), so there is one implementation and three thin adapters.

    python app.py            -> http://localhost:7860
"""
from __future__ import annotations

import html
import json
import uuid

import gradio as gr
from langgraph.types import Command

from . import store
from .agent.graph import (get_graph, initial_state, mermaid, new_thread_id, snapshot, stream_run)
from .ask import ask, log_feedback
from .config import LOG_DIR
from .ragcore import llm

EXAMPLES = [
    "How should I greet my host in Japan?",
    "What gifts should I avoid there?",
    "What is the proper way to greet someone in Japan?",
    "What do people eat to break the fast during Ramadan?",
    "Can I serve a cheeseburger to guests who keep kosher?",
    "What is the GDP of Peru?",
]
BRIEF_EXAMPLES = [
    "Business lunch with Saudi partners in Riyadh on 2026-03-05",
    "Dinner at a colleague's home in Tokyo, Japan on 8 November 2026",
    "Hosting Indian guests for a Diwali dinner on 2026-11-08; one guest is vegetarian",
]


# --- Ask tab -----------------------------------------------------------------------------------
def _scores(rows: list[dict], n: int = 10) -> str:
    return ", ".join(f"`{r['id']}` {r['score']}" for r in rows[:n]) or "-"


def explain(res: dict) -> str:
    """Markdown 'How this answer was built' panel."""
    g, c, r = res["guards"], res["cache"], res["retrieval"]
    tox = g.get("toxicity", {})
    lines = [f"**Status:** `{res['status']}` | **brain:** {res['model']} | **tokens:** {res['tokens']} | "
             f"**total:** {res['latency_ms'].get('total', '-')} ms | **trace id:** `{res['request_id']}`", "",
             "| stage | result |", "|---|---|",
             f"| 1. input guard | PII masked: {g.get('pii_masked') or 'none'}; toxicity: {tox.get('flagged')} "
             f"{tox.get('reasons') or ''}; injection: {g.get('injection') or 'none'} |",
             f"| 2. semantic cache | entities {c.get('entities')}; hit **{c.get('hit')}**; similarity {c.get('similarity')} "
             f"{'(' + c.get('reason') + ')' if c.get('reason') else ''} |"]
    if r:
        lat = r.get("latency_ms", {})
        lines += [
            f"| 3a. rewrite ({r.get('rewrite_mode', 'off')}) | {' / '.join(r.get('queries', []))} ({lat.get('rewrite')} ms) |",
            f"| 3b. BM25 top 10 | {_scores(r.get('bm25_top10', []))} ({lat.get('bm25')} ms) |",
            f"| 3c. dense top 10 | {_scores(r.get('dense_top10', []))} ({lat.get('dense')} ms) |",
            f"| 3d. RRF fused (k=60) | {_scores(r.get('fused', []))} |",
            f"| 3e. cross-encoder rerank | {_scores(r.get('reranked', []), 5)} ({lat.get('rerank')} ms) |",
            f"| 3f. abstain? | top score {r.get('top_score')} vs threshold {r.get('config', {}).get('abstain_threshold')}: "
            f"**{r.get('abstain')}** |"]
    cr = res.get("citation_report") or {}
    if cr:
        lines.append(f"| 5. citation check | ok **{cr.get('ok')}**; coverage {cr.get('coverage')}; cited {cr.get('cited')}; "
                     f"invalid {cr.get('invalid')} |")
    return "\n".join(lines)


def sources_html(citations: list[dict]) -> str:
    if not citations:
        return "<p><i>No sources for this answer.</i></p>"
    parts = []
    for c in citations:
        body = html.escape(c["text"].split("\n\n", 1)[-1])
        parts.append(f"<details><summary><b>[{c['n']}]</b> {html.escape(c['title'])} "
                     f"<small>(rerank {c['score']})</small></summary><p>{body}</p>"
                     f"<p><a href='{html.escape(c['url'])}' target='_blank'>{html.escape(c['url'])}</a></p></details>")
    return "\n".join(parts)


def on_ask(message, chat, session_id, results, use_rewrite, use_bm25, use_dense, use_rerank, use_cache):
    if not message.strip():
        return gr.update(), chat, results, gr.update(), gr.update()
    if not (use_bm25 or use_dense):
        use_dense = True
    res = ask(message, session_id, use_dense, use_bm25, use_rerank, use_rewrite, use_cache).to_dict()
    res["question"] = message
    chat = chat + [{"role": "user", "content": message}, {"role": "assistant", "content": res["answer"]}]
    results = results + [res]
    return "", chat, results, explain(res), sources_html(res["citations"])


def on_like(data: gr.LikeData, results):
    idx = data.index if isinstance(data.index, int) else data.index[0]
    res_index = (idx - 1) // 2
    if 0 <= res_index < len(results):
        r = results[res_index]
        log_feedback(r["request_id"], r.get("question", ""), r["answer"], r["citations"], "up" if data.liked else "down")
        gr.Info("Feedback saved to data/logs/ask_log.jsonl")


def new_session():
    return uuid.uuid4().hex[:12], [], [], "", ""


# --- Brief me tab ----------------------------------------------------------------------------
def _progress(events: list[dict]) -> str:
    if not events:
        return "_Waiting to start._"
    rows = ["| # | node | what happened | ms |", "|---|---|---|---|"]
    rows += [f"| {i} | `{e['node']}` | {e['summary']} | {e['ms']} |" for i, e in enumerate(events, 1)]
    return "\n".join(rows)


def _flags_md(flags: list[dict]) -> str:
    if not flags:
        return "No sensitivity flags."
    return "\n".join(f"- **{f['kind']}** ({f.get('by', '')}): {f['text']}  \n  _{f['reason']}_" for f in flags)


def _sources_md(sources: list[dict]) -> str:
    return "\n".join(f"{s['n']}. " + (f"[{s['title']}]({s['url']})" if s["url"].startswith("http") else
                                      f"{s['title']} ({s['url']})") + f" `{s['id']}`" for s in sources or [])


def _brief_outputs(thread_id: str, updates: list, pending):
    snap = snapshot(get_graph(), thread_id)
    st = snap["state"]
    waiting = pending is not None or snap["status"] == "awaiting_review"
    draft = st.get("final") or st.get("draft") or ""
    status = {"awaiting_review": "**Paused for human review**: approve, edit or reject below.",
              "approved": f"**Approved.** Saved to playbook: entry #{st.get('saved_id')}" if st.get("saved_id") else "**Approved** (not saved: playbook not requested).",
              "rejected": "**Rejected.** Nothing was saved.", "done": "**Done**: no flags and no save requested, so no review needed."
              }.get(snap["status"], f"Running... ({snap['status']})")
    return (thread_id, _progress(st.get("events", [])), updates, status, draft, _flags_md(st.get("flags", [])),
            _sources_md(st.get("sources", [])), gr.update(value=draft, visible=waiting),
            gr.update(visible=waiting))


def on_brief_start(request, user_id, save):
    graph, thread_id = get_graph(), new_thread_id()
    updates, pending = [], None
    for kind, node, value in stream_run(graph, initial_state(request, user_id or "demo", bool(save)), thread_id):
        if kind == "interrupt":
            pending = value
        else:
            updates.append({"node": node, "update": value})
        yield _brief_outputs(thread_id, updates, pending)


def on_resume(thread_id, decision, edited_text, updates):
    if not thread_id:
        raise gr.Error("Start a briefing first.")
    graph = get_graph()
    if "human_review" not in (graph.get_state({"configurable": {"thread_id": thread_id}}).next or ()):
        raise gr.Error("This briefing is not waiting for review.")
    payload = Command(resume={"decision": decision, "edited_text": edited_text})
    updates = list(updates or [])
    for kind, node, value in stream_run(graph, payload, thread_id):
        if kind == "update":
            updates.append({"node": node, "update": value})
    yield _brief_outputs(thread_id, updates, None)


def on_approve(thread_id, edited_text, updates):
    yield from on_resume(thread_id, "approve", "", updates)


def on_edit(thread_id, edited_text, updates):
    yield from on_resume(thread_id, "edit", edited_text, updates)


def on_reject(thread_id, edited_text, updates):
    yield from on_resume(thread_id, "reject", "", updates)


def load_profile(user_id):
    p = store.get_profile(user_id or "demo")
    return p["home_culture"], p["dietary_needs"], p["notes"], json.dumps(p["past_trips"], indent=1)


def save_profile(user_id, home, diet, notes):
    p = store.get_profile(user_id or "demo")
    p.update(home_culture=home, dietary_needs=diet, notes=notes)
    store.save_profile(p)
    gr.Info("Profile saved")
    return load_profile(user_id)


# --- Evals tab ---------------------------------------------------------------------------------
def load_report():
    path = LOG_DIR / "eval_report.md"
    return path.read_text(encoding="utf-8") if path.exists() else "_No report yet. Click 'Run quick evals'._"


def run_quick_evals():
    from .evals.run_all import main
    main(quick=True)
    return load_report()


# --- layout ------------------------------------------------------------------------------------
def build() -> gr.Blocks:
    with gr.Blocks(title="BridgeBot v2") as demo:
        gr.Markdown(f"# BridgeBot v2: cultural intelligence for global teams\n"
                    f"Cited, respectful guidance from curated public sources. **Brain:** `{llm.badge()}`")
        with gr.Tab("Ask"):
            session = gr.State(uuid.uuid4().hex[:12])
            results = gr.State([])
            with gr.Row():
                with gr.Column(scale=3):
                    chat = gr.Chatbot(height=420, label="BridgeBot", feedback_options=("Like", "Dislike"))
                    msg = gr.Textbox(placeholder="Ask about etiquette, festivals, food customs, greetings...",
                                     show_label=False)
                    gr.Examples(EXAMPLES, msg)
                with gr.Column(scale=1):
                    gr.Markdown("**Pipeline switches**")
                    use_rewrite = gr.Checkbox(True, label="Query rewrite + multi-query")
                    use_bm25 = gr.Checkbox(True, label="BM25 (lexical)")
                    use_dense = gr.Checkbox(True, label="Dense (bge-small)")
                    use_rerank = gr.Checkbox(True, label="Cross-encoder rerank + abstain")
                    use_cache = gr.Checkbox(True, label="Semantic cache")
                    new_btn = gr.Button("New session")
            with gr.Accordion("How this answer was built", open=False):
                explain_md = gr.Markdown()
            with gr.Accordion("Sources (click to expand)", open=True):
                sources = gr.HTML()
            switches = [use_rewrite, use_bm25, use_dense, use_rerank, use_cache]
            msg.submit(on_ask, [msg, chat, session, results, *switches], [msg, chat, results, explain_md, sources])
            chat.like(on_like, [results], None)
            new_btn.click(new_session, None, [session, chat, results, explain_md, sources])

        with gr.Tab("Brief me"):
            thread = gr.State("")
            updates = gr.State([])
            with gr.Row():
                with gr.Column(scale=2):
                    request = gr.Textbox(label="Meeting request", lines=2, value=BRIEF_EXAMPLES[0])
                    gr.Examples(BRIEF_EXAMPLES, request)
                    with gr.Row():
                        user_id = gr.Textbox("demo", label="User id", scale=1)
                        save = gr.Checkbox(True, label="Save to team playbook (requires review)", scale=2)
                    start = gr.Button("Prepare briefing", variant="primary")
                with gr.Column(scale=1):
                    gr.Markdown("**Profile (long-term memory)**")
                    home = gr.Textbox(label="Home culture")
                    diet = gr.Textbox(label="Dietary needs")
                    notes = gr.Textbox(label="Notes")
                    trips = gr.Code(label="Past trips (written by the agent)", language="json")
                    with gr.Row():
                        load_p = gr.Button("Load")
                        save_p = gr.Button("Save profile")
            status = gr.Markdown()
            progress = gr.Markdown("_Waiting to start._")
            with gr.Accordion("State updates per node (raw)", open=False):
                raw = gr.JSON()
            with gr.Row():
                with gr.Column(scale=3):
                    draft = gr.Markdown(label="Draft")
                with gr.Column(scale=2):
                    gr.Markdown("**Sensitivity flags**")
                    flags = gr.Markdown()
                    gr.Markdown("**Sources**")
                    srcs = gr.Markdown()
            edited = gr.Textbox(label="Edit the draft before approving", lines=10, visible=False)
            with gr.Row(visible=False) as review_row:
                approve = gr.Button("Approve", variant="primary")
                edit = gr.Button("Save edit")
                reject = gr.Button("Reject", variant="stop")
            with gr.Accordion("Agent graph (Mermaid)", open=False):
                gr.Markdown(f"```mermaid\n{mermaid()}\n```")
            outs = [thread, progress, updates, status, draft, flags, srcs, edited, review_row]
            start.click(on_brief_start, [request, user_id, save], outs)
            approve.click(on_approve, [thread, edited, updates], outs)
            edit.click(on_edit, [thread, edited, updates], outs)
            reject.click(on_reject, [thread, edited, updates], outs)
            updates.change(lambda u: u, updates, raw)
            load_p.click(load_profile, [user_id], [home, diet, notes, trips])
            save_p.click(save_profile, [user_id, home, diet, notes], [home, diet, notes, trips])
            demo.load(load_profile, [user_id], [home, diet, notes, trips])

        with gr.Tab("Evals"):
            gr.Markdown("Full suite: `python -m bridgebot.evals.run_all` (includes answer-level evals). "
                        "The button runs the quick suite (retrieval, safety, cache, agent trajectories), about 1-2 minutes.")
            with gr.Row():
                load_btn = gr.Button("Load latest report")
                run_btn = gr.Button("Run quick evals", variant="primary")
            report = gr.Markdown(load_report())
            load_btn.click(load_report, None, report)
            run_btn.click(run_quick_evals, None, report)
    return demo


def launch(port: int = 7860) -> None:
    from .corpus import get_indexes
    get_indexes()  # build/load the index before the first request
    build().queue().launch(server_name="127.0.0.1", server_port=port)
