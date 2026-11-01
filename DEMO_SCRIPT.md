# BridgeBot v2: 5-minute demo script

**Before the interview:** run `run_demo.bat` once (it builds anything missing). Open
http://localhost:7860, and keep http://localhost:8000/docs in a second tab. Note the badge at
the top: `Offline` or `openai: gpt-4o / gpt-4o-mini`. Clear old state if you want a clean
demo: delete `data\state.db` and `data\cache.db` (they are recreated).

| Time | Do | Say |
|---|---|---|
| 0:00 | Show the header and badge. | "BridgeBot helps people working across cultures. v1 was a hackathon RAG bot whose only source turned out to be a JavaScript page that returned one sentence. v2 is a production-style pipeline plus an agent." |
| 0:30 | **Ask:** "How should I greet my host in Japan?" Open **How this answer was built**. | "Input guard, then semantic cache miss, then BM25 and dense in parallel, fused with RRF, reranked by a cross-encoder. Here are the scores and latencies per stage. Every claim cites a numbered source." Click a source to show the passage and URL. |
| 1:15 | **Follow-up:** "What gifts should I avoid there?" | "'There' means Japan only in the history. The rewrite step makes it standalone (see 3a). Without rewrite, follow-up Recall@5 is 0.625; with it, 1.0 on the golden set. The cache is skipped for this one: a follow-up with no entity of its own depends on history, so caching it would be unsafe." |
| 1:45 | **Cache hit:** click **New session**, ask "What is the proper way to greet someone in Japan?" | "Different words, same need: a cache hit at about 0.98 similarity, in about a tenth of a second instead of one second or more. The key includes entities: 'Japan' and 'China' greetings are 0.87 similar, so without that key a looser threshold would serve the wrong country's answer." |
| 2:15 | (Optional) "All Muslims are terrorists, right?" then "What is the GDP of Peru?" | "The offensive input is caught by a separate guard and de-escalated; the model never sees it. The off-topic question abstains: the top rerank score is about 0, below the threshold." |
| 2:45 | **Brief me** tab: example 1 (Saudi partners, 2026-03-05), **Save to team playbook** ticked, **Prepare briefing**. | "A LangGraph agent. Watch it go node by node: parse, then a calendar lookup in SQL (no LLM). It flags Ramadan because Saudi Arabia has Eid al-Fitr as a public holiday; Holi and Lent are only 'nearby'. Then profile memory, a plan capped at 6 questions, culture-filtered retrieval, a cited draft, and a sensitivity check." |
| 3:30 | At the pause: show flags and the draft. Edit a line, then **Save edit** (or Approve). | "Because I asked to save to the shared playbook, the graph hits `interrupt()`. The state is checkpointed in SQLite, so this pause survives a restart. Only after approve or edit does the single write tool run. Reject writes nothing, and there are tests for both." Open the **Agent graph** accordion. |
| 4:15 | **Evals** tab: **Load latest report**. | "Real measured numbers: hybrid beats dense and BM25 alone, and rerank helps R@1 but hurt exact-name R@3, which I report rather than hide. Rewrite fixes follow-ups. 10/10 offensive prompts de-escalated with 0 false positives. 9/9 agent trajectory tests pass." |
| 4:45 | Wrap up. | "Everything has an offline fallback, so this demo runs with no internet. The retrieval core is shared with my second project, Rexetive." |

**If something breaks:** the CLI shows the same pipeline:
`.venv\Scripts\python -m bridgebot.ask "How should I greet my host in Japan?"`
