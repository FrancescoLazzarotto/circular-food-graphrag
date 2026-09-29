# product/

The thing a person actually uses. Two surfaces over one engine.

| File | What it is |
|---|---|
| `app.py` | Browser UI (Streamlit): the answer, what it was built from, what it cited, and the evidence behind it |
| `console.py` | Terminal REPL: same answers, no tunnel needed |
| `config.py` | The settings both of them run on, how they reach the graph and the models, and what the collection holds |
| `ui.py` | Interface strings in two languages, and the pure helpers that turn one `result` into a page |

## Running it

```bash
bash scripts/serving/start_demo.sh              # encoder + generator + UI, with a preflight
bash scripts/serving/start_demo.sh --list       # which models can be served
bash scripts/serving/stop_demo.sh               # everything down
bash scripts/serving/stop_demo.sh streamlit     # only the browser UI
```

Or one surface at a time, against servers that are already up:

```bash
conda run -n graphllm streamlit run product/app.py
conda run -n graphllm python product/console.py --strategy hybrid
```

## Where the line is

`src/graphrag/` is the **engine**: retrieval, the agent state machine, the LLM
layer. It is what the thesis measured, and changing its defaults makes future
runs incomparable with the campaigns already reported.

`product/` is **how the engine is presented**: which settings the demo runs on,
what the answers look like, how the whole thing starts. Change demo behaviour
here, never in `graphrag.config` or `graphrag.strategies`.

Every setting in `config.py` is an environment variable with the value the demo
ships with, so nothing needs editing to try something. The full list, with each
default, is in
[../docs/configuration.md](../docs/configuration.md#demo-settings):

```bash
DEMO_STRATEGY=default DEMO_COMPLEXITY=medium \
  conda run -n graphllm streamlit run product/app.py
```

A demo answer is still not a retrieval measurement: the demo runs its own
settings — the answer first and at most two short paragraphs after it, 2048
new tokens, a dense text backend — and a
campaign runs the frozen ones. Read a demo answer as a presentation of the
engine, never as a data point.

`config.py` is also where the graph connection lives. The primary graph is a
hosted instance that suspends itself after three idle days; when it does not
answer, `DEMO_NEO4J_FALLBACK_URL` takes over so a booked session does not die
because a database went to sleep. An instance that suspends *mid-session* is
caught too: the first question that fails on it rebuilds the agent once, which
is what moves the session onto the mirror.

## What the page shows

Everything on screen is read from the `result` the agent returns. The answer's
own text is split on its section headings — the limits section, and the source
list the engine appends — and nothing else is recovered from the prose.

| On screen | Where it comes from |
|---|---|
| The answer, with its limits section in its own box | `answer`, split by `ui.split_answer` |
| In the sidebar, "Dettagli della risposta" for the chosen answer: `8 passaggi · 20 fatti dal grafo · 4 documenti · 31 s` and `31 citazioni, tutte verificate` | `retrieved_text_sources`, `kg_triples`, `evidence_index`, `citation_report` |
| "Approfondisci": the same question answered again in the long form, from more passages and without the thread; the "Approfondita" switch beside the send arrow gives the long form straight away | `config.build_deep_agent`, `mode: deep` in the session log |
| Sources, one block per document, each cited passage readable in place | `evidence_index` + `citation_report.cited_refs` |
| Under the details: every passage and graph fact retrieved, cited or not | `evidence_index` |
| `● Sistema operativo` / `● Modalità ridotta` | which graph answered, and whether the cross-lingual channel was skipped |
| The orientation shown on a refused question | `out_of_scope`, plus the corpus manifest and `DEMO_EXAMPLE_QUESTIONS` |

Two deliberate absences. **`insufficient_answer` is not used anywhere in the
interface**: measured on this repository's own data it flags invented answers
that hedge in the tail, so it cannot carry a reliability claim — the citation
check does that alone. And **the strategy, the model id and the graph URL are
not on the page**; they are behind `DEMO_DEBUG=1`, and stay in the session log
where they are needed.

The count in "sulla base di N documenti" is read from the stage0 manifest of the
runs in `DEMO_TEXT_STAGE0_RUNS` (`config.corpus_manifest`), so it follows the
collection instead of being typed into a sentence. The manifest's
`publication_year` is deliberately not used: on the current runs it spans
1943-2026, because the extractor picks up any four-digit number on a cover.

What the page still cannot show, because the data is not in `result`: how many
claims each document supports (`citation_report` counts distinct references and
total tags, never occurrences per reference), and what the agent is doing while
a question runs — `node_timings` is declared in `RAGState` and never populated,
and `invoke()` returns only at the end. Real progress needs `graph.stream()`,
which is engine work and the same work an HTTP API would need.

## Appearance

`.streamlit/config.toml` at the repository root (Streamlit reads it from the
working directory, not from here) holds a restrained base palette. The page
sets its own name, icon and tagline from `DEMO_PRODUCT_NAME`,
`DEMO_PRODUCT_ICON` and `DEMO_PRODUCT_TAGLINE`, and lays itself out in two
columns — a fixed-width reading column and a panel reserved for the evidence.

There is no injected CSS and no logo. A visual identity is not something a
Streamlit theme can carry: that is frontend work, and this file will say so
until it is done.
