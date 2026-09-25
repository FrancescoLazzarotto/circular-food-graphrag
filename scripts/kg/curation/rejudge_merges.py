"""Re-judge every star merge of stage 4 with the strict judge, and filter the cache.

Run after stage 4 has written ``stage4_merge_approved.json``. It rebuilds the
mention groups and the star clusters exactly as stage 4 does, and asks
`merge_judge` (with reasoning) whether each member whose name differs from its
centre is the same entity. Rejected pairs leave the approvals; a released
member can land under another centre, so the loop repeats until no new pair is
left. A pair the model does not answer is asked again the next round; if it
is still without a verdict it is stored as null and treated as rejected: a
merge nobody confirmed is not made. A round where no pair at all gets a
verdict for the first time stops the run instead (the servers are down).

Writes to ``--run-dir``:

* ``stage4_merge_approved_unfiltered.json`` — the approvals stage 4 wrote,
  kept once and read on every later run;
* ``merge_verdicts.json`` — every verdict, keyed ``"member|centre"``; with it
  the script replays without calling the model;
* ``stage4_merge_approved.json`` — the filtered approvals, same group
  fingerprint, so stage 4 loads them instead of asking the LLM again.

Afterwards stage 4 and 5 must run again from the filtered cache
(``replay_curation.sh`` does it).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from merge_judge import judge_in_slices  # noqa: E402

from kg_pipeline.models.types import KGTriple  # noqa: E402
from kg_pipeline.stages import resolution as res  # noqa: E402


def fold(s: str) -> str:
    """Accent-folded, lowercased alphanumeric key of a name."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s)


def main() -> None:
    """Re-judge the star merges round by round and write the filtered approvals."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--endpoints", default="http://localhost:8000/v1,http://localhost:8003/v1",
                   help="comma-separated vLLM endpoints; the ones that answer share the work")
    p.add_argument("--model", default="Qwen/Qwen3-32B-AWQ")
    p.add_argument("--context-jaccard-floor", type=float, default=0.15,
                   help="must match resolution.context_jaccard_floor of the run")
    a = p.parse_args()
    run = a.run_dir
    endpoints = [u.strip() for u in a.endpoints.split(",") if u.strip()]

    unfiltered = run / "stage4_merge_approved_unfiltered.json"
    if not unfiltered.exists():
        shutil.copy(run / "stage4_merge_approved.json", unfiltered)
    cache = json.loads(unfiltered.read_text())

    triples = [KGTriple.model_validate(t) for t in json.loads((run / "stage3_triples_raw.json").read_text())]
    acr = json.loads((run / "stage3_acronyms.json").read_text())
    mentions = res._build_mentions(triples)
    groups = res._initial_groups(mentions, acr, context_jaccard_floor=a.context_jaccard_floor)
    if res._group_fingerprint(mentions, groups) != cache["group_fingerprint"]:
        raise SystemExit("the approvals were built for another grouping: rerun stage 4 first")
    approved = res._drop_number_mismatches({tuple(p) for p in cache["pairs"]}, mentions, groups)
    names = [Counter(mentions[m]["name"] for m in g).most_common(1)[0][0] for g in groups]

    store = run / "merge_verdicts.json"
    judged: dict[str, bool | None] = json.loads(store.read_text()) if store.exists() else {}
    for key, v in judged.items():
        if v is False:
            m, c = map(int, key.split("|"))
            approved.discard(tuple(sorted((m, c))))

    rounds = []
    asked: set[str] = set()
    while True:
        clusters = res._centre_clusters(groups, approved)
        todo = [(m, c) for c, ms in clusters.items() for m in ms[1:]
                if fold(names[m]) != fold(names[c]) and f"{m}|{c}" not in judged]
        if not todo:
            break
        verdicts = judge_in_slices([(names[m], names[c]) for m, c in todo], endpoints, a.model)
        vetoed = missing = 0
        for (m, c), v in zip(todo, verdicts):
            if v is None:
                missing += 1  # no verdict: asked again next round, never taken as a veto
                continue
            judged[f"{m}|{c}"] = v
            if v is not True:
                approved.discard(tuple(sorted((m, c))))
                vetoed += 1
        keys = [f"{m}|{c}" for m, c in todo]
        if missing == len(todo):
            if not asked.issuperset(keys):
                raise SystemExit("no verdict in this round: do the model servers answer?")
            # Asked twice without a verdict: stored as null so a replay does not ask
            # again; the close below detaches them like a rejection.
            judged.update(dict.fromkeys(keys, None))
        asked.update(keys)
        store.write_text(json.dumps(judged, ensure_ascii=False))
        rounds.append({"judged": len(todo) - missing, "rejected": vetoed, "no_verdict": missing})
        print(f"round {len(rounds)}: judged {len(todo) - missing}, rejected {vetoed}, no verdict {missing}", flush=True)
        if missing == len(todo):
            break

    # Close: a member still under a centre it was never confirmed with is detached.
    unjudged = []
    while True:
        clusters = res._centre_clusters(groups, approved)
        left = [(m, c) for c, ms in clusters.items() for m in ms[1:]
                if fold(names[m]) != fold(names[c]) and judged.get(f"{m}|{c}") is not True]
        if not left:
            break
        for m, c in left:
            approved.discard(tuple(sorted((m, c))))
            unjudged.append((names[m], names[c]))

    (run / "stage4_merge_approved.json").write_text(json.dumps({
        "group_fingerprint": cache["group_fingerprint"],
        "n_groups": cache["n_groups"],
        "pairs": sorted(approved),
    }))
    kept = sum(1 for v in judged.values() if v is True)
    rejected = sum(1 for v in judged.values() if v is False)
    print(f"judged {len(judged)}, kept {kept}, rejected {rejected}; "
          f"detached without a verdict {len(unjudged)} {unjudged}; approvals left {len(approved)}")


if __name__ == "__main__":
    main()
