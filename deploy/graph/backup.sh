#!/usr/bin/env bash
# Back up the local production graph.
#
#   backup.sh daily    online JSON export (kg_backup.py), no downtime; keeps 7
#   backup.sh weekly   neo4j-admin dump with the database stopped for about a
#                      minute; keeps 4. The dump is the exact copy (vectors and
#                      record ids included) and the fast way back.
#
# A run writes into <stamp>.partial and renames it only once what it wrote is
# checked, so pruning, which counts finished backups only, never trades good
# copies for failed ones. A run records itself in last_<mode>.json, which the
# health check reads. Every job of this folder takes the same lock: a run that
# finds another one going (a load, the other backup) skips and says so.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETTINGS="${GRAPH_SETTINGS:-$HOME/.config/graphrag/graph-db.env}"
set -a
# shellcheck disable=SC1090
source "$SETTINGS"
set +a
export PYTHONNOUSERSITE=1
MODE="${1:?usage: backup.sh daily|weekly}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$GRAPH_BACKUPS/backup.log"
mkdir -p "$GRAPH_BACKUPS/daily" "$GRAPH_BACKUPS/weekly"
say() { echo "$(date -u +%FT%TZ) [$MODE] $*" | tee -a "$LOG"; }

exec 9>"$GRAPH_BACKUPS/.lock"
if ! flock -n 9; then
  say "un altro lavoro sul grafo è in corso: salto questo giro"
  exit 0
fi

prune() {  # keep the newest $2 finished backups of $1; drop leftovers of failed runs
  local dir="$1" keep="$2"
  find "$dir" -maxdepth 1 -name '*.partial' -mmin +60 -exec rm -rf -- {} + 2>/dev/null || true
  ls -1 "$dir" | grep -E '^[0-9]{8}T[0-9]{6}Z$' | sort | head -n "-$keep" | while read -r old; do
    rm -rf -- "${dir:?}/$old"; say "rimosso il vecchio $dir/$old"
  done
}

case "$MODE" in
  daily)
    out="$GRAPH_BACKUPS/daily/$STAMP"
    say "inizio export in $out"
    "$GRAPH_PYTHON" "$REPO/scripts/kg/kg_backup.py" --output-dir "$out.partial" >> "$LOG" 2>&1
    "$GRAPH_PYTHON" - "$out.partial" "$out" "$GRAPH_BACKUPS/last_daily.json" <<'PY'
import json, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
partial, final, last = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
manifest = json.loads((partial / "manifest.json").read_text())
nodes = json.loads((partial / "nodes.json").read_text())
edges = json.loads((partial / "edges.json").read_text())
ids = Counter(n["id"] for n in nodes)
problems = []
if (len(nodes), len(edges)) != (manifest["nodes"], manifest["relationships"]):
    problems.append(f"{len(nodes)}/{manifest['nodes']} nodi, {len(edges)}/{manifest['relationships']} archi")
if any(c > 1 for c in ids.values()):
    problems.append("nodi ripetuti")
if any(c > 1 for c in Counter(e.get("id") for e in edges).values()):
    problems.append("archi ripetuti")
loose = sum(1 for e in edges if e["src"] not in ids or e["dst"] not in ids)
if loose:
    problems.append(f"{loose} archi con un estremo assente (il grafo è cambiato durante l'export?)")
if problems:
    sys.exit("export non valido: " + "; ".join(problems))
last.write_text(json.dumps({"path": final, "nodes": len(nodes), "edges": len(edges),
                            "at": datetime.now(timezone.utc).isoformat()}) + "\n")
PY
    mv "$out.partial" "$out"
    say "export verificato"
    prune "$GRAPH_BACKUPS/daily" 7
    ;;
  weekly)
    out="$GRAPH_BACKUPS/weekly/$STAMP"
    # A job writing to the graph would be cut off by the stop: this week's dump
    # waits for the next one instead, and the health check reports its age.
    busy="$("$GRAPH_PYTHON" "$REPO/scripts/kg/graph_health.py" --active-transactions)"
    if [[ "$busy" != "0" ]]; then
      say "$busy transazioni aperte sul grafo: niente dump questa settimana"
      exit 1
    fi
    mkdir -p "$out.partial"
    say "fermo il database per il dump"
    systemctl --user stop graph-db.service
    # The database comes back whatever happens to the dump.
    trap 'systemctl --user start graph-db.service' EXIT
    JAVA_HOME="$JAVA_HOME" "$GRAPH_HOME/bin/neo4j-admin" database dump neo4j --to-path="$out.partial" >> "$LOG" 2>&1
    systemctl --user start graph-db.service
    trap - EXIT
    [[ -s "$out.partial/neo4j.dump" ]] || { say "dump mancante o vuoto in $out.partial"; exit 1; }
    mv "$out.partial" "$out"
    printf '{"path": "%s", "bytes": %s, "at": "%s"}\n' "$out/neo4j.dump" "$(stat -c %s "$out/neo4j.dump")" \
      "$(date -u +%FT%TZ)" > "$GRAPH_BACKUPS/last_weekly.json"
    say "dump verificato ($(du -h "$out/neo4j.dump" | cut -f1))"
    prune "$GRAPH_BACKUPS/weekly" 4
    flock -u 9
    bash "$REPO/deploy/graph/health.sh" --wait 300 >> "$LOG" 2>&1 || say "ATTENZIONE: dopo il dump il controllo di salute non passa"
    ;;
  *) echo "usage: backup.sh daily|weekly" >&2; exit 2 ;;
esac
