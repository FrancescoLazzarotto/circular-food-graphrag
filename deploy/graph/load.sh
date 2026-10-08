#!/usr/bin/env bash
# Replace the content of the local production graph with another graph.
#
#   load.sh --from-home <neo4j home>   dump a STOPPED instance (staging, lots) and load it
#   load.sh --from-dump <dir>          load a neo4j-admin dump (a folder holding neo4j.dump)
#
# A neo4j-admin dump copies the store as it is: indexes, constraints, vectors
# and record ids come across unchanged. Only the database id differs, so the
# vector carriers' pointers are rewritten to the new one (relink_vectors.py),
# which also checks that each lands on its node. The current production graph
# is dumped first, as the rollback point, and any failure prints the command
# that goes back to it. The database is down for the load, about a minute at
# today's size. Without CONFIRM=yes it only prints the plan.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETTINGS="${GRAPH_SETTINGS:-$HOME/.config/graphrag/graph-db.env}"
set -a
# shellcheck disable=SC1090
source "$SETTINGS"
set +a
export PYTHONNOUSERSITE=1
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$GRAPH_BACKUPS/backup.log"
say() { echo "$(date -u +%FT%TZ) [load] $*" | tee -a "$LOG"; }

case "${1:-}" in
  --from-home) SOURCE_HOME="${2:?}"; SOURCE_DUMP="$GRAPH_BACKUPS/imports/$STAMP" ;;
  --from-dump) SOURCE_DUMP="${2:?}" ;;
  *) echo "usage: load.sh --from-home <neo4j home> | --from-dump <dir>" >&2; exit 2 ;;
esac
SAFETY="$GRAPH_BACKUPS/pre_load/$STAMP"

if [[ "${CONFIRM:-no}" != "yes" ]]; then
  cat <<EOF
PROVA A SECCO. Con CONFIRM=yes:
  ${SOURCE_HOME:+0. dump della istanza ferma $SOURCE_HOME in $SOURCE_DUMP}
  1. ferma graph-db e salva il grafo di produzione attuale in $SAFETY (punto di ritorno)
  2. carica $SOURCE_DUMP/neo4j.dump al suo posto
  3. riavvia graph-db, ricollega i vettori ai nodi, lancia il controllo di salute
EOF
  exit 0
fi

exec 9>"$GRAPH_BACKUPS/.lock"
flock -n 9 || { echo "un backup o un altro caricamento è in corso: riprovare dopo" >&2; exit 1; }

if [[ -n "${SOURCE_HOME:-}" ]]; then
  # A running source would be read while it changes; neo4j-admin refuses too.
  if NEO4J_HOME="$SOURCE_HOME" JAVA_HOME="$JAVA_HOME" "$SOURCE_HOME/bin/neo4j" status >/dev/null 2>&1; then
    echo "$SOURCE_HOME è acceso: fermarlo prima" >&2; exit 1
  fi
  mkdir -p "$SOURCE_DUMP"
  say "dump di $SOURCE_HOME in $SOURCE_DUMP"
  NEO4J_HOME="$SOURCE_HOME" JAVA_HOME="$JAVA_HOME" "$SOURCE_HOME/bin/neo4j-admin" database dump neo4j \
    --to-path="$SOURCE_DUMP" >> "$LOG" 2>&1
fi
[[ -s "$SOURCE_DUMP/neo4j.dump" ]] || { echo "manca $SOURCE_DUMP/neo4j.dump" >&2; exit 1; }
# Room for the rollback dump and the loaded store, checked before production stops.
need=$(( $(stat -c %s "$SOURCE_DUMP/neo4j.dump") * 4 ))
free=$(( $(df --output=avail -B1 "$GRAPH_BACKUPS" | tail -1) ))
(( free > need )) || { echo "spazio insufficiente: servono ~$((need / 1024**2)) MB, liberi $((free / 1024**2)) MB" >&2; exit 1; }

mkdir -p "$SAFETY"
say "fermo graph-db; punto di ritorno in $SAFETY"
systemctl --user stop graph-db.service
loaded=no
on_exit() {
  systemctl --user start graph-db.service || true
  if [[ "$loaded" != "ok" ]]; then
    say "CARICAMENTO NON RIUSCITO. Per tornare al grafo di prima: CONFIRM=yes $0 --from-dump $SAFETY"
  fi
}
trap on_exit EXIT
JAVA_HOME="$JAVA_HOME" "$GRAPH_HOME/bin/neo4j-admin" database dump neo4j --to-path="$SAFETY" >> "$LOG" 2>&1
say "carico $SOURCE_DUMP/neo4j.dump"
JAVA_HOME="$JAVA_HOME" "$GRAPH_HOME/bin/neo4j-admin" database load neo4j --from-path="$SOURCE_DUMP" \
  --overwrite-destination=true >> "$LOG" 2>&1
systemctl --user start graph-db.service
# The service answers queries from here on; the pointers are rewritten right
# after it opens, before the health check vouches for the graph.
bash "$REPO/deploy/graph/health.sh" --wait 300 --empty-ok > /dev/null || true
"$GRAPH_PYTHON" "$REPO/scripts/kg/relink_vectors.py" | tee -a "$LOG"
bash "$REPO/deploy/graph/health.sh" --wait 300 > /dev/null
loaded=ok
say "caricato e sano"

for dir in imports pre_load; do  # each holds a full dump: the last three are enough
  [[ -d "$GRAPH_BACKUPS/$dir" ]] || continue
  ls -1 "$GRAPH_BACKUPS/$dir" | grep -E '^[0-9]{8}T[0-9]{6}Z$' | sort | head -n -3 | while read -r old; do
    rm -rf -- "${GRAPH_BACKUPS:?}/$dir/$old"; say "rimosso il vecchio $dir/$old"
  done
done
