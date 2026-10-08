#!/usr/bin/env bash
# Install or update the local production graph database: Neo4j Community as a
# systemd user service on loopback, with nightly and weekly backups and a
# health check on timers.
#
#   bash deploy/graph/install.sh --home /data/neo4j_prod/neo4j-community-5.26.0 \
#       --backups /data/graph_backups --tarball neo4j-community-5.26.0-unix.tar.gz \
#       --apoc apoc-5.26.0-core.jar --java-home /opt/jdk21 --python /path/to/python
#   bash deploy/graph/install.sh            # again: re-applies config and units
#
# Paths, ports and the generated password live in a settings file outside the
# repository (default ~/.config/graphrag/graph-db.env, mode 600): every other
# script in this folder reads it, and a second run takes its defaults from it.
# The service survives logout and reboot only with `loginctl enable-linger`.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETTINGS="${GRAPH_SETTINGS:-$HOME/.config/graphrag/graph-db.env}"
UNITS="$HOME/.config/systemd/user"

# shellcheck disable=SC1090
[[ -f "$SETTINGS" ]] && source "$SETTINGS"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --home) GRAPH_HOME="$2"; shift 2 ;;
    --backups) GRAPH_BACKUPS="$2"; shift 2 ;;
    --tarball) NEO4J_TARBALL="$2"; shift 2 ;;
    --apoc) APOC_JAR="$2"; shift 2 ;;
    --java-home) JAVA_HOME="$2"; shift 2 ;;
    --python) GRAPH_PYTHON="$2"; shift 2 ;;
    --bolt-port) BOLT_PORT="$2"; shift 2 ;;
    --http-port) HTTP_PORT="$2"; shift 2 ;;
    --heap) HEAP="$2"; shift 2 ;;
    --pagecache) PAGECACHE="$2"; shift 2 ;;
    *) echo "opzione sconosciuta: $1" >&2; exit 2 ;;
  esac
done
BOLT_PORT="${BOLT_PORT:-7687}"; HTTP_PORT="${HTTP_PORT:-7474}"
HEAP="${HEAP:-4g}"; PAGECACHE="${PAGECACHE:-4g}"
for var in GRAPH_HOME GRAPH_BACKUPS NEO4J_TARBALL APOC_JAR JAVA_HOME GRAPH_PYTHON; do
  [[ -n "${!var:-}" ]] || { echo "manca $var (prima installazione: vedi l'intestazione)" >&2; exit 2; }
done
[[ -x "$JAVA_HOME/bin/java" ]] || { echo "Java non trovato in $JAVA_HOME" >&2; exit 2; }
# The service must never answer on the ports of the other instances.
case "$BOLT_PORT" in 7689|7690) echo "la porta $BOLT_PORT è di staging o dei lotti" >&2; exit 2 ;; esac

# 1. Settings, with a password generated once and kept.
mkdir -p "$(dirname "$SETTINGS")" "$GRAPH_BACKUPS"
# A restart in the middle of a backup or a load would cut it off.
exec 9>"$GRAPH_BACKUPS/.lock"
flock -n 9 || { echo "un backup o un caricamento è in corso: riprovare dopo" >&2; exit 1; }
NEO4J_PASSWORD="${NEO4J_PASSWORD:-$(openssl rand -base64 24 | tr -d '/+=' | cut -c1-24)}"
umask 077
cat > "$SETTINGS" <<EOF
# Local production graph database (written by deploy/graph/install.sh).
GRAPH_HOME=$GRAPH_HOME
GRAPH_BACKUPS=$GRAPH_BACKUPS
NEO4J_TARBALL=$NEO4J_TARBALL
APOC_JAR=$APOC_JAR
JAVA_HOME=$JAVA_HOME
GRAPH_PYTHON=$GRAPH_PYTHON
BOLT_PORT=$BOLT_PORT
HTTP_PORT=$HTTP_PORT
HEAP=$HEAP
PAGECACHE=$PAGECACHE
NEO4J_URI=bolt://localhost:$BOLT_PORT
NEO4J_URL=bolt://localhost:$BOLT_PORT
NEO4J_USER=neo4j
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=$NEO4J_PASSWORD
NEO4J_DATABASE=neo4j
NEO4J_DB=neo4j
EOF
chmod 600 "$SETTINGS"
umask 022

# 2. Binaries and APOC.
if [[ ! -x "$GRAPH_HOME/bin/neo4j" ]]; then
  mkdir -p "$(dirname "$GRAPH_HOME")"
  tar xzf "$NEO4J_TARBALL" -C "$(dirname "$GRAPH_HOME")"
  [[ -x "$GRAPH_HOME/bin/neo4j" ]] || { echo "$NEO4J_TARBALL non contiene $(basename "$GRAPH_HOME")" >&2; exit 1; }
fi
cp -f "$APOC_JAR" "$GRAPH_HOME/plugins/"

# 3. Configuration: the managed block replaced, the rest of neo4j.conf untouched.
conf="$GRAPH_HOME/conf/neo4j.conf"
sed -i '/^# --- graph-db: managed/,/^# --- end graph-db ---/d' "$conf"
# Neo4j refuses a setting declared twice: the stock line of every setting the
# block sets is commented out (server.jvm.additional may repeat).
grep -oE '^[a-z][a-z0-9_.]+=' "$REPO/deploy/graph/neo4j.conf.fragment" | grep -v '^server.jvm.additional=' | sort -u |
  while read -r key; do sed -i "s|^${key//./\\.}|#&|" "$conf"; done
sed -e "s|@BOLT_PORT@|$BOLT_PORT|; s|@HTTP_PORT@|$HTTP_PORT|; s|@HEAP@|$HEAP|g; s|@PAGECACHE@|$PAGECACHE|" \
  "$REPO/deploy/graph/neo4j.conf.fragment" >> "$conf"

# 4. Password, before the first start only. neo4j-admin takes it as an
# argument, visible in the process list for the second it runs, once.
if [[ ! -f "$GRAPH_HOME/data/dbms/auth.ini" && ! -f "$GRAPH_HOME/data/dbms/auth" ]]; then
  JAVA_HOME="$JAVA_HOME" "$GRAPH_HOME/bin/neo4j-admin" dbms set-initial-password "$NEO4J_PASSWORD"
fi

# 5. Units, then (re)start.
mkdir -p "$UNITS"
for unit in "$REPO"/deploy/graph/systemd/*; do
  sed -e "s|@GRAPH_HOME@|$GRAPH_HOME|g; s|@JAVA_HOME@|$JAVA_HOME|g; s|@REPO@|$REPO|g" "$unit" > "$UNITS/$(basename "$unit")"
done
systemctl --user daemon-reload
systemctl --user enable graph-db.service graph-db-backup.timer graph-db-dump.timer graph-db-health.timer >/dev/null
systemctl --user restart graph-db.service
systemctl --user start graph-db-backup.timer graph-db-dump.timer graph-db-health.timer

if bash "$REPO/deploy/graph/health.sh" --wait 180 --empty-ok; then
  echo "graph-db attivo su bolt://localhost:$BOLT_PORT (impostazioni in $SETTINGS)"
else
  echo "graph-db non risponde: journalctl --user -u graph-db -n 50" >&2
  exit 1
fi
