#!/usr/bin/env bash
# Local Neo4j staging instance, where a graph is rebuilt and measured first.
#
# Aura 588fe1bc serves the demo. Rebuilding aliases and adding edges there would
# change what the demo answers while the work is in progress, so a graph is
# built here and promoted only after it is measured; the two can then be
# compared head to head.
#
# Docker is not usable on this host (the account is not in the docker group), so
# this is the unpacked Community tarball run as a user process, with a JDK 21
# from the `neo4jrt` conda env because the system JDK is 11 and Neo4j 5 needs 17+.
#
# Bolt is on 7689, HTTP on 7476. 7687 is the default and 7688/7475 are held by
# another staging instance under /mnt/storage/flazzarotto/neo4j-staging, which
# is left alone.
set -euo pipefail

NEO_HOME="${NEO_HOME:-/mnt/storage/flazzarotto/neo4j_v2/neo4j-community-5.26.0}"
JAVA_HOME="${JAVA_HOME:-/home/flazzarotto/.conda/envs/neo4jrt}"
BOLT_PORT="${BOLT_PORT:-7689}"
HTTP_PORT="${HTTP_PORT:-7476}"
PASSWORD="${STAGING_PASSWORD:-staging-kg-v2}"

export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"

conf="$NEO_HOME/conf/neo4j.conf"

if ! grep -q "^# kg-v2 staging" "$conf"; then
  cat >> "$conf" <<EOF

# kg-v2 staging
server.bolt.listen_address=:${BOLT_PORT}
server.http.listen_address=:${HTTP_PORT}
server.memory.heap.initial_size=2g
server.memory.heap.max_size=6g
server.memory.pagecache.size=2g
db.logs.query.enabled=OFF
EOF
  echo "conf patched"
fi

# Idempotent: the second call is a no-op once the auth store exists.
if [ ! -f "$NEO_HOME/data/dbms/auth" ]; then
  "$NEO_HOME/bin/neo4j-admin" dbms set-initial-password "$PASSWORD"
fi

"$NEO_HOME/bin/neo4j" start

for _ in $(seq 1 60); do
  if (exec 3<>"/dev/tcp/localhost/${BOLT_PORT}") 2>/dev/null; then
    echo "bolt://localhost:${BOLT_PORT} is up"
    exit 0
  fi
  sleep 2
done
echo "neo4j did not open bolt on ${BOLT_PORT} within 120s" >&2
tail -40 "$NEO_HOME/logs/neo4j.log" >&2 || true
exit 1
