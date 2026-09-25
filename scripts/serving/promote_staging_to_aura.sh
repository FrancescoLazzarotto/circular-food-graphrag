#!/usr/bin/env bash
# Replace the Aura graph with the staging graph, once staging has been measured.
#
# Deliberately a full replace, not a delta. A delta pass would have to match
# staging nodes to Aura nodes by name, and the graph has dozens of groups of
# nodes that share a normalised name — every one a chance to write an alias onto
# the wrong node, silently. A dump and restore has no such ambiguity, and every
# step below already exists and is already used elsewhere.
#
# The cost of a full replace is that Aura reassigns element identifiers, which
# breaks the vector index: the carriers survive, their `of` pointers go stale,
# and retrieval quietly degrades to lexical matching
# instead of failing. Hence the rebuild and the guard at the end, in that order,
# and hence the refusal to skip them.
#
#   bash scripts/serving/promote_staging_to_aura.sh          # dry run: prints the plan
#   CONFIRM=yes bash scripts/serving/promote_staging_to_aura.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

STAGING_URL="${STAGING_URL:-bolt://localhost:7689}"
STAGING_USER="${STAGING_USER:-neo4j}"
STAGING_PASSWORD="${STAGING_PASSWORD:-staging-kg-v2}"
STAMP="$(date +%Y%m%d_%H%M%S)"
STAGING_DUMP="artifacts/kg_backups/staging_v2_${STAMP}"
AURA_SAFETY="artifacts/kg_backups/aura_pre_promote_${STAMP}"

if [ "${CONFIRM:-no}" != "yes" ]; then
  cat <<EOF
DRY RUN. This would:

  1. dump Aura to ${AURA_SAFETY}            (the rollback point)
  2. dump staging to ${STAGING_DUMP}
  3. WIPE Aura
  4. restore the staging dump into Aura
  5. rebuild the vector index on Aura
  6. run the resolution guard, and fail loudly if it does not pass

Aura is the graph the expert demo talks to. Re-run with CONFIRM=yes when the
staging graph has actually won its comparison.
EOF
  exit 0
fi

echo "=== 1/6 safety dump of Aura : $(date -Is) ==="
conda run --no-capture-output -n graphllm python scripts/kg/kg_backup.py \
  --output-dir "$AURA_SAFETY"

echo "=== 2/6 dump of staging ==="
NEO4J_URL="$STAGING_URL" NEO4J_USERNAME="$STAGING_USER" \
NEO4J_PASSWORD="$STAGING_PASSWORD" NEO4J_DATABASE="neo4j" \
  conda run --no-capture-output -n graphllm python scripts/kg/kg_backup.py \
    --output-dir "$STAGING_DUMP"

# kg_wipe.py loads kg_pipeline/.env with override=True and so always targets
# Aura, whatever is exported in this shell. That is what we want here, and it is
# worth knowing before running it anywhere else.
echo "=== 3/6 wipe Aura ==="
conda run --no-capture-output -n graphllm python scripts/kg/kg_wipe.py --yes

echo "=== 4/6 restore staging into Aura ==="
AURA_URL="$(grep -E '^NEO4J_URL=' .env | cut -d= -f2- | tr -d '"'"'"'')"
AURA_USER="$(grep -E '^NEO4J_USERNAME=' .env | cut -d= -f2- | tr -d '"'"'"'')"
AURA_PASSWORD="$(grep -E '^NEO4J_PASSWORD=' .env | cut -d= -f2- | tr -d '"'"'"'')"
AURA_DB="$(grep -E '^NEO4J_DATABASE=' .env | cut -d= -f2- | tr -d '"'"'"'')"
conda run --no-capture-output -n graphllm python scripts/kg/kg_restore.py \
  --backup-dir "$STAGING_DUMP" \
  --uri "$AURA_URL" --user "$AURA_USER" --password "$AURA_PASSWORD" \
  --database "$AURA_DB"

echo "=== 5/6 rebuild the vector index on Aura ==="
conda run --no-capture-output -n graphllm python scripts/kg/kg_vector_index.py --context-chars 300

echo "=== 6/6 guard ==="
conda run --no-capture-output -n graphllm python scripts/kg/check_vector_index.py --min-resolving 1000

echo "=== promoted : $(date -Is) ==="
echo "rollback, if needed: kg_restore.py --backup-dir ${AURA_SAFETY} into Aura"
