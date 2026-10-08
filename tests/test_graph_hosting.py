"""The production graph's configuration keeps it on loopback, and its health check reads backup ages right.

The settings block is appended to Neo4j's own configuration, which refuses a
setting declared twice; a listener on 0.0.0.0 would expose an unauthenticated
admin surface on the network.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("graph_health", ROOT / "scripts" / "kg" / "graph_health.py")
graph_health = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("graph_health", graph_health)
_spec.loader.exec_module(graph_health)


def test_the_managed_block_listens_on_loopback_only_and_declares_each_setting_once():
    lines = [
        line for line in (ROOT / "deploy" / "graph" / "neo4j.conf.fragment").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    keys = [line.split("=", 1)[0] for line in lines if not line.startswith("server.jvm.additional=")]

    assert len(keys) == len(set(keys))
    listeners = [line for line in lines if re.match(r"server\.(default_listen_address|bolt\.listen_address|http\.listen_address)=", line)]
    assert len(listeners) == 3
    assert all("127.0.0.1" in line for line in listeners)
    assert "server.https.enabled=false" in lines


def test_a_backup_age_is_read_from_its_record_and_a_missing_one_is_none(tmp_path):
    at = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    (tmp_path / "last_daily.json").write_text(json.dumps({"path": "/b/daily/x", "at": at}))

    age, path = graph_health._backup_age(tmp_path, "daily")

    assert 4.9 < age < 5.1 and path == "/b/daily/x"
    assert graph_health._backup_age(tmp_path, "weekly") == (None, "")

