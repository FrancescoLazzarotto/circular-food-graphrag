"""Where a turn's seconds went, recorded per turn rather than probed once.

`latency_ms` says how long a turn took and nothing about which stage spent
the time. A one-off probe answers that for one day; this makes the split a
property of every turn.

No LLM and no graph: the agent's nodes are replaced by functions that sleep a
known amount.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from graphrag.agent.core import KGRAGAgent
from graphrag.config import AgentConfig


def _agent(**overrides: Any) -> KGRAGAgent:
    """Agent with no retriever or LLM, warmup and cache off."""
    base: dict[str, Any] = {"llm_warmup": False, "enable_cache": False}
    base.update(overrides)
    return KGRAGAgent(config=AgentConfig(**base), kg_retriever=None, llm=None)


def _clocked(agent: KGRAGAgent):
    """Reset the agent's stage timings and return the dict they fill."""
    agent._stage_clock.timings = {}
    return agent._stage_clock.timings


# --- the timer itself ------------------------------------------------------


def test_a_node_reports_the_time_it_spent():
    agent = _agent()
    timings = _clocked(agent)

    agent._timed("retrieve", lambda state: time.sleep(0.02) or {})({})

    assert timings["retrieve"] >= 20.0


def test_each_node_is_counted_under_its_own_name():
    agent = _agent()
    timings = _clocked(agent)

    agent._timed("retrieve", lambda state: {})({})
    agent._timed("generate", lambda state: {})({})

    assert sorted(timings) == ["generate", "retrieve"]


def test_a_node_that_runs_twice_accumulates():
    # `grade` sends the question back to `rewrite` up to three times, so the
    # times have to add up rather than replace each other.
    agent = _agent()
    timings = _clocked(agent)
    node = agent._timed("rewrite", lambda state: time.sleep(0.01) or {})

    node({})
    first = timings["rewrite"]
    node({})

    assert timings["rewrite"] > first


def test_a_node_that_raises_is_still_timed():
    agent = _agent()
    timings = _clocked(agent)

    def _boom(state):
        time.sleep(0.01)
        raise RuntimeError("vLLM down")

    with pytest.raises(RuntimeError):
        agent._timed("generate", _boom)({})

    assert timings["generate"] >= 10.0


def test_a_node_returns_what_it_returned():
    agent = _agent()
    _clocked(agent)

    out = agent._timed("scope", lambda state: {"in_domain": False})({})

    assert out == {"in_domain": False}


def test_timing_outside_an_invocation_is_dropped_rather_than_failing():
    # The clock is set per invocation; a node called directly by a test or a
    # script has nowhere to write.
    agent = _agent()
    if hasattr(agent._stage_clock, "timings"):
        del agent._stage_clock.timings

    assert agent._timed("scope", lambda state: {"ok": True})({}) == {"ok": True}


# --- one agent, several sessions -------------------------------------------


def test_two_threads_do_not_add_their_stages_together():
    # One agent serves every browser session, so the running totals cannot
    # live on the instance.
    agent = _agent()
    seen: dict[str, dict[str, float]] = {}
    ready = threading.Barrier(2)

    def _session(name: str, sleep_s: float) -> None:
        agent._stage_clock.timings = {}
        ready.wait()
        agent._timed("retrieve", lambda state: time.sleep(sleep_s) or {})({})
        seen[name] = dict(agent._stage_clock.timings)

    threads = [
        threading.Thread(target=_session, args=("slow", 0.05)),
        threading.Thread(target=_session, args=("fast", 0.0)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert seen["slow"]["retrieve"] >= 50.0
    assert seen["fast"]["retrieve"] < 50.0


def test_every_graph_node_is_wrapped():
    # A node added later and left unwrapped is a stage that silently reports
    # nothing, which reads as "it was free".
    agent = _agent()
    nodes = set(agent.graph.get_graph().nodes) - {"__start__", "__end__"}

    assert nodes == {
        "scope",
        "refuse",
        "decompose",
        "route",
        "retrieve",
        "grade",
        "rewrite",
        "generate",
    }
