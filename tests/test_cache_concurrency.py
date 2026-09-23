"""The retrieval cache is shared by every browser session, in separate threads.

Streamlit gives each session its own thread and `@st.cache_resource` gives them
all the same agent, so the same `LRUCache`. `OrderedDict` is not safe against
that: without the lock, `get` looks a key up, moves it to the end and then
reads it, and an eviction landing between the lookup and the read raises
`KeyError` — a failed turn for whoever loses the race, on a cache whose whole
purpose is to make turns cheaper.

That window is three bytecodes wide, so it does not open at the default thread
switch interval. `maxsize=1` keeps eviction constant and a nanosecond switch
interval puts a context switch almost everywhere, which is what makes the test
a guard rather than a formality.
"""

from __future__ import annotations

import sys
import threading

import pytest

from graphrag.agent.cache import LRUCache

THREADS = 16
ROUNDS = 2000
KEYS = 512
MAXSIZE = 1


@pytest.fixture
def switch_everywhere():
    """Force a context switch almost between every bytecode, then restore."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-9)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


def _hammer(cache: LRUCache, errors: list[BaseException], seed: int) -> None:
    """Put and get from one thread, collecting any exception instead of raising."""
    try:
        for i in range(ROUNDS):
            key = str((i * 13 + seed) % KEYS)
            cache.put(key, "m", {"nodes": [key]})
            got = cache.get(key, "m")
            if got is not None:
                assert got["nodes"] == [key]
    except BaseException as exc:  # noqa: BLE001 - reported per thread, not raised
        errors.append(exc)


def _stress(cache: LRUCache) -> list[BaseException]:
    """Run `_hammer` on `THREADS` threads at once and return what they raised."""
    errors: list[BaseException] = []
    threads = [
        threading.Thread(target=_hammer, args=(cache, errors, seed))
        for seed in range(THREADS)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def test_a_concurrent_eviction_does_not_fail_a_turn(switch_everywhere) -> None:
    """Without the lock this raises KeyError in nearly every thread."""
    errors = _stress(LRUCache(maxsize=MAXSIZE))
    assert not errors, f"{len(errors)}/{THREADS} threads failed, first: {errors[0]!r}"


def test_the_size_limit_holds_under_contention(switch_everywhere) -> None:
    """Two concurrent puts could each see the size over the limit and evict."""
    cache = LRUCache(maxsize=MAXSIZE)
    assert not _stress(cache)
    assert len(cache._cache) <= MAXSIZE


def test_a_stored_value_is_not_the_callers_object() -> None:
    """The copy-on-write contract holds under the lock too."""
    cache = LRUCache(maxsize=2)
    value = {"nodes": ["a"]}
    cache.put("q", "m", value)
    value["nodes"].append("b")
    assert cache.get("q", "m") == {"nodes": ["a"]}
    got = cache.get("q", "m")
    got["nodes"].append("c")
    assert cache.get("q", "m") == {"nodes": ["a"]}
