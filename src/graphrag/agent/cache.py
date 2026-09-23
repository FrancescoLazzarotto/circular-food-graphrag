"""Thread-safe LRU cache for retrieval results."""

from __future__ import annotations

import copy
import hashlib
import threading
from collections import OrderedDict
from typing import Any


class LRUCache:
    """Retrieval-result cache that hands out copies, never its own objects.

    The cached value is a dict of retrieved nodes and triples that the caller
    merges into LangGraph state, where downstream nodes are free to mutate it.
    Values are deep-copied on the way in and on the way out, so an in-place
    edit never reaches a later turn that hits the same key.

    Every operation holds a lock: in the Streamlit demo one agent, and so one
    cache, is shared by every browser session, each running in its own
    thread, and ``OrderedDict`` is not safe under concurrent lookup, reorder
    and eviction.
    """

    def __init__(self, maxsize: int = 256) -> None:
        """Create an empty cache holding at most ``maxsize`` entries."""
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._maxsize = maxsize
        # Held across the copy too: releasing it earlier would let an eviction
        # land between the lookup and the read, which is the race itself.
        self._lock = threading.Lock()

    @staticmethod
    def _key(query: str, mode: str) -> str:
        """SHA-256 hex digest of ``mode::query``."""
        raw = f"{mode}::{query}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, query: str, mode: str) -> Any | None:
        """Look up a cached result and mark it most recently used.

        Args:
            query: Retrieval query.
            mode: Retrieval mode the result was computed for.

        Returns:
            A deep copy of the cached value, or ``None`` on a miss.
        """
        key = self._key(query, mode)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return copy.deepcopy(self._cache[key])
            return None

    def put(self, query: str, mode: str, value: Any) -> None:
        """Store a deep copy of ``value``, evicting the least recently used.

        Args:
            query: Retrieval query.
            mode: Retrieval mode the result was computed for.
            value: Result to cache.
        """
        key = self._key(query, mode)
        # Copied before the lock: the value belongs to the caller and nothing
        # else can reach it yet, so the copy needs no protection and the lock
        # is held only for the dictionary work.
        stored = copy.deepcopy(value)
        with self._lock:
            self._cache[key] = stored
            self._cache.move_to_end(key)
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)
