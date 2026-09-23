"""Head-and-tail trimming of the retrieval context to a token budget."""

from __future__ import annotations

import logging

logger = logging.getLogger("graphrag")


class ContextCompressor:
    """Bounds retrieval context to a token budget by keeping head and tail.

    The budget applies to the context only, not the full prompt (system
    message and question are added on top). When trimming occurs, the middle
    section of the context is dropped entirely — information located there is
    lost. Retrieval orders evidence by relevance, so the head carries the
    strongest signal, but raise ``max_tokens`` if mid-context evidence matters.
    """

    def __init__(self, max_tokens: int, ratio: float = 0.25) -> None:
        """Create a compressor.

        Args:
            max_tokens: Token budget of the context.
            ratio: Estimated tokens per character. Subword tokenizers average
                about four characters per token, so 0.25; a larger value
                over-estimates tokens and trims far too aggressively.
        """
        self.max_tokens = max_tokens
        self.ratio = ratio

    # Evidence blocks are separated by a blank line; lines are the fallback
    # boundary when a single block is larger than half the budget.
    _BLOCK_SEP = "\n\n"

    def _estimate_tokens(self, text: str) -> int:
        """Estimate the token count of ``text`` from its length."""
        return int(len(text) * self.ratio)

    @staticmethod
    def _snap_head(head: str) -> str:
        """Drop a trailing partial block from ``head``."""
        for sep in (ContextCompressor._BLOCK_SEP, "\n"):
            cut = head.rfind(sep)
            if cut > 0:
                return head[:cut]
        return head

    @staticmethod
    def _snap_tail(tail: str) -> str:
        """Drop a leading partial block from ``tail``."""
        for sep in (ContextCompressor._BLOCK_SEP, "\n"):
            cut = tail.find(sep)
            if 0 <= cut < len(tail) - len(sep):
                return tail[cut + len(sep) :]
        return tail

    def compress(self, text: str) -> str:
        """Fit ``text`` into the budget by dropping its middle.

        Args:
            text: Rendered retrieval context.

        Returns:
            ``text`` unchanged when it fits; otherwise its head and tail, each
            about half the budget and cut on block boundaries, joined by a
            ``[... context trimmed ...]`` marker.
        """
        estimated = self._estimate_tokens(text)
        if estimated <= self.max_tokens:
            return text

        char_budget = int(self.max_tokens / self.ratio)
        half = char_budget // 2

        # Cut on block boundaries: a raw character cut would leave a
        # half-rendered evidence entry at each seam (a reference id over a
        # truncated passage, or a passage with no id), which the model then
        # cites or mis-cites.
        head = self._snap_head(text[:half]) or text[:half]
        tail = self._snap_tail(text[-half:]) or text[-half:]

        compressed = head + "\n\n[... context trimmed ...]\n\n" + tail
        logger.warning(
            "Context compressed: %d to %d estimated tokens (middle section dropped)",
            estimated,
            self.max_tokens,
        )
        return compressed
