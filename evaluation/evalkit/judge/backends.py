"""Judge backends: vLLM, local HuggingFace, Anthropic/OpenAI API, Claude Code CLI."""

from __future__ import annotations

import logging
import os
from typing import Any

from evalkit.judge.base import JudgeBackend

logger = logging.getLogger("graphrag")


def _backoff(attempt: int) -> None:
    """Exponential backoff with jitter between retry attempts."""
    import random
    import time

    time.sleep(min(2.0**attempt + random.random(), 30.0))


class VLLMBackend:
    """Judge backend using an OpenAI-compatible vLLM endpoint.

    Reuses the same connection pattern as LLMManager in src/graphrag/llm/manager.py.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str = "",
        api_key: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        """Configure the backend; the client is built on first use.

        Args:
            model_id: Served judge model.
            base_url: vLLM endpoint; ``VLLM_BASE_URL`` when empty.
            api_key: Endpoint key; from the environment when empty.
            max_new_tokens: Judge output budget.
            temperature: Sampling temperature.
        """
        self.model_id = model_id
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._client: Any = None

    def _get_client(self) -> Any:
        """The ChatOpenAI client, built on first use."""
        if self._client is None:
            from langchain_openai import ChatOpenAI  # type: ignore

            # The engine's own client sets both of these on purpose. Without a
            # timeout the OpenAI SDK waits 600 s, and with its default two
            # retries on top of the three below a wedged server -- not a dead
            # one, a wedged one -- costs 600 x 3 x 3, about ninety minutes, with
            # a scoring run apparently still running. Retries belong to the
            # loop in `complete`, which logs them; the SDK's own are invisible.
            self._client = ChatOpenAI(
                model=self.model_id,
                base_url=self.base_url,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                timeout=float(os.getenv("GRAPHRAG_LLM_HTTP_TIMEOUT_SEC", "300")),
                max_retries=0,
            )
        return self._client

    def complete(self, system: str, user: str) -> str:
        """Ask the judge, up to three attempts; ``""`` when all fail."""
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

        client = self._get_client()
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        for attempt in range(3):
            try:
                output = client.invoke(messages)
                return str(output.content).strip()
            except Exception as exc:
                logger.warning("VLLMBackend attempt %d failed: %s", attempt + 1, exc)
                if attempt < 2:
                    _backoff(attempt)
        logger.error("VLLMBackend gave up after 3 attempts; row will carry no score")
        return ""


class LocalHFBackend:
    """Judge backend using a local HuggingFace model.

    Wraps LLMManager from src/graphrag/llm/manager.py to reuse retry logic.
    """

    def __init__(self, model_id: str, max_new_tokens: int = 256) -> None:
        """Configure the backend; the model is loaded on first use."""
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._llm: Any = None

    def _get_llm(self) -> Any:
        """The graphrag LLMManager's model, loaded on first use."""
        if self._llm is None:
            import sys
            from pathlib import Path

            # Ensure src/ is importable. parents[3] is the repo root
            # (evaluation/evalkit/judge/backends.py -> up 3 = repo); normally a
            # no-op because graphrag is installed editable, but it must point at
            # a real directory for environments that skip `pip install -e .`.
            src_path = Path(__file__).resolve().parents[3] / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))

            from graphrag.llm.manager import LLMManager  # type: ignore

            mgr = LLMManager(
                model_id=self.model_id,
                max_new_tokens=self.max_new_tokens,
                use_vllm=False,
            )
            self._llm = mgr.load_llm()
        return self._llm

    def complete(self, system: str, user: str) -> str:
        """Ask the judge, up to three attempts; ``""`` when all fail."""
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

        llm = self._get_llm()
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        for attempt in range(3):
            try:
                output = llm.invoke(messages)
                return str(output.content if hasattr(output, "content") else output).strip()
            except Exception as exc:
                logger.warning("LocalHFBackend attempt %d failed: %s", attempt + 1, exc)
                if attempt < 2:
                    _backoff(attempt)
        logger.error("LocalHFBackend gave up after 3 attempts; row will carry no score")
        return ""


class APIBackend:
    """Judge backend using an external API (Anthropic or OpenAI).

    Requires the appropriate SDK to be installed and API key set.
    """

    def __init__(
        self,
        model_id: str,
        provider: str = "anthropic",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        """Configure the backend.

        Args:
            model_id: Judge model at the provider.
            provider: ``anthropic`` or ``openai``.
            max_tokens: Judge output budget.
            temperature: Sampling temperature.
        """
        self.model_id = model_id
        self.provider = provider.lower()
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        """Ask the configured provider.

        Raises:
            ValueError: If the provider is unknown.
        """
        if self.provider == "anthropic":
            return self._anthropic(system, user)
        if self.provider == "openai":
            return self._openai(system, user)
        raise ValueError(f"Unknown API provider: {self.provider!r}")

    def _anthropic(self, system: str, user: str) -> str:
        """Ask Anthropic, up to three attempts; ``""`` when all fail.

        Raises:
            ImportError: If the ``anthropic`` SDK is not installed.
            anthropic.APIStatusError: On auth or request errors, not retried.
        """
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise ImportError("Install anthropic: pip install anthropic") from exc

        # Auth/invalid-request failures are deterministic: retrying them burns
        # time and hides a misconfiguration behind per-row empty results.
        fatal = (anthropic.AuthenticationError, anthropic.BadRequestError,
                 anthropic.PermissionDeniedError, anthropic.NotFoundError)

        client = anthropic.Anthropic()
        for attempt in range(3):
            try:
                message = client.messages.create(
                    model=self.model_id,
                    max_tokens=self.max_tokens,
                    # temperature must reach the API or the judge samples at the
                    # provider default (1.0) and scores stop being reproducible.
                    temperature=self.temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return str(message.content[0].text).strip()
            except fatal as exc:
                logger.error("APIBackend (anthropic) non-retryable error: %s", exc)
                raise
            except Exception as exc:
                logger.warning("APIBackend (anthropic) attempt %d failed: %s", attempt + 1, exc)
                if attempt < 2:
                    _backoff(attempt)
        logger.error("APIBackend (anthropic) gave up after 3 attempts; row will carry no score")
        return ""

    def _openai(self, system: str, user: str) -> str:
        """Ask OpenAI, up to three attempts; ``""`` when all fail.

        Raises:
            ImportError: If the ``openai`` SDK is not installed.
            openai.APIStatusError: On auth or request errors, not retried.
        """
        try:
            import openai  # type: ignore
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc

        fatal = (openai.AuthenticationError, openai.BadRequestError,
                 openai.PermissionDeniedError, openai.NotFoundError)

        client = openai.OpenAI()
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=self.model_id,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return str(response.choices[0].message.content or "").strip()
            except fatal as exc:
                logger.error("APIBackend (openai) non-retryable error: %s", exc)
                raise
            except Exception as exc:
                logger.warning("APIBackend (openai) attempt %d failed: %s", attempt + 1, exc)
                if attempt < 2:
                    _backoff(attempt)
        logger.error("APIBackend (openai) gave up after 3 attempts; row will carry no score")
        return ""


class ClaudeCodeBackend:
    """Judge backend driving the Claude Code CLI (`claude -p`) headless.

    Uses the Claude Code subscription auth (Pro/Max) rather than a metered API
    key: the `claude` binary must be installed and logged in to a claude.ai
    account. This is the only way to use the subscription programmatically.

    The user prompt is piped via stdin (avoids arg-length/quoting limits on
    large prompts); the system prompt is appended via --append-system-prompt.

    Env overrides:
        CLAUDE_CODE_BIN: path to the claude binary (default "claude").
        CLAUDE_CODE_TIMEOUT: per-call timeout in seconds (default 300).
        CLAUDE_CODE_EXTRA_ARGS: extra CLI args, whitespace-separated.
    """

    def __init__(
        self,
        model_id: str = "sonnet",
        max_tokens: int = 256,  # noqa: ARG002 - kept for backend interface parity
        bin_path: str = "",
        timeout: int = 300,
    ) -> None:
        """Configure the CLI call.

        Args:
            model_id: Claude Code model alias; ``sonnet`` when empty.
            max_tokens: Unused; kept for interface parity.
            bin_path: Path to ``claude``; ``CLAUDE_CODE_BIN`` when empty.
            timeout: Per-call timeout; ``CLAUDE_CODE_TIMEOUT`` overrides it.
        """
        self.model_id = model_id or "sonnet"
        self.bin = bin_path or os.getenv("CLAUDE_CODE_BIN", "claude")
        self.timeout = int(os.getenv("CLAUDE_CODE_TIMEOUT", str(timeout)))
        extra = os.getenv("CLAUDE_CODE_EXTRA_ARGS", "").strip()
        self.extra_args = extra.split() if extra else []

    def complete(self, system: str, user: str) -> str:
        """Run ``claude -p`` on the prompts, up to three attempts; ``""`` when all fail.

        Raises:
            RuntimeError: If the ``claude`` binary is not found.
        """
        import json as _json
        import subprocess

        cmd = [self.bin, "-p", "--output-format", "json", "--model", self.model_id, "--max-turns", "1"]
        if system:
            cmd += ["--append-system-prompt", system]
        cmd += self.extra_args

        for attempt in range(3):
            try:
                proc = subprocess.run(
                    cmd,
                    input=user,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"claude CLI not found ({self.bin!r}). Install Claude Code and run "
                    "`claude login` with your Pro/Max account, or set CLAUDE_CODE_BIN."
                ) from exc
            except subprocess.TimeoutExpired:
                logger.warning("ClaudeCodeBackend attempt %d timed out", attempt + 1)
                _backoff(attempt)
                continue

            if proc.returncode != 0:
                logger.warning(
                    "ClaudeCodeBackend attempt %d exit=%d stderr=%s",
                    attempt + 1, proc.returncode, proc.stderr[:200],
                )
                _backoff(attempt)
                continue

            out = proc.stdout.strip()
            try:
                payload = _json.loads(out)
            except _json.JSONDecodeError:
                return out  # already plain text
            if isinstance(payload, dict):
                if payload.get("is_error"):
                    logger.warning("ClaudeCodeBackend is_error: %s", str(payload)[:200])
                    _backoff(attempt)
                    continue
                return str(payload.get("result", "")).strip()
            return out
        return ""


def make_backend(
    backend: str,
    model_id: str,
    vllm_base_url: str = "",
    vllm_api_key: str = "",
    api_provider: str = "anthropic",
    max_new_tokens: int = 256,
    claude_code_bin: str = "",
) -> JudgeBackend:
    """Factory: return the appropriate JudgeBackend based on *backend* string."""
    if backend in ("vllm", "local_hf", "api") and not model_id.strip():
        # JudgeConfig defaults model_id to "": failing here beats a per-row 404
        # three-retries-then-empty-string loop at scoring time.
        raise ValueError(
            f"backend {backend!r} requires an explicit judge model_id "
            "(JudgeConfig.model_id / --model)"
        )
    if backend == "vllm":
        return VLLMBackend(
            model_id=model_id,
            base_url=vllm_base_url,
            api_key=vllm_api_key,
            max_new_tokens=max_new_tokens,
        )
    if backend == "local_hf":
        return LocalHFBackend(model_id=model_id, max_new_tokens=max_new_tokens)
    if backend == "api":
        return APIBackend(
            model_id=model_id,
            provider=api_provider,
            max_tokens=max_new_tokens,
        )
    if backend == "claude_code":
        return ClaudeCodeBackend(
            model_id=model_id,
            max_tokens=max_new_tokens,
            bin_path=claude_code_bin,
        )
    raise ValueError(
        f"Unknown backend: {backend!r}. Choose from: vllm, local_hf, api, claude_code"
    )
