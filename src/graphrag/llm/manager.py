"""LLM backend: a vLLM endpoint or a local Hugging Face model, with retries,
the domain and evidence gates, answer generation and answer-language control.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

import torch

from graphrag import questions
from graphrag.config import AgentConfig, DEFAULT_MODEL_ID
from graphrag.llm.prompts import PromptLibrary
from graphrag.llm.refusal import looks_like_refusal

logger = logging.getLogger("graphrag")


def _content_log_level() -> int:
    """Log level of the rendered-prompt and raw-output lines of each generation.

    At INFO they flood the operational log: the prompt line repeats the same
    system prompt on every call, and the output line carries text quoted from
    third-party documents, which makes the log unshareable. Both stay useful
    when a bad answer needs explaining, so they are logged at DEBUG rather than
    dropped.

    Returns:
        ``logging.INFO`` when ``GRAPHRAG_LOG_PROMPT_TEXT=1``, else
        ``logging.DEBUG``.
    """
    return (
        logging.INFO
        if os.getenv("GRAPHRAG_LOG_PROMPT_TEXT", "") == "1"
        else logging.DEBUG
    )

# Orthographic markers of Italian used to break ties on short questions, where
# function words alone are too thin a signal.
_ITALIAN_ACCENTED = re.compile(r"[àèéìòù]")
_ITALIAN_ELISION = re.compile(r"\b(?:l|dell|nell|all|sull|dall|un|quell|c|d)'")
# An imperative with the pronoun attached: "spiegameli", "dammene", "mostrameli".
# No English word takes this shape, and such questions ("Spiegameli meglio")
# carry no function words at all, so without this the marker counts both score
# zero and the tie goes to English.
_ITALIAN_IMPERATIVE_CLITIC = re.compile(
    r"\b(?:damm|dimm|famm|spiegam|elencam|indicam|riportam|mostram|parlam)"
    r"(?:i|e(?:l[oaie]|ne))\b"
)


class LLMManager:
    """Loads a chat model once and generates answers with it.

    With ``use_vllm`` the model is served by an OpenAI-compatible vLLM
    endpoint; otherwise it is loaded locally through Hugging Face, 4-bit
    quantised when possible. Generation is greedy in both cases. Prompts come
    only from :class:`~graphrag.llm.prompts.PromptLibrary`, so both backends
    see identical text.
    """

    # Models at least this large (in billions of parameters, read from the
    # model id) never silently fall back to unquantised fp16.
    _LARGE_MODEL_THRESHOLD_B = 30.0

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        warmup: bool = False,
        max_new_tokens: int = 256,
        gpu_memory_fraction: float = 0.92,
        allow_large_model_fp16_fallback: bool = False,
        use_vllm: bool = False,
        vllm_base_url: str = "http://localhost:8000/v1",
    ) -> None:
        """Configure the backend; the model is loaded lazily unless ``warmup``.

        Retries are configured by ``GRAPHRAG_LLM_GENERATE_RETRIES`` (default 2)
        and ``GRAPHRAG_LLM_GENERATE_RETRY_BACKOFF_SEC`` (default 1.0); the API
        key comes from ``VLLM_API_KEY`` or ``OPENAI_API_KEY``.

        Args:
            model_id: Hugging Face model id, or the served model name.
            warmup: Load the model now.
            max_new_tokens: Generation budget per call.
            gpu_memory_fraction: Share of each GPU the local model may use.
            allow_large_model_fp16_fallback: Allow unquantised fp16 loading of
                a large model when 4-bit loading fails; also enabled by
                ``GRAPHRAG_ALLOW_LARGE_MODEL_FP16_FALLBACK``.
            use_vllm: Use the vLLM endpoint instead of a local model.
            vllm_base_url: Base URL of the vLLM OpenAI-compatible API.

        Raises:
            ValueError: If an argument is out of range.
        """
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if gpu_memory_fraction <= 0 or gpu_memory_fraction > 1:
            raise ValueError("gpu_memory_fraction must be in (0, 1]")
        if not vllm_base_url.strip():
            raise ValueError("vllm_base_url must be non-empty")

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.gpu_memory_fraction = gpu_memory_fraction
        self.use_vllm = bool(use_vllm)
        self.vllm_base_url = vllm_base_url.strip().rstrip("/")
        self.vllm_api_key = (
            os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        )
        env_allow_fallback = (
            os.getenv("GRAPHRAG_ALLOW_LARGE_MODEL_FP16_FALLBACK", "").strip().lower()
        )
        self.allow_large_model_fp16_fallback = (
            allow_large_model_fp16_fallback
            or env_allow_fallback
            in {
                "1",
                "true",
                "yes",
                "on",
            }
        )

        self._cached_model: Any | None = None
        self._cached_model_id: str | None = None
        self._load_lock = threading.Lock()
        self._vllm_endpoint_checked = False

        try:
            self.generate_retry_attempts = max(
                1, int(os.getenv("GRAPHRAG_LLM_GENERATE_RETRIES", "2"))
            )
        except ValueError:
            self.generate_retry_attempts = 2
        try:
            self.generate_retry_backoff_sec = max(
                0.0, float(os.getenv("GRAPHRAG_LLM_GENERATE_RETRY_BACKOFF_SEC", "1.0"))
            )
        except ValueError:
            self.generate_retry_backoff_sec = 1.0

        if self.use_vllm:
            logger.info(
                "vLLM mode enabled: gpu_memory_fraction and allow_large_model_fp16_fallback are ignored "
                "for client-side inference (server controls memory/precision)."
            )

        if warmup:
            self.warmup()

    @staticmethod
    def _import_vllm_stack() -> Any:
        """Import ``ChatOpenAI``, with an install hint when it is missing.

        Raises:
            RuntimeError: If ``langchain-openai`` is not installed.
        """
        try:
            from langchain_openai import ChatOpenAI
        except Exception as exc:
            raise RuntimeError(
                "vLLM mode requires langchain-openai. Install it in your runtime env, for example: "
                'conda run -n graphllm python -m pip install "langchain-openai>=0.2,<0.4"'
            ) from exc

        return ChatOpenAI

    @staticmethod
    def _import_hf_stack() -> tuple[Any, Any, Any, Any, Any, Any]:
        """Import the LangChain and transformers classes the local backend needs.

        Returns:
            ``(ChatHuggingFace, HuggingFacePipeline, AutoModelForCausalLM,
            AutoTokenizer, BitsAndBytesConfig, pipeline)``.

        Raises:
            RuntimeError: On the known transformers/huggingface-hub version
                mismatch, with the fix.
        """
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                pipeline,
            )
        except Exception as exc:  # pragma: no cover - depends on runtime env
            text = str(exc).lower()
            if "huggingface-hub" in text and "required" in text:
                raise RuntimeError(
                    "Incompatible transformers/huggingface-hub versions detected. "
                    'Fix with: conda run -n graphllm python -m pip install "huggingface-hub>=0.34.0,<1.0"'
                ) from exc
            raise

        return (
            ChatHuggingFace,
            HuggingFacePipeline,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline,
        )

    @staticmethod
    def _hf_token() -> str | None:
        """Hugging Face token from ``HF_TOKEN`` or ``HUGGINGFACE_HUB_TOKEN``."""
        return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    @staticmethod
    def _model_size_billions(model_id: str) -> float | None:
        """Parameter count in billions read from the model id (``32B``), if any."""
        match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?:\b|[-_/])", model_id)
        if match is None:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @classmethod
    def _is_large_model(cls, model_id: str) -> bool:
        """Whether the model id names at least ``_LARGE_MODEL_THRESHOLD_B`` billions."""
        size_b = cls._model_size_billions(model_id)
        return size_b is not None and size_b >= cls._LARGE_MODEL_THRESHOLD_B

    @staticmethod
    def _is_awq_model(model_id: str) -> bool:
        """Whether the model id names an AWQ-quantised checkpoint."""
        return bool(re.search(r"[-_/]awq(?:[-_/]|$)", model_id, re.IGNORECASE))

    def _build_max_memory(self) -> dict[int | str, str] | None:
        """Per-device memory caps for ``device_map="auto"``.

        Returns:
            ``gpu_memory_fraction`` of each GPU plus a CPU offload budget from
            ``GRAPHRAG_CPU_OFFLOAD_GIB`` (default 64, at least 4); ``None``
            without CUDA.
        """
        if not torch.cuda.is_available():
            return None

        gpu_count = torch.cuda.device_count()
        if gpu_count < 1:
            return None

        max_memory: dict[int | str, str] = {}
        for index in range(gpu_count):
            total_gib = torch.cuda.get_device_properties(index).total_memory / (1024**3)
            usable_gib = max(1, int(total_gib * self.gpu_memory_fraction))
            max_memory[index] = f"{usable_gib}GiB"

        try:
            cpu_offload_gib = int(os.getenv("GRAPHRAG_CPU_OFFLOAD_GIB", "64"))
        except ValueError:
            cpu_offload_gib = 64
        max_memory["cpu"] = f"{max(4, cpu_offload_gib)}GiB"
        return max_memory

    @staticmethod
    def _offload_folder() -> str:
        """Create and return the weight offload directory (``GRAPHRAG_OFFLOAD_DIR``)."""
        offload_dir = Path(os.getenv("GRAPHRAG_OFFLOAD_DIR", "/tmp/graphrag-offload"))
        offload_dir.mkdir(parents=True, exist_ok=True)
        return str(offload_dir)

    @staticmethod
    def _fp16_fallback_message(model_id: str, root_exc: BaseException) -> RuntimeError:
        """Error explaining that fp16 fallback is disabled for a large model."""
        return RuntimeError(
            "4-bit quantized load failed for large model '"
            + model_id
            + "'. To keep production reliability, fp16 fallback is disabled for large models by default. "
            + "Set --allow-large-model-fp16-fallback (or GRAPHRAG_ALLOW_LARGE_MODEL_FP16_FALLBACK=1) "
            + "if you explicitly want this fallback. Root cause: "
            + str(root_exc)
        )

    @staticmethod
    def _is_hf_auth_error(exc: BaseException) -> bool:
        """Whether ``exc`` or its causes look like a gated-model or token error."""
        details: list[str] = []
        current: BaseException | None = exc
        depth = 0
        while current is not None and depth < 6:
            details.append(f"{type(current).__name__}: {current}")
            current = current.__cause__ or current.__context__
            depth += 1

        text = " ".join(details).lower()
        markers = (
            "gated repo",
            "cannot access gated repo",
            "unauthorized",
            "401",
            "localtokennotfounderror",
            "access to model",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _raise_hf_access_error(model_id: str, exc: BaseException) -> None:
        """Re-raise an access failure with instructions for setting a token.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(
            "Cannot load Hugging Face model '"
            + model_id
            + "': access denied or authentication missing.\n"
            + "If the model is gated, request access at https://huggingface.co/"
            + model_id
            + "\n"
            + "Fast path (recommended): export HF_TOKEN and rerun the same command.\n"
            + '  export HF_TOKEN="<your-hf-token>"\n'
            + "Optional persistent login from this conda env:\n"
            + '  $CONDA_PREFIX/bin/python -m huggingface_hub.commands.huggingface_cli login --token "$HF_TOKEN"\n'
            + "You can also use HUGGINGFACE_HUB_TOKEN instead of HF_TOKEN.\n"
            + "You can also switch to an ungated model, for example Qwen/Qwen2.5-7B-Instruct."
        ) from exc

    def _build_llm(self, model_id: str) -> Any:
        """Build the chat model for ``model_id``.

        Locally, with CUDA: AWQ checkpoints load in fp16, others in 4-bit
        bitsandbytes, falling back to fp16 unless the model is large and the
        fallback is not allowed; Flash Attention 2 is used when installed and
        supported, and ``GRAPHRAG_TORCH_COMPILE`` compiles the fp16 model.
        Without CUDA the model loads in fp32 on CPU.

        Args:
            model_id: Model to load.

        Returns:
            A LangChain chat model.

        Raises:
            RuntimeError: If loading fails for access, dependency or fallback
                reasons.
        """
        if self.use_vllm:
            return self._build_vllm_llm(model_id)

        (
            ChatHuggingFace,
            HuggingFacePipeline,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            hf_pipeline,
        ) = self._import_hf_stack()

        logger.info("Loading LLM model: %s", model_id)
        hf_token = self._hf_token()
        model_is_large = self._is_large_model(model_id)

        common_load_kwargs: dict[str, Any] = {
            "token": hf_token,
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            common_load_kwargs["device_map"] = "auto"
            max_memory = self._build_max_memory()
            if max_memory:
                common_load_kwargs["max_memory"] = max_memory
                common_load_kwargs["offload_folder"] = self._offload_folder()

            # Enable Flash Attention 2 if flash-attn package is installed (A40/Ampere+ supported)
            try:
                importlib.metadata.version("flash-attn")
                common_load_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 enabled.")
            except importlib.metadata.PackageNotFoundError:
                pass

        torch_compile = (
            os.getenv("GRAPHRAG_TORCH_COMPILE", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )

        def _load_model_fa2_safe(**kw: Any) -> Any:
            """Load model; strip flash_attention_2 and retry if architecture doesn't support it."""
            try:
                return AutoModelForCausalLM.from_pretrained(model_id, **kw)
            except (ValueError, NotImplementedError) as exc:
                if "attn_implementation" in kw and "flash" in str(exc).lower():
                    kw.pop("attn_implementation")
                    logger.warning(
                        "Flash Attention 2 not supported for '%s', retrying without it: %s",
                        model_id,
                        exc,
                    )
                    return AutoModelForCausalLM.from_pretrained(model_id, **kw)
                raise

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            if torch.cuda.is_available():
                if self._is_awq_model(model_id):
                    try:
                        importlib.metadata.version("autoawq")
                    except importlib.metadata.PackageNotFoundError:
                        raise RuntimeError(
                            "autoawq is required to load AWQ models. "
                            "Install it with: pip install autoawq>=0.2"
                        ) from None
                    # AWQ model: weights already quantized on HuggingFace; skip bitsandbytes
                    logger.info("AWQ model detected; loading in fp16 (skipping bitsandbytes).")
                    base_model = _load_model_fa2_safe(
                        torch_dtype=torch.float16,
                        **common_load_kwargs,
                    )
                    if torch_compile:
                        logger.info("Applying torch.compile(mode='reduce-overhead') to AWQ model.")
                        base_model = torch.compile(base_model, mode="reduce-overhead")
                else:
                    try:
                        importlib.metadata.version("bitsandbytes")
                        base_model = _load_model_fa2_safe(
                            quantization_config=BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True,
                            ),
                            **common_load_kwargs,
                        )
                    except importlib.metadata.PackageNotFoundError:
                        if model_is_large and not self.allow_large_model_fp16_fallback:
                            raise RuntimeError(
                                "bitsandbytes is required for large models (>=30B) in this production profile. "
                                "Install bitsandbytes or use a smaller model, or explicitly allow fp16 fallback."
                            )
                        logger.warning(
                            "bitsandbytes not installed: loading model on GPU without 4-bit quantization."
                        )
                        base_model = _load_model_fa2_safe(
                            torch_dtype=torch.float16,
                            **common_load_kwargs,
                        )
                        if torch_compile:
                            logger.info("Applying torch.compile(mode='reduce-overhead').")
                            base_model = torch.compile(base_model, mode="reduce-overhead")
                    except (
                        Exception
                    ) as exc:  # pragma: no cover - depends on GPU/runtime setup
                        if self._is_hf_auth_error(exc):
                            raise
                        if model_is_large and not self.allow_large_model_fp16_fallback:
                            raise self._fp16_fallback_message(
                                model_id=model_id, root_exc=exc
                            ) from exc
                        logger.warning(
                            "bitsandbytes is installed, but 4-bit loading failed (%s). "
                            "Falling back to standard fp16 GPU loading.",
                            exc,
                        )
                        base_model = _load_model_fa2_safe(
                            torch_dtype=torch.float16,
                            **common_load_kwargs,
                        )
                        if torch_compile:
                            logger.info("Applying torch.compile(mode='reduce-overhead').")
                            base_model = torch.compile(base_model, mode="reduce-overhead")
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    **common_load_kwargs,
                )
        except Exception as exc:
            if self._is_hf_auth_error(exc):
                self._raise_hf_access_error(model_id=model_id, exc=exc)
            raise

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "return_full_text": False,
            "repetition_penalty": 1.05,
        }
        if tokenizer.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = tokenizer.pad_token_id

        generation = hf_pipeline(
            "text-generation",
            model=base_model,
            tokenizer=tokenizer,
            **generation_kwargs,
        )
        return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=generation))

    def classify_in_domain(
        self, question: str, config: Any, known_entities: Sequence[str] = ()
    ) -> bool:
        """Whether the question belongs to the corpus domain.

        One classification call before retrieval, answering a single word.

        The prompt asks for one token, but nothing enforces it: `_build_vllm_llm`
        binds `max_tokens=self.max_new_tokens` on the shared client, so the gate
        call carries the full answer budget and a reasoning model will spend it
        on a <think> block. The parsing below is written for that.

        Args:
            question: The question as typed.
            config: The agent config, read for ``domain_scope``.
            known_entities: Names the graph holds that this question mentions,
                so the model is not asked to recognise an acronym it has never
                seen. Empty leaves the prompt exactly as validated.

        Returns:
            ``True`` when in domain, and on any failure — a broken gate must not
            silence a working demo.
        """
        prompt = PromptLibrary.domain_gate_prompt(
            getattr(config, "domain_scope", ""), known_entities
        )
        try:
            model = self.load_llm()
            output = self._invoke_with_retry(
                model, prompt.invoke({"question": question})
            )
        except Exception as exc:
            logger.warning("Domain gate failed (%s); treating question as in domain", exc)
            return True

        return self._read_gate_verdict(output, question)

    @staticmethod
    def _read_gate_verdict(output: Any, question: str) -> bool:
        """Parse a one-word IN/OUT completion, defaulting to IN.

        Reasoning models open with a <think> block, so the verdict is not
        necessarily at the start: the reasoning block is stripped, then the
        verdict is looked for as a whole word anywhere in what remains.

        Args:
            output: Backend response.
            question: The question, for the log line.

        Returns:
            ``False`` only when the model said OUT (and, if it said both, said
            OUT last); ``True`` otherwise.
        """
        verdict = str(output.content if hasattr(output, "content") else output).strip()
        logger.info("Gate: %r -> %s", question[:80], verdict[:64])
        cleaned = re.sub(
            r"<think>.*?</think>", " ", verdict, flags=re.DOTALL | re.IGNORECASE
        )
        cleaned = re.sub(r"<think>.*\Z", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
        upper = cleaned.upper()
        out_of_domain = re.search(r"\bOUT(?:[_\- ]?OF[_\- ]?DOMAIN)?\b", upper)
        in_domain = re.search(r"\bIN(?:[_\- ]?DOMAIN)?\b", upper)
        if out_of_domain and not in_domain:
            return False
        if out_of_domain and in_domain:
            # Both words present: trust whichever the model said last, which is
            # its conclusion rather than its restatement of the options.
            return in_domain.start() > out_of_domain.start()
        return True

    def classify_answerable(
        self,
        question: str,
        entity_names: Sequence[str] = (),
        passages: Sequence[str] = (),
        sources: Sequence[str] = (),
    ) -> bool:
        """Whether what the collection returned is about the question.

        The evidence-grounded counterpart of `classify_in_domain`: it names no
        domain, so it does not have to be rewritten when the collection grows.

        Args:
            question: The question as typed.
            entity_names: Names the collection holds for the question's terms.
            passages: Snippets of the passages retrieval returned.
            sources: Documents those passages came from.

        Returns:
            ``True`` when the material is about the question, and on any
            failure — a broken gate must not silence a working demo.
        """
        prompt = PromptLibrary.evidence_gate_prompt(entity_names, passages, sources)
        try:
            model = self.load_llm()
            output = self._invoke_with_retry(
                model, prompt.invoke({"question": question})
            )
        except Exception as exc:  # noqa: BLE001 - never let the gate break the demo
            logger.warning("Evidence gate failed (%s); treating question as in domain", exc)
            return True
        return self._read_gate_verdict(output, question)

    def _build_vllm_llm(self, model_id: str) -> Any:
        """Build a greedy ``ChatOpenAI`` client for the vLLM endpoint.

        The request timeout is ``GRAPHRAG_LLM_HTTP_TIMEOUT_SEC`` (default 300).

        Args:
            model_id: Served model name.

        Returns:
            The chat client.
        """
        ChatOpenAI = self._import_vllm_stack()
        logger.info(
            "Using vLLM OpenAI-compatible endpoint at %s for model %s",
            self.vllm_base_url,
            model_id,
        )
        # Without an explicit timeout the OpenAI SDK waits 600 s and retries
        # twice on its own, under this class's own retry loop, so a wedged
        # server can hang an interactive session for most of an hour. Generous
        # on purpose: a long answer from a large model takes minutes, and a cap
        # that fires on a slow-but-working answer is worse than the hang it
        # prevents. The worst case is this value times
        # GRAPHRAG_LLM_GENERATE_RETRIES, since a timeout counts as transient.
        # The KG pipeline has its own, much longer budget (VLLM_HTTP_TIMEOUT):
        # nobody is watching a batch run.
        timeout_sec = float(os.getenv("GRAPHRAG_LLM_HTTP_TIMEOUT_SEC", "300"))
        return ChatOpenAI(
            model=model_id,
            base_url=self.vllm_base_url,
            api_key=self.vllm_api_key,
            temperature=0,
            max_tokens=self.max_new_tokens,
            timeout=timeout_sec,
            # This class already retries transient failures with backoff;
            # the SDK's own retries only multiply the wait.
            max_retries=0,
        )

    @staticmethod
    def _models_url(base_url: str) -> str:
        """The ``/models`` URL under ``base_url``."""
        return urllib.parse.urljoin(base_url.rstrip("/") + "/", "models")

    def _check_vllm_endpoint(self, target_model_id: str) -> None:
        """Health-check the vLLM endpoint before the first call.

        The timeout is ``GRAPHRAG_VLLM_HEALTHCHECK_TIMEOUT_SEC`` (default 5). A
        model missing from the server's list is only logged.

        Args:
            target_model_id: Model the caller is about to use.

        Raises:
            RuntimeError: If ``/models`` is unreachable or returns an error.
        """
        models_url = self._models_url(self.vllm_base_url)
        request = urllib.request.Request(models_url, method="GET")
        if self.vllm_api_key:
            request.add_header("Authorization", f"Bearer {self.vllm_api_key}")

        timeout_sec = float(os.getenv("GRAPHRAG_VLLM_HEALTHCHECK_TIMEOUT_SEC", "5"))
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                status_code = int(getattr(response, "status", 200))
                if status_code >= 400:
                    raise RuntimeError(
                        f"vLLM endpoint returned HTTP {status_code} on {models_url}"
                    )
                payload = response.read().decode("utf-8", errors="ignore")
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            raise RuntimeError(
                "Cannot reach vLLM endpoint at '"
                + models_url
                + "'. Ensure server is running and reachable from this process. "
                + "If using sbatch mode, enable USE_VLLM=1 and verify port forwarding/bind address."
            ) from exc

        reported_models: list[str] = []
        try:
            parsed = json.loads(payload)
            data = parsed.get("data", []) if isinstance(parsed, dict) else []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and isinstance(item.get("id"), str):
                        reported_models.append(item["id"])
        except json.JSONDecodeError:
            reported_models = []

        if reported_models and target_model_id not in reported_models:
            preview = ", ".join(reported_models[:5])
            logger.warning(
                "Requested model '%s' not listed by vLLM /models. Reported models: %s",
                target_model_id,
                preview,
            )

    def load_llm(self, model_id: str | None = None) -> Any:
        """Return the chat model, building it once per model id.

        Thread-safe: concurrent callers share one load.

        Args:
            model_id: Model to load; defaults to ``self.model_id``, which is
                updated to the loaded one.

        Returns:
            The cached chat model.
        """
        target_model_id = model_id or self.model_id

        if self._cached_model is not None and self._cached_model_id == target_model_id:
            return self._cached_model

        with self._load_lock:
            if (
                self._cached_model is not None
                and self._cached_model_id == target_model_id
            ):
                return self._cached_model

            if self.use_vllm and not self._vllm_endpoint_checked:
                self._check_vllm_endpoint(target_model_id)
                self._vllm_endpoint_checked = True

            self._cached_model = self._build_llm(target_model_id)
            self._cached_model_id = target_model_id
            self.model_id = target_model_id
            return self._cached_model

    def warmup(self) -> None:
        """Load the model now instead of on the first call."""
        self.load_llm()

    @staticmethod
    def _is_transient_error(exc: BaseException) -> bool:
        """Whether ``exc`` looks like a timeout, connection or server error."""
        text = f"{type(exc).__name__}: {exc}".lower()
        markers = (
            "timeout",
            "timed out",
            "connection",
            "temporarily unavailable",
            "service unavailable",
            "serviceunavailable",
            "bad gateway",
            "gateway timeout",
            "rate limit",
            "ratelimiterror",
            "apiconnectionerror",
            "apitimeouterror",
            "internalservererror",
        )
        return any(marker in text for marker in markers)

    def _invoke_with_retry(
        self, model: Any, payload: Any, on_token: Callable[[str | None], None] | None = None
    ) -> Any:
        """Invoke the model, retrying only on transient network/server errors.

        Args:
            model: The chat backend.
            payload: The rendered prompt.
            on_token: Called with each piece of text as it arrives, so a reader
                sees the answer fill instead of waiting. Called with ``None``
                when a retry discards what was already emitted, so the caller
                can drop it instead of appending the answer twice.

        Returns:
            The backend response. Streaming returns the summed chunks, which
            carry the same ``content`` and ``response_metadata`` — including
            the ``finish_reason`` the token-limit check reads.

        Raises:
            Exception: The backend's error, when it is not transient or the
                retries are exhausted.
        """
        attempts = max(1, self.generate_retry_attempts)
        for attempt in range(1, attempts + 1):
            try:
                if on_token is None or not hasattr(model, "stream"):
                    return model.invoke(payload)
                if attempt > 1:
                    # The previous attempt may have emitted a partial answer.
                    on_token(None)
                merged = None
                for chunk in model.stream(payload):
                    merged = chunk if merged is None else merged + chunk
                    piece = getattr(chunk, "content", "") or ""
                    if piece:
                        on_token(str(piece))
                if merged is None:
                    # A stream that yielded nothing is not an answer; fall back
                    # to the blocking call rather than returning an empty one.
                    return model.invoke(payload)
                return merged
            except Exception as exc:
                if attempt >= attempts or not self._is_transient_error(exc):
                    raise
                backoff_sec = self.generate_retry_backoff_sec * attempt
                logger.warning(
                    "LLM invoke transient failure. retry=%d/%d backoff_sec=%.2f error=%s",
                    attempt,
                    attempts,
                    backoff_sec,
                    exc,
                )
                if backoff_sec > 0:
                    time.sleep(backoff_sec)

    def generate(
        self,
        query: str,
        context: str,
        config: AgentConfig,
        transcript: str = "",
        on_token: Callable[[str | None], None] | None = None,
    ) -> dict[str, str]:
        """Generate the answer to ``query`` from ``context``.

        The answer cut by the token cap is trimmed to its last sentence. An
        empty or refusing answer is retried once with a stricter prompt, unless
        parametric fallback is allowed. With ``enforce_language``, an answer in
        the wrong language is regenerated once.

        Args:
            query: The question as typed.
            context: Rendered retrieval context.
            config: Agent configuration.
            transcript: Conversation so far; empty adds nothing to the prompt.
            on_token: Streaming callback for the first generation only; see
                :meth:`_invoke_with_retry`.

        Returns:
            ``answer``, ``pre_retry_answer`` (before the refusal retry) and
            ``refusal_retry_applied``.
        """
        response_language = self._answer_language(query, transcript, config.fallback_language)

        # PromptLibrary is the single source of truth for prompts: both the
        # vLLM and local HF backends must see the same prompt so their answers
        # stay comparable across experiments.
        # Definitional questions are detected on the question, not asked of
        # the model: one regex instead of a classifier call on every turn.
        definitional = config.prefer_verbatim_definitions and questions.is_definitional(
            query
        )
        if definitional:
            logger.info("Definitional question detected; asking for a verbatim opening")

        prompt = PromptLibrary.answer_prompt(
            config,
            language=response_language if config.enforce_language else None,
            definitional=definitional,
            transcript=bool(transcript),
        )
        slots: dict[str, str] = {"question": query, "context": context}
        if transcript:
            slots["transcript"] = transcript
        rendered = prompt.invoke(slots)

        logger.log(
            _content_log_level(),
            "Rendered prompt (first 500 chars): %s",
            str(rendered)[:500],
        )
        logger.info("Context length (chars): %d", len(context))

        model = self.load_llm()
        # Only the first generation streams. The rescue retry and the language
        # pass below rewrite the answer wholesale, and streaming those would
        # show a reader two answers for one question; the caller replaces what
        # it streamed with the final text either way.
        # `on_token` is passed only when someone is listening, so test doubles
        # of `_invoke_with_retry` without that parameter keep working.
        output = (
            self._invoke_with_retry(model, rendered, on_token=on_token)
            if on_token is not None
            else self._invoke_with_retry(model, rendered)
        )

        answer = str(output.content if hasattr(output, "content") else output).strip()
        logger.log(
            _content_log_level(), "LLM raw output (first 800 chars): %s", answer[:800]
        )
        # The shape of the turn stays at INFO: it is what an operator reads to
        # tell a truncated answer from an empty one, and it carries no text.
        logger.info("Answer length (chars): %d", len(answer))

        # Kept so abstention can be measured on what the model said before any
        # rescue retry rewrote it.
        pre_retry_answer = answer
        refusal_retry_applied = False

        if self._hit_token_limit(output):
            logger.warning(
                "Answer hit max_new_tokens=%d and was cut mid-sentence; trimming "
                "to the last complete sentence",
                self.max_new_tokens,
            )
            answer = self._trim_to_last_sentence(answer)

        # If model returned empty or a generic refusal, try a stricter fallback prompt once.
        #
        # Skipped when parametric fallback is authorised: there the model may
        # answer from its own knowledge as long as it marks the statement, so a
        # refusal means it has nothing to offer from either source. Retrying with
        # "use only the provided context" would talk it out of a decision it was
        # entitled to make and turn a correct abstention into an answer.
        if (
            looks_like_refusal(answer)
            and context
            and str(context).strip()
            and not config.allow_parametric_fallback
        ):
            try:
                logger.info("LLM refusal detected; attempting fallback retry...")
                fallback_prompt = PromptLibrary.refusal_retry_prompt(
                    language=response_language
                ).invoke({"question": query, "context": context})
                output2 = self._invoke_with_retry(model, fallback_prompt)
                answer2 = str(
                    output2.content if hasattr(output2, "content") else output2
                ).strip()
                if answer2 and not looks_like_refusal(answer2):
                    logger.log(
                        _content_log_level(),
                        "Fallback retry succeeded: %s",
                        answer2[:500],
                    )
                    # The pre-retry answer is kept: abstention measured on the
                    # final answer would measure post-retry behaviour.
                    refusal_retry_applied = True
                    answer = answer2
                elif answer2:
                    # A second refusal is not a rescue. Keeping the first
                    # formulation preserves the model's own wording.
                    logger.info("Fallback retry also refused; keeping first answer")
                else:
                    logger.info("Fallback retry returned empty answer")
            except Exception as exc:
                # Best-effort retry: on error keep the original answer.
                logger.warning("Fallback retry failed: %s", exc)
                pass

        # Do not apply aggressive rule-based extraction when the model refuses
        # to answer. Preserve the model's refusal so failures remain transparent
        # to the caller instead of being converted into questionable lists.

        if config.enforce_language and answer:
            answer = self._enforce_answer_language(
                model=model,
                query=query,
                context=context,
                config=config,
                answer=answer,
                target_language=response_language,
                transcript=transcript,
            )

        return {
            "answer": answer,
            "pre_retry_answer": pre_retry_answer,
            "refusal_retry_applied": refusal_retry_applied,
        }

    def _enforce_answer_language(
        self,
        model: Any,
        query: str,
        context: str,
        config: AgentConfig,
        answer: str,
        target_language: str,
        transcript: str = "",
    ) -> str:
        """Regenerate once when the answer came back in the wrong language.

        A single retry: a second wrong-language answer means the constraint is
        not what is failing, and further calls only add latency.

        Args:
            model: The already-loaded chat model.
            query: The user question, as sent to the first attempt.
            context: The retrieved context, unchanged.
            config: Agent configuration.
            answer: The answer produced by the first attempt.
            target_language: ``"it"`` or ``"en"``, detected on the question.
            transcript: Conversation so far, passed to the retry prompt.

        Returns:
            The retried answer when it is in the target language, otherwise the
            original one — a wrong-language answer still beats a worse answer.
        """
        if self._detect_text_language(answer) == target_language:
            return answer

        logger.warning(
            "Answer language mismatch: expected=%s. Retrying once with a "
            "reinforced constraint.",
            target_language,
        )
        try:
            retry_slots: dict[str, str] = {"question": query, "context": context}
            if transcript:
                retry_slots["transcript"] = transcript
            retry_prompt = PromptLibrary.answer_prompt(
                config,
                language=target_language,
                reinforce_language=True,
                definitional=(
                    config.prefer_verbatim_definitions
                    and questions.is_definitional(query)
                ),
                transcript=bool(transcript),
            ).invoke(retry_slots)
            output = self._invoke_with_retry(model, retry_prompt)
            retried = str(
                output.content if hasattr(output, "content") else output
            ).strip()
        except Exception as exc:  # noqa: BLE001 - best effort, keep first answer
            logger.warning("Language retry failed: %s", exc)
            return answer

        if retried and self._detect_text_language(retried) == target_language:
            logger.info("Language retry produced a %s answer", target_language)
            return retried

        logger.warning(
            "Language retry did not fix the language; keeping the first answer"
        )
        return answer

    @staticmethod
    def _hit_token_limit(output: Any) -> bool:
        """Whether generation stopped because it ran out of tokens.

        Args:
            output: The backend response; only OpenAI-compatible backends carry
                ``response_metadata['finish_reason']``.

        Returns:
            ``True`` when the backend reported a length stop.
        """
        metadata = getattr(output, "response_metadata", None)
        if not isinstance(metadata, dict):
            return False
        return str(metadata.get("finish_reason", "")).lower() == "length"

    @staticmethod
    def _trim_to_last_sentence(answer: str) -> str:
        """Drop a dangling half-sentence left by the token cap.

        Only the trailing fragment goes: a reader seeing "…e coerente con le
        fonti, la specifica" reads it as a bug, which it is, and the fragment
        carries no information anyway.

        Args:
            answer: The generated answer, possibly cut mid-sentence.

        Returns:
            The answer up to the last sentence end, or unchanged when no
            sentence boundary is left to cut back to.
        """
        text = str(answer or "").rstrip()
        if not text or text[-1] in ".!?:»\"')":
            return text

        cut = max(text.rfind(mark) for mark in (". ", ".\n", "! ", "? ", ".", "!", "?"))
        if cut <= 0:
            return text
        # Keep at least a paragraph: an answer trimmed down to one sentence is
        # worse than the fragment it was meant to hide.
        if cut < len(text) * 0.4:
            return text
        return text[: cut + 1].rstrip()

    @staticmethod
    def _detect_text_language(text: str) -> str:
        """Detect the language of generated prose.

        Strips what carries no language signal but plenty of foreign tokens:
        reference tags, the trailing source list whose document titles are
        mostly English even under an Italian answer, and verbatim quotations,
        which are deliberately left in the source's language.

        Args:
            text: The generated answer.

        Returns:
            ``"it"`` or ``"en"``.
        """
        body = re.split(r"\n\s*(?:Fonti|Sources)\s*:", str(text or ""))[0]
        body = re.sub(r"\[[STst]\s*\d+(?:\s*,\s*[STst]\s*\d+)*\]", " ", body)
        body = re.sub(r"«[^»]*»", " ", body)
        return LLMManager._detect_query_language(body)

    @staticmethod
    def _answer_language(query: str, transcript: str = "", fallback: str = "en") -> str:
        """The language to answer in, using the conversation when the turn is mute.

        A continuation can carry no marker at all: "Non ho capito niente" scores
        zero on both sides, and the tie in `_detect_query_language` would send
        an Italian conversation to English. The words the user typed stay the
        first authority — a mid-conversation switch of language is honoured —
        and the transcript is consulted only when they say nothing.

        Args:
            query: The question as typed.
            transcript: Conversation so far.
            fallback: The language when neither carries a signal, as a bare
                term ("scotta") on the first turn does.

        Returns:
            ``"it"`` or ``"en"``; ``fallback`` when neither carries a signal.
        """
        it_score, en_score = LLMManager._language_scores(query)
        if it_score or en_score:
            return "it" if it_score > en_score else "en"
        it_score, en_score = LLMManager._language_scores(transcript)
        if it_score or en_score:
            return "it" if it_score > en_score else "en"
        return fallback

    @staticmethod
    def _language_scores(query: str) -> tuple[int, int]:
        """Italian and English marker counts for ``query``.

        Split out of `_detect_query_language` so a caller can tell a decision
        from the absence of one: both scores zero means the text carried no
        evidence either way, not that it is English.

        Args:
            query: Text to score.

        Returns:
            ``(italian_score, english_score)``.
        """
        text = str(query or "").strip().lower()
        if not text:
            return 0, 0

        italian_markers = {
            # Articulated prepositions are a closed class and are listed in
            # full paradigms: a missing form can leave a short question
            # ("Sai dirmi qualcosa sul system thinking?") with no marker on
            # either side, and the tie goes to English.
            # "i" is deliberately left out: lowercased, it is also English "I".
            "sul", "sullo", "sugli",
            "dal", "dallo", "dalla", "dai", "dagli", "dalle",
            "nello", "nei",
            "al", "allo", "alla", "ai",
            "lo", "le",
            # Frequent in real questions.
            "sai", "dirmi", "qualcosa", "cui", "ha", "hanno",
            "il",
            "la",
            "gli",
            "della",
            "delle",
            "perche",
            "perché",
            "quali",
            "quale",
            "come",
            "sono",
            "rispetto",
            "tra",
            "sulla",
            # Elided forms tokenize with the apostrophe attached, so plain
            # articles never match them; accented "è" and "cos'è" are the
            # strongest single-token signals in short Italian questions.
            "è",
            "che",
            "cosa",
            "cos'è",
            "qual",
            "dove",
            "chi",
            "quando",
            "quanto",
            "quanti",
            "quante",
            "del",
            "dei",
            "dello",
            "degli",
            "nel",
            "nella",
            "una",
            "uno",
            "più",
            "può",
            "viene",
            "vengono",
            # Bare prepositions/articles that have no English homograph:
            # they carry verbless fragments like "Esempi di X a Torino?".
            "di",
            "da",
            "su",
            "un",
            # Imperative openings common in follow-ups ("Mi indichi…",
            # "Approfondisci…"): often the only Italian tokens in a short turn.
            "mi",
            "dimmi",
            "dammi",
            "elenca",
            "indica",
            "indichi",
            "spiega",
            "approfondisci",
            "riporta",
            "esempi",
            "secondo",
            "per",
            "con",
            "questo",
            "questa",
            "questi",
            "queste",
            "quel",
            "quella",
            "anche",
            "invece",
            "quindi",
            "senza",
            "sulle",
            "sui",
            "nelle",
            "negli",
            "alle",
            "agli",
        }
        # Kept at comparable coverage to the Italian set above: a much smaller
        # set would give Italian a structural advantage on every mixed sentence
        # and, with `enforce_language` on, answer English questions in Italian.
        # Homographs are deliberately absent from both sets: "in" and "a" are
        # high-frequency function words in Italian too, and scoring them as
        # English would turn "Approfondisci l'economia circolare in Piemonte"
        # into an English question.
        english_markers = {
            "about", "all", "an", "and", "any", "are", "as", "at", "be",
            "been", "between", "both", "but", "by", "can", "could", "did", "do",
            "does", "each", "for", "from", "give", "has", "have", "how",
            "into", "is", "it", "its", "list", "many", "may", "much", "must",
            "of", "on", "or", "other", "our", "over", "regarding", "should",
            "some", "such", "than", "that", "the", "their", "them", "then",
            "there", "these", "they", "this", "those", "through", "to", "under",
            "was", "were", "what", "what's", "when", "where", "which", "while",
            "who", "why", "will", "with", "within", "would",
        }

        tokens = re.findall(r"[a-zà-öø-ÿ']+", text)
        it_score = sum(1 for token in tokens if token in italian_markers)
        en_score = sum(1 for token in tokens if token in english_markers)

        # Short questions carry almost no function words ("Definizione di SEeD?",
        # "Cos'è la coevoluzione?"): orthography is then the strongest remaining
        # signal. It is only decisive when no English function word appeared at
        # all — otherwise a borrowed or accented noun ("café", "Fassio's
        # coevoluzione") would hand a free point to Italian in a plainly English
        # sentence.
        if en_score == 0:
            if _ITALIAN_ACCENTED.search(text):
                it_score += 1
            if _ITALIAN_ELISION.search(text):
                it_score += 1
            if _ITALIAN_IMPERATIVE_CLITIC.search(text):
                it_score += 1

        return it_score, en_score

    @staticmethod
    def _detect_query_language(query: str) -> str:
        """The language to answer ``query`` in.

        A tie goes to English, which is right for a lone question — but wrong
        for a turn that carries no signal at all; see `_answer_language`.

        Args:
            query: Text to classify.

        Returns:
            ``"it"`` when Italian markers outnumber English ones, else ``"en"``.
        """
        it_score, en_score = LLMManager._language_scores(query)
        return "it" if it_score > en_score else "en"

