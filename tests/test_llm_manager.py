"""What the LLM layer decides before, around and after a generation call.

`llm/manager.py` holds the two gates' plumbing, the retry policy, the endpoint
check and the hand-written language detector — the one that answers an Italian
question in English when the tie goes the wrong way.

No model is loaded and no endpoint is contacted: `load_llm` and the HTTP probe
are replaced. What is pinned is the deciding.
"""

from __future__ import annotations

import json
import logging
import urllib.error
from typing import Any

import pytest

from graphrag.llm.manager import LLMManager


class _Output:
    """Chat-model output with an optional finish reason."""

    def __init__(self, content: str, finish_reason: str | None = None) -> None:
        self.content = content
        self.response_metadata = (
            {"finish_reason": finish_reason} if finish_reason is not None else {}
        )


class _Model:
    """Returns, or raises, one outcome per invoke."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.payloads: list[Any] = []

    def invoke(self, payload: Any) -> Any:
        self.payloads.append(payload)
        outcome = self.outcomes.pop(0) if self.outcomes else _Output("IN")
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _manager(monkeypatch, outcomes: list[Any] | None = None, **kwargs: Any) -> LLMManager:
    """An `LLMManager` over a `_Model` of `outcomes`, retries bypassed."""
    manager = LLMManager(warmup=False, **kwargs)
    model = _Model(outcomes or [])
    monkeypatch.setattr(manager, "load_llm", lambda model_id=None: model)
    monkeypatch.setattr(LLMManager, "_invoke_with_retry", lambda self, m, p: m.invoke(p))
    manager._fake_model = model  # type: ignore[attr-defined]
    return manager


class _Config:
    """Config carrying only the domain scope."""

    domain_scope = "circular economy for food"


# --- the constructor refuses nonsense --------------------------------------


@pytest.mark.parametrize("value", [0, -1])
def test_a_token_budget_below_one_is_refused(value):
    with pytest.raises(ValueError, match="max_new_tokens"):
        LLMManager(max_new_tokens=value)


@pytest.mark.parametrize("value", [0.0, -0.5, 1.5])
def test_a_memory_fraction_outside_the_unit_interval_is_refused(value):
    with pytest.raises(ValueError, match="gpu_memory_fraction"):
        LLMManager(gpu_memory_fraction=value)


def test_an_empty_endpoint_is_refused():
    with pytest.raises(ValueError, match="vllm_base_url"):
        LLMManager(vllm_base_url="   ")


# --- reading a gate verdict ------------------------------------------------


@pytest.mark.parametrize("verdict", ["IN", "in", " IN ", "IN_DOMAIN", "IN-DOMAIN"])
def test_an_in_verdict_is_read_as_in_domain(verdict):
    assert LLMManager._read_gate_verdict(_Output(verdict), "q") is True


@pytest.mark.parametrize("verdict", ["OUT", "out", "OUT_OF_DOMAIN", "OUT-OF-DOMAIN"])
def test_an_out_verdict_is_read_as_out_of_domain(verdict):
    assert LLMManager._read_gate_verdict(_Output(verdict), "q") is False


def test_a_reasoning_block_before_the_verdict_does_not_flip_it():
    # Reading only the first three characters (`startswith("OUT")`) would let
    # any preamble turn a refusal into an acceptance — and reasoning models
    # open with a <think> block.
    output = _Output("<think>The user asks about pasta, which is food but…</think>\nOUT")

    assert LLMManager._read_gate_verdict(output, "q") is False


def test_an_unterminated_reasoning_block_is_still_stripped():
    output = _Output("<think>reasoning that never closes and mentions OUT")

    assert LLMManager._read_gate_verdict(output, "q") is True


def test_when_both_words_appear_the_last_one_is_the_conclusion():
    assert LLMManager._read_gate_verdict(_Output("OUT or IN? IN"), "q") is True
    assert LLMManager._read_gate_verdict(_Output("IN or OUT? OUT"), "q") is False


def test_a_reply_saying_neither_is_read_as_in_domain():
    # A broken gate must not silence a working demo.
    assert LLMManager._read_gate_verdict(_Output("I am not sure"), "q") is True
    assert LLMManager._read_gate_verdict(_Output(""), "q") is True


def test_a_bare_string_reply_is_read_too():
    assert LLMManager._read_gate_verdict("OUT", "q") is False


# --- the two gates ---------------------------------------------------------


def test_the_domain_gate_asks_the_model_and_reports_its_verdict(monkeypatch):
    manager = _manager(monkeypatch, [_Output("OUT")])

    assert manager.classify_in_domain("come si fa la carbonara?", _Config()) is False


def test_a_domain_gate_that_cannot_reach_the_model_lets_the_question_in(monkeypatch, caplog):
    manager = _manager(monkeypatch, [RuntimeError("connection refused")])

    with caplog.at_level(logging.WARNING):
        assert manager.classify_in_domain("q", _Config()) is True

    assert "Domain gate failed" in caplog.text


def test_the_evidence_gate_asks_the_model_and_reports_its_verdict(monkeypatch):
    manager = _manager(monkeypatch, [_Output("OUT")])

    assert manager.classify_answerable("q", ["Scotta"], ["un passaggio"]) is False


def test_an_evidence_gate_that_cannot_reach_the_model_lets_the_question_in(
    monkeypatch, caplog
):
    manager = _manager(monkeypatch, [RuntimeError("connection refused")])

    with caplog.at_level(logging.WARNING):
        assert manager.classify_answerable("q", ["Scotta"]) is True

    assert "Evidence gate failed" in caplog.text


@pytest.mark.parametrize("name", ["Progetto {LIFE}", "a {b} c", "}{"])
def test_a_node_name_with_a_brace_no_longer_silences_the_domain_gate(monkeypatch, name):
    # The names come from the graph and the template parses what it is given,
    # so an unescaped `{` raises KeyError inside prompt.invoke — which the
    # caller swallows by returning "in domain", and the gate silently does not
    # run for that question.
    manager = _manager(monkeypatch, [_Output("OUT")])

    assert manager.classify_in_domain("q", _Config(), [name]) is False
    assert manager._fake_model.payloads  # the model was reached


def test_an_escaped_name_still_reads_as_itself_in_the_prompt(monkeypatch):
    manager = _manager(monkeypatch, [_Output("IN")])

    manager.classify_in_domain("q", _Config(), ["Progetto {LIFE}"])

    rendered = str(manager._fake_model.payloads[0])
    assert "Progetto {LIFE}" in rendered
    assert "{{" not in rendered


def test_the_evidence_gate_survives_the_same_name(monkeypatch):
    manager = _manager(monkeypatch, [_Output("OUT")])

    assert manager.classify_answerable("q", ["Progetto {LIFE}"]) is False
    assert manager._fake_model.payloads  # it did reach the model


# --- retrying a generation call --------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "APITimeoutError: timed out",
        "connection reset",
        "Service Unavailable",
        "502 Bad Gateway",
        "gateway timeout",
        "rate limit exceeded",
        "InternalServerError",
    ],
)
def test_a_transport_failure_is_worth_another_attempt(message):
    assert LLMManager._is_transient_error(RuntimeError(message)) is True


@pytest.mark.parametrize(
    "message", ["ValueError: bad prompt", "context length exceeded", "invalid model id"]
)
def test_a_refusal_from_the_server_is_not_retried(message):
    assert LLMManager._is_transient_error(ValueError(message)) is False


def test_a_transient_failure_is_retried_until_it_works(monkeypatch):
    monkeypatch.setattr("graphrag.llm.manager.time.sleep", lambda _s: None)
    manager = LLMManager(warmup=False)
    manager.generate_retry_attempts = 3
    model = _Model([RuntimeError("timed out"), _Output("va bene")])

    assert manager._invoke_with_retry(model, "payload").content == "va bene"
    assert len(model.payloads) == 2


def test_a_permanent_failure_is_raised_at_once():
    manager = LLMManager(warmup=False)
    model = _Model([ValueError("bad prompt")])

    with pytest.raises(ValueError, match="bad prompt"):
        manager._invoke_with_retry(model, "payload")

    assert len(model.payloads) == 1


def test_the_retries_run_out(monkeypatch):
    monkeypatch.setattr("graphrag.llm.manager.time.sleep", lambda _s: None)
    manager = LLMManager(warmup=False)
    manager.generate_retry_attempts = 2
    model = _Model([RuntimeError("timed out")] * 5)

    with pytest.raises(RuntimeError, match="timed out"):
        manager._invoke_with_retry(model, "payload")

    assert len(model.payloads) == 2


# --- reaching the endpoint -------------------------------------------------


@pytest.mark.parametrize(
    "base, expected",
    [
        ("http://localhost:8000/v1", "http://localhost:8000/v1/models"),
        ("http://localhost:8000/v1/", "http://localhost:8000/v1/models"),
    ],
)
def test_the_models_url_is_built_from_the_base(base, expected):
    assert LLMManager._models_url(base) == expected


class _Response:
    """HTTP response over a fixed payload, usable as a context manager."""

    def __init__(self, payload: str, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def read(self) -> bytes:
        return self._payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_an_unreachable_endpoint_says_which_url_it_tried(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("refused")),
    )
    manager = LLMManager(warmup=False, vllm_base_url="http://localhost:8000/v1")

    with pytest.raises(RuntimeError, match="localhost:8000/v1/models"):
        manager._check_vllm_endpoint("Qwen/Qwen2.5-32B-Instruct-AWQ")


def test_an_http_error_from_the_endpoint_is_reported(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Response("{}", 503))
    manager = LLMManager(warmup=False)

    with pytest.raises(RuntimeError, match="Cannot reach vLLM endpoint"):
        manager._check_vllm_endpoint("any-model")


def test_a_served_model_that_matches_passes_quietly(monkeypatch, caplog):
    payload = json.dumps({"data": [{"id": "Qwen/Qwen2.5-32B-Instruct-AWQ"}]})
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Response(payload))
    manager = LLMManager(warmup=False)

    with caplog.at_level(logging.WARNING):
        manager._check_vllm_endpoint("Qwen/Qwen2.5-32B-Instruct-AWQ")

    assert caplog.text == ""


def test_a_different_model_than_configured_is_flagged(monkeypatch, caplog):
    # `kg_pipeline/.env` naming one model while another is served is a real and
    # recurring state; the warning is what makes it visible.
    payload = json.dumps({"data": [{"id": "RedHatAI/Qwen3.8-27B-INT4"}]})
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Response(payload))
    manager = LLMManager(warmup=False)

    with caplog.at_level(logging.WARNING):
        manager._check_vllm_endpoint("Qwen/Qwen2.5-32B-Instruct-AWQ")

    assert "not listed by vLLM /models" in caplog.text


def test_an_unparseable_listing_is_not_treated_as_a_mismatch(monkeypatch, caplog):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Response("not json"))
    manager = LLMManager(warmup=False)

    with caplog.at_level(logging.WARNING):
        manager._check_vllm_endpoint("any-model")

    assert caplog.text == ""


def test_the_api_key_is_sent_when_there_is_one(monkeypatch):
    seen: list[Any] = []

    def _urlopen(request, timeout=None):
        seen.append(request)
        return _Response(json.dumps({"data": []}))

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    manager = LLMManager(warmup=False)
    manager.vllm_api_key = "secret-token"

    manager._check_vllm_endpoint("any-model")

    assert seen[0].get_header("Authorization") == "Bearer secret-token"


# --- which model is being loaded -------------------------------------------


@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("Qwen/Qwen2.5-32B-Instruct-AWQ", 32.0),
        ("meta-llama/Llama-3.1-8B", 8.0),
        ("RedHatAI/Qwen3.8-27B-INT4", 27.0),
        ("some/model-1.5b-chat", 1.5),
        ("sentence-transformers/all-MiniLM-L6-v2", None),
    ],
)
def test_the_parameter_count_is_read_out_of_the_model_id(model_id, expected):
    assert LLMManager._model_size_billions(model_id) == expected


def test_a_large_model_is_recognised_as_large():
    assert LLMManager._is_large_model("Qwen/Qwen2.5-32B-Instruct-AWQ") is True
    assert LLMManager._is_large_model("Qwen/Qwen2.5-7B-Instruct") is False


def test_a_model_id_with_no_size_is_not_assumed_large():
    assert LLMManager._is_large_model("some/unnamed-model") is False


@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("Qwen/Qwen2.5-32B-Instruct-AWQ", True),
        ("Qwen/Qwen2.5-32B-awq", True),
        ("Qwen/Qwen2.5-32B-Instruct", False),
        ("some/awqmodel", False),
    ],
)
def test_a_quantised_model_is_recognised(model_id, expected):
    assert LLMManager._is_awq_model(model_id) is expected


@pytest.mark.parametrize("var", ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"])
def test_either_hugging_face_token_variable_is_read(monkeypatch, var):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv(var, "tok")

    assert LLMManager._hf_token() == "tok"


def test_no_token_at_all_is_none(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    assert LLMManager._hf_token() is None


@pytest.mark.parametrize(
    "message",
    ["Cannot access gated repo", "401 Unauthorized", "LocalTokenNotFoundError", "access to model"],
)
def test_an_access_failure_is_told_apart_from_a_load_failure(message):
    assert LLMManager._is_hf_auth_error(RuntimeError(message)) is True


def test_an_ordinary_load_failure_is_not_an_access_failure():
    assert LLMManager._is_hf_auth_error(RuntimeError("out of memory")) is False


def test_an_access_failure_wrapped_in_another_exception_is_still_found():
    # The real one arrives several layers deep inside the hub client.
    root = RuntimeError("cannot access gated repo")
    try:
        try:
            raise root
        except RuntimeError as exc:
            raise OSError("could not load model") from exc
    except OSError as wrapped:
        assert LLMManager._is_hf_auth_error(wrapped) is True


def test_the_access_error_says_how_to_fix_it():
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        LLMManager._raise_hf_access_error("meta-llama/Llama-3.1-8B", RuntimeError("401"))


def test_the_fp16_fallback_message_names_the_flag_that_enables_it():
    message = str(LLMManager._fp16_fallback_message("big/model-70B", RuntimeError("oom")))

    assert "--allow-large-model-fp16-fallback" in message
    assert "oom" in message


# --- the token cap ---------------------------------------------------------


def test_a_length_stop_is_recognised():
    assert LLMManager._hit_token_limit(_Output("una risposta", "length")) is True


def test_a_normal_stop_is_not_a_length_stop():
    assert LLMManager._hit_token_limit(_Output("una risposta", "stop")) is False


def test_a_backend_that_reports_nothing_is_not_assumed_truncated():
    assert LLMManager._hit_token_limit(_Output("una risposta")) is False
    assert LLMManager._hit_token_limit("plain string") is False


# --- which language the answer is written in -------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Cosa contiene la scotta prodotta dai caseifici?",
        "Quali sono le tre componenti principali del capitale?",
        "Mi puoi spiegare che cosa si intende per simbiosi industriale?",
    ],
)
def test_an_italian_question_is_answered_in_italian(query):
    assert LLMManager._detect_query_language(query) == "it"


@pytest.mark.parametrize(
    "query",
    [
        "What does rice husk contain?",
        "Which of these are the three components of capital?",
        "Can you explain what industrial symbiosis means?",
    ],
)
def test_an_english_question_is_answered_in_english(query):
    assert LLMManager._detect_query_language(query) == "en"


@pytest.mark.parametrize("query", ["", "   ", "3C?", "CO2"])
def test_a_question_with_no_signal_falls_to_english(query):
    # The tie goes to English. Right for a lone question, and the reason a
    # terse imperative like "Spiegameli meglio" needs the conversation: the
    # detector counts function words from two lists and it has none.
    assert LLMManager._detect_query_language(query) == "en"


def test_the_score_pair_is_what_the_verdict_is_read_from():
    it_score, en_score = LLMManager._language_scores("Cosa contiene la scotta?")

    assert it_score > en_score


# --- the language of a turn that says nothing -------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Non ho capito niente",
        "E SeED?",
        "Ah okok, grazie mille. ora ho capito",
        "Non ho capito molto, puoi approfondirlo?",
        "Sicuro? Guarda bene se esiste materia rinnovabile numero 33",
    ],
)
def test_a_turn_with_no_marker_takes_the_conversations_language(query):
    # The detector's misses are 0-0 ties like these, not wrong word lists, so
    # the conversation's language is what settles a tie.
    italian = "Utente: Cosa contiene la scotta?\nAssistente: E' il residuo liquido."

    assert LLMManager._detect_query_language(query) == "en"
    assert LLMManager._answer_language(query, italian) == "it"


def test_the_same_turn_in_an_english_conversation_stays_english():
    english = "User: What does rice husk contain?\nAssistant: It is the outer shell."

    assert LLMManager._answer_language("Not sure I follow", english) == "en"


def test_the_words_typed_outrank_the_conversation():
    # A mid-conversation switch of language is honoured.
    italian = "Utente: Cosa contiene la scotta?\nAssistente: E' il residuo liquido."

    assert LLMManager._answer_language("What is whey exactly?", italian) == "en"


def test_with_no_conversation_behind_it_a_mute_turn_is_english():
    assert LLMManager._answer_language("Non ho capito niente", "") == "en"


@pytest.mark.parametrize(
    "query, it_wins",
    [
        ("Sai dirmi qualcosa sul system thinking?", True),
        ("Approfondisci l'economia circolare in Piemonte", True),
        ("Spiegameli meglio", True),
        ("What is the role of industrial symbiosis?", False),
    ],
)
def test_the_marker_lists_carry_the_cases_they_were_corrected_for(query, it_wins):
    it_score, en_score = LLMManager._language_scores(query)

    assert (it_score > en_score) is it_wins
