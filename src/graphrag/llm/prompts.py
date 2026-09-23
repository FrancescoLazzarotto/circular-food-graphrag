"""Every prompt the agent and the LLM backends send, and the fixed replies."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.prompts import ChatPromptTemplate

from graphrag.config import AgentConfig, OUTPUT_COMPLEXITY, OUTPUT_TONE


class PromptLibrary:
    """Single source of every prompt template and fixed reply.

    The vLLM and local HF backends render prompts only from here, so both send
    identical text. Changing a template changes what experiments measure.
    """

    # Written in the target language on purpose: an Italian instruction holds
    # an Italian answer far better than an English sentence asking for
    # Italian, especially when the retrieved context is mostly English.
    LANGUAGE_DIRECTIVES = {
        "it": (
            "Rispondi SEMPRE in italiano, anche quando il contesto e in inglese: "
            "traduci le evidenze invece di cambiare lingua. Restano nella lingua "
            "originale solo i nomi propri, i titoli dei documenti e i termini "
            "tecnici privi di traduzione corrente."
        ),
        "en": (
            "ALWAYS answer in English, even when the context is written in "
            "Italian: translate the evidence instead of switching language. Only "
            "proper names, document titles and technical terms with no current "
            "translation stay in the original language."
        ),
    }
    # The language directive is repeated as the very last line of the prompt,
    # the position models obey, so it overrides the "copy word for word" rule
    # for quoted definitions unless the exception travels with it; otherwise a
    # quoted definition comes back translated and the quote gate strips it.
    QUOTE_LANGUAGE_EXCEPTIONS = {
        "it": (
            " Unica eccezione: il passaggio citato fra «...» va copiato nella "
            "lingua originale della fonte, senza tradurlo, perche' una citazione "
            "tradotta non e' piu' una citazione; la traduzione va subito dopo, "
            "fuori dalle virgolette."
        ),
        "en": (
            " One exception: the passage quoted between «...» must be copied in "
            "the source's original language, untranslated, because a translated "
            "quotation is no longer a quotation; put your translation "
            "immediately after, outside the guillemets."
        ),
    }
    LANGUAGE_REINFORCEMENTS = {
        "it": (
            "VINCOLO ASSOLUTO: la risposta precedente era nella lingua sbagliata. "
            "Scrivi TUTTA la risposta in italiano, dalla prima all'ultima parola. "
        ),
        "en": (
            "ABSOLUTE CONSTRAINT: the previous answer was in the wrong language. "
            "Write the ENTIRE answer in English, from first word to last. "
        ),
    }

    @staticmethod
    def language_directive(
        language: str,
        reinforced: bool = False,
        quote_exception: bool = False,
    ) -> str:
        """Return the answer-language constraint, written in that language.

        Args:
            language: ``"it"`` or ``"en"``; anything else yields an empty string.
            reinforced: Prefix the stronger wording used on the retry that
                follows a wrong-language answer.
            quote_exception: Add the carve-out that keeps a quoted passage in
                the source's language. Only for definitional questions: on
                every other answer there is nothing to quote and the clause
                would just invite the model to leave English prose in.

        Returns:
            The directive, or an empty string when the language is unknown.
        """
        key = str(language or "").lower()
        directive = PromptLibrary.LANGUAGE_DIRECTIVES.get(key)
        if not directive:
            return ""
        if quote_exception:
            directive += PromptLibrary.QUOTE_LANGUAGE_EXCEPTIONS[key]
        if reinforced:
            return PromptLibrary.LANGUAGE_REINFORCEMENTS[key] + directive
        return directive

    @staticmethod
    def answer_prompt(
        config: AgentConfig,
        language: str | None = None,
        reinforce_language: bool = False,
        definitional: bool = False,
        transcript: bool = False,
    ) -> ChatPromptTemplate:
        """Build the answer prompt.

        Args:
            config: Agent configuration; ``complexity`` drives answer depth and
                ``cite_evidence`` the citation protocol.
            language: Target answer language (``"it"``/``"en"``). ``None`` adds
                no language directive, which is the prompt baselines and gold
                runs without ``enforce_language`` see.
            reinforce_language: Use the stronger constraint (retry after a
                wrong-language answer). Ignored when ``language`` is ``None``.
            definitional: The question asks what something is. Adds the
                quote-then-explain structure.
            transcript: Add a ``transcript`` slot carrying the conversation so
                far. ``False``, the default and what every experiment run
                uses, adds nothing to the template.

        Returns:
            The chat prompt template with ``question`` and ``context`` slots,
            plus ``transcript`` when that flag is set. ``config.answer_prompt``,
            when set, replaces the whole template.
        """
        if config.answer_prompt:
            return ChatPromptTemplate.from_template(config.answer_prompt)

        tone_map = {
            OUTPUT_TONE.TECHNICAL: "Use precise technical terminology.",
            OUTPUT_TONE.SIMPLIFIED: "Explain in simple, accessible terms.",
            OUTPUT_TONE.FORMAL: "Use a formal, academic register.",
        }

        complexity_map = {
            OUTPUT_COMPLEXITY.LOW: "Keep the answer brief (2-3 sentences).",
            OUTPUT_COMPLEXITY.MEDIUM: "Provide a well-structured paragraph.",
            OUTPUT_COMPLEXITY.HIGH: "Provide a thorough, multi-paragraph analysis.",
        }

        structured = ""
        if config.use_structured_response:
            structured = (
                "\nAnswer using this structure:\n"
                "## Key Concepts\n## Relationships\n## Reasoning Chain\n## Conclusions\n"
            )

        # System message with explicit response rules
        if config.allow_parametric_fallback:
            # "ONLY the provided context" makes graph context harmful when
            # retrieval misses: a block of triples with confidence scores and
            # page citations reads as authoritative enough to override what the
            # model already knows. The permission is only safe with the marking
            # requirement: an unmarked fallback is indistinguishable from a
            # hallucination, and the groundedness measurement depends on
            # telling them apart.
            grounding_rule = (
                "You are a knowledge graph assistant. Ground the answer in the "
                "provided context whenever the context covers the question. "
                "The context is retrieved automatically and is often incomplete: "
                "when it does not cover the question, you may answer from your "
                "own knowledge of the domain instead of refusing. "
                "Never present the two as the same: mark every statement that "
                "the context does not support with '(not in the retrieved "
                "evidence)'. If you know nothing reliable either, say so plainly "
                "rather than inventing. "
            )
        else:
            grounding_rule = (
                "You are a knowledge graph assistant. Answer using ONLY the provided context. "
                "If context does not answer the question, state this plainly. "
                "Do not invent or generate content outside the context. "
            )
        focus_rule = ""
        if config.focused_answer:
            focus_rule = (
                "Answer only what was asked. The evidence is retrieved in bulk "
                "and carries neighbouring material: mention a concept only if it "
                "is part of the answer, not because it appears in the context. "
                "Prefer naming fewer things precisely over listing everything "
                "related. "
            )
        system_message = (
            grounding_rule
            + focus_rule
            + "Preserve all entity names exactly as given. "
            "Respond in the same language as the question (English or Italian), "
            "even when the context is written in the other language: translate "
            "the evidence into the question's language, never switch language. "
            "Prefer a natural, human explanation over a mechanical list. "
            "If you mention a fact, tie it to an exact node, triple, or other "
            "explicit evidence from the context."
        )
        language_block = (
            PromptLibrary.language_directive(
                language,
                reinforced=reinforce_language,
                quote_exception=definitional,
            )
            if language
            else ""
        )
        if language_block:
            system_message += " " + language_block
        if config.cite_evidence:
            system_message += (
                " Evidence items in the context are numbered: reference them by "
                "their id so each specific claim stays traceable to its source "
                "document."
            )

        # Without this the model has no record of its own prose, so a question
        # that quotes it — "hai scritto X, quali?" — arrives as a bare claim,
        # and the grounding rule above is precisely what turns a bare claim into
        # a denial of the assistant's own earlier answer.
        if transcript:
            system_message += (
                " The conversation so far is given under 'Conversation so far'. "
                "Those are the user's earlier questions and your own earlier "
                "replies: they are a record of what was said, never evidence. "
                "Never cite them, and never treat a statement as supported "
                "because you made it earlier. When the user refers to something "
                "you said, take it as said — do not deny it and do not call the "
                "premise unsupported — then answer by re-grounding that point in "
                "the context below, and say plainly if the context does not "
                "support it after all."
            )

        # An English heading on top of an Italian answer is a language leak, so
        # the title follows the answer language.
        limits_title = (
            "Limiti e affidabilità" if language == "it" else "Limits and confidence"
        )
        if config.always_include_limits:
            limits_block = (
                f"Always include a short section titled '{limits_title}' "
                "assessing how strong the supporting evidence is, in at most "
                "three sentences: it closes the answer, it is not the answer. "
            )
            # Only meaningful without the citation protocol below, which replaces
            # the trailing evidence section with per-claim reference tags.
            no_inline_block = (
                "Keep the main paragraphs free of inline triple citations in "
                "parentheses; cite nodes and triples only in the dedicated "
                "evidence section. "
            )
        else:
            limits_block = (
                "If context is sparse, include a short section titled "
                f"'{limits_title}'. "
            )
            no_inline_block = ""

        if config.cite_evidence:
            # Deliberately restrictive: a tag on every sentence reads as noise
            # and stops carrying information. Citations belong on claims a
            # reader could want to check.
            evidence_block = (
                "Evidence items in the context are numbered: [S1], [S2], ... for "
                "source passages and [T1], [T2], ... for knowledge-graph facts. "
                "Put the id of the evidence you used in square brackets at the end "
                "of the sentence it supports. "
                "Cite only claims carrying specific content: figures, percentages, "
                "dates, proper names, article or standard numbers, definitions, and "
                "statements attributable to an author or a document. "
                "Do not cite generic, connective or summarising sentences, and use at "
                "most one tag per sentence. "
                "When several evidence items support the same claim, cite the single "
                "most specific one instead of stacking ids: never put more than two "
                "ids in a tag. "
                "Never write an id that is not in the context, and never cite the "
                "entity sections, which carry no source. "
                "When consecutive sentences rest on the same evidence, cite it "
                "once for the whole passage instead of repeating it. "
                "Do not write a source list at the end: it is generated automatically. "
            )
        else:
            evidence_block = (
                no_inline_block
                + "When possible, add a short 'Evidence in graph' section with the "
                "exact node or triple names that support the answer. "
            )

        if config.complexity is OUTPUT_COMPLEXITY.HIGH:
            # "1-2 short paragraphs" would contradict a HIGH complexity setting
            # and turn answers into abstract summaries, which drop exactly the
            # figures, names and article numbers the reader asks for.
            depth_block = (
                "If context has at least some factual evidence, provide the best "
                "grounded answer possible, developing every point the evidence "
                "supports across several paragraphs. "
                "Avoid a checklist style unless the user explicitly asks for a list. "
                "Stay concrete: use the figures, proper names, years, percentages "
                "and article or standard numbers that appear in the evidence, and "
                "never generalise when a specific one is available — write 'a 65% "
                "impact reduction at Terra Madre Salone del Gusto', not 'a "
                "significant impact reduction'. "
            )
        else:
            depth_block = (
                "If context has at least some factual evidence, provide the best "
                "grounded answer possible in 1-2 short paragraphs. "
                "Avoid a checklist style unless the user explicitly asks for a list. "
            )

        # A definition *is* its wording, and the graph channel tends to replace
        # it with relations. The "only if it is there" clause is not
        # politeness: the instruction to quote is also an invitation to invent
        # a quote, and the quote gate downstream strips the guillemets off
        # anything invented.
        definition_block = ""
        if definitional:
            definition_block = (
                "This question asks what something is. Open the answer with the "
                "source's own definition between «guillemets», followed by its "
                "reference tag — including the expansion of an acronym when the "
                "source gives one. "
                # Models tend to reorder the source's words inside the
                # guillemets, which reads as a quotation and is not one. The
                # gate strips those guillemets, so the instruction has to be
                # about copying, not about quoting.
                "Inside the guillemets copy the source word for word, in its "
                "original order, changing nothing: no reordering, no rewording, "
                "no shortening except a [...] for an omitted middle. "
                # The corpus is bilingual and the answer language is pinned to
                # the question's, so a definition taken from an English document
                # and answered in Italian can never be verbatim. Quoting in the
                # original and translating outside the guillemets is the only
                # form that satisfies both constraints.
                "This is the one place where you do not translate: copy the "
                "passage in the language the source wrote it in, then give your "
                "translation immediately after, outside the guillemets. "
                "When you cannot copy a passage exactly, or when no passage "
                "defines the term, use no guillemets at all: say in one sentence "
                "that the context carries no explicit definition and answer, in "
                "your own words and as a complete sentence, from what the context "
                "does say. Never invent a definition to quote. "
                "Then explain the definition and what it means in practice, and "
                "only after that use the graph facts, as a complement to the "
                "definition and never as a replacement for it. "
            )

        # Ahead of the question and the context on purpose. The transcript only
        # ever grows at its end, so turn N's prefix extends turn N-1's and the
        # served prefix cache (--enable-prefix-caching on both endpoints) reuses
        # it; placed after the context, which is rebuilt every turn, nothing
        # below it would ever be reusable.
        transcript_block = (
            "Conversation so far (what was already said; not evidence, never "
            "cite it):\n{transcript}\n\n"
            if transcript
            else ""
        )
        human_message_template = (
            f"Target audience: {config.target_audience}.\n"
            f"{tone_map[config.tone]}\n{complexity_map[config.complexity]}\n"
            f"{structured}\n"
            + transcript_block
            + "Question:\n{question}\n\n"
            "Context:\n{context}\n\n"
            + definition_block
            + depth_block
            + limits_block
            + evidence_block
            # The closing line is the one models follow, so it must match the
            # grounding rule at the top. The legacy wording — "state that
            # context is insufficient only when context is empty or lacks
            # factual evidence" — tells the model not to call an unrelated but
            # factual context insufficient, and the retriever has no score
            # floor, so an out-of-domain question still arrives with a full
            # context of unrelated chunks.
            + (
                "State that context is insufficient only when context is empty "
                "or lacks factual evidence."
                if config.legacy_insufficiency_wording
                else "When the context does not cover the question, say so and mark "
                "every statement it does not support with '(not in the retrieved "
                "evidence)'. Never pass one off as the other."
                if config.allow_parametric_fallback
                else "State that the context is insufficient whenever it does "
                "not cover what was asked — an unrelated context is an "
                "insufficient one, however many facts it contains."
            )
            + (f"\n\n{language_block}" if language_block else "")
        )

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("human", human_message_template),
            ]
        )

    @staticmethod
    def rewrite_prompt(config: AgentConfig) -> ChatPromptTemplate:
        """Prompt that rewrites a question to retrieve better.

        Args:
            config: Agent configuration; ``rewrite_prompt``, when set, replaces
                the template.

        Returns:
            A template with a ``question`` slot.
        """
        if config.rewrite_prompt:
            return ChatPromptTemplate.from_template(config.rewrite_prompt)
        return ChatPromptTemplate.from_template(
            "You rewrite a question so that it retrieves better over a knowledge "
            "base, without changing what it asks.\n\n"
            "Original: {question}\n\n"
            "Rules:\n"
            "- Keep the user's intent and the user's language.\n"
            "- Never guess what an acronym or a shorthand stands for. Asked about "
            "the 3C, both served models supplied an expansion from marketing and "
            "sent retrieval to documents the question was not about; the letters "
            "alone retrieve the right passage, an invented expansion does not.\n"
            "- Add a synonym or a domain term only when it certainly belongs to "
            "the question's subject.\n"
            "- Do not add facts, do not answer, and do not explain your choices.\n"
            "- If you cannot improve it, repeat the question unchanged.\n"
            "- Reply with the rewritten question only, on a single line.\n\n"
            "Rewritten question:"
        )

    @staticmethod
    def followup_rewrite_prompt(config: AgentConfig) -> ChatPromptTemplate:
        """Make an elliptical follow-up self-contained, for retrieval only.

        The output never reaches the answer prompt: it only feeds the
        retriever, so the instructions optimise for search terms, not for
        phrasing. The "repeat it unchanged" escape hatch matters — the rewrite
        can run on a question that turns out to need nothing, and the model
        must be free to say so.

        Args:
            config: Agent configuration (unused).

        Returns:
            A template with ``entities``, ``previous_question`` and
            ``question`` slots.
        """
        return ChatPromptTemplate.from_template(
            "You rewrite a follow-up question so that it can be understood on its own, "
            "outside the conversation.\n"
            "Topics active in this conversation: {entities}\n"
            "Previous question: {previous_question}\n"
            "Follow-up question: {question}\n\n"
            "Rules:\n"
            "- Keep the user's intent and the user's language.\n"
            "- Resolve pronouns and implicit references using the active topics above.\n"
            "- Use a topic only when the follow-up actually refers to it, and ignore "
            "the others: an unrelated topic pulls retrieval away from the question.\n"
            "- Add only the missing context. Do not add facts, and do not answer.\n"
            "- If the follow-up already stands on its own, repeat it unchanged.\n"
            "- Reply with the rewritten question only, on a single line.\n\n"
            "Rewritten question:"
        )

    @staticmethod
    def decomposition_prompt(config: AgentConfig) -> ChatPromptTemplate:
        """Prompt that splits a question into sub-questions, as a JSON array.

        Args:
            config: Agent configuration; ``decomposition_prompt``, when set,
                replaces the template.

        Returns:
            A template with a ``question`` slot.
        """
        if config.decomposition_prompt:
            return ChatPromptTemplate.from_template(config.decomposition_prompt)
        return ChatPromptTemplate.from_template(
            "Break this complex question into 2-4 simpler, self-contained sub-questions "
            "that together cover the full scope of the original.\n"
            "Return ONLY a JSON array of strings.\n\n"
            "Question: {question}\n\nSub-questions:"
        )

    @staticmethod
    def reflection_prompt(config: AgentConfig) -> ChatPromptTemplate:
        """Prompt that checks an answer against its context, as JSON.

        Not called by the agent graph, which has no reflection node.

        Args:
            config: Agent configuration; ``reflection_prompt``, when set,
                replaces the template.

        Returns:
            A template with ``context`` and ``answer`` slots.
        """
        if config.reflection_prompt:
            return ChatPromptTemplate.from_template(config.reflection_prompt)
        return ChatPromptTemplate.from_template(
            "You are a grounding verifier. Check whether the answer is faithful to "
            "the provided context. Look for hallucinations, unsupported claims, or "
            "logical errors.\n\n"
            "Context:\n{context}\n\n"
            "Answer:\n{answer}\n\n"
            "Respond with a JSON object:\n"
            '{{"passed": true/false, "confidence": 0.0-1.0, "feedback": "..."}}'
        )

    @staticmethod
    def adaptive_router_prompt(config: AgentConfig) -> ChatPromptTemplate:
        """Prompt that picks TEXT, KG, HYBRID or MULTIHOP retrieval.

        Args:
            config: Agent configuration; ``adaptive_router_prompt``, when set,
                replaces the template.

        Returns:
            A template with a ``question`` slot.
        """
        if config.adaptive_router_prompt:
            return ChatPromptTemplate.from_template(config.adaptive_router_prompt)
        return ChatPromptTemplate.from_template(
            "Given this question, choose the best retrieval strategy.\n"
            "Options:\n"
            "- TEXT: factual lookup, keyword-heavy queries\n"
            "- KG: relationship or reasoning queries\n"
            "- HYBRID: complex questions needing both facts and relationships\n"
            "- MULTIHOP: questions requiring chain reasoning across multiple concepts\n\n"
            "Question: {question}\n\n"
            "Respond with ONLY one word: TEXT, KG, HYBRID, or MULTIHOP."
        )

    # Frozen wording, validated by scripts/domain_gate/eval_domain_gate_llm.py
    # and scripts/domain_gate/eval_domain_gate_heldout.py. The by-product
    # composition clause admits questions about what a residue contains; the
    # framework-vocabulary clause admits questions that never mention food on
    # their surface. Editing this text invalidates both measurements — rerun
    # them.
    DEFAULT_DOMAIN_SCOPE = (
        "circular economy principles and frameworks applied to food, food systems "
        "and supply chains, agri-food by-products and residues and their "
        "valorisation — including their chemical composition and their "
        "pharmaceutical, nutraceutical, cosmetic, energy and material uses — food "
        "waste, food and beverage packaging and its materials, sustainability "
        "indicators and policy, and territorial or regional food projects"
    )

    @staticmethod
    def domain_gate_prompt(
        scope: str = "", known_entities: Sequence[str] = ()
    ) -> ChatPromptTemplate:
        """Single-token in/out classification of a question against the corpus.

        Args:
            scope: What the collection covers. Empty uses
                :data:`DEFAULT_DOMAIN_SCOPE`.
            known_entities: Names the graph actually holds that this question
                mentions. Empty leaves the prompt byte-identical to the
                validated wording.

        Returns:
            A prompt whose completion is ``IN`` or ``OUT``.
        """
        scope_text = scope.strip() or PromptLibrary.DEFAULT_DOMAIN_SCOPE
        system_message = (
            "You classify whether a question can be answered from a document "
            f"collection about the following domain: {scope_text}.\n\n"
            "Answer with exactly one word:\n"
            "IN — the question is about that domain\n"
            "OUT — the question is about something else (programming, "
            "mathematics, geography, entertainment, general knowledge, or any "
            "other field)\n\n"
            "A question is IN whenever its subject is a food, a crop, a "
            "food-industry residue or by-product, or a food supply chain — "
            "whatever is being asked about it. Asking what compounds rice bran "
            "contains, or what a food package is made of, is a question about "
            "the domain, not about pharmacology or materials science.\n\n"
            "A question is also IN when it asks about the theoretical vocabulary "
            "of the Circular Economy for Food framework itself, even when it "
            "never mentions food: the three C's (Capital, Cyclicality, "
            "Co-evolution), metabolisation and its implementation cycles, "
            "extension, cascading, ecodesign, industrial symbiosis, and the "
            "relations between these concepts.\n\n"
            "Answer IN whenever the question plausibly belongs to the domain, "
            "even if you doubt the collection holds the specific detail asked "
            "for: the retrieval step decides that, not you. Answer with the "
            "single word only."
        )
        # Names are the one thing the model cannot judge from world knowledge:
        # a project acronym it has never seen looks like no domain at all.
        # When the graph itself reports that it holds a node by that name, say
        # so — and say only that. The verdict stays with the model, because a
        # node named "Torino" does not turn "consigliami un ristorante a
        # Torino" into a question this collection answers.
        names = [str(name).strip() for name in known_entities if str(name).strip()]
        if names:
            # Braces are escaped for the same reason as in `evidence_gate_prompt`
            # below: these names come from the graph and the template parses
            # what it is given. An unescaped "Progetto {LIFE}" raises KeyError
            # inside `prompt.invoke`, and `classify_in_domain` swallows that by
            # returning "in domain", so the gate would silently not run.
            listed = "; ".join(dict.fromkeys(names)).replace("{", "{{").replace("}", "}}")
            system_message += (
                "\n\nThe collection is known to contain entries named: "
                f"{listed}. These are real names from this collection, so a "
                "question asking what one of them is, or what it does, is IN "
                "even if the name means nothing to you. This tells you the "
                "name exists here, nothing more: judge the question itself as "
                "above."
            )
        return ChatPromptTemplate.from_messages(
            [("system", system_message), ("human", "{question}")]
        )

    @staticmethod
    def evidence_gate_prompt(
        entity_names: Sequence[str] = (),
        passages: Sequence[str] = (),
        sources: Sequence[str] = (),
    ) -> ChatPromptTemplate:
        """Judge a question against what the collection actually returned.

        The other gate describes the domain in the prompt — food, crops,
        by-products, the three C's, ecodesign — and that description goes stale
        as the collection grows: a question about a newly added document is
        refused because the description predates it. This one never names a
        domain. It shows what the collection returned for the question and asks
        whether that material is about it, a judgement that widens on its own
        as documents are added.

        Retrieval matches strings, so an unrelated question still returns
        something: "come si fa la carbonara" comes back with "carbonio",
        "Carrara", "impronta di carbonio". Telling that apart from a real match
        is what a reader can do and a similarity threshold cannot: the vector
        score does not separate the two.

        Args:
            entity_names: Names the collection holds for this question's terms.
            passages: Short snippets from the passages retrieval returned.
            sources: Documents those passages came from.

        Returns:
            A prompt whose completion is ``IN`` or ``OUT``.
        """
        system_message = (
            "A document collection was searched for the user's question. You "
            "are shown what it returned. Decide whether the question belongs "
            "to the same subject area as that material.\n\n"
            "Answer with exactly one word:\n"
            "IN — the question belongs to the same subject area\n"
            "OUT — the question is about a plainly different subject\n\n"
            "IN is the normal answer. Say OUT only when the question is about "
            "a plainly different subject from everything shown.\n\n"
            "In particular, answer IN when the material is about the right "
            "subject but does not contain the specific figure, percentage, "
            "price, name or detail the question asks for. Whether the answer "
            "is actually in there is decided later by another step, not by "
            "you: a question asking how much of something there is, or what "
            "share, or what it costs, is IN as long as the material is about "
            "that thing.\n\n"
            "Search matches text, not meaning, so a question about a "
            "completely different subject still returns results anyway. Before "
            "answering IN, take each word the question and the material have in "
            "common and ask whether it is used for the SAME THING in both:\n"
            "- one spelling can name two unrelated things, and a technical term "
            "in the material is often the everyday word in the question\n"
            "- a fragment of a longer word is not the word\n"
            "- a year, a number, a place name or a common verb appearing in "
            "both is not a connection at all\n"
            "When every word in common is an accident of spelling like these, "
            "or the material merely mentions a word the question contains "
            "without being about it, answer OUT. That is what OUT is for.\n\n"
            "Also answer OUT when answering the question would need a field the "
            "documents listed below are plainly not about — someone holding "
            "only these documents could not answer it at all.\n\n"
            "Judge only against the material shown, never against any idea you "
            "have of what this collection is for. Answer with the single word "
            "only."
        )

        def _block(title: str, items: Sequence[str]) -> str:
            """Render a titled bullet list, each item cut to 200 characters."""
            kept = [" ".join(str(i).split())[:200] for i in items if str(i).strip()]
            if not kept:
                return f"\n\n{title}: nothing.\n"
            # Braces are escaped because these strings come from the graph and
            # the template parses them: an unescaped "{" raises KeyError, which
            # the caller swallows by returning "in domain".
            body = "\n".join(f"- {k}" for k in kept)
            return f"\n\n{title}:\n{body}\n".replace("{", "{{").replace("}", "}}")

        system_message += _block("Entries the collection holds", entity_names)
        system_message += _block("Passages the collection returned", passages)
        # The documents these came from: the collection describing itself from
        # its own data, so nothing about the domain has to be written here.
        system_message += _block("Documents these came from", sources)
        return ChatPromptTemplate.from_messages(
            [("system", system_message), ("human", "{question}")]
        )

    @staticmethod
    def out_of_scope_message(language: str = "en", scope_hint: str = "") -> str:
        """The fixed reply for a question the gate rejected.

        Fixed text, not a generated one: the point of the gate is that no answer
        is produced, and a model asked to phrase its own refusal will smuggle a
        partial answer into it.

        Args:
            language: ``"it"`` or anything else, which is answered in English.
            scope_hint: What the collection covers, appended when given.

        Returns:
            The refusal text.
        """
        if language == "it":
            base = (
                "Questa domanda è fuori dall'ambito dei documenti che ho a "
                "disposizione, quindi non rispondo: qualsiasi risposta non "
                "sarebbe fondata su di essi."
            )
            if scope_hint:
                return f"{base}\n\nLa raccolta copre: {scope_hint}."
            return base
        base = (
            "This question falls outside the documents I have, so I am not "
            "answering it: any answer would not be grounded in them."
        )
        if scope_hint:
            return f"{base}\n\nThe collection covers: {scope_hint}."
        return base

    @staticmethod
    def identity_message(
        language: str = "en", examples: Sequence[str] = ()
    ) -> str:
        """The reply to a question about the assistant itself.

        Fixed text, like the refusal above and for a stronger reason: asked
        "chi sei?" with an empty context, a served model invents a product,
        answers as a generic assistant, and names no subject the reader could
        ask about next.

        Args:
            language: ``"it"`` or anything else, which is answered in English.
            examples: Questions to offer. Empty prints the invitation without
                a list, which is what the console demo does when its operator
                configured none.

        Returns:
            The whole answer, ready to display.
        """
        listed = [" ".join(str(e).split()) for e in examples if str(e).strip()]
        if language == "it":
            text = (
                "Sono un assistente sull'economia circolare applicata al cibo. "
                "Rispondo solo a partire dai documenti che ho a disposizione e "
                "cito il documento da cui prendo ogni affermazione, quindi su "
                "tutto il resto non rispondo."
            )
            if listed:
                text += "\n\nPuoi chiedermi per esempio:\n"
                text += "\n".join(f"- {q}" for q in listed)
            else:
                text += (
                    "\n\nChiedimi pure di sottoprodotti agroalimentari, scarti "
                    "di filiera, packaging, indicatori di sostenibilità o dei "
                    "progetti territoriali descritti nei documenti."
                )
            return text
        text = (
            "I am an assistant on the circular economy applied to food. I "
            "answer only from the documents I have, and I cite the document "
            "each statement comes from, so anything else is outside what I can "
            "answer."
        )
        if listed:
            text += "\n\nYou could ask, for example:\n"
            text += "\n".join(f"- {q}" for q in listed)
        else:
            text += (
                "\n\nAsk me about agri-food by-products, supply-chain "
                "residues, packaging, sustainability indicators or the "
                "territorial projects the documents describe."
            )
        return text

    @staticmethod
    def refusal_retry_prompt(language: str = "en") -> ChatPromptTemplate:
        """Stricter prompt used for the single fallback attempt after a refusal.

        Lives here (not inline in the backend) so vLLM and local HF keep
        rendering identical prompts — the invariant the whole PromptLibrary
        exists for.

        Args:
            language: ``"it"`` or anything else, which is answered in English.

        Returns:
            A template with ``context`` and ``question`` slots.
        """
        if language == "it":
            return ChatPromptTemplate.from_template(
                "Usa solo il contesto fornito per rispondere in modo naturale e conciso alla domanda. "
                "Rispondi SEMPRE in italiano, anche se il contesto è in inglese: traduci le evidenze. "
                "Evita un elenco meccanico; preferisci una breve spiegazione in 1-2 paragrafi. "
                "Se possibile, aggiungi una piccola sezione 'Evidence in graph' con i nomi esatti dei nodi o dei tripletti che supportano la risposta. "
                "Contesto:\n{context}\n\nDomanda:\n{question}\n\nRisposta:"
            )
        return ChatPromptTemplate.from_template(
            "Use only the provided context to answer the question naturally and concisely. "
            "ALWAYS answer in English, even if the context is in Italian: translate the evidence. "
            "Avoid a mechanical list; prefer a short 1-2 paragraph explanation. "
            "When possible, add a short 'Evidence in graph' section with the exact node or triple names that support the answer. "
            "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )

    @staticmethod
    def multihop_steer_prompt() -> ChatPromptTemplate:
        """Prompt that decides whether multi-hop exploration has enough, as JSON.

        Not called by the agent graph.

        Returns:
            A template with ``hop_history`` and ``question`` slots.
        """
        return ChatPromptTemplate.from_template(
            "You are exploring a knowledge graph to answer a question.\n"
            "So far you have gathered:\n{hop_history}\n\n"
            "Question: {question}\n\n"
            "Based on what you know so far, do you have enough information?\n"
            # Doubled braces: single ones are parsed as template variables by
            # ChatPromptTemplate.
            'Respond with JSON: {{"enough": true/false, "next_entities": ["..."], '
            '"reasoning": "..."}}'
        )
