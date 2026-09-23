"""Canonical relationship vocabulary shared by the post-processing and repair passes.

Every pass that renames or reclassifies a relationship type must agree on the
allowed types, so they all import them from here.

The list order is significant: it is serialised into the LLM reclassification
prompts, and changing it changes those prompts.
"""

from __future__ import annotations

CANONICAL_RELATION_TYPES: list[str] = [
    "RELATED_TO",
    "AFFECTS",
    "IMPACTS",
    "INFLUENCES",
    "CAUSES",
    "CAUSED_BY",
    "CONTRIBUTES_TO",
    "LEADS_TO",
    "DRIVEN_BY",
    "DEPENDS_ON",
    "ASSOCIATED_WITH",
    "BASED_ON",
    "DERIVED_FROM",
    "PART_OF",
    "HAS_PART",
    "HAS_COMPONENT",
    "COMPOSED_OF",
    "INCLUDES",
    "CONTAINS_DATA",
    "IS_TYPE_OF",
    "DEFINED_AS",
    "HAS_MAXIMUM_LEVEL",
    "HAS_MINIMUM_LEVEL",
    "HAS_VALUE",
    "HAS_UNIT",
    "VALUE_OF",
    "MEASURES",
    "INDICATES",
    "APPLIES_TO",
    "TARGETS",
    "TARGET_OF",
    "REQUIRES",
    "REQUIRED_BY",
    "USES",
    "USED_BY",
    "USES_METHOD",
    "HAS_METHOD",
    "MANAGES",
    "MANAGED_BY",
    "REGULATES",
    "REGULATED_BY",
    "GOVERNS",
    "GOVERNED_BY",
    "COMPLIES_WITH",
    "SHOULD_BE_MANAGED_BY",
    "ENSURES",
    "AIMS_TO_ACHIEVE",
    "NEEDED_FOR",
    "PUBLISHED",
    "WORKED_WITH",
    "EXCHANGES_INFO_WITH",
    "TAKE_INTO_ACCOUNT",
    "PRODUCES",
    "LOCATED_IN",
    "OCCURS_IN",
    "BELONGS_TO",
    "HAS_MEMBER",
    "MEMBER_OF",
    "ANALYZES",
    "ESTABLISHES",
    "ESTABLISHED_BY",
]

CANONICAL_RELATION_SET: frozenset[str] = frozenset(CANONICAL_RELATION_TYPES)
