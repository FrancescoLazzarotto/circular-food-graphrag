"""Resolution of the Neo4j connection target from arguments and the environment.

Every process that opens a Neo4j driver resolves its target here, so they all
accept the same variable names, apply the same precedence and fail with the
same error. Because several local instances and a hosted one are reachable,
the resolved target can describe where it points without exposing the
password.
"""

from __future__ import annotations

import os
from typing import Any, NamedTuple
from urllib.parse import urlsplit

from neo4j import Driver, GraphDatabase

# Accepted environment variable names, in precedence order. Both spellings of
# each setting are in use in existing `.env` files.
URI_VARS = ("NEO4J_URI", "NEO4J_URL")
USER_VARS = ("NEO4J_USER", "NEO4J_USERNAME")
PASSWORD_VARS = ("NEO4J_PASSWORD",)
DATABASE_VARS = ("NEO4J_DATABASE", "NEO4J_DB")


def _first_env(names: tuple[str, ...]) -> str:
    """Return the first non-blank value among ``names``, stripped, or ``""``."""
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return ""


def _or_names(names: tuple[str, ...]) -> str:
    """Join variable names as ``"A or B"`` for error messages."""
    return " or ".join(names)


class Neo4jTarget(NamedTuple):
    """A resolved Neo4j connection target.

    Attributes:
        uri: Bolt or Neo4j URI.
        user: User name.
        password: Password.
        database: Database name, or ``None`` for the server default.
    """

    uri: str
    user: str
    password: str
    database: str | None

    @property
    def auth(self) -> tuple[str, str]:
        """The ``(user, password)`` pair expected by the driver."""
        return (self.user, self.password)

    @property
    def host(self) -> str:
        """Lower-cased hostname of ``uri``, or ``""`` when it has none."""
        parsed = urlsplit(self.uri if "://" in self.uri else f"//{self.uri}")
        return (parsed.hostname or "").lower()

    @property
    def is_local(self) -> bool:
        """Whether the target is on the loopback interface."""
        return self.host in {"localhost", "127.0.0.1", "::1", ""}

    def session_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``driver.session()``.

        Returns:
            ``{"database": name}``, or an empty dict when no database is set.
        """
        return {"database": self.database} if self.database else {}

    def describe(self) -> str:
        """Describe the target for logs, without the password."""
        return f"{self.uri} (user {self.user}, database {self.database or '<default>'})"


def resolve_target(
    *,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
    require: bool = True,
) -> Neo4jTarget:
    """Resolve the target from explicit arguments first, then the environment.

    Args:
        uri: Overrides ``NEO4J_URI`` / ``NEO4J_URL`` when non-empty. Scripts
            with a ``--uri`` flag pass it here rather than applying their own
            precedence.
        user: Overrides ``NEO4J_USER`` / ``NEO4J_USERNAME``.
        password: Overrides ``NEO4J_PASSWORD``.
        database: Overrides ``NEO4J_DATABASE`` / ``NEO4J_DB``. An empty value
            means the server's default database, not a missing setting.
        require: Raise when the URI, user or password cannot be found. Pass
            ``False`` only to report the configuration without connecting.

    Returns:
        The resolved target.

    Raises:
        ValueError: When ``require`` is set and a setting is missing. The
            message names every variable that would satisfy it.
    """
    resolved_uri = (uri or "").strip() or _first_env(URI_VARS)
    resolved_user = (user or "").strip() or _first_env(USER_VARS)
    resolved_password = password if password else _first_env(PASSWORD_VARS)
    resolved_database = (database or "").strip() or _first_env(DATABASE_VARS)

    if require:
        missing = []
        if not resolved_uri:
            missing.append(_or_names(URI_VARS))
        if not resolved_user:
            missing.append(_or_names(USER_VARS))
        if not resolved_password:
            missing.append(_or_names(PASSWORD_VARS))
        if missing:
            raise ValueError(
                "Missing Neo4j connection settings: "
                + "; ".join(missing)
                + ". Set them in the environment or kg_pipeline/.env."
            )

    return Neo4jTarget(
        uri=resolved_uri,
        user=resolved_user,
        password=resolved_password,
        database=resolved_database or None,
    )


def connect(target: Neo4jTarget | None = None, **overrides: Any) -> Driver:
    """Build a Neo4j driver.

    The driver is a context manager, so callers use
    ``with connect() as driver:``. The target's database is not applied here:
    pass it to ``driver.session()``.

    Args:
        target: Target to connect to. When ``None``, it is resolved with
            :func:`resolve_target`.
        **overrides: Keyword arguments for :func:`resolve_target`, used only
            when ``target`` is ``None``.

    Returns:
        A driver for ``target``.

    Raises:
        ValueError: If ``target`` is ``None`` and the settings are incomplete.
    """
    if target is None:
        target = resolve_target(**overrides)
    return GraphDatabase.driver(target.uri, auth=target.auth)
