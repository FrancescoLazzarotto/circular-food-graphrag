"""Refuse to modify a graph that nobody asked to modify.

Helper module, not an entrypoint.

The kg_repair passes rewrite the graph in place: they delete nodes, reverse
relationships and rename relationship types. They take no arguments and read
their target from ``kg_pipeline/.env`` — which points at the hosted graph the
demo serves. Run with ``--help`` by someone checking what it does, a pass
without an argument parser ignores the flag and does the work, and one such
run can cost the demo graph its vector carriers, entities and whole
relationship types.

So the rule here is narrow and absolute: **asking a repair pass what it does
must never make it do it.** Confirmation is explicit, and the target is printed
before anything is written, because "which graph is this pointing at" is the
question that actually matters.

And confirming the action is not confirming the target. These passes read their
target from ``kg_pipeline/.env``, which points at the hosted graph the demo
serves, while the local mirror they are meant for is ``bolt://localhost:7689``.
``--yes`` becomes muscle memory; a write to a remote instance therefore needs
that instance's host named in ``KG_ALLOW_HOSTED_WRITES``, which cannot be typed
by habit.
"""

from __future__ import annotations

import os
import sys

CONFIRM_ENV = "KG_REPAIR_CONFIRM"
CONFIRM_FLAG = "--yes"
# Confirming the action is not confirming the target. `--yes` becomes muscle
# memory after the tenth run against staging, and the eleventh is the one that
# is pointed at the graph the demo is serving. Writing to a remote instance
# needs its host named, so the confirmation cannot be typed by habit.
HOSTED_ENV = "KG_ALLOW_HOSTED_WRITES"
_HELP_FLAGS = {"-h", "--help"}


def _host_of(uri: str) -> str:
    """The host part of a Neo4j URI, without scheme, port or credentials."""
    remainder = uri.split("://", 1)[-1]
    remainder = remainder.rsplit("@", 1)[-1]
    return remainder.split("/", 1)[0].split(":", 1)[0].strip().lower()


def _looks_hosted(uri: str) -> bool:
    """Whether this URI is a remote instance rather than a local one."""
    lowered = uri.lower()
    local_markers = ("localhost", "127.0.0.1", "0.0.0.0", "::1")
    return not any(marker in lowered for marker in local_markers)


def require_hosted_target_named(title: str, uri: str, database: str | None = None) -> None:
    """Return unless this is a remote graph nobody named.

    Local targets pass straight through. A hosted one needs its host in
    ``KG_ALLOW_HOSTED_WRITES``, because a script that reads its target from a
    dotenv file will point at the graph the demo serves by default, and the
    operator typing the confirmation is usually not thinking about which of the
    three live instances is on the other end.

    Raises:
        SystemExit: with code 1, if the target is hosted and unnamed.
    """
    if not _looks_hosted(uri):
        return
    target = f"{uri or '<unset>'} (db: {database or '<default>'})"
    host = _host_of(uri)
    allowed = os.getenv(HOSTED_ENV, "").strip().lower()
    if allowed and allowed == host:
        print(f"!! {title}: writing to the HOSTED graph {target}", file=sys.stderr)
        return

    print(f"{title}: refusing to write to a hosted graph on {CONFIRM_FLAG} alone.")
    print()
    print(f"Target: {target}")
    print("        ^ this is a hosted instance, not a scratch copy. The local")
    print("        mirror is bolt://localhost:7689; 7688 is a different live")
    print("        instance, so 'localhost' alone does not name the target.")
    print()
    if allowed:
        print(f"{HOSTED_ENV} is set to {allowed!r}, which is not {host!r}.")
        print()
    print("If this graph really is the one you mean, name it:")
    print(f"  {HOSTED_ENV}={host} {CONFIRM_ENV}=yes python {sys.argv[0]}")
    raise SystemExit(1)


def require_confirmation(
    title: str,
    what_it_does: str,
    uri: str,
    database: str | None,
    argv: list[str] | None = None,
) -> None:
    """Return only if the caller has explicitly confirmed a destructive run.

    Args:
        title: Human name of the pass, e.g. "KG Repair 3".
        what_it_does: One or more lines describing the writes it performs.
        uri: The Neo4j URI the pass would write to.
        database: The target database, or None for the server default.
        argv: Command line to inspect; defaults to ``sys.argv[1:]``.

    Raises:
        SystemExit: Always, unless the run was confirmed. Exit code 0 — being
            asked for a plan and getting one is a success, not a failure.
    """
    args = sys.argv[1:] if argv is None else argv
    target = f"{uri or '<unset>'} (db: {database or '<default>'})"
    confirmed = CONFIRM_FLAG in args or os.getenv(CONFIRM_ENV, "").strip().lower() == "yes"

    if not any(flag in args for flag in _HELP_FLAGS) and confirmed:
        require_hosted_target_named(title, uri, database)
        return

    asked_for_help = any(flag in args for flag in _HELP_FLAGS)
    print(f"{title} — modifies the graph in place. Nothing has been written.")
    print()
    print("What it would do:")
    for line in what_it_does.strip().splitlines():
        print(f"  {line.strip()}")
    print()
    print(f"Target: {target}")
    if _looks_hosted(uri):
        print("        ^ this is the hosted graph the demo serves, not a scratch copy.")
        print("        Export NEO4J_URL/NEO4J_USERNAME/NEO4J_PASSWORD to point elsewhere;")
        print("        exported values win over kg_pipeline/.env.")
    print()
    print("To run it for real:")
    if _looks_hosted(uri):
        print(f"  {HOSTED_ENV}={_host_of(uri)} python {sys.argv[0]} {CONFIRM_FLAG}")
    else:
        print(f"  python {sys.argv[0]} {CONFIRM_FLAG}")
        print(f"  {CONFIRM_ENV}=yes python {sys.argv[0]}")
    if asked_for_help and confirmed:
        print()
        print(f"({CONFIRM_FLAG} was given together with a help flag; help wins.)")
    raise SystemExit(0)
