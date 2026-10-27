#!/usr/bin/env python3
"""Share one budget of expert minutes among the quality dimensions.

Each dimension (accuracy, interoperability, coverage, conformity) is the
share of its units that are right: edges, mappings, answers, rule
violations. An automatic meter (a judge, the rules themselves) puts every
unit in a stratum for free; a few human labels per stratum then estimate the
share with a stratified estimator, whose variance is
``sum_h W_h^2 S_h^2 / n_h`` (W the stratum's share of the units, S its
standard deviation estimated from the labels already there).

Given the minutes ``B`` and the cost of one label in each dimension, three
ways of spending them are compared:

- optimal: labels go to every (dimension, stratum) in proportion to
  ``W S / sqrt(cost)``, which minimises the summed variance (Neyman
  allocation with costs, here across dimensions too);
- uniform: the minutes split equally among the dimensions, Neyman within;
- expert only: no meter, simple random labels, minutes split equally.

It also reports the minutes needed to bring every dimension under a given
half-width with the optimal allocation.

Input: a JSON spec ``{"dimensions": {name: {"cost_min": c, "units":
[{"stratum": s, "y": 0 | 1 | null}, ...]}}}``; ``y`` is null for a unit no
person has labelled yet. A stratum without labels is given S = 0.5, the
largest possible, so it is never starved by an optimistic guess.

Usage:
    python scripts/kg/quality/expert_budget.py --spec spec.json --budget 120 [--target 0.05]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

Z = 1.959964


def strata(units: list[dict]) -> dict[str, dict]:
    """Share, labelled count and estimated standard deviation of each stratum.

    Args:
        units: The dimension's units, with ``stratum`` and ``y`` (or None).

    Returns:
        Stratum → ``{"W", "labelled", "p", "S"}``.
    """
    total = len(units)
    out: dict[str, dict] = {}
    for unit in units:
        entry = out.setdefault(str(unit["stratum"]), {"count": 0, "ys": []})
        entry["count"] += 1
        if unit.get("y") is not None:
            entry["ys"].append(int(unit["y"]))
    result = {}
    for name, entry in out.items():
        ys = entry["ys"]
        p = sum(ys) / len(ys) if ys else 0.5
        # With few labels a pure stratum would get S = 0 and never be labelled
        # again; one pseudo-observation of each kind keeps it in play.
        s = math.sqrt((sum(ys) + 1) * (len(ys) - sum(ys) + 1)) / (len(ys) + 2) if ys else 0.5
        result[name] = {"W": entry["count"] / total, "labelled": len(ys), "p": p, "S": s}
    return result


def half_width(table: dict[str, dict], labels: dict[str, float]) -> float:
    """Half-width of the 95 % interval of the stratified share with these labels per stratum."""
    variance = sum(t["W"] ** 2 * t["S"] ** 2 / max(labels.get(h, 0.0), 1e-9) for h, t in table.items() if t["W"])
    return Z * math.sqrt(variance)


def neyman(tables: dict[str, dict], costs: dict[str, float], minutes: float) -> dict[str, dict[str, float]]:
    """Labels per (dimension, stratum) minimising the summed variance for these minutes."""
    weights = {(d, h): t["W"] * t["S"] / math.sqrt(costs[d]) for d, table in tables.items() for h, t in table.items()}
    spend = sum(t["W"] * t["S"] * math.sqrt(costs[d]) for d, table in tables.items() for h, t in table.items())
    out: dict[str, dict[str, float]] = {d: {} for d in tables}
    for (d, h), w in weights.items():
        out[d][h] = minutes * w / spend if spend else 0.0
    return out


def _effective_costs(dims: dict) -> dict[str, float]:
    # Labels drawn in clusters (several edges of one passage) are worth 1/deff
    # of independent ones, so an independent-equivalent label costs deff times
    # as many minutes. Allocations and widths below count those labels.
    return {d: float(v["cost_min"]) * float(v.get("deff", 1.0)) for d, v in dims.items()}


def compare(spec: dict, budget: float) -> dict:
    """Half-width per dimension under the three ways of spending ``budget`` minutes.

    Args:
        spec: The dimensions, as described in the module docstring; each may
            carry a ``deff``, the design effect of its labelling.
        budget: Expert minutes.

    Returns:
        Per dimension: the current estimate, its strata, and for each way of
        spending the minutes the labels to collect and the half-width reached.
    """
    dims = spec["dimensions"]
    tables = {d: strata(v["units"]) for d, v in dims.items()}
    costs = _effective_costs(dims)
    deff = {d: float(v.get("deff", 1.0)) for d, v in dims.items()}
    optimal = neyman(tables, costs, budget)
    uniform = {d: neyman({d: tables[d]}, {d: costs[d]}, budget / len(dims))[d] for d in dims}
    result = {}
    for d in dims:
        single = strata([{**u, "stratum": "all"} for u in dims[d]["units"]])
        expert_labels = budget / len(dims) / costs[d]
        result[d] = {
            "estimate": round(sum(t["W"] * t["p"] for t in tables[d].values()), 3),
            "strata": {h: {k: round(v, 3) for k, v in t.items()} for h, t in tables[d].items()},
            "optimal": {"labels": round(sum(optimal[d].values()) * deff[d], 1),
                        "half_width": round(half_width(tables[d], optimal[d]), 3)},
            "uniform": {"labels": round(sum(uniform[d].values()) * deff[d], 1),
                        "half_width": round(half_width(tables[d], uniform[d]), 3)},
            "expert_only": {"labels": round(expert_labels * deff[d], 1),
                            "half_width": round(half_width(single, {"all": expert_labels}), 3)},
        }
    return result


def minutes_for(spec: dict, target: float) -> float:
    """Fewest minutes for every dimension under ``target`` half-width, optimal allocation scaled up."""
    dims = spec["dimensions"]
    tables = {d: strata(v["units"]) for d, v in dims.items()}
    costs = _effective_costs(dims)
    low, high = 0.0, 1.0
    while max(half_width(tables[d], neyman(tables, costs, high)[d]) for d in dims) > target:
        high *= 2
        if high > 1e7:
            return math.inf
    for _ in range(60):
        mid = (low + high) / 2
        widest = max(half_width(tables[d], neyman(tables, costs, mid)[d]) for d in dims)
        low, high = (mid, high) if widest > target else (low, mid)
    return high


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--budget", type=float, required=True, help="expert minutes")
    parser.add_argument("--target", type=float, default=0.05, help="half-width every dimension must reach")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    report = {
        "budget_min": args.budget,
        "dimensions": compare(spec, args.budget),
        "minutes_for_target": {"target": args.target, "optimal": round(minutes_for(spec, args.target), 1)},
    }
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
