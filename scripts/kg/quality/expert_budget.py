#!/usr/bin/env python3
"""Share one budget of expert minutes among the quality dimensions.

Each dimension (accuracy, interoperability, coverage, conformity) is the
share of its units that are right: edges, mappings, answers, rule
violations. An automatic meter (a judge, the rules themselves) puts every
unit in a stratum for free; a few human labels per stratum then estimate the
share with a stratified estimator, whose variance is
``sum_h W_h^2 S_h^2 / n_h * (1 - n_h / N_h)`` (W the stratum's share of the
units, S its standard deviation estimated from the labels already there, N
its units). A dimension with few units (twenty mappings) is read in full
for a few minutes, so no stratum is given more labels than it has units.

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
        result[name] = {"W": entry["count"] / total, "labelled": len(ys), "p": p, "S": s, "N": entry["count"]}
    return result


def half_width(table: dict[str, dict], labels: dict[str, float], deff: float = 1.0) -> float:
    """Half-width of the 95 % interval of the stratified share with these labels per stratum.

    ``labels`` count independent-equivalent labels; ``deff`` turns them back
    into labels read, which is what exhausts a stratum.
    """
    variance = 0.0
    for h, t in table.items():
        if not t["W"]:
            continue
        n = max(labels.get(h, 0.0), 1e-9)
        read = min(1.0, n * deff / t.get("N", math.inf))
        variance += t["W"] ** 2 * t["S"] ** 2 / n * (1 - read)
    return Z * math.sqrt(variance)


def neyman(tables: dict[str, dict], costs: dict[str, float], minutes: float,
           deffs: dict[str, float] | None = None) -> dict[str, dict[str, float]]:
    """Labels per (dimension, stratum) minimising the summed variance for these minutes.

    A stratum whose share would exceed its units is read in full, and the
    minutes left go to the others in the same proportions.
    """
    deffs = deffs or {}
    cap = {(d, h): t.get("N", math.inf) / deffs.get(d, 1.0) for d, table in tables.items() for h, t in table.items()}
    out: dict[str, dict[str, float]] = {d: {h: 0.0 for h in table} for d, table in tables.items()}
    free = set(cap)
    left = minutes
    while free and left > 0:
        weight = {(d, h): tables[d][h]["W"] * tables[d][h]["S"] / math.sqrt(costs[d]) for d, h in free}
        spend = sum(tables[d][h]["W"] * tables[d][h]["S"] * math.sqrt(costs[d]) for d, h in free)
        if not spend:
            break
        share = {k: left * w / spend for k, w in weight.items()}
        full = {k for k, n in share.items() if n >= cap[k]}
        if not full:
            for (d, h), n in share.items():
                out[d][h] = n
            break
        for d, h in full:
            out[d][h] = cap[(d, h)]
            left -= cap[(d, h)] * costs[d]
        free -= full
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
    optimal = neyman(tables, costs, budget, deff)
    uniform = {d: neyman({d: tables[d]}, {d: costs[d]}, budget / len(dims), deff)[d] for d in dims}
    result = {}
    for d in dims:
        single = strata([{**u, "stratum": "all"} for u in dims[d]["units"]])
        expert_labels = min(budget / len(dims) / costs[d], len(dims[d]["units"]) / deff[d])
        result[d] = {
            "estimate": round(sum(t["W"] * t["p"] for t in tables[d].values()), 3),
            "strata": {h: {k: round(v, 3) for k, v in t.items()} for h, t in tables[d].items()},
            "optimal": {"labels": round(sum(optimal[d].values()) * deff[d], 1),
                        "half_width": round(half_width(tables[d], optimal[d], deff[d]), 3)},
            "uniform": {"labels": round(sum(uniform[d].values()) * deff[d], 1),
                        "half_width": round(half_width(tables[d], uniform[d], deff[d]), 3)},
            "expert_only": {"labels": round(expert_labels * deff[d], 1),
                            "half_width": round(half_width(single, {"all": expert_labels}, deff[d]), 3)},
        }
    return result


def minutes_for(spec: dict, target: float) -> float:
    """Fewest minutes for every dimension under ``target`` half-width, optimal allocation scaled up."""
    dims = spec["dimensions"]
    tables = {d: strata(v["units"]) for d, v in dims.items()}
    costs = _effective_costs(dims)
    deff = {d: float(v.get("deff", 1.0)) for d, v in dims.items()}

    def widest(minutes: float) -> float:
        allocation = neyman(tables, costs, minutes, deff)
        return max(half_width(tables[d], allocation[d], deff[d]) for d in dims)

    low, high = 0.0, 1.0
    while widest(high) > target:
        high *= 2
        if high > 1e7:
            return math.inf
    for _ in range(60):
        mid = (low + high) / 2
        widest_mid = widest(mid)
        low, high = (mid, high) if widest_mid > target else (low, mid)
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
