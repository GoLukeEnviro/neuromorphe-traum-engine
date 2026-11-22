"""Misst die Suchqualität der semantischen Suche gegen ein festes Prompt-Set.

Aufruf:
    PYTHONPATH=src .venv/bin/python scripts/evaluate_search.py [--topk 5]

Bewertet wird Precision@k: Wie viele der k besten Treffer gehören in die
Kategorie, die der Prompt meint? Das ist ein grober, aber ehrlicher Indikator —
die Ground Truth ist die Kategorie des Stems, nicht ein menschliches Urteil
darüber, ob der Klang zum Prompt passt.

Die Quote ist damit eine OBERGRENZE für die wahrgenommene Qualität: Ein
Kick-Sample in der Kategorie „kick" kann trotzdem nicht zum Prompt passen.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Prompt -> Kategorien, die als Treffer zählen.
EVAL_SET = [
    ("dark driving kick", {"kick"}),
    ("punchy techno kick drum", {"kick"}),
    ("deep sub bass kick", {"kick"}),
    ("tight closed hihat", {"hihat"}),
    ("open hihat groove", {"hihat"}),
    ("metallic percussion", {"percussion"}),
    ("wooden percussion hit", {"percussion"}),
    ("acid bassline", {"bass"}),
    ("deep rolling sub bass", {"bass"}),
    ("dark atmospheric pad", {"atmosphere"}),
    ("eerie background texture", {"atmosphere"}),
    ("industrial clap", {"snare"}),
    ("sharp snare hit", {"snare"}),
    ("hypnotic synth stab", {"synth"}),
    ("chord stab techno", {"synth"}),
    ("four on the floor drum loop", {"loop"}),
    ("134 bpm techno loop", {"loop"}),
    ("hard warehouse kick", {"kick"}),
    ("clanging metal hit", {"percussion"}),
    ("slow evolving drone", {"atmosphere"}),
]


async def run(topk: int) -> None:
    from search.service import SearchService
    from search.schemas import SearchRequest

    service = SearchService()
    rows = []
    hits = 0
    total = 0

    print(f"{'Prompt':34} {'Treffer':>8} {'Precision':>10}  Top-Ergebnisse")
    print("-" * 100)

    for prompt, expected in EVAL_SET:
        try:
            result = await service.search_by_text(
                SearchRequest(query=prompt, limit=topk)
            )
            items = getattr(result, "results", None) or []
            items = [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in items
            ]
        except Exception as exc:
            print(f"{prompt:34} FEHLER: {type(exc).__name__}: {str(exc)[:60]}")
            continue

        found = 0
        names = []
        for item in items[:topk]:
            category = (
                item.get("category") if isinstance(item, dict) else getattr(item, "category", None)
            )
            name = (
                item.get("filename") or item.get("name")
                if isinstance(item, dict)
                else getattr(item, "filename", "?")
            )
            names.append(f"{name}:{category}")
            if category in expected:
                found += 1

        precision = found / topk if topk else 0.0
        hits += found
        total += topk
        rows.append({"prompt": prompt, "expected": sorted(expected), "found": found, "precision": precision})
        print(f"{prompt:34} {found:>3}/{topk:<4} {precision:>9.0%}  {', '.join(names[:3])}")

    overall = hits / total if total else 0.0
    print("-" * 100)
    print(f"Precision@{topk} gesamt: {overall:.1%}  ({hits}/{total})")

    out = Path("evaluation_results.json")
    out.write_text(json.dumps({"topk": topk, "precision": overall, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Details: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run(args.topk))


if __name__ == "__main__":
    main()
