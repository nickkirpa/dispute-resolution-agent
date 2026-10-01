"""Human review of generated data: golden-set complaints and synthetic router messages.

Shows every auto-flagged item plus a random sample per scenario, and writes your verdict back into the file
(review_status: approved | rejected, optional review_note). Rejected items are excluded from evals and router splits.

    uv run python scripts/review.py golden            # eval/golden_v1.jsonl
    uv run python scripts/review.py router            # data/synthetic/not_received.jsonl
    uv run python scripts/review.py golden --sample 3 # per-scenario sample size (flagged items are always shown)
    uv run python scripts/review.py golden --flags-only
    uv run python scripts/review.py golden --stats    # progress only, no prompts

Keys: [a]pprove  [r]eject (asks for a note)  [s]kip  [q]uit. Progress is saved after every answer.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {"golden": ROOT / "eval" / "golden_v1.jsonl", "router": ROOT / "data" / "synthetic" / "not_received.jsonl"}

CONTACT = r"\b(contact\w*|emailed|e-mailed|called|phoned|messaged|reached out|spoke|wrote to|complained to|asked them|told them)\b"
TWICE = r"\b(twice|two times|double|duplicate|again|second time|2 times)\b"
UNAUTH = r"(recogni[sz]e|authori[sz]|did ?n[o']t make|didn'?t make|never made|fraud|compromis|stolen|not me)"
AMOUNT = r"\d+[.,]\d{2}"


def golden_flags(r: dict) -> list[str]:
    """Cheap faithfulness heuristics per scenario. A flag means 'look carefully', not 'wrong'."""
    t, sc = r["narrative"].lower(), r["scenario"]
    flags = []
    if sc == "not_received/request_info_no_contact" and re.search(CONTACT, t):
        flags.append("no-contact case mentions contact")
    if sc.startswith("not_received/") and sc != "not_received/request_info_no_contact" and not re.search(CONTACT, t):
        flags.append("merchant contact not stated")
    if sc.startswith("duplicate/") and not re.search(TWICE, t):
        flags.append("does not say charged twice")
    if sc.startswith("unauthorized/") and not re.search(UNAUTH, t):
        flags.append("does not say payment unrecognised")
    if sc.startswith("refund_not_processed/") and "refund" not in t:
        flags.append("does not mention a refund")
    if sc == "wrong_amount/request_info_no_expected" and len(set(re.findall(AMOUNT, t))) > 1:
        flags.append("states a second amount (could be read as the expected price)")
    if sc == "wrong_amount/refund" and not re.search(r"instead of|agreed|should (have )?been|quoted|price was", t):
        flags.append("expected price phrasing unclear")
    return flags


def router_flags(r: dict) -> list[str]:
    t = r["text"].lower()
    flags = []
    if re.search(r"\d", t):
        flags.append("contains digits (amount/date leak)")
    if re.search(TWICE, t):
        flags.append("sounds like duplicate charge")
    if re.search(UNAUTH, t):
        flags.append("sounds like unauthorized")
    if re.search(r"refund.{0,30}(promised|approved|confirmed|not (yet )?(arrived|received|shown))", t):
        flags.append("sounds like refund_not_processed")
    if not re.search(r"\b(never|not|nothing|no|missed|haven'?t|havent|hasn'?t|didn'?t|didnt|wasn'?t)\b|n't", t):
        flags.append("no negation: non-delivery may not be stated")
    return flags


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path: Path, rows: list[dict]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")  # atomic write: never leave a half-written file
    with os.fdopen(fd, "w") as f:
        f.write("".join(json.dumps(r, default=str) + "\n" for r in rows))
    os.replace(tmp, path)


def show(kind: str, r: dict, flags: list[str], pos: str) -> None:
    print("\n" + "=" * 78)
    if kind == "golden":
        led = r["ledger"]["transactions"]
        case_txns = [t for t in led if t[2] not in {"GroceryCo", "FuelStop", "CafeLumen", "PharmaPlus"}]
        refund = f" {r['expected_refund']:.2f} EUR" if r["expected_refund"] else ""
        print(f"{pos}  {r['case_id']}  [{r['scenario']}]  -> expected {r['expected_decision'].upper()}{refund}")
        print(f"ledger (case txns): {[(t[2], t[3], t[4]) for t in case_txns]}"
              + (f"   prior disputes: {len(r['ledger']['disputes'])}" if r["ledger"]["disputes"] else ""))
        print(f"date rule: {r['generator']['date_rule']}")
        print(f"\n  \"{r['narrative']}\"\n")
        print("Check: does the text state the facts this scenario needs, and nothing that changes the outcome?")
    else:
        print(f"{pos}  [{r['angle']} | {r['style']}]")
        print(f"\n  \"{r['text']}\"\n")
        print("Check: clearly 'paid but not delivered/provided', and not a duplicate/unrecognised/refund-promised message?")
    if flags:
        print(f"FLAGS: {'; '.join(flags)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=list(FILES))
    ap.add_argument("--sample", type=int, default=3, help="random unflagged items per scenario/angle to review")
    ap.add_argument("--flags-only", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = FILES[args.kind]
    rows = load(path)
    flag_fn = golden_flags if args.kind == "golden" else router_flags
    group = (lambda r: r["scenario"]) if args.kind == "golden" else (lambda r: r["angle"])

    status = Counter(r.get("review_status", "unreviewed") for r in rows)
    flagged_all = [r for r in rows if flag_fn(r)]
    print(f"{path.relative_to(ROOT)}: {len(rows)} items | {dict(status)} | auto-flagged: {len(flagged_all)}")
    if args.stats:
        return

    rng = random.Random(args.seed)
    by_group = defaultdict(list)
    for r in rows:
        if r.get("review_status", "unreviewed") == "unreviewed" and not flag_fn(r):
            by_group[group(r)].append(r)
    queue = [r for r in flagged_all if r.get("review_status", "unreviewed") == "unreviewed"]
    if not args.flags_only:
        for g in sorted(by_group):
            queue += rng.sample(by_group[g], min(args.sample, len(by_group[g])))
    if not queue:
        print("Nothing left to review with these settings.")
        return
    print(f"Reviewing {len(queue)} items ({sum(bool(flag_fn(r)) for r in queue)} flagged first, then samples).")

    for i, r in enumerate(queue, 1):
        show(args.kind, r, flag_fn(r), f"[{i}/{len(queue)}]")
        while True:
            try:
                ans = input("[a]pprove  [r]eject  [s]kip  [q]uit > ").strip().lower()
            except EOFError:  # not an interactive terminal
                ans = "q"
            if ans in {"a", "r", "s", "q"}:
                break
        if ans == "q":
            break
        if ans == "s":
            continue
        r["review_status"] = "approved" if ans == "a" else "rejected"
        if ans == "r":
            r["review_note"] = input("why? > ").strip()
        save(path, rows)

    status = Counter(r.get("review_status", "unreviewed") for r in rows)
    print(f"\nSaved. {dict(status)}")


if __name__ == "__main__":
    main()
