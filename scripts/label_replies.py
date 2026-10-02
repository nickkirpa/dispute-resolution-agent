"""Human labels for validating the LLM judge. Run in your own terminal:

    uv run python scripts/label_replies.py            # label (resumes where you stopped)
    uv run python scripts/label_replies.py --stats    # progress only
    uv run python scripts/label_replies.py --redo R33 # forget the label of R33 (or several: --redo R33 R07) and label it again

For each reply you see the case facts and the reply the agent would send, and answer 5 yes/no questions:
  1 facts     facts correct: merchant, amount, dates match the case; nothing invented
  2 decision  matches the decision (refund states amount / reject explains / request_info says what is needed /
              escalate says a person will review)
  3 promises  no invented promises (deadlines, guarantees, amounts not in the decision or policy)
  4 clarity   clear and polite: the customer understands what happens next
  5 send      OK to send to a real customer as is?
Keys: y / n for each question, s = skip reply, q = quit. Labels are saved after every reply.
Whether a reply was deliberately altered is NOT shown: judge each reply on its own.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLIES = ROOT / "eval" / "judge" / "replies.jsonl"
LABELS = ROOT / "eval" / "judge" / "labels.jsonl"
QUESTIONS = [
    ("facts", "Facts correct (merchant, amount, dates; nothing invented)?"),
    ("decision", "Matches the decision?"),
    ("promises", "No invented promises?"),
    ("clarity", "Clear and polite?"),
    ("send", "OK to send as is?"),
]


def ask(prompt: str) -> str:
    while True:
        try:
            a = input(f"  {prompt} [y/n/s/q] > ").strip().lower()
        except EOFError:
            return "q"
        if a in {"y", "n", "s", "q"}:
            return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--redo", nargs="+", default=[], metavar="ID", help="drop these labels and show the replies again")
    args = ap.parse_args()
    items = [json.loads(l) for l in REPLIES.read_text().splitlines() if l.strip()]
    done = {json.loads(l)["id"]: json.loads(l) for l in LABELS.read_text().splitlines() if l.strip()} if LABELS.exists() else {}
    for rid in args.redo:
        if done.pop(rid, None) is not None:
            print(f"dropped label {rid}; it will be shown again")
    if args.redo:
        LABELS.write_text("".join(json.dumps(v) + "\n" for v in done.values()))
    print(f"{len(done)}/{len(items)} replies labelled")
    if args.stats:
        return
    for n, it in enumerate(items, 1):
        if it["id"] in done:
            continue
        f = it["facts"]
        print("\n" + "=" * 90)
        print(f"[{n}/{len(items)}] {it['id']}   DECISION: {f['decision'].upper()}"
              + (f"   REFUND: {f['refund_amount']:.2f} EUR" if f["decision"] == "refund" else ""))
        print("Case transactions: " + "; ".join(f"{t['merchant']} {t['amount']:.2f} on {t['date']}" for t in f["case_transactions"])
              + (f"   | prior disputes: {f['prior_disputes']}" if f["prior_disputes"] else ""))
        print("Customer wrote:\n" + textwrap.indent(textwrap.fill(f["complaint"], 100), "   "))
        print("\nREPLY THE AGENT WOULD SEND:\n" + textwrap.indent(textwrap.fill(it["reply"], 100), "   ") + "\n")
        label, quit_ = {"id": it["id"]}, False
        for key, q in QUESTIONS:
            a = ask(q)
            if a == "q":
                quit_ = True
                break
            if a == "s":
                label = None
                break
            label[key] = a == "y"
        if quit_:
            break
        if label is None:
            continue
        try:
            label["note"] = input("  note (optional, Enter to skip) > ").strip()
        except EOFError:
            label["note"] = ""
        done[it["id"]] = label
        LABELS.write_text("".join(json.dumps(v) + "\n" for v in done.values()))
    print(f"\nSaved. {len(done)}/{len(items)} labelled -> {LABELS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
