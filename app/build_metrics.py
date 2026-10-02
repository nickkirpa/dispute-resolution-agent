"""Snapshot the headline eval results into app/data/metrics.json (committed; eval/results/ is not).

Every number is read from an eval report, never typed in; `source` records which file it came from.

    uv run python app/build_metrics.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "eval" / "results"
sys.path.insert(0, str(ROOT))
from eval.failure_taxonomy import ORDER, summarize  # noqa: E402


def latest(pattern: str, golden: str | None = "golden_v1.jsonl") -> tuple[dict, str]:
    for path in sorted(glob.glob(str(RES / pattern)), reverse=True):
        r = json.loads(Path(path).read_text())
        g = r.get("summary", {}).get("golden") or r.get("aggregate", {}).get("golden")
        if golden is None or g == golden:
            return r, Path(path).name
    raise SystemExit(f"no report matches {pattern} on {golden}")


def config_row(label: str, pattern: str, note: str = "") -> dict:
    r, src = latest(pattern)
    if "aggregate" in r:  # repeated runs: mean of each metric
        m = r["aggregate"]["metrics"]
        return {"config": label, "decision_accuracy": m["decision_accuracy"]["mean"], "runs": r["aggregate"]["runs"],
                "unsafe_refund_rate": m["unsafe_refund_rate"]["mean"], "cost_per_case_usd": m["cost_per_case_usd"]["mean"],
                "latency_s": m["avg_latency_s"]["mean"], "note": note, "source": src}
    s = r["summary"]
    return {"config": label, "decision_accuracy": s["decision_accuracy"], "runs": 1, "unsafe_refund_rate": s["unsafe_refund_rate"],
            "cost_per_case_usd": s["total_cost_usd"] / s["n_cases"], "latency_s": s["avg_latency_s"], "note": note, "source": src}


def main() -> None:
    configs = [
        config_row("Rule baseline (regex)", "*-rules_plan.json"),
        config_row("gpt-5.4-nano · plan", "*gpt-5.4-nano_plan.json", "after the POL-UNA-02 guard fix"),
        config_row("gpt-5.4-mini · plan", "*gpt-5.4-mini_plan-x3.json"),
        config_row("gpt-5.4-mini · plan + router", "*gpt-5.4-mini_plan_router0.9-x3.json"),
        config_row("gpt-5.4-mini · agent", "*gpt-5.4-mini_agent.json"),
        config_row("gpt-5.4-mini · agent + hybrid", "*gpt-5.4-mini_agent_hybrid.json"),
        config_row("gpt-5.5 · plan", "*gpt-5.5_plan.json"),
    ]

    total: dict[str, int] = {}
    for path in glob.glob(str(RES / "2*.json")):
        name = Path(path).stem
        if "-x" in name or "gpt-5.4-mini" not in name:  # the chosen model only; nano/5.5 are in the comparison table
            continue
        r = json.loads(Path(path).read_text())
        if r.get("summary", {}).get("golden") not in ("golden_v1.jsonl", "golden_policy_shift.jsonl", "golden_policy_shift_policy_v2.jsonl"):
            continue
        if "nofill" in name:  # the ablation fails on purpose; it is reported separately
            continue
        for k, v in summarize(r)["primary"].items():
            total[k] = total.get(k, 0) + v
    taxonomy = [{"cause": k, "count": total.get(k, 0)} for k in ORDER]

    shift_v1, src1 = latest("*gpt-5.4-mini_agent.json", "golden_policy_shift.jsonl")
    shift_v2, src2 = latest("*gpt-5.4-mini_agent.json", "golden_policy_shift_policy_v2.jsonl")
    d1 = {c["case_id"]: c["decision"] for c in shift_v1["cases"]}
    flips = sum(d1[c["case_id"]] != c["decision"] for c in shift_v2["cases"])
    policy = {"n": len(shift_v2["cases"]), "v1_correct": sum(c["decision_ok"] for c in shift_v1["cases"]),
              "v2_correct": sum(c["decision_ok"] for c in shift_v2["cases"]), "decisions_changed": flips, "source": [src1, src2]}

    router = json.loads((RES / "router-ModernBERT-base.json").read_text())
    base = json.loads(next(RES.glob("router-baseline-*.json")).read_text())
    router_cmp = {"router": {"intent_accuracy": router["test"]["banking77"]["intent"]["accuracy"],
                             "type_accuracy": router["test"]["all"]["dispute_type"]["accuracy"],
                             "type_macro_f1": router["test"]["all"]["dispute_type"]["macro_f1"],
                             "latency_ms": router["latency_ms"]["single_message"], "train_minutes": router["train_minutes"]},
                  "llm": {"type_accuracy": base["results"]["all"]["accuracy"], "type_macro_f1": base["results"]["all"]["macro_f1"],
                          "latency_ms": base["latency_s"]["mean"] * 1000, "cost_per_1k_usd": base["cost_per_1k_usd"]}}

    judge = {}
    for v in ("v1", "v2"):
        path = next(p for p in sorted(RES.glob(f"judge-*-{v}-x3.json")) if "screen" not in p.name)  # validation runs only
        j = json.loads(path.read_text())
        judge[v] = {q: round(m["kappa"], 3) for q, m in j["per_question"].items()} | {
            "send_agreement": j["per_question"]["send"]["agreement"]}

    retrieval = json.loads(next(RES.glob("retrieval-v2-openai*.json")).read_text())["results"]

    out = ROOT / "app" / "data" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"configs": configs, "failure_taxonomy": taxonomy, "failure_taxonomy_scope": "gpt-5.4-mini single runs on golden_v1 + policy-change sets", "policy_change": policy,
                               "router": router_cmp, "judge": judge, "retrieval": retrieval}, indent=1))
    for c in configs:
        print(f"{c['config']:<32} acc {c['decision_accuracy']:.3f} unsafe {c['unsafe_refund_rate']:.3f} "
              f"${c['cost_per_case_usd']:.4f} {c['latency_s']:.1f}s  ({c['runs']} run(s), {c['source']})")
    print("taxonomy:", total, "| policy:", policy["v1_correct"], policy["v2_correct"], flips)
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
