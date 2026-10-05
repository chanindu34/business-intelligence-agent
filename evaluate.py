"""
Evaluate the agent on eval/questions.yaml.

    python3 evaluate.py --check     # verify report figures appear in the index, no API calls
    python3 evaluate.py             # full run, roughly 2 to 4 API calls per question

Metrics
  tool selection   the agent used exactly the expected set of tools (or a listed alternative)
  answer correct   expected values present, forbidden ones absent
  over-searching   searched when no tool was needed
  turns, latency   median and 95th percentile
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
QUESTIONS = ROOT / "eval" / "questions.yaml"
RESULTS = ROOT / "eval" / "results"
RETRY_WAIT_SECONDS = 40


def normalise(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\*\*|__|`", "", t)
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    t = re.sub(r"\s*(?:per cent|percent)\b", "%", t)
    t = re.sub(r"\brs\.?\s*(?=\d)", "rs ", t)
    return re.sub(r"\s+", " ", t)


def contains(text: str, needle: str) -> bool:
    t, n = normalise(text), normalise(needle)
    if re.search(r"\d", n):
        return re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", t) is not None
    return n in t


def answer_ok(answer: str, q: dict):
    problems = [m for m in q.get("must_include", []) if not contains(answer, m)]
    if q.get("any_of") and not any(contains(answer, a) for a in q["any_of"]):
        problems.append("one of " + " / ".join(q["any_of"]))
    problems += [f"should not say {m}" for m in q.get("must_not", []) if contains(answer, m)]
    return not problems, problems


def tools_match(used, q: dict) -> bool:
    allowed = [q["tools"]] + q.get("also_ok_tools", [])
    return any(sorted(set(used)) == sorted(set(a)) for a in allowed)


def load():
    qs = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))
    assert len({q["id"] for q in qs}) == len(qs), "duplicate ids"
    return qs


def check(qs) -> int:
    from retrieval import ReportIndex

    corpus = " ".join(ReportIndex.load().docs)
    bad = 0
    for q in qs:
        if q.get("computed") or not q["tools"]:
            continue
        needles = q.get("must_include", [])
        ok = all(contains(corpus, n) for n in needles) and (
            not q.get("any_of") or any(contains(corpus, a) for a in q["any_of"]))
        if not ok:
            bad += 1
            print(f"  PROBLEM {q['id']}: expected values not found in the index")
    print(f"{len(qs)} questions. " + ("Answer key OK." if not bad else f"{bad} need fixing."))
    return bad


def pct(values, p):
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def run(qs):
    from agent import run_agent
    from service import get_client, get_embedder, get_index

    client, index, embed = get_client(), get_index(), get_embedder()
    rows = []
    for i, q in enumerate(qs, 1):
        r = run_agent(q["question"], client, index=index, embed=embed)
        if r.stopped == "no_model":
            # Usually a per-minute limit: wait it out once rather than score a non-answer.
            print(f"   [{i:>2}/{len(qs)}] {q['id']}: no model available, waiting {RETRY_WAIT_SECONDS}s and retrying once")
            time.sleep(RETRY_WAIT_SECONDS)
            r = run_agent(q["question"], client, index=index, embed=embed)
        if r.stopped == "no_model":
            rows.append({"id": q["id"], "question": q["question"], "expected_tools": q["tools"],
                         "not_run": True, "answer": r.answer, "model": None, "stopped": r.stopped})
            print(f"-- [{i:>2}/{len(qs)}] {q['id']:<24} NOT RUN: no model available (quota)")
            continue
        tools_ok = tools_match(r.tools_used, q)
        ok, problems = answer_ok(r.answer, q)
        rows.append({"id": q["id"], "question": q["question"], "expected_tools": q["tools"],
                     "tools_used": r.tools_used, "tools_ok": tools_ok, "answer_ok": ok, "problems": problems,
                     "answer": r.answer, "model": r.model, "turns": r.turns, "stopped": r.stopped,
                     "seconds": r.seconds, "pages": r.pages})
        print(f"{'OK ' if tools_ok and ok else 'XX '}[{i:>2}/{len(qs)}] {q['id']:<24} tools={r.tools_used} "
              f"{r.seconds:.1f}s {problems if problems else ''}")
    return rows


def report(all_rows):
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{stamp}.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    rows = [r for r in all_rows if not r.get("not_run")]
    skipped = [r["id"] for r in all_rows if r.get("not_run")]
    n = len(rows)
    if not n:
        path = RESULTS / f"{stamp}.md"
        path.write_text(f"# Agent evaluation {stamp}\n\nNo question could run: every model was out of quota.\n",
                        encoding="utf-8")
        return path
    models = {}
    for r in rows:
        models[r["model"]] = models.get(r["model"], 0) + 1
    tool_acc = sum(r["tools_ok"] for r in rows) / n
    ans_acc = sum(r["answer_ok"] for r in rows) / n
    no_tool = [r for r in rows if not r["expected_tools"]]
    over = sum(1 for r in no_tool if "search_knowledge_base" in r["tools_used"])
    secs = [r["seconds"] for r in rows]
    scored = f"{n} questions scored" + (f"; {len(skipped)} not run (no model available): {', '.join(skipped)}"
                                        if skipped else "")
    lines = [f"# Agent evaluation {stamp}", "", scored, "",
             "Models that answered: " + ", ".join(f"{m} ({c})" for m, c in sorted(models.items())), "",
             "| Metric | Result |", "|---|---|",
             f"| Tool selection accuracy | {tool_acc:.0%} ({sum(r['tools_ok'] for r in rows)}/{n}) |",
             f"| Answer accuracy | {ans_acc:.0%} ({sum(r['answer_ok'] for r in rows)}/{n}) |",
             f"| Searched when no tool was needed | {over}/{len(no_tool)} |",
             f"| Turns, median | {pct([r['turns'] for r in rows], 50):.0f} |",
             f"| Latency, median / p95 | {pct(secs, 50):.1f}s / {pct(secs, 95):.1f}s |",
             "", "## Failures", ""]
    for r in rows:
        if not (r["tools_ok"] and r["answer_ok"]):
            lines.append(f"- **{r['id']}**: expected tools {r['expected_tools']}, used {r['tools_used']}; {r['problems']}")
    path = RESULTS / f"{stamp}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", help="comma-separated ids")
    args = ap.parse_args()
    qs = load()
    if args.only:
        qs = [q for q in qs if q["id"] in set(args.only.split(","))]
    if args.check:
        sys.exit(1 if check(qs) else 0)
    if check(qs):
        sys.exit("Fix the answer key first.")
    path = report(run(qs))
    print("\n" + path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
