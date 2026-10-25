#!/usr/bin/env python3
"""One student-facing command for preflight, original results, tests and smoke runs.

The results command reads committed original metrics. It never recreates or
modifies model checkpoints and never accesses the once-opened final test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
G10 = ROOT / "docs/experiments/g10_five_seed/g10_seed_summary.json"
MATCHED_300K = ROOT / "docs/experiments/matched_300k/summary.json"
REPAIR = ROOT / "docs/experiments/q2_repair_mediation/summary.json"
PATCH = ROOT / "docs/experiments/q2_activation_patching/five_seed_summary.json"
REQUIRED = (
    "README.txt", "requirements.txt", "scripts/check_data.py",
    "scripts/base_recursive.py", "scripts/extended_recursive.py",
    "scripts/evaluate_sealed.py", "external/icl-dynamics",
    str(G10.relative_to(ROOT)), str(REPAIR.relative_to(ROOT)),
    str(PATCH.relative_to(ROOT)),
)

def preflight():
    print("REPOSITORY PREFLIGHT (presence only, not a reproduction)")
    missing = []
    for name in REQUIRED:
        exists = (ROOT / name).exists()
        print(f"{'OK' if exists else 'MISSING':7} {name}")
        if not exists:
            missing.append(name)
    data = ROOT / "external/icl-dynamics/omniglot_resnet18_randomized_order_s0.h5"
    print(f"{'OK' if data.is_file() else 'MISSING':7} original Omniglot features")
    if not data.is_file():
        missing.append(str(data.relative_to(ROOT)))
    print("Run tests and smoke commands separately to verify execution.")
    return bool(missing)

def load(path):
    if not path.is_file():
        raise FileNotFoundError(f"Original metric absent: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))

def index_evidence(out):
    """Inventory committed original evidence without modifying any source file."""
    source = ROOT / "docs/experiments"
    if not source.is_dir():
        raise FileNotFoundError("Committed experiments directory is missing")
    groups = {}
    records = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        group = path.relative_to(source).parts[0]
        ext = path.suffix.lower() or "(none)"
        groups.setdefault(group, {})
        groups[group][ext] = groups[group].get(ext, 0) + 1
        records.append({
            "file": path.relative_to(ROOT).as_posix(),
            "group": group,
            "format": ext,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    if not records:
        raise ValueError("No original experimental outputs were found")
    lines = [
        "# Original experimental evidence index", "",
        "This is an inventory of original experimental results, not a fresh run.", "",
        "| Experiment family | Files | JSON | CSV | Figures |",
        "|:--|--:|--:|--:|--:|",
    ]
    for group, types in sorted(groups.items()):
        files = sum(types.values())
        figures = sum(types.get(k, 0) for k in (".svg", ".png", ".pdf"))
        lines.append(f"| {group} | {files} | {types.get('.json', 0)} | {types.get('.csv', 0)} | {figures} |")
    lines.extend(["", f"Total original evidence files indexed: {len(records)}", "",
                  "The CSV manifest records SHA-256 and exact byte length for each file.",
                  "Large historical checkpoints remain available separately in GitHub",
                  "Actions artifacts or releases and are not indexed here.", ""])
    (out / "RESULTS_INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    with (out / "RESULTS_FILE_MANIFEST.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("file", "group", "format", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(records)
    print(f"Indexed {len(records)} original evidence files in {len(groups)} families")


def summarize(out):
    g10, repair, patch = load(G10), load(REPAIR), load(PATCH)
    if g10["n_model_seeds_per_task"] != 5 or patch["n_independent_model_pairs"] != 5:
        raise ValueError("Five-seed provenance mismatch")
    if repair["n_independent_paired_training_seed_experiments"] != 5:
        raise ValueError("Repair seed-count mismatch")
    if g10["independent_test_set_for_G10"] is not False or repair["final_test_accessed"] is not False:
        raise ValueError("Final-test isolation requirement not met")
    matched = load(MATCHED_300K)
    base_data = matched["base"]
    if base_data["n_independent_model_seeds"] != 5 or matched["base_new_G1_G10_models"] != 100 or matched["historical_final_test_accessed_by_this_study"] is not False:
        raise ValueError("Matched 300k Base provenance or test isolation mismatch")
    base = {"mean_clean_minus_recursive": base_data["mean_clean_minus_recursive"], "indicative_paired_t95_interval": base_data["descriptive_t95_interval"]}
    extended = g10["paired_effects"]["extended"]["generated_next_symbol_validity"]
    mass = repair["metrics"]["valid_mass"]
    logit = repair["metrics"]["context_preference_logit_units"]
    measures = [
        ("Base G10", "Label accuracy (control minus recursive)", base["mean_clean_minus_recursive"], base["indicative_paired_t95_interval"], "pp", MATCHED_300K),
        ("Extended G10", "Sampled q2 validity (control minus recursive)", extended["mean_clean_minus_recursive"], extended["indicative_paired_t95_interval"], "pp", G10),
        ("Target repair", "Valid-symbol mass (repaired minus raw)", mass["mean"], mass["paired_t95_interval"], "pp", REPAIR),
        ("Target repair", "Context-switch preference (repaired minus raw)", logit["mean"], logit["paired_t95_interval"], "logits", REPAIR),
    ]
    out.mkdir(parents=True, exist_ok=True)
    md = [
        "# Original model-collapse results (five paired seeds)", "",
        "This report reads existing original experiment JSON. It does not rerun research.", "",
        "| Study | Outcome | Paired effect | Descriptive 95% interval |",
        "|:--|:--|--:|--:|",
    ]
    records = []
    print("ORIGINAL RESULTS (n=5 seed pairs, validation-only G10)\n")
    for study, metric, value, interval, unit, source in measures:
        if unit == "pp":
            shown = f"{100 * value:+.2f} pp"
            ci = f"[{100 * interval[0]:.2f}, {100 * interval[1]:.2f}] pp"
        else:
            shown = f"{value:+.2f} logits"
            ci = f"[{interval[0]:.2f}, {interval[1]:.2f}] logits"
        print(f"{study:<16}{shown:>14}  {metric}")
        md.append(f"| {study} | {metric} | {shown} | {ci} |")
        records.append({"study": study, "metric": metric, "paired_effect": shown,
                        "descriptive_95_interval": ci, "source": str(source.relative_to(ROOT))})
    notes = [
        "G10 findings are validation-only: the original 100-class final test was already opened at G3.",
        "The synthetic loop replaces training targets, not the genuine Omniglot input context.",
        "The reported Base and Extended G10 primary comparisons each use 300,000 episodes per generation and five seed pairs; their metrics, tasks, and difficulty still differ.",
        "Base clean G10 accuracy is 61.56% in the matched-budget sample; one of five clean seeds is below 60%.",
        "The q2 vocabulary is the 50 training identities and is not defined for unseen class IDs.",
        "The five-seed intervals describe variation across model seeds; they are not a new independent test result.",
        "Repair changes all trainable weights; attention activation patching does not prove an isolated unique head mechanism.",
    ]
    md.extend(["", "## Limitations", ""] + ["- " + n for n in notes])
    md.extend(["", "## Provenance", ""] + sorted(set("- " + r["source"] for r in records)))
    (out / "RESULTS_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (out / "RESULTS_SUMMARY.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    index_evidence(out)
    from scripts.audit_methodology import audit, write_report
    write_report(audit(ROOT), out)
    print(f"\nWrote {out / 'RESULTS_SUMMARY.md'} and {out / 'RESULTS_SUMMARY.csv'}")
    return False

def run_command(args):
    print("$ " + " ".join(args), flush=True)
    return subprocess.call(args, cwd=ROOT) != 0

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "results", "methodology", "tests", "data-check", "verify-checkpoints", "restore-checkpoints", "smoke-base", "smoke-extended", "fresh-plan", "fresh-aggregate", "matched-plan", "matched-aggregate", "base300k-plan", "base300k-aggregate"))
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/reports")
    args = parser.parse_args()
    if args.command == "check":
        return preflight()
    if args.command == "results":
        return summarize(args.out.resolve())
    if args.command == "methodology":
        from scripts.audit_methodology import audit, write_report
        write_report(audit(ROOT), args.out.resolve())
        print("METHOD_CONTROLS_VERIFIED=original single-seed controls")
        return False
    if args.command == "verify-checkpoints":
        return run_command([sys.executable, "scripts/checkpoint_registry.py", "verify"])
    if args.command == "restore-checkpoints":
        return run_command([sys.executable, "scripts/checkpoint_registry.py", "restore"])
    if args.command == "tests":
        return run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    if args.command == "data-check":
        return run_command([sys.executable, "scripts/check_data.py"])
    if args.command == "base300k-plan":
        return run_command([sys.executable, "-m", "scripts.base_300k_replication", "plan"])
    if args.command == "base300k-aggregate":
        return run_command([sys.executable, "-m", "scripts.aggregate_base_300k"])
    if args.command == "matched-plan":
        return run_command([sys.executable, "scripts/fresh_five_seed.py", "plan",
                            "--extended-exposures", "1000000"])
    if args.command == "matched-aggregate":
        return run_command([sys.executable, "scripts/fresh_five_seed.py", "aggregate",
                            "--root", "outputs/matched_five_inputs",
                            "--out", "outputs/matched_five_results",
                            "--extended-exposures", "1000000",
                            "--base-reference-run-id", "40194253117"])
    if args.command == "fresh-plan":
        return run_command([sys.executable, "scripts/fresh_five_seed.py", "plan"])
    if args.command == "fresh-aggregate":
        return run_command([sys.executable, "scripts/fresh_five_seed.py", "aggregate",
                            "--root", "outputs/fresh_five_inputs",
                            "--out", "outputs/fresh_five_results"])
    if args.command == "smoke-base":
        return run_command([sys.executable, "scripts/base_recursive.py", "--artifact-root", "outputs/base_full_1m",
                            "--generations", "1", "--steps", "4", "--batch-size", "8",
                            "--eval-every", "4", "--eval-count", "32",
                            "--arms", "clean_only", "recursive_replacement", "--out", "outputs/smoke_base"])
    if args.command == "smoke-extended":
        return run_command([sys.executable, "scripts/extended_recursive.py",
                            "--base-artifact", "outputs/base_full_1m",
                            "--extended-g0-artifact", "outputs/extended_g0_full",
                            "--generations", "1", "--steps", "4", "--batch-size", "8",
                            "--eval-every", "4", "--eval-count", "32",
                            "--arms", "clean_only", "recursive_replacement",
                            "--allow-incomplete-g0-for-smoke", "--out", "outputs/smoke_extended"])
    return True

if __name__ == "__main__":
    try:
        sys.exit(int(main()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print("ERROR:", error, file=sys.stderr)
        sys.exit(1)
