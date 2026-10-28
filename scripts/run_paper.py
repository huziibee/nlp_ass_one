import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEEDS = (6001, 5054, 6054, 7054, 8054)
ARMS = ("clean_only", "recursive_replacement")


def read_json(path):
    return json.loads(path.read_text())


def complete(paths):
    return bool(paths) and all(
        path.is_file() and path.stat().st_size > 0 for path in paths
    )


def task_complete(task, refresh_summaries=True):
    if not complete(task["outputs"]):
        return False
    name = task["name"]
    if name.startswith("evidence-"):
        report = task["outputs"][0].parent / "report.json"
        if not report.is_file() or read_json(report).get("complete") is not True:
            return False
    if name == "figure-fresh-results":
        figure = task["outputs"][0]
        folder = figure.parent / "paper_results"
        sources = [
            folder / name
            for name in (
                "summary.json",
                "base_300k_seed_outcomes.csv",
                "five_seed_generation_summary.csv",
            )
        ]
        if not complete(sources) or figure.stat().st_mtime < max(
            path.stat().st_mtime for path in sources
        ):
            return False
    if refresh_summaries and name.startswith(
        ("aggregate-", "summarize-", "figure-", "evidence-")
    ):
        return False
    if name.startswith("base-g0-") or name.startswith("base-chain-"):
        from scripts.base_300k_replication import check_g0, check_chain

        try:
            parts = name.split("-")
            index = int(parts[2])
            if name.startswith("base-g0-"):
                check_g0(task["directory"], index)
            else:
                check_chain(task["directory"], parts[3], index)
        except (ValueError, KeyError, OSError, AssertionError, IndexError):
            return False
    for path in task["outputs"]:
        if path.suffix != ".json":
            continue
        try:
            data = read_json(path)
        except (ValueError, OSError):
            return False
        if not isinstance(data, dict):
            return False
        if any(
            data.get(key) is True
            for key in (
                "final_test_evaluated",
                "final_test_accessed",
                "final_test_access",
            )
        ):
            return False
        if path.name == "metrics.json":
            budget = data.get(
                "training_exposures",
                data.get("optimizer_exposures", data.get("train_exposures")),
            )
            if budget is None and "steps" in data and "batch_size" in data:
                budget = data["steps"] * data["batch_size"]
            expected = 100000 if name == "cumulative-pilot" else 300000
            if budget != expected:
                return False
            if data.get("history"):
                final = data["history"][-1]
                seen = final.get(
                    "sequence_exposures", final.get("seen_training_sequences", budget)
                )
                if seen != expected:
                    return False
    return True


def generation_outputs(root, arms, count=10):
    return [
        root / arm / f"g{generation:02d}" / name
        for arm in arms
        for generation in range(1, count + 1)
        for name in ("metrics.json", "model.eqx")
    ]


def extended_paths(out, seed):
    if seed == 6001:
        root = out / "extended_primary"
        return root / "g00", root, root
    return (
        out / f"extended_g0_{seed}",
        out / f"extended_{seed}_clean_only",
        out / f"extended_{seed}_recursive_replacement",
    )


def tasks(out):
    base = out / "base_300k_g0_0"
    python = sys.executable

    def script(name, *args):
        return [python, str(ROOT / "scripts" / name), *map(str, args)]

    def task(name, command, outputs, directory=None):
        return {
            "name": name,
            "command": command,
            "outputs": list(outputs),
            "directory": directory,
        }

    for index in range(5):
        g0 = out / f"base_300k_g0_{index}"
        yield task(
            f"base-g0-{index}",
            [
                python,
                "-m",
                "scripts.base_300k_replication",
                "g0",
                "--seed-index",
                str(index),
                "--g0-root",
                str(g0),
            ],
            [
                g0 / "safe_train_val_smoke/checkpoints/00000300000.eqx",
                g0 / "independent_protocol.json",
            ],
            g0,
        )
        for arm in ARMS:
            root = out / f"base_300k_{index}_{arm}"
            yield task(
                f"base-chain-{index}-{arm}",
                [
                    python,
                    "-m",
                    "scripts.base_300k_replication",
                    "chain",
                    "--seed-index",
                    str(index),
                    "--arm",
                    arm,
                    "--g0-root",
                    str(g0),
                    "--out",
                    str(root),
                ],
                generation_outputs(root, [arm]),
                root,
            )
    common = [
        "--steps",
        9375,
        "--batch-size",
        32,
        "--eval-count",
        512,
        "--learning-rate",
        0.0003,
        "--temperature",
        1.0,
    ]
    for seed in SEEDS:
        g0, clean, recursive = extended_paths(out, seed)
        init = 4054 if seed == 6001 else seed
        every = 3125 if seed == 6001 else 9375
        yield task(
            f"extended-g0-{seed}",
            script(
                "extended_g0.py",
                "--artifact-root",
                base,
                "--seed",
                init,
                "--eval-every",
                every,
                "--out",
                g0,
                *common,
            ),
            [g0 / "metrics.json", g0 / "extended_g0_model.eqx"],
            g0,
        )
        for arm, root in zip(ARMS, (clean, recursive)):
            yield task(
                f"extended-chain-{seed}-{arm}",
                script(
                    "extended_recursive.py",
                    "--base-artifact",
                    base,
                    "--extended-g0-artifact",
                    g0,
                    "--include-all-replication-seeds",
                    "--generations",
                    10,
                    "--seed",
                    seed,
                    "--initialization-seed",
                    init,
                    "--eval-every",
                    every,
                    "--arms",
                    arm,
                    "--out",
                    root,
                    *common,
                ),
                generation_outputs(root, [arm]),
                root if seed != 6001 else root / arm,
            )
    primary_g0, primary, _ = extended_paths(out, 6001)
    for arm in ("frozen_g0", "real_anchor_half"):
        root = out / "extended_controls" / arm
        yield task(
            f"extended-control-{arm}",
            script(
                "extended_recursive.py",
                "--base-artifact",
                base,
                "--extended-g0-artifact",
                primary_g0,
                "--generations",
                10,
                "--seed",
                6001,
                "--initialization-seed",
                4054,
                "--eval-every",
                3125,
                "--arms",
                arm,
                "--out",
                root,
                *common,
            ),
            generation_outputs(root, [arm]),
            root,
        )
    for rate in ("0.0001", "0.0003"):
        root = out / "lr_pilot" / rate
        yield task(
            f"lr-pilot-{rate}",
            script(
                "extended_g0.py",
                "--artifact-root",
                base,
                "--steps",
                9375,
                "--batch-size",
                32,
                "--eval-every",
                3125,
                "--eval-count",
                512,
                "--seed",
                4054,
                "--learning-rate",
                rate,
                "--temperature",
                1.0,
                "--out",
                root,
            ),
            [root / "metrics.json", root / "extended_g0_model.eqx"],
            root,
        )
    for seed in SEEDS:
        g0, clean, recursive = extended_paths(out, seed)
        init = 4054 if seed == 6001 else seed
        repair = out / "q2_study/repairs" / str(seed)
        yield task(
            f"repair-{seed}",
            script(
                "extended_g10_repair.py",
                "--include-all-replication-seeds",
                "--base-root",
                base,
                "--g0-root",
                g0,
                "--g10-root",
                recursive,
                "--repair",
                "fix_invalid_q2",
                "--seed",
                seed,
                "--init-seed",
                init,
                "--steps",
                9375,
                "--learning-rate",
                0.0003,
                "--temperature",
                1.0,
                "--out",
                repair,
            ),
            [
                repair / "fix_invalid_q2/g10/metrics.json",
                repair / "fix_invalid_q2/g10/model.eqx",
            ],
            repair,
        )
        for arm, model in (
            ("clean_only", clean / "clean_only/g10"),
            ("recursive_replacement", recursive / "recursive_replacement/g10"),
            ("fix_invalid_q2", repair / "fix_invalid_q2/g10"),
        ):
            folder = "repaired_patches" if arm == "fix_invalid_q2" else "raw_patches"
            target = out / "q2_study" / folder / f"{seed}-{arm}.json"
            yield task(
                f"patch-{seed}-{arm}",
                script(
                    "q2_context_activation_patch_pilot.py",
                    "--base-root",
                    base,
                    "--arm",
                    arm,
                    "--model-root",
                    model,
                    "--n",
                    128,
                    "--out",
                    target,
                ),
                [target],
            )
    for arm in ARMS:
        shared = [
            "--base-root",
            base,
            "--g0-root",
            primary_g0,
            "--generation-root",
            primary,
            "--arm",
            arm,
            "--n",
            128,
        ]
        for label, filename in (
            ("distribution", "analyze_contextual_symbol_softmax.py"),
            ("joint-heads", "extended_joint_head_ablations.py"),
            ("head-selection", "compare_extended_circuits_g10.py"),
            ("counterfactual", "abc_counterfactual_head_audit.py"),
        ):
            target = (
                out
                / "probes"
                / label
                / (arm if label == "distribution" else f"{arm}.json")
            )
            expected = target / "summary.json" if label == "distribution" else target
            yield task(
                f"probe-{label}-{arm}",
                script(
                    filename,
                    *(
                        [
                            "--g10-root" if value == "--generation-root" else value
                            for value in shared
                        ]
                        if label == "counterfactual"
                        else shared
                    ),
                    "--out",
                    target,
                ),
                [expected],
            )
    for mode in ("symbol_only", "number_only"):
        target = out / "factorial" / mode
        yield task(
            f"factorial-{mode}",
            script(
                "factorial_invalid_symbol_repair.py",
                "--base-root",
                base,
                "--g0-root",
                primary_g0,
                "--g10-root",
                primary,
                "--mode",
                mode,
                "--out",
                target,
            ),
            [target / "metrics.json", target / "model.eqx"],
            target,
        )
    yield task(
        "aggregate-factorial",
        script(
            "aggregate_factorial_repair.py",
            "--raw",
            primary / "recursive_replacement/g10",
            "--symbol-only",
            out / "factorial/symbol_only",
            "--number-only",
            out / "factorial/number_only",
            "--joint",
            out / "q2_study/repairs/6001/fix_invalid_q2/g10",
            "--out",
            out / "factorial/summary.json",
        ),
        [out / "factorial/summary.json"],
    )
    target = out / "cumulative"
    yield task(
        "cumulative-pilot",
        script(
            "cumulative_corpus_extended.py",
            "--base-root",
            base,
            "--g0-root",
            primary_g0,
            "--steps",
            3125,
            "--pool-block",
            2048,
            "--seed",
            7712,
            "--out",
            target,
        ),
        generation_outputs(target, ["clean_history", "cumulative_recursive"]),
        target,
    )
    yield task(
        "aggregate-patching",
        script(
            "aggregate_q2_patching.py",
            "--input",
            out / "q2_study/raw_patches",
            "--out",
            out / "q2_study/patch_summary.json",
        ),
        [out / "q2_study/patch_summary.json"],
    )
    yield task(
        "aggregate-mediation",
        script(
            "aggregate_q2_repair_mediation.py",
            "--raw-root",
            out / "q2_study/raw_patches",
            "--repaired-root",
            out / "q2_study/repaired_patches",
            "--out",
            out / "q2_study/mediation",
        ),
        [out / "q2_study/mediation/summary.json"],
    )
    yield task(
        "summarize-fresh-training",
        script(
            "aggregate_paper_runs.py", "--root", out, "--out", out / "paper_results"
        ),
        [out / "paper_results/summary.json"],
    )
    yield task(
        "figure-fresh-results",
        script(
            "plot_paper_figure1.py",
            "--base",
            out / "paper_results/base_300k_seed_outcomes.csv",
            "--extended",
            out / "paper_results/five_seed_generation_summary.csv",
            "--output",
            out / "figure1.png",
        ),
        [out / "figure1.png"],
    )


GROUPS = {
    "train": "Main experiment: five Base and five Extended seed pairs, G0–G10",
    "results": "Main result: Table 1 and Figure 1",
    "controls": "Training controls and learning-rate pilot",
    "distribution": "Generated-symbol error and probability-distribution evidence",
    "repair": "Joint correction of invalid generated targets",
    "factorial-repair": "Table 2: symbol-only, number-only and joint target repair",
    "head-ablation": "Individual attention-head selection and ablation",
    "joint-ablation": "Joint attention-head ablation",
    "counterfactual": "Response to edited support symbols and labels",
    "patching": "Five-seed activation patching and repair mediation",
}


def task_group(name):
    if name.startswith(("base-", "extended-g0-", "extended-chain-")):
        return "train"
    if name.startswith(("summarize-fresh-", "figure-fresh-")):
        return "results"
    if name.startswith(("extended-control-", "lr-pilot-", "cumulative-")):
        return "controls"
    if name.startswith("repair-"):
        return "repair"
    if name.startswith(("patch-", "aggregate-patching", "aggregate-mediation")):
        return "patching"
    if name.startswith(("factorial-", "aggregate-factorial")):
        return "factorial-repair"
    for prefix, group in (
        ("probe-distribution-", "distribution"),
        ("probe-head-selection-", "head-ablation"),
        ("probe-joint-heads-", "joint-ablation"),
        ("probe-counterfactual-", "counterfactual"),
    ):
        if name.startswith(prefix):
            return group
    raise ValueError(f"Unknown task: {name}")


def task_label(task):
    name = task["name"]
    if "label" in task:
        return task["label"]
    condition = (
        "correct targets" if name.endswith("clean_only") else "recursive targets"
    )
    if name.startswith("base-g0-"):
        return f"Base seed {int(name.split('-')[2]) + 1} of 5: train G0"
    if name.startswith("base-chain-"):
        return (
            f"Base seed {int(name.split('-')[2]) + 1} of 5: train {condition}, G1–G10"
        )
    if name.startswith(("extended-g0-", "extended-chain-")):
        seed = int(name.split("-")[2])
        number = SEEDS.index(seed) + 1
        suffix = (
            "train G0"
            if name.startswith("extended-g0-")
            else f"train {condition}, G1–G10"
        )
        return f"Extended seed {number} of 5: {suffix}"
    if name.startswith("extended-control-"):
        return (
            "Train fixed-G0 control, G1–G10"
            if name.endswith("frozen_g0")
            else "Train 50/50 real/generated-target control, G1–G10"
        )
    if name.startswith("lr-pilot-"):
        return f"Train learning-rate pilot at {name.rsplit('-', 1)[1]}"
    if name.startswith("repair-"):
        number = SEEDS.index(int(name.split("-")[1])) + 1
        return f"Repair invalid G9 targets and retrain G10, seed {number} of 5"
    if name.startswith("patch-"):
        _, seed, arm = name.split("-", 2)
        model = {
            "clean_only": "clean",
            "recursive_replacement": "recursive",
            "fix_invalid_q2": "repaired",
        }[arm]
        return f"Patch {model} G10 model, seed {SEEDS.index(int(seed)) + 1} of 5"
    if name.startswith("probe-"):
        return f"Measure {GROUPS[task_group(name)].lower()} ({condition})"
    labels = {
        "factorial-symbol_only": "Retrain G10 after correcting invalid symbols only",
        "factorial-number_only": "Retrain G10 after correcting associated numbers only",
        "aggregate-factorial": "Format the four target-repair conditions for Table 2",
        "cumulative-pilot": "Train the accumulating-corpus control, G1–G10",
        "aggregate-patching": "Summarize activation patching across five seed pairs",
        "aggregate-mediation": "Summarize repair effects and paired uncertainty",
        "summarize-fresh-training": "Calculate Table 1 from the five newly trained seed pairs",
        "figure-fresh-results": "Plot Figure 1 from the newly trained results",
    }
    return labels.get(name, name)


def ordered_tasks(out, command="all"):
    pending = list(tasks(out))
    selected = GROUPS if command in ("all", "status") else {command: GROUPS[command]}
    for group in selected:
        for task in pending:
            if task_group(task["name"]) == group:
                task["group"] = group
                yield task
        if group != "train":
            component = "main-results" if group == "results" else group
            folder = out / "evidence" / component
            yield {
                "name": f"evidence-{component}",
                "label": f"Write the readable evidence report: {GROUPS[group]}",
                "group": group,
                "command": [
                    sys.executable,
                    str(ROOT / "scripts/format_evidence.py"),
                    component,
                    "--root",
                    str(out),
                ],
                "outputs": [
                    folder / "README.md",
                    folder / "results.csv",
                    folder / "report.json",
                ],
                "directory": None,
            }


def show_status(plan, verbose=False, progress=None):
    finished = 0
    for group, description in GROUPS.items():
        selected = [task for task in plan if task["group"] == group]
        if not selected:
            continue
        ready = {
            task["name"]: task_complete(task, refresh_summaries=False)
            for task in selected
        }
        done = sum(ready.values())
        finished += done
        if group == "train":
            print("Main experiment (G0–G10):")
            for title, prefix, seeds in (
                ("Base", "base", range(5)),
                ("Extended", "extended", SEEDS),
            ):
                pairs = 0
                for seed in seeds:
                    seed_tasks = [
                        task
                        for task in selected
                        if task["name"].startswith(
                            (f"{prefix}-g0-{seed}", f"{prefix}-chain-{seed}-")
                        )
                    ]
                    pairs += bool(seed_tasks) and all(
                        ready[task["name"]] for task in seed_tasks
                    )
                print(f"  {title}: {pairs} of 5 seed pairs complete")
        else:
            state = (
                "Complete"
                if done == len(selected)
                else "Partial results" if done else "Pending"
            )
            print(f"{description}: {state}")
        if verbose:
            for task in selected:
                print(
                    f"  {'Complete' if ready[task['name']] else 'Pending'}: {task_label(task)}"
                )
    active = [
        task
        for task in plan
        if (progress or {}).get(task["name"], {}).get("status") == "running"
        and not task_complete(task, refresh_summaries=False)
    ]
    if active:
        print("\nRunning steps recorded:")
        for task in active:
            print(f"  {task_label(task)}")
    if finished != len(plan):
        print("\nPending work is still being run or is waiting for its prerequisites.")
        print("Workflow: train → results → individual analyses and evidence reports.")
    else:
        print("\nAll selected experiments and evidence reports are complete.")
    return int(finished != len(plan))


def check_prerequisites(command, out):
    if command in ("all", "train", "status"):
        return True
    available = list(tasks(out))
    primary = {
        "base-g0-0",
        "extended-g0-6001",
        "extended-chain-6001-clean_only",
        "extended-chain-6001-recursive_replacement",
    }
    if command in ("results", "repair", "patching"):
        required = [task for task in available if task_group(task["name"]) == "train"]
    else:
        names = primary if command != "controls" else {"base-g0-0", "extended-g0-6001"}
        required = [task for task in available if task["name"] in names]
    if command == "factorial-repair":
        required += [task for task in available if task["name"] == "repair-6001"]
    if command == "patching":
        required += [task for task in available if task_group(task["name"]) == "repair"]
    missing = [
        task for task in required if not task_complete(task, refresh_summaries=False)
    ]
    if not missing:
        return True
    print(f"{GROUPS[command]} needs completed model training before it can run.")
    print(
        f"{len(required) - len(missing)} of {len(required)} prerequisite steps are ready."
    )
    if any(task_group(task["name"]) == "train" for task in missing):
        print("Complete 'python scripts/run_paper.py train' first.")
    if any(task_group(task["name"]) == "repair" for task in missing):
        print("Then complete 'python scripts/run_paper.py repair'.")
    print("Use 'python scripts/run_paper.py status' to check progress.")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Train G0–G10, format the main results, or investigate one paper result"
    )
    parser.add_argument(
        "command", nargs="?", choices=(*GROUPS, "all", "status"), default="all"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show individual steps and command details",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Validate saved outputs without running or changing anything",
    )
    parser.add_argument("--only", help="Run tasks whose names contain this text")
    parser.add_argument("--restart-incomplete", action="store_true")
    args = parser.parse_args()
    args.status = args.status or args.command == "status"
    if args.status and args.dry_run:
        parser.error("Use either --status or --dry-run")
    args.out = args.out.resolve()
    plan = [
        task
        for task in ordered_tasks(args.out, args.command)
        if not args.only or args.only in task["name"]
    ]
    if not plan:
        parser.error("No matching tasks")
    progress = {}
    status_path = args.out / "run_status.json"
    if status_path.is_file():
        progress = read_json(status_path)
    if args.status:
        return show_status(plan, verbose=args.verbose, progress=progress)
    if (
        not args.dry_run
        and not args.only
        and not check_prerequisites(args.command, args.out)
    ):
        return 1

    def save_status():
        latest = read_json(status_path) if status_path.is_file() else {}
        for name, record in progress.items():
            if record.get("updated", 0) >= latest.get(name, {}).get("updated", 0):
                latest[name] = record
        temporary = status_path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(latest, indent=2) + "\n")
        temporary.replace(status_path)

    for step, task in enumerate(plan, start=1):
        name, command = task["name"], task["command"]
        finished = task_complete(task)
        print(
            f"[{step}/{len(plan)}] {'Already complete' if finished else 'Run'}: {task_label(task)}",
            flush=True,
        )
        if args.verbose or args.dry_run:
            print(f"  {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        args.out.mkdir(parents=True, exist_ok=True)
        record = {
            "command": command,
            "expected_outputs": list(map(str, task["outputs"])),
            "status": "complete" if finished else "running",
            "updated": time.time(),
        }
        progress[name] = record
        save_status()
        if finished:
            continue
        directory = task["directory"]
        if directory and directory.exists() and any(directory.iterdir()):
            if not args.restart_incomplete:
                record["status"] = "blocked-incomplete-output"
                save_status()
                raise RuntimeError(
                    f"A partial run exists for {task_label(task)} at {directory}. Use --restart-incomplete to archive and restart it."
                )
            archive = ROOT / "unneeded/interrupted_runs" / str(time.time_ns())
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(directory), str(archive))
        log = args.out / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(
            {
                "WANDB_MODE": "disabled",
                "JAX_PLATFORMS": "cpu",
                "MPLCONFIGDIR": "/tmp/nlp_matplotlib",
            }
        )
        try:
            with log.open("a") as handle:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode:
                raise RuntimeError(
                    f"{task_label(task)} failed with exit {result.returncode}; see {log}"
                )
            if not complete(task["outputs"]) or (
                not name.startswith(
                    ("aggregate-", "summarize-", "figure-", "evidence-")
                )
                and not task_complete(task)
            ):
                raise RuntimeError(
                    f"{task_label(task)} exited without all expected outputs; see {log}"
                )
        except BaseException as error:
            record.update(
                {
                    "status": (
                        "interrupted"
                        if isinstance(error, KeyboardInterrupt)
                        else "failed"
                    ),
                    "error": str(error),
                    "updated": time.time(),
                    "log": str(log),
                }
            )
            save_status()
            raise
        record.update({"status": "complete", "updated": time.time(), "log": str(log)})
        save_status()
    print(
        (
            "Plan only; no experiments were run."
            if args.dry_run
            else "All selected steps completed."
        ),
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
