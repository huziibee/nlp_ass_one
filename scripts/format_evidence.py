import argparse
import csv
import json
import os
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("clean_only", "recursive_replacement")
SEEDS = (6001, 5054, 6054, 7054, 8054)
LABELS = {
    "natural": "Unmodified",
    "clean_only": "Real targets",
    "recursive_replacement": "Recursive",
    "fix_invalid_q2": "Joint target repair",
    "symbol_only": "Symbol-only repair",
    "number_only": "Number-only repair",
    "raw": "No repair",
    "joint": "Joint repair",
    "val_relabel": "Held-out label assignments",
    "val_exemplar": "Held-out handwriting",
    "val_classes": "Held-out character classes",
    "train": "Training prompts used for head selection",
    "frozen_g0": "Fixed G0 generator",
    "real_anchor_half": "Half real targets",
    "clean_history": "Real-target history",
    "cumulative_recursive": "Accumulating recursive corpus",
}
COMPONENTS = {
    "main-results": (
        "Table 1 and Figure 1",
        "Run python scripts/run_paper.py train, then python scripts/run_paper.py results.",
    ),
    "distribution": (
        "Where next-character probability goes",
        "Run python scripts/run_paper.py distribution after the primary G0–G10 experiment.",
    ),
    "head-ablation": (
        "Single-head ablation evidence",
        "Run python scripts/run_paper.py head-ablation after the primary G0–G10 experiment.",
    ),
    "joint-ablation": (
        "Joint-head ablation evidence",
        "Run python scripts/run_paper.py joint-ablation after the primary G0–G10 experiment.",
    ),
    "counterfactual": (
        "Does retrieval follow the prompt?",
        "Run python scripts/run_paper.py counterfactual after the primary G0–G10 experiment.",
    ),
    "factorial-repair": (
        "Table 2: which target needs repair?",
        "Run python scripts/run_paper.py factorial-repair after the primary recursive G9 and G10 checkpoints and joint repair exist.",
    ),
    "repair": (
        "Repairing generated targets",
        "Run python scripts/run_paper.py repair after the clean and recursive G0–G10 experiments.",
    ),
    "patching": (
        "Attention activation-patching evidence",
        "Run python scripts/run_paper.py patching after the clean, recursive and repaired G10 checkpoints exist.",
    ),
    "controls": (
        "Training controls and learning-rate pilot",
        "Run python scripts/run_paper.py controls after the primary Extended G0 checkpoint exists.",
    ),
}


def label(value):
    return LABELS.get(str(value), str(value).replace("_", " "))


def read(path):
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    if any(
        data.get(key) is True
        for key in (
            "final_test_evaluated",
            "final_test_accessed",
            "final_test_access",
            "test_evaluated",
            "original_final_test_access",
        )
    ):
        raise ValueError(
            f"Final-test results cannot enter the validation report: {path}"
        )
    if path.name == "metrics.json" and data.get("history"):
        expected = data.get(
            "training_exposures",
            data.get("train_exposures", data.get("optimizer_exposures")),
        )
        if expected is None and "steps" in data and "batch_size" in data:
            expected = data["steps"] * data["batch_size"]
        final = data["history"][-1]
        observed = final.get(
            "sequence_exposures",
            final.get("seen_training_sequences", final.get("train_exposures")),
        )
        if expected is not None and observed is not None and observed != expected:
            raise FileNotFoundError(f"Training has not finished for {path}")
    return data


def add(rows, path, metric, value, unit="%", scale=100, **details):
    if value is None:
        return
    if not math.isfinite(float(value)):
        raise ValueError(f"Non-finite evidence value for {metric}: {path}")
    rows.append(
        {
            "condition": label(details.get("condition", "")),
            "seed": details.get("seed", ""),
            "generation": details.get("generation", ""),
            "validation": label(details.get("role", "")),
            "intervention": label(details.get("intervention", "")),
            "metric": metric,
            "value": float(value) * scale,
            "unit": unit,
            "change_pp": details.get("change_pp", ""),
            "interval": details.get("interval", ""),
            "source": str(path),
        }
    )


def scores(rows, path, data, **details):
    metrics = (
        ("first_label_accuracy", "First numerical-label accuracy"),
        (
            "second_label_teacher_forced_accuracy",
            "Teacher-forced second-label accuracy",
        ),
        (
            "teacher_forced_second_label_accuracy",
            "Teacher-forced second-label accuracy",
        ),
        ("next_symbol_valid_probability_mass", "Context-valid symbol probability mass"),
        ("free_next_symbol_context_validity", "Sampled symbol validity"),
        (
            "free_second_label_context_consistency_all",
            "Valid symbol and correct second number",
        ),
    )
    natural = details.pop("natural", {})
    for key, name in metrics:
        if key in data:
            delta = 100 * (data[key] - natural[key]) if key in natural else ""
            add(rows, path, name, data[key], change_pp=delta, **details)


def distribution(root):
    rows, missing = [], []
    for arm in ARMS:
        path = root / "probes/distribution" / arm / "summary.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        data = read(path)
        for generation in data.get("generations", (0, 10)):
            roles = data.get("results", {}).get(str(generation), {})
            for role in data.get("roles", ("val_relabel", "val_exemplar")):
                if role not in roles:
                    missing.append(f"{path}: generation {generation}, {role}")
        for generation, roles in data["results"].items():
            for role, result in roles.items():
                details = dict(condition=arm, generation=generation, role=role)
                for key, name in (
                    (
                        "probability_mass_valid_supports",
                        "Context-valid symbol probability mass",
                    ),
                    (
                        "probability_mass_off_context",
                        "Off-context symbol probability mass",
                    ),
                    (
                        "fraction_invalid_symbol_is_top1",
                        "Greedy choice outside context",
                    ),
                    (
                        "q2_teacher_forced_first_label_valid_mass",
                        "Valid symbol mass with correct first label",
                    ),
                ):
                    if key in result:
                        add(rows, path, name, result[key], **details)
                for key, name in (
                    (
                        "mean_per_episode_symbol_entropy_nats",
                        "Mean per-prompt symbol entropy",
                    ),
                    ("global_marginal_symbol_entropy_nats", "Global symbol entropy"),
                ):
                    if key in result:
                        add(
                            rows,
                            path,
                            name,
                            result[key],
                            unit="nats",
                            scale=1,
                            **details,
                        )
    return (
        rows,
        missing,
        [
            "Probability mass and greedy validity are different measurements from freely sampled symbol validity. This diagnostic uses one model seed and shared validation prompts."
        ],
    )


def ablation(root, joint=False):
    rows, missing = [], []
    folder = "joint-heads" if joint else "head-selection"
    for arm in ARMS:
        path = root / "probes" / folder / f"{arm}.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        data = read(path)
        if not joint and data.get("g0_training_attention"):
            attention = data["g0_training_attention"]
            choices = (
                (
                    "l1_prev_token_mean_by_head",
                    data["selected_l1_previous_head"],
                    1,
                    "Previous-token attention",
                ),
                (
                    "l2_query_first_number_matching_margin_by_head",
                    data["selected_l2_first_head"],
                    2,
                    "First-label matching margin",
                ),
                (
                    "l2_query_second_number_matching_margin_by_head",
                    data["selected_l2_second_head"],
                    2,
                    "Second-label matching margin",
                ),
            )
            for metric, head, layer, title in choices:
                add(
                    rows,
                    path,
                    title,
                    attention[metric][head],
                    unit="score",
                    scale=1,
                    condition=arm,
                    generation=0,
                    role="train",
                    intervention=f"Layer {layer}, head {head}",
                )
        for generation in data.get("generations", (0, 10)):
            for role in ("val_relabel", "val_exemplar", "val_classes"):
                observed = (
                    data.get("validation", {}).get(role, {}).get(str(generation))
                    if joint
                    else data.get("results", {}).get(str(generation), {}).get(role)
                )
                if observed is None:
                    missing.append(f"{path}: generation {generation}, {role}")
        if joint:
            entries = (
                (generation, role, results)
                for role, generations in data["validation"].items()
                for generation, results in generations.items()
            )
        else:
            entries = (
                (generation, role, results)
                for generation, roles in data["results"].items()
                for role, results in roles.items()
            )
        for generation, role, results in entries:
            natural = results["natural"]
            if not joint and results.get("attention"):
                attention = results["attention"]
                for key, head, title in (
                    (
                        "l2_query_first_number_matching_margin_by_head",
                        data["selected_l2_first_head"],
                        "First-label matching margin",
                    ),
                    (
                        "l2_query_second_number_matching_margin_by_head",
                        data["selected_l2_second_head"],
                        "Second-label matching margin",
                    ),
                ):
                    add(
                        rows,
                        path,
                        title,
                        attention[key][head],
                        unit="score",
                        scale=1,
                        condition=arm,
                        generation=generation,
                        role=role,
                        intervention=f"Layer 2, head {head}",
                    )
            for intervention, result in results.items():
                if intervention == "attention" or not isinstance(result, dict):
                    continue
                intervention_label = intervention
                if not joint and intervention == "ablated_first_candidate":
                    intervention_label = f"Remove layer 2 head {data['selected_l2_first_head']} (first-label candidate)"
                elif not joint and intervention == "ablated_second_candidate":
                    intervention_label = f"Remove layer 2 head {data['selected_l2_second_head']} (second-label candidate)"
                elif joint:
                    first, second, third = data["top_heads"][:3]
                    removed = {
                        "head1": [first],
                        "runner_up": [second],
                        "pair_top2": [first, second],
                        "pair_top3": [first, second, third],
                        "head1_plus_third": [first, third],
                    }.get(intervention, [])
                    intervention_label = (
                        "Unmodified"
                        if not removed
                        else "Remove layer 2 heads " + ", ".join(map(str, removed))
                    )
                scores(
                    rows,
                    path,
                    result,
                    condition=arm,
                    generation=generation,
                    role=role,
                    intervention=intervention_label,
                    natural=natural,
                )
    return (
        rows,
        missing,
        [
            "Change (pp) is intervention minus the natural model on the same prompts. A negative value means the intervention reduced the metric. Candidate heads are selected on G0 training prompts, not validation outcomes. Head indices are zero-based. Ablations are one-seed component tests; they do not prove a unique circuit."
        ],
    )


def counterfactual(root):
    rows, missing = [], []
    for arm in ARMS:
        path = root / "probes/counterfactual" / f"{arm}.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        data = read(path)
        for role in ("val_relabel", "val_classes"):
            for generation in ("g0", "g10"):
                if generation not in data.get("roles", {}).get(role, {}):
                    missing.append(f"{path}: {generation}, {role}")
        for role, generations in data["roles"].items():
            for generation, result in generations.items():
                for test, metrics in result["counterfactual_tests"].items():
                    for key, name in (
                        (
                            "accuracy_against_new_oracle",
                            "Accuracy against edited prompt",
                        ),
                        ("fraction_prediction_changed", "Predictions changed"),
                        (
                            "on_orig_correct_fraction_follows_changed_oracle",
                            "Edited-prompt accuracy among originally correct predictions",
                        ),
                    ):
                        add(
                            rows,
                            path,
                            name,
                            metrics.get(key),
                            condition=arm,
                            generation=generation,
                            role=role,
                            intervention=test,
                        )
    return (
        rows,
        missing,
        [
            "Label swaps and support-order edits use the same validation prompts. Replacing a query with the other support exemplar is an easier task than a new handwriting query. These are one-seed retrieval probes."
        ],
    )


def factorial(root):
    path = root / "factorial/summary.json"
    if not path.is_file():
        return [], [str(path)], []
    data = read(path)
    rows = []
    for result in data["results"]:
        for key, name in (
            ("context_validity", "Sampled symbol validity"),
            (
                "valid_symbol_plus_correct_number",
                "Valid symbol and correct second number",
            ),
        ):
            add(
                rows,
                path,
                name,
                result[key],
                condition=result["condition"],
                seed=6001,
                generation=10,
                role="val_relabel",
            )
    return (
        rows,
        [],
        [
            "This is the one-seed factorial repair comparison. Symbol-only and associated-number-only repairs are distinct from jointly repairing the invalid symbol and its label. The joint-validity metric checks q2 and y2 and does not score y1."
        ],
    )


def paired_repair_results(rows, path):
    if not path.is_file():
        return
    data = read(path)
    if data.get("n_independent_paired_training_seed_experiments") != 5:
        raise ValueError(f"Expected five matched repair experiments: {path}")
    selected = {
        "valid_mass": ("Context-valid symbol probability mass", 100),
        "context_preference_logit_units": ("Context-switch logit response", 1),
        "layer2_patch_fraction_of_context_preference": (
            "Fraction transferred by layer 2",
            100,
        ),
        "layer2_patch_shift_logit_units": ("Layer 2 raw patch margin shift", 1),
    }
    for key, (title, scale) in selected.items():
        result = data["metrics"][key]
        for condition, field in (
            ("Recursive mean", "raw_mean"),
            ("Repaired mean", "repaired_mean"),
            ("Paired repair effect", "mean"),
        ):
            unit = "logits" if scale == 1 else "pp" if field == "mean" else "%"
            interval = ""
            if field == "mean":
                lower, upper = result["paired_t95_interval"]
                interval = f"[{lower * scale:.2f}, {upper * scale:.2f}]"
            add(
                rows,
                path,
                f"Five-seed matched comparison: {title}",
                result[field],
                unit=unit,
                scale=scale,
                condition=condition,
                generation=10,
                role="val_relabel",
                interval=interval,
            )


def repair(root):
    rows, missing = [], []
    for seed in SEEDS:
        path = root / "q2_study/repairs" / str(seed) / "fix_invalid_q2/g10/metrics.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        data = read(path)
        budget = data.get("train_exposures", data.get("training_exposures"))
        if budget != 300000:
            raise ValueError(f"Expected completed 300,000-episode repair: {path}")
        metrics = data["history"][-1]["validation"]["val_relabel"]
        baseline_path = (
            root / "extended_primary/recursive_replacement/g10/metrics.json"
            if seed == 6001
            else root
            / f"extended_{seed}_recursive_replacement/recursive_replacement/g10/metrics.json"
        )
        baseline = {}
        if baseline_path.is_file():
            baseline = read(baseline_path)["history"][-1]["validation"]["val_relabel"]
            scores(
                rows,
                baseline_path,
                baseline,
                condition="recursive_replacement",
                seed=seed,
                generation=10,
                role="val_relabel",
            )
        else:
            missing.append(str(baseline_path))
        scores(
            rows,
            path,
            metrics,
            condition="fix_invalid_q2",
            seed=seed,
            generation=10,
            role="val_relabel",
            natural=baseline,
        )
        counts = data.get("total_synthetic_target_quality", {})
        for key, name in (
            ("source_off_context_symbols", "Source target: off-context character"),
            (
                "source_context_valid_wrong_second_number",
                "Source target: valid character but wrong second number",
            ),
            ("source_wrong_first_numbers", "Source target: wrong first number"),
        ):
            if key in counts:
                add(
                    rows,
                    path,
                    name,
                    counts[key] / budget,
                    condition="recursive_replacement",
                    seed=seed,
                    generation=9,
                )
    paired_repair_results(rows, root / "q2_study/mediation/summary.json")
    return (
        rows,
        missing,
        [
            "Only completed, observed seed runs appear below. Source-error percentages use the generated 300,000-answer training stream and their categories may overlap. Joint repair changes both invalid symbols and their associated numbers, so it does not isolate a symbol-only mediated effect. A partial seed set is descriptive evidence, not the paper’s five-seed estimate."
        ],
    )


def patching(root):
    rows, missing = [], []
    for seed in SEEDS:
        for arm in (*ARMS, "fix_invalid_q2"):
            directory = "repaired_patches" if arm == "fix_invalid_q2" else "raw_patches"
            path = root / "q2_study" / directory / f"{seed}-{arm}.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            data = read(path)
            if data["n"] != 128:
                raise ValueError(f"Expected 128 fixed validation prompts: {path}")
            for layer, result in data["layers"].items():
                if (
                    result["identity_patch_max_error"] >= 3e-5
                    or result["future_token_patch_max_error"] >= 3e-5
                ):
                    raise ValueError(f"Patch sanity check failed: {path}")
                details = dict(
                    condition=arm,
                    seed=seed,
                    generation=10,
                    role="val_relabel",
                    intervention=f"Layer {layer}",
                )
                preference = (
                    result["recipient_margin_mean"] - result["donor_margin_mean"]
                )
                shift = result["whole_layer_patch_margin_shift"]
                add(
                    rows,
                    path,
                    "Context-switch logit response",
                    preference,
                    unit="logits",
                    scale=1,
                    **details,
                )
                add(
                    rows,
                    path,
                    "Transferred logit response",
                    -shift,
                    unit="logits",
                    scale=1,
                    **details,
                )
                if preference != 0:
                    add(
                        rows,
                        path,
                        "Fraction of context response transferred",
                        -shift / preference,
                        **details,
                    )
                add(
                    rows,
                    path,
                    "Recipient context-valid symbol mass",
                    result["recipient_valid_mass"],
                    **details,
                )
                for head, values in result.get("head_level", {}).items():
                    head_details = {
                        **details,
                        "intervention": f"Layer {layer}, head {head}",
                    }
                    add(
                        rows,
                        path,
                        "Patched head logit-margin shift",
                        values["q2_old_B_vs_new_C_margin_shift"],
                        unit="logits",
                        scale=1,
                        **head_details,
                    )
    summary = root / "q2_study/mediation/summary.json"
    if summary.is_file():
        paired_repair_results(rows, summary)
    else:
        missing.append(str(summary))
    return (
        rows,
        missing,
        [
            "Patches transfer activations between edited contexts within each trained model. Logit response and probability mass have different units. The sign of the transferred response is reversed from the raw recipient-margin shift so a positive value indicates transfer in the intended context direction. Missing seeds are listed explicitly; no five-seed average is inferred from a partial set."
        ],
    )


def controls(root):
    rows, missing = [], []
    for rate in ("0.0001", "0.0003"):
        path = root / "lr_pilot" / rate / "metrics.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        data = read(path)
        scores(
            rows,
            path,
            data["history"][-1]["validation"]["val_relabel"],
            condition=f"Learning rate {rate}",
            seed=4054,
            generation=0,
            role="val_relabel",
        )
    for arm in ("frozen_g0", "real_anchor_half"):
        path = root / "extended_controls" / arm / arm / "g10/metrics.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        data = read(path)
        scores(
            rows,
            path,
            data["history"][-1]["validation"]["val_relabel"],
            condition=arm,
            seed=6001,
            generation=10,
            role="val_relabel",
        )
    for arm in ("clean_history", "cumulative_recursive"):
        path = root / "cumulative" / arm / "g10/metrics.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        data = read(path)
        scores(
            rows,
            path,
            data["validation"]["val_relabel"],
            condition=arm,
            seed=7712,
            generation=10,
            role="val_relabel",
        )
    return (
        rows,
        missing,
        [
            "Fixed-generator and mixed-target controls use the primary 300,000-episode protocol. The accumulating-corpus pilot uses 100,000 episodes per generation and is not directly comparable. The learning-rate comparison uses one seed."
        ],
    )


def main_results(root):
    path = root / "paper_results/table1.csv"
    if not path.is_file():
        return [], [str(path)], []
    with path.open(newline="") as handle:
        data = list(csv.DictReader(handle))
    rows = []
    for result in data:
        for key, metric in (
            ("clean_mean", "Real-target mean"),
            ("recursive_mean", "Recursive mean"),
            ("gap_pp", "Clean minus recursive"),
            ("lower_95_pp", "95% paired interval lower bound"),
            ("upper_95_pp", "95% paired interval upper bound"),
        ):
            if key in result:
                add(
                    rows,
                    path,
                    metric,
                    result[key],
                    condition=result.get("outcome", ""),
                    unit="pp" if key.endswith("_pp") else "%",
                    scale=1 if key.endswith("_pp") else 100,
                )
    details = root / "paper_results/extended_seed_outcomes.csv"
    missing = []
    if details.is_file():
        with details.open(newline="") as handle:
            seed_rows = list(csv.DictReader(handle))
        for seed in seed_rows:
            for key, metric, role in (
                (
                    "G10_first_label_heldout_labels",
                    "First-label accuracy by seed",
                    "val_relabel",
                ),
                (
                    "G10_first_label_heldout_classes",
                    "First-label accuracy by seed",
                    "val_classes",
                ),
                (
                    "G10_sampled_q2_validity_heldout_exemplars",
                    "Sampled symbol validity by seed",
                    "val_exemplar",
                ),
                (
                    "G10_valid_q2_correct_y2",
                    "Valid symbol and correct second number by seed",
                    "val_relabel",
                ),
            ):
                if key not in seed:
                    missing.append(f"Regenerate {details} with the results command")
                    break
                add(
                    rows,
                    details,
                    metric,
                    seed[key],
                    condition=seed["arm"],
                    seed=seed["seed"],
                    generation=10,
                    role=role,
                )
    else:
        missing.append(str(details))
    return (
        rows,
        missing,
        [
            "Table 1 uses five matched seed pairs. Intervals are descriptive paired 95% t intervals. Figure 1 is at ../../figure1.png relative to the evidence folder; the table and figure must use the same freshly aggregated run set."
        ],
    )


BUILDERS = {
    "main-results": main_results,
    "distribution": distribution,
    "head-ablation": lambda root: ablation(root),
    "joint-ablation": lambda root: ablation(root, joint=True),
    "counterfactual": counterfactual,
    "factorial-repair": factorial,
    "repair": repair,
    "patching": patching,
    "controls": controls,
}


def display(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value).replace("|", "/")


def write_report(component, root, destination):
    rows, missing, notes = BUILDERS[component](root)
    if not rows:
        raise FileNotFoundError(
            f'No completed evidence for {component}. {COMPONENTS[component][1]} Missing: {", ".join(missing)}'
        )
    folder = destination / component
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(
        json.dumps(
            {
                "component": component,
                "complete": not missing,
                "missing_inputs": missing,
                "source_files": sorted({row["source"] for row in rows}),
            },
            indent=2,
        )
        + "\n"
    )
    with (folder / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    title, prerequisite = COMPONENTS[component]
    lines = [
        f"# {title}",
        "",
        f"Prerequisite: {prerequisite}",
        "",
        f'Status: {"partial; required inputs are still missing" if missing else "all expected inputs are present"}.',
        "",
    ]
    if notes:
        lines.extend(notes + [""])
    if component == "main-results":
        lines += ["![Figure 1](../../figure1.png)", ""]
    if component == "main-results":
        with (root / "paper_results/table1.csv").open(newline="") as handle:
            table = list(csv.DictReader(handle))
        lines += [
            "## Table 1",
            "",
            "| Outcome | Real targets (%) | Recursive (%) | Gap (pp) | 95% paired interval (pp) |",
            "|:--|--:|--:|--:|:--|",
        ]
        for row in table:
            lines.append(
                f"| {label(row['outcome'])} | {100 * float(row['clean_mean']):.2f} | {100 * float(row['recursive_mean']):.2f} | {float(row['gap_pp']):.2f} | [{float(row['lower_95_pp']):.2f}, {float(row['upper_95_pp']):.2f}] |"
            )
        lines.append("")
        details = root / "paper_results/extended_seed_outcomes.csv"
        if details.is_file() and not missing:
            with details.open(newline="") as handle:
                seed_rows = list(csv.DictReader(handle))
            lines += [
                "## Extended G10: retrieval and handwriting checks",
                "",
                "These checks distinguish numerical retrieval from sampled character validity. Character identity accuracy is undefined on unseen classes.",
                "",
                "| Seed | Targets | First label: held-out labels (%) | First label: held-out classes (%) | Valid symbol: held-out handwriting (%) | Valid symbol + second number (%) |",
                "|:--|:--|--:|--:|--:|--:|",
            ]
            for row in seed_rows:
                values = [
                    f"{100 * float(row[key]):.2f}"
                    for key in (
                        "G10_first_label_heldout_labels",
                        "G10_first_label_heldout_classes",
                        "G10_sampled_q2_validity_heldout_exemplars",
                        "G10_valid_q2_correct_y2",
                    )
                ]
                lines.append(
                    "| " + " | ".join([row["seed"], label(row["arm"]), *values]) + " |"
                )
            lines.append("")
    elif component == "factorial-repair":
        lines += [
            "## Table 2",
            "",
            "| G10 target correction | Valid symbol (%) | Valid symbol and correct second number (%) |",
            "|:--|--:|--:|",
        ]
        for condition in dict.fromkeys(row["condition"] for row in rows):
            values = {
                row["metric"]: row["value"]
                for row in rows
                if row["condition"] == condition
            }
            lines.append(
                f"| {condition} | {values['Sampled symbol validity']:.2f} | {values['Valid symbol and correct second number']:.2f} |"
            )
        lines.append("")
    else:
        for metric in dict.fromkeys(row["metric"] for row in rows):
            selected = [row for row in rows if row["metric"] == metric]
            fields = [
                ("condition", "Condition"),
                ("seed", "Seed"),
                ("generation", "Generation"),
                ("validation", "Validation"),
                ("intervention", "Intervention"),
            ]
            fields = [
                (key, title)
                for key, title in fields
                if any(row[key] != "" for row in selected)
            ]
            show_change = any(row["change_pp"] != "" for row in selected)
            show_interval = any(row["interval"] for row in selected)
            headers = (
                [title for _, title in fields]
                + ["Value"]
                + (["Change (pp)"] if show_change else [])
                + (["95% paired interval"] if show_interval else [])
            )
            lines += [
                f"## {metric}",
                "",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(":--" for _ in headers) + " |",
            ]
            for row in selected:
                values = [row[key] for key, _ in fields]
                measured = (
                    f'{row["value"]:.3f}'
                    if row["unit"] in ("score", "nats")
                    else display(row["value"])
                )
                values += [f'{measured} {row["unit"]}']
                if show_change:
                    values.append(display(row["change_pp"]))
                if show_interval:
                    values.append(row["interval"])
                lines.append(
                    "| " + " | ".join(display(value) for value in values) + " |"
                )
            lines.append("")
    if missing:
        lines += ["## Missing inputs", ""]
        for path in missing:
            item = Path(path)
            display_path = (
                item.relative_to(ROOT).as_posix() if item.is_relative_to(ROOT) else path
            )
            lines.append(f"- `{display_path}`")
        lines.append("")
    lines += ["## Source files", ""]
    for source in sorted({row["source"] for row in rows}):
        path = Path(source)
        display_path = (
            path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else source
        )
        link = Path(os.path.relpath(path, folder)).as_posix()
        lines.append(f"- [{display_path}]({link})")
    lines.append("")
    (folder / "README.md").write_text("\n".join(lines))
    print(f'Evidence report: {folder / "README.md"}')
    return folder / "README.md"


def main():
    parser = argparse.ArgumentParser(
        description="Format observed experiment results as readable evidence tables"
    )
    parser.add_argument("component", choices=(*COMPONENTS, "all"))
    parser.add_argument("--root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    destination = args.out or args.root / "evidence"
    selected = list(COMPONENTS) if args.component == "all" else [args.component]
    for component in selected:
        try:
            write_report(component, args.root, destination)
        except FileNotFoundError as error:
            if args.component != "all":
                parser.exit(1, str(error) + "\n")
    destination.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Evidence for the paper",
        "",
        "These reports use saved outputs from the current runs.",
        "",
    ]
    for component, (title, _) in COMPONENTS.items():
        folder = destination / component
        if not (folder / "README.md").is_file():
            lines.append(f"- {title}: awaiting experiment outputs.")
            continue
        metadata = (
            read(folder / "report.json") if (folder / "report.json").is_file() else {}
        )
        status = "Complete" if metadata.get("complete") else "Partial"
        lines.append(f"- [{title}]({component}/README.md): {status}.")
    (destination / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
