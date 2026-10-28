"""Aggregate and visualize *only* verified G1-G10 validation results.

Consumes 8 trajectory CSVs exported by scripts/compile_g10_validation.py.
Checks complete set of models/conditions and generations, then creates
reproducible master CSV/JSON and crisp SVG performance figures.

Original G0 is deliberately excluded unless independently evaluated on the
same episode seed: no misleading plot based on different validation samples.
"""
from __future__ import annotations
import argparse
import csv
import html
import json
import math
from pathlib import Path

ARMS=("clean_only","frozen_g0","real_anchor_half","recursive_replacement")
TASKS=("base","extended")
COLORS={"clean_only":"#1959a6","frozen_g0":"#8c579a",
        "real_anchor_half":"#17845c","recursive_replacement":"#c34b45"}
DESCRIPTIONS={
    "base_pair":("Base: held-out label-pair accuracy","base","val_relabel_accuracy"),
    "base_classes":("Base: held-out validation-class accuracy","base","val_classes_accuracy"),
    "extended_first":("Extended: first-number accuracy","extended",
                      "val_relabel_first_label_accuracy"),
    "extended_second":("Extended: teacher-forced second-number accuracy","extended",
                       "val_relabel_second_teacher_forced_accuracy"),
    "extended_valid":("Extended: freely generated symbol is in context","extended",
                      "val_relabel_generated_symbol_context_validity"),
    "extended_continuation":("Extended: valid symbol and correct second number","extended",
                             "val_relabel_generated_second_context_consistency"),
    "extended_valid_mass":("Extended: probability mass on two contextual symbols","extended",
                           "val_relabel_next_symbol_valid_probability_mass"),
}

def load_trajectories(folder:Path):
    paths=sorted(folder.rglob("*.csv"))
    rows={}
    for path in paths:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("task") not in TASKS or row.get("arm") not in ARMS:
                    raise ValueError("Unrecognized task/arm in "+str(path))
                task,arm,gen=row["task"],row["arm"],int(row["generation"])
                key=(task,arm,gen)
                if key in rows:
                    raise ValueError("Duplicate generation row "+str(key))
                if gen not in range(1,11):
                    raise ValueError("Unknown generation "+str(key))
                if row.get("validation_only")!="True":
                    raise ValueError("Attempt to mix held-out test into validation dashboard")
                for col,value in row.items():
                    if col.startswith("val_"):
                        z=float(value)
                        if not math.isfinite(z):
                            raise ValueError(f"Non-finite metric {col} for {key}")
                        if col.endswith("accuracy") or col.endswith("validity") or col.endswith("consistency") or col.endswith("mass"):
                            if not 0<=z<=1:
                                raise ValueError("Outside [0,1] "+col+" "+str(key))
                rows[key]=row
    expected={(task,arm,g) for task in TASKS for arm in ARMS
              for g in range(1,11)}
    if set(rows)!=expected:
        missing=sorted(expected-set(rows))
        excess=sorted(set(rows)-expected)
        raise ValueError(f"Incomplete G10 analysis: missing={missing}; extra={excess}")
    return rows

def save_svg(out:Path, name:str, title:str, task:str, metric:str, records):
    width,height=970,500
    left,top,right,bottom=80,65,35,110
    w=width-left-right
    h=height-top-bottom
    series={
        arm:[float(records[(task,arm,g)][metric]) for g in range(1,11)]
        for arm in ARMS
    }
    lo=min(min(a) for a in series.values())
    hi=max(max(a) for a in series.values())
    axis_min=max(0,math.floor((lo-0.025)*20)/20)
    axis_max=min(1,math.ceil((hi+0.025)*20)/20)
    if axis_max-axis_min<0.1:
        axis_min=max(0,axis_max-0.1)
    def xs(g): return left+(g-1)/9*w
    def ys(v): return top+(axis_max-v)/(axis_max-axis_min)*h
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#263345}.ttl{font-size:22px;font-weight:600}.axis{font-size:13px}.leg{font-size:13px;font-weight:600}.note{font-size:12px;fill:#657387}</style>',
         f'<text x="{left}" y="36" class="ttl">{html.escape(title)}</text>']
    for i in range(5):
        v=axis_min+(axis_max-axis_min)*i/4
        y=ys(v)
        svg+=[f'<line x1="{left}" x2="{width-right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e2e8f0"/>',
              f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" class="axis">{v*100:.0f}%</text>']
    for g in range(1,11):
        x=xs(g)
        svg.append(f'<text x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle" class="axis">G{g}</text>')
    svg.append(f'<line x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}" stroke="#8f9cb0"/>')
    svg.append(f'<text x="{(left+width-right)/2}" y="{height-61}" class="axis" text-anchor="middle">Model generation (validation only)</text>')
    for i,arm in enumerate(ARMS):
        points=" ".join(f"{xs(g):.2f},{ys(series[arm][g-1]):.2f}" for g in range(1,11))
        color=COLORS[arm]
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
        for g,value in enumerate(series[arm],1):
            svg.append(f'<circle cx="{xs(g):.2f}" cy="{ys(value):.2f}" r="3" fill="{color}"/>')
        lx=left+(i%2)*360
        ly=height-39+(i//2)*22
        svg.append(f'<line x1="{lx}" x2="{lx+24}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{lx+31}" y="{ly+4}" class="leg">{html.escape(arm.replace("_"," "))}</text>')
    svg.append('</svg>')
    out.write_text("\n".join(svg))

def build_dashboard(folder:Path,out:Path):
    rows=load_trajectories(folder)
    out.mkdir(parents=True,exist_ok=True)
    keys=sorted({k for r in rows.values() for k in r.keys()})
    with (out/"g10_all_validation.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore")
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])
    summary={
        "source":"completed G1-G10 original Omniglot training checkpoints",
        "evaluation":"validation-only; no final-test use",
        "independent_model_seeds":1,
        "g0_excluded":"different episode sampling; evaluate on matched episodes before including",
        "interpretive_limit":"target-only synthetic feedback; not full input recursion",
        "base_training_exposures_per_generation":1_000_000,
        "extended_training_exposures_per_generation":300_000,
        "visualized_metrics":{},
        "g10":{},
    }
    for slug,(title,task,metric) in DESCRIPTIONS.items():
        save_svg(out/f"{slug}.svg",slug,title,task,metric,rows)
        summary["visualized_metrics"][slug]=metric
    for task in TASKS:
        summary["g10"][task]={}
        for arm in ARMS:
            metrics={k:float(v) for k,v in rows[(task,arm,10)].items()
                     if k.startswith("val_")}
            summary["g10"][task][arm]=metrics
    (out/"g10_summary.json").write_text(json.dumps(summary,indent=2))
    print("G10_CONSOLIDATED="+json.dumps({
        "validated_rows":len(rows),
        "figures":list(DESCRIPTIONS),
        "g10":summary["g10"],
        "final_test_evaluated":False,
    }),flush=True)
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    build_dashboard(a.input,a.out)
if __name__=="__main__":
    main()
