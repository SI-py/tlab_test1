#!/usr/bin/env python3
"""Four-fit random window-subset control on the existing development snapshots.

This is an exploratory control added after the original test was observed.
It never opens a test snapshot, changes the original selection, or runs a refit.
Run after the KION development stage; saved article results are not overwritten.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CARDINALITIES = [1, 2, 3, 5]
PARAMS = {"depth": 7, "l2_leaf_reg": 15}


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def annotate(evidence):
    note = ("Random feature blocks use canonical day order; original importance blocks use importance order. "
            "For these fixed draws the orders differ only for the five-window random trial. "
            "Column serialization can affect fixed-seed CatBoost behavior, so this is a bounded procedural comparison, "
            "not a pure causal estimate of the selection rule. No fits were repeated to remove this nuisance.")
    if note not in evidence["limitations"]:
        evidence["limitations"].append(note)
    shared = evidence["design"]["shared_precompute"]["fit_seconds"]
    for method in ["importance", "random"]:
        if "total_fit_seconds" in evidence[method]:
            evidence[method]["total_fit_seconds_including_shared_precompute"] = (
                shared + evidence[method]["total_fit_seconds"])


def summary(trials):
    best = max(trials, key=lambda trial: trial["metrics"]["ndcg@10"])
    return {"trials": trials,
            "subset_fit_count": len(trials),
            "total_fit_seconds": sum(t["fit_seconds"] for t in trials),
            "best": {k: best[k] for k in ["trial", "groups", "cardinality", "trees", "features", "metrics"]}}


def compact(record, method, number):
    groups = record["selected_window_groups"]
    return {"trial": f"{method}_{number}", "groups": groups,
            "cardinality": len(groups), "params": record["params"],
            "seed": record["seed"], "trees": record["trees"],
            "features": record["features"], "fit_seconds": record["fit_seconds"],
            "metrics": record["metrics"], "fit_record": record["fit_record"]}


def make_plan(base):
    import numpy as np
    from recsys.v2_experiment import features_for

    selection = read(base / "selection.json")
    config = selection["config"]
    stats = read(base / "snapshots" / ("train_" + config["protocol"]["train_cutoffs"][0]) / "stats.json")
    universe = sorted((g for g in stats["groups"] if g.startswith("window_")),
                      key=lambda group: int(group.removeprefix("window_")))
    if len(universe) != 9:
        raise ValueError("The declared control requires the original nine-window grammar")
    old = read(base / "validation_search.json")["trials"]
    importance = [r for r in old if r["variant"] == "auto_selected"]
    if [len(r["selected_window_groups"]) for r in importance] != CARDINALITIES:
        raise ValueError("Original importance search must contain exactly cardinalities 1,2,3,5")
    for record in importance:
        if (record["params"] != PARAMS or record["seed"] != 17 or
                record["fit_record"]["trained_through_iteration"] != 2000 or
                record["fit_record"]["staged_period"] != 50):
            raise ValueError("Original subset fits do not match the fixed control budget")
    manual = features_for(stats, "manual")
    rng = np.random.default_rng(20261025)
    random_subsets = []
    for number, cardinality in enumerate(CARDINALITIES, start=1):
        chosen = set(rng.choice(universe, cardinality, replace=False).tolist())
        groups = [group for group in universe if group in chosen]
        columns = list(dict.fromkeys(manual + [c for g in groups for c in stats["groups"][g]]))
        random_subsets.append({"trial": f"random_{number}", "cardinality": cardinality,
                               "groups": groups, "columns": columns})
    common = [{k: r[k] for k in ["params", "seed", "trees", "fit_seconds", "metrics"]}
              for r in old if r["variant"] == "auto_windows" and r["family"] == "catboost"]
    return {
        "status": "planned", "created_at": now(),
        "scope": "Exploratory equal-subset-fit-budget control added after original test access; development only. One random seed is one search-variance probe, not evidence of universal superiority.",
        "base_output": str(base.resolve()),
        "design": {
            "random_subset_seed": 20261025, "training_seed": 17,
            "cardinalities": CARDINALITIES, "window_universe": universe,
            "sampling": "Four independent uniform subsets; no replacement within a subset; groups stored in canonical day order.",
            "params": PARAMS, "iterations_per_fit": 2000, "staged_period": 50,
            "subset_fits_per_procedure": 4,
            "objective": "Equal mean of the two full-denominator temporal validation NDCG@10 values; stable item-ID tie breaking.",
            "train_cutoffs": config["protocol"]["train_cutoffs"],
            "validation_cutoffs": config["protocol"]["validation_cutoffs"],
            "manual_features": len(manual), "features_per_window": {g: len(stats["groups"][g]) for g in universe},
            "pool": "Existing V4 sampled training rows and complete common validation candidates/truths; no resampling or target injection.",
            "importance_order": selection["feature_specification"]["grammar"]["generated_windows"],
            "random_subsets": random_subsets,
            "shared_precompute": {"description": "The original two all-window CatBoost parameter fits supply the shared fixed hyperparameters; their train-fitted importance supplies the importance order. Cost is common to both conditional searches, not hidden or repeated.",
                                  "fits": common, "fit_count": len(common),
                                  "fit_seconds": sum(r["fit_seconds"] for r in common)},
            "budget_interpretation": "Matched subset-fit count, cardinalities, fixed parameters and 2000-tree cap; elapsed CPU time need not be identical. Shared all-window precompute cost is reported separately. No search expansion after observing random results."},
        "importance": summary([compact(r, "importance", n) for n, r in enumerate(importance, start=1)]),
        "random": {"trials": []},
        "limitations": ["Four subsets from one preset RNG seed do not characterize random-search variance.",
                        "The same development weeks select tree counts and the best subset for both procedures.",
                        "Point estimates compare these bounded procedures; no new significance or confirmatory-test claim.",
                        "Original final test was already observed before this control; it is not a new untouched confirmation."]}


def projected_snapshot(path, columns):
    import numpy as np
    import pandas as pd

    stats = read(path / "stats.json")
    requested = list(dict.fromkeys(["query_id", "item_id", "label"] + columns))
    frame = pd.concat([pd.read_parquet(path / name, columns=requested)
                       for name in stats["feature_parts"]], ignore_index=True)
    query, item = frame.query_id.to_numpy(), frame.item_id.to_numpy()
    if np.any(query[1:] < query[:-1]) or np.any((query[1:] == query[:-1]) & (item[1:] <= item[:-1])):
        raise ValueError("Snapshots must preserve unique query/item ordering")
    return frame, pd.read_parquet(path / "truth.parquet"), stats


def development(base, design, columns):
    import pandas as pd

    train, validation, truths = [], [], []
    for role, key, destination in [("train", "train_cutoffs", train),
                                   ("validation", "validation_cutoffs", validation)]:
        for cutoff in design[key]:
            frame, truth, stats = projected_snapshot(base / "snapshots" / f"{role}_{cutoff}", columns)
            destination.append(frame)
            if role == "validation":
                truths.append(truth)
    return pd.concat(train, ignore_index=True), pd.concat(validation, ignore_index=True), truths, stats


def write_csv(path, evidence):
    fields = ["method", "trial", "cardinality", "groups", "features", "trees", "fit_seconds",
              "ndcg@10", "recall@10", "week1_ndcg@10", "week2_ndcg@10"]
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in ["importance", "random"]:
            for trial in evidence[method]["trials"]:
                row = {k: trial[k] for k in ["trial", "cardinality", "features", "trees", "fit_seconds"]}
                row.update(method=method, groups=",".join(trial["groups"]),
                           **{k: trial["metrics"][k] for k in ["ndcg@10", "recall@10"]})
                for n, week in enumerate(trial["metrics"]["weeks"], start=1):
                    row[f"week{n}_ndcg@10"] = week["ndcg@10"]
                writer.writerow(row)


def run(args, evidence):
    from recsys.v2_search import FullRankingMetric
    from recsys.v3_catboost import fit_classifier

    config = read(args.base / "selection.json")["config"]
    config = {**config, "catboost_params": PARAMS}
    completed = {r["trial"] for r in evidence["random"]["trials"]}
    if len(completed) == 4 and evidence["status"] == "complete":
        print("RANDOM_CONTROL_ALREADY_COMPLETE", flush=True)
        return
    args.work.mkdir(parents=True, exist_ok=True)
    evidence["status"] = "running"
    evidence.setdefault("started_at", now())
    save(args.results, evidence)
    for trial in evidence["design"]["random_subsets"]:
        if trial["trial"] in completed:
            continue
        started = time.perf_counter()
        columns = trial["columns"]
        train, val, truths, stats = development(args.base, evidence["design"], columns)
        metric = FullRankingMetric(val, truths, evidence["design"]["validation_cutoffs"])
        print("RANDOM_CONTROL_START", json.dumps({k: v for k, v in trial.items() if k != "columns"}),
              "train_rows", len(train), "validation_rows", len(val), flush=True)
        model, fit_record = fit_classifier(train, columns, stats["cats"], config,
                                          seed=17, iterations=2000, val=val,
                                          metric=metric, staged_period=50)
        model_file = args.work / (trial["trial"] + ".cbm")
        model.save_model(str(model_file))
        record = {k: v for k, v in trial.items() if k != "columns"}
        record.update(params=PARAMS, seed=17, trees=int(model.tree_count_), features=len(columns),
                      columns=columns, categoricals=[c for c in stats["cats"] if c in columns],
                      fit_seconds=fit_record["fit_seconds"], fit_record=fit_record,
                      metrics=fit_record["metrics"], model_file=str(model_file.resolve()),
                      wall_seconds_with_snapshot_load=time.perf_counter() - started,
                      training_rows=len(train), validation_rows=len(val))
        evidence["random"]["trials"].append(record)
        save(args.results, evidence)
        write_csv(args.results, evidence)
        print("RANDOM_CONTROL_TRIAL", json.dumps({k: v for k, v in record.items()
              if k not in ["columns", "categoricals", "fit_record"]}), flush=True)
        del model, train, val, truths, metric
        gc.collect()
    evidence["random"] = summary(evidence["random"]["trials"])
    importance, random = evidence["importance"]["best"], evidence["random"]["best"]
    evidence["comparison"] = {
        "random_minus_importance_ndcg@10": random["metrics"]["ndcg@10"] - importance["metrics"]["ndcg@10"],
        "random_minus_importance_recall@10": random["metrics"]["recall@10"] - importance["metrics"]["recall@10"],
        "random_minus_importance_week_ndcg@10": [r["ndcg@10"] - i["ndcg@10"]
            for r, i in zip(random["metrics"]["weeks"], importance["metrics"]["weeks"])],
        "interpretation": "Best-of-four development point estimates under the fixed conditional search budget; not a significance claim or an untouched-test result."}
    evidence.update(status="complete", completed_at=now())
    annotate(evidence)
    save(args.results, evidence)
    write_csv(args.results, evidence)
    print("RANDOM_CONTROL_COMPLETE", json.dumps(evidence["comparison"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "artifacts/v4")
    parser.add_argument("--work", type=Path, default=ROOT / "artifacts/random_control")
    parser.add_argument("--results", type=Path, default=ROOT / "artifacts/random_control/results.json")
    parser.add_argument("--stage", choices=["plan", "develop"], default="plan")
    args = parser.parse_args()
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    plan = make_plan(args.base)
    if args.results.exists():
        evidence = read(args.results)
        if evidence["design"] != plan["design"] or evidence["base_output"] != plan["base_output"]:
            raise ValueError("Existing random control has another fixed design; use a fresh output")
    else:
        evidence = plan
    annotate(evidence)
    save(args.results, evidence)
    if args.stage == "plan":
        print(json.dumps({"status": evidence["status"], "design": evidence["design"]}, ensure_ascii=False, indent=2))
    else:
        run(args, evidence)


if __name__ == "__main__":
    main()
