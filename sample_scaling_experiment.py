#!/usr/bin/env python3
"""
Experiment to test how success probability and skeleton SHD vary with samples per object.

For a given environment and displacement, tests increasing numbers of samples per object
(1, 2, 3, ...) and computes success probability and average skeleton SHD across seeds.

With --evaluate-baselines, also compares against PC algorithm and ad hoc collision methods
at the minimum sample count that achieves 95% success.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np

from rube.level_parser import LevelParser
from eval import load_edges_from_file
from baseline_pc import run_pc_algorithm


def print_latex_table(result: Dict, synthetic: bool):
    """Print results in LaTeX table format."""
    env = result["environment"]
    disp = result["displacement"]
    m = result["min_samples_per_action"]
    n = result["num_variables"]

    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")

    if synthetic:
        # Synthetic mode: show both SHD and Skeleton SHD
        print("Method & Precision & Recall & F1 & SHD & Skel. SHD & Time (s) \\\\")
        print("\\midrule")
        # Our method
        if not np.isnan(result.get("our_precision", np.nan)):
            print(
                f"Our Method (M={m}) & {result['our_precision']:.3f} & {result['our_recall']:.3f} & {result['our_f1']:.3f} & {result['our_shd']:.2f} & {result['our_skeleton_shd']:.2f} & {result['our_time']:.3f} \\\\"
            )
        else:
            print(f"Our Method (M={m}) & - & - & - & - & - & - \\\\")
        # PC
        if not np.isnan(result["pc_precision"]):
            print(
                f"PC (N={result['total_samples']}) & {result['pc_precision']:.3f} & {result['pc_recall']:.3f} & {result['pc_f1']:.3f} & {result['pc_shd']:.2f} & {result['pc_skeleton_shd']:.2f} & {result['pc_time']:.3f} \\\\"
            )
    else:
        # Physics mode: show both SHD and Skeleton SHD for all methods
        print("Method & Precision & Recall & F1 & SHD & Skel. SHD & Time (s) \\\\")
        print("\\midrule")
        # Our method
        if not np.isnan(result.get("our_precision", np.nan)):
            print(
                f"Our Method (M={m}) & {result['our_precision']:.3f} & {result['our_recall']:.3f} & {result['our_f1']:.3f} & {result['our_shd']:.2f} & {result['our_skeleton_shd']:.2f} & {result['our_time']:.3f} \\\\"
            )
        else:
            print(f"Our Method (M={m}) & - & - & - & - & - & - \\\\")
        # PC
        if not np.isnan(result["pc_precision"]):
            print(
                f"PC (N={result['total_samples']}) & {result['pc_precision']:.3f} & {result['pc_recall']:.3f} & {result['pc_f1']:.3f} & {result['pc_shd']:.2f} & {result['pc_skeleton_shd']:.2f} & {result['pc_time']:.3f} \\\\"
            )
        # Ad hoc methods
        if not np.isnan(result["ad_hoc_simple_precision"]):
            print(
                f"Ad hoc (simple) & {result['ad_hoc_simple_precision']:.3f} & {result['ad_hoc_simple_recall']:.3f} & {result['ad_hoc_simple_f1']:.3f} & {result['ad_hoc_simple_shd']:.2f} & {result['ad_hoc_simple_skeleton_shd']:.2f} & {result['ad_hoc_simple_time']:.3f} \\\\"
            )
        if not np.isnan(result["ad_hoc_time_precision"]):
            print(
                f"Ad hoc (time) & {result['ad_hoc_time_precision']:.3f} & {result['ad_hoc_time_recall']:.3f} & {result['ad_hoc_time_f1']:.3f} & {result['ad_hoc_time_shd']:.2f} & {result['ad_hoc_time_skeleton_shd']:.2f} & {result['ad_hoc_time_time']:.3f} \\\\"
            )

    print("\\bottomrule")
    print("\\end{tabular}")
    print(
        f"\\caption{{Baseline comparison for {env} at displacement {disp} (N={n} variables, {result['seeds_tested']} seeds)}}"
    )
    print("\\label{tab:comparison_" + env + "_" + str(disp).replace(".", "_") + "}")
    print("\\end{table}")


def run_simulation_to_generate_data(
    level_file: str,
    seed: int,
    displacement: float,
    num_intervention_samples: int,
    output_dir: Path,
    max_steps: int = 1000,
    settling_steps: int = 10,
    no_intervention_samples: int = 0,
    synthetic: bool = False,
    synthetic_graph: str = None,
    synthetic_probs: str = None,
) -> Path:
    """Run simulation to generate intervention data."""
    cmd = [
        sys.executable,
        "run_simulation.py",
        level_file,
        "--seed",
        str(seed),
        "--displacement",
        str(displacement),
        "--max-steps",
        str(max_steps),
        "--settling-steps",
        str(settling_steps),
        "--intervention-samples",
        str(num_intervention_samples),
        "--no-intervention-samples",
        str(no_intervention_samples),
        "--per-action",
        "--output-dir",
        str(output_dir),
        "--no-gif",
    ]

    # Add synthetic mode arguments if enabled
    if synthetic:
        cmd.append("--synthetic")
        if synthetic_graph:
            cmd.extend(["--synthetic-graph", synthetic_graph])
        if synthetic_probs:
            cmd.extend(["--synthetic-probs", synthetic_probs])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Simulation failed: {result.stderr}")

    # Find the output directory
    csv_paths = list(output_dir.glob("*/intervention_samples.csv"))
    if not csv_paths:
        raise RuntimeError(f"No intervention_samples.csv found under {output_dir}")

    return csv_paths[0].parent


def run_causal_discovery(
    intervention_csv: Path,
    true_graph: Path,
    output_dir: Path,
) -> dict:
    """Run causal discovery and return evaluation metrics."""
    cmd = [
        sys.executable,
        "causal_discovery.py",
        str(intervention_csv),
        "--true-graph",
        str(true_graph),
        "--output-dir",
        str(output_dir),
        "--quiet",
        "--force-tree",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    # Load predicted edges
    pred_file = output_dir / "edge_predictions.txt"
    if not pred_file.exists():
        return None

    pred_edges = load_edges_from_file(str(pred_file))
    true_edges = load_edges_from_file(str(true_graph))

    # Evaluate
    true_set = set(true_edges)
    pred_set = set(pred_edges)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    # Skeleton SHD
    def to_skeleton(edges):
        skel = set()
        for u, v in edges:
            if u != v:
                skel.add(frozenset((u, v)))
        return skel

    true_skel = to_skeleton(true_set)
    pred_skel = to_skeleton(pred_set)
    skeleton_shd = len(pred_skel - true_skel) + len(true_skel - pred_skel)

    # Success = perfect match
    success = tp == len(true_set) and fp == 0 and fn == 0

    return {
        "success": success,
        "skeleton_shd": skeleton_shd,
    }


def evaluate_ad_hoc_method(
    ad_hoc_file: Path,
    true_graph: Path,
) -> Dict:
    """
    Evaluate ad hoc method.

    Returns:
        Dictionary with evaluation metrics
    """
    if not ad_hoc_file.exists():
        return None

    pred_edges = load_edges_from_file(str(ad_hoc_file))
    true_edges = load_edges_from_file(str(true_graph))

    true_set = set(true_edges)
    pred_set = set(pred_edges)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )

    # Compute directed SHD (counting reversed edges correctly)
    # SHD = missing edges + extra edges + reversed edges
    true_only = true_set - pred_set
    pred_only = pred_set - true_set

    # Find reversed edges: edges that appear in both graphs but with reversed direction
    reversal_pairs = set()
    for u, v in true_set:
        if (v, u) in pred_set:
            reversal_pairs.add(frozenset((u, v)))
    num_reversals = len(reversal_pairs)

    # SHD = |true \ pred| + |pred \ true| - |reversals|
    # (reversals are counted in both true_only and pred_only, so we subtract once)
    shd = len(true_only) + len(pred_only) - num_reversals

    # Skeleton SHD
    def to_skeleton(edges):
        skel = set()
        for u, v in edges:
            if u != v:
                skel.add(frozenset((u, v)))
        return skel

    true_skel = to_skeleton(true_set)
    pred_skel = to_skeleton(pred_set)
    skeleton_shd = len(pred_skel - true_skel) + len(true_skel - pred_skel)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "shd": shd,
        "skeleton_shd": skeleton_shd,
    }


def test_samples_per_object(
    level_file: str,
    seeds: List[int],
    displacement: float,
    samples_per_object: int,
    num_objects: int,
    temp_dir: Path,
    max_steps: int = 1000,
    settling_steps: int = 10,
    synthetic: bool = False,
    synthetic_graph: str = None,
    synthetic_probs: str = None,
) -> Tuple[float, float]:
    """
    Test a specific number of samples per object across all seeds.

    Returns:
        (success_probability, avg_skeleton_shd)
    """
    successes = 0
    skeleton_shds = []

    for seed in seeds:
        run_dir = temp_dir / f"samples_{samples_per_object}_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            output_dir = run_simulation_to_generate_data(
                level_file=level_file,
                seed=seed,
                displacement=displacement,
                num_intervention_samples=samples_per_object,
                output_dir=run_dir,
                max_steps=max_steps,
                settling_steps=settling_steps,
                synthetic=synthetic,
                synthetic_graph=synthetic_graph,
                synthetic_probs=synthetic_probs,
            )

            # Get the true graph (solution) from this specific run
            true_graph = output_dir / "solution.txt"
            if not true_graph.exists():
                print(f"      Warning: Solution file not found for seed {seed}")
                continue

            # Run causal discovery
            intervention_csv = output_dir / "intervention_samples.csv"
            cd_output = run_dir / "cd"

            metrics = run_causal_discovery(
                intervention_csv=intervention_csv,
                true_graph=true_graph,
                output_dir=cd_output,
            )

            if metrics:
                if metrics["success"]:
                    successes += 1
                skeleton_shds.append(metrics["skeleton_shd"])

        except Exception as e:
            print(f"      Error with seed {seed}: {e}")
            continue

    # Calculate statistics
    total_tested = len(skeleton_shds)
    success_prob = successes / total_tested if total_tested > 0 else 0.0
    avg_skeleton_shd = np.mean(skeleton_shds) if skeleton_shds else float("nan")

    return success_prob, avg_skeleton_shd


def evaluate_baselines_at_min_samples(
    level_file: str,
    seeds: List[int],
    displacement: float,
    min_samples_per_action: int,
    num_vars: int,
    temp_dir: Path,
    max_steps: int,
    settling_steps: int,
    synthetic: bool,
    synthetic_graph: str,
    synthetic_probs: str,
    skip_pc: bool,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Evaluate baseline methods at the minimum sample count.

    Returns:
        Tuple of (averaged_metrics_dict, timing_dict, our_method_metrics_dict, our_method_timing_dict)
    """
    total_samples = min_samples_per_action * num_vars

    baseline_metrics_per_seed = {"pc": [], "simple": [], "time": []}
    baseline_times = {"pc": [], "simple": [], "time": []}

    # Track our method's metrics
    our_method_metrics_per_seed = []
    our_method_times = []

    print(
        f"\n  Evaluating our method and baselines with M={min_samples_per_action} per action (total={total_samples} observational samples)..."
    )
    print(f"  Testing across {len(seeds)} seeds...")

    for seed in seeds:
        # Generate data with minimum samples
        eval_dir = temp_dir / f"eval_seed_{seed}"
        eval_dir.mkdir(parents=True, exist_ok=True)

        try:
            output_data_dir = run_simulation_to_generate_data(
                level_file=level_file,
                seed=seed,
                displacement=displacement,
                num_intervention_samples=min_samples_per_action,
                output_dir=eval_dir,
                max_steps=max_steps,
                settling_steps=settling_steps,
                no_intervention_samples=total_samples,  # Use equivalent total samples for baselines
                synthetic=synthetic,
                synthetic_graph=synthetic_graph,
                synthetic_probs=synthetic_probs,
            )

            # Get the true graph (solution) from this specific run
            true_graph_for_eval = output_data_dir / "solution.txt"
            if not true_graph_for_eval.exists():
                print(f"    Seed {seed}: WARNING - Solution file not found")
                continue

            # Evaluate our method first
            intervention_csv = output_data_dir / "intervention_samples.csv"
            cd_output = eval_dir / "cd"

            start_time = time.time()
            metrics_our = run_causal_discovery(
                intervention_csv=intervention_csv,
                true_graph=true_graph_for_eval,
                output_dir=cd_output,
            )
            our_method_time = time.time() - start_time

            if metrics_our:
                # Load edges to compute full metrics
                pred_file = cd_output / "edge_predictions.txt"
                true_edges = load_edges_from_file(str(true_graph_for_eval))
                pred_edges = load_edges_from_file(str(pred_file))

                true_set = set(true_edges)
                pred_set = set(pred_edges)

                tp = len(true_set & pred_set)
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)

                precision = tp / (tp + fp) if tp + fp > 0 else 0.0
                recall = tp / (tp + fn) if tp + fn > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall > 0
                    else 0.0
                )

                # Compute directed SHD (counting reversed edges correctly)
                true_only = true_set - pred_set
                pred_only = pred_set - true_set

                # Find reversed edges
                reversal_pairs = set()
                for u, v in true_set:
                    if (v, u) in pred_set:
                        reversal_pairs.add(frozenset((u, v)))
                num_reversals = len(reversal_pairs)

                # SHD = |true \ pred| + |pred \ true| - |reversals|
                shd = len(true_only) + len(pred_only) - num_reversals

                our_method_metrics_per_seed.append(
                    {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "shd": shd,
                        "skeleton_shd": metrics_our["skeleton_shd"],
                    }
                )
                our_method_times.append(our_method_time)

            # Run PC algorithm (both modes unless skipped)
            if not skip_pc:
                obs_csv = output_data_dir / "no_intervention_samples.csv"
                if not obs_csv.exists():
                    print(
                        f"    Seed {seed}: WARNING - Observational CSV not found for PC: {obs_csv}"
                    )
                else:
                    start_time = time.time()
                    metrics_pc = run_pc_algorithm(obs_csv, true_graph_for_eval)
                    pc_time = time.time() - start_time
                    if metrics_pc:
                        baseline_metrics_per_seed["pc"].append(metrics_pc)
                        baseline_times["pc"].append(pc_time)
                    else:
                        print(f"    Seed {seed}: WARNING - PC algorithm returned None")

            # Run ad hoc methods (physics mode only)
            if not synthetic:
                ad_hoc_simple = output_data_dir / "ad_hoc_collisions.txt"
                ad_hoc_time = output_data_dir / "ad_hoc_collisions_with_time.txt"

                start_time = time.time()
                metrics_simple = evaluate_ad_hoc_method(
                    ad_hoc_simple, true_graph_for_eval
                )
                simple_time = time.time() - start_time
                if metrics_simple:
                    baseline_metrics_per_seed["simple"].append(metrics_simple)
                    baseline_times["simple"].append(simple_time)

                start_time = time.time()
                metrics_time = evaluate_ad_hoc_method(ad_hoc_time, true_graph_for_eval)
                time_method_time = time.time() - start_time
                if metrics_time:
                    baseline_metrics_per_seed["time"].append(metrics_time)
                    baseline_times["time"].append(time_method_time)

        except Exception as e:
            print(f"    Seed {seed}: ERROR ({e})")
            continue

    # Average baseline metrics across seeds
    avg_metrics = {}
    avg_times = {}

    for method_name in ["pc", "simple", "time"]:
        if baseline_metrics_per_seed[method_name]:
            metrics_list = baseline_metrics_per_seed[method_name]
            avg_metrics[method_name] = {
                "precision": np.mean([m["precision"] for m in metrics_list]),
                "recall": np.mean([m["recall"] for m in metrics_list]),
                "f1": np.mean([m["f1"] for m in metrics_list]),
                "skeleton_shd": np.mean([m["skeleton_shd"] for m in metrics_list]),
            }

            # Add SHD (all methods now have this metric)
            if "shd" in metrics_list[0]:
                avg_metrics[method_name]["shd"] = np.mean(
                    [m["shd"] for m in metrics_list]
                )

            # Average timing
            if baseline_times[method_name]:
                avg_times[method_name] = np.mean(baseline_times[method_name])

            # Print results
            time_str = (
                f", Time={avg_times.get(method_name, 0):.3f}s"
                if method_name in avg_times
                else ""
            )
            if method_name == "pc":
                method_label = "PC"
                print(
                    f"  {method_label}: P={avg_metrics[method_name]['precision']:.3f}, "
                    f"R={avg_metrics[method_name]['recall']:.3f}, "
                    f"F1={avg_metrics[method_name]['f1']:.3f}, "
                    f"SHD={avg_metrics[method_name]['shd']:.2f}, "
                    f"Skeleton SHD={avg_metrics[method_name]['skeleton_shd']:.2f}"
                    f"{time_str}"
                )
            elif method_name == "simple":
                method_label = "Ad hoc (simple)"
                print(
                    f"  {method_label}: P={avg_metrics[method_name]['precision']:.3f}, "
                    f"R={avg_metrics[method_name]['recall']:.3f}, "
                    f"F1={avg_metrics[method_name]['f1']:.3f}, "
                    f"SHD={avg_metrics[method_name]['shd']:.2f}, "
                    f"Skeleton SHD={avg_metrics[method_name]['skeleton_shd']:.2f}"
                    f"{time_str}"
                )
            else:  # time
                method_label = "Ad hoc (time)"
                print(
                    f"  {method_label}: P={avg_metrics[method_name]['precision']:.3f}, "
                    f"R={avg_metrics[method_name]['recall']:.3f}, "
                    f"F1={avg_metrics[method_name]['f1']:.3f}, "
                    f"SHD={avg_metrics[method_name]['shd']:.2f}, "
                    f"Skeleton SHD={avg_metrics[method_name]['skeleton_shd']:.2f}"
                    f"{time_str}"
                )
        else:
            avg_metrics[method_name] = None

    # Average our method metrics
    our_method_avg_metrics = {}
    our_method_avg_time = np.nan
    if our_method_metrics_per_seed:
        our_method_avg_metrics = {
            "precision": np.mean([m["precision"] for m in our_method_metrics_per_seed]),
            "recall": np.mean([m["recall"] for m in our_method_metrics_per_seed]),
            "f1": np.mean([m["f1"] for m in our_method_metrics_per_seed]),
            "shd": np.mean([m["shd"] for m in our_method_metrics_per_seed]),
            "skeleton_shd": np.mean(
                [m["skeleton_shd"] for m in our_method_metrics_per_seed]
            ),
        }
        if our_method_times:
            our_method_avg_time = np.mean(our_method_times)

        # Print our method results
        print(
            f"  Our Method: P={our_method_avg_metrics['precision']:.3f}, "
            f"R={our_method_avg_metrics['recall']:.3f}, "
            f"F1={our_method_avg_metrics['f1']:.3f}, "
            f"SHD={our_method_avg_metrics['shd']:.2f}, "
            f"Skeleton SHD={our_method_avg_metrics['skeleton_shd']:.2f}, "
            f"Time={our_method_avg_time:.3f}s"
        )

    return avg_metrics, avg_times, our_method_avg_metrics, {"time": our_method_avg_time}


def run_scaling_experiment(
    level_file: str,
    displacement: float,
    seeds: List[int],
    output_dir: Path,
    min_samples: int = 1,
    max_samples: int = 100,
    max_steps: int = 1000,
    settling_steps: int = 10,
    synthetic: bool = False,
    synthetic_graph: str = None,
    synthetic_probs: str = None,
    evaluate_baselines: bool = False,
    skip_pc: bool = False,
    success_threshold: float = 0.95,
) -> Tuple[str, Optional[Dict]]:
    """
    Run scaling experiment: test increasing samples per object until 100% success.

    Returns:
        Tuple of (path_to_scaling_csv, comparison_result_dict or None)
    """
    env_name = Path(level_file).stem

    if synthetic:
        # In synthetic mode, level_file is the graph file
        level_path = Path(level_file)
    else:
        # In physics mode, level_file is in levels/ directory
        level_path = Path("levels") / level_file

    if not level_path.exists():
        raise ValueError(f"File not found: {level_path}")

    print(f"\n{'='*70}")
    print(f"Sample Scaling Experiment")
    print(f"{'='*70}")

    # Print command for easy re-run
    cmd_parts = [sys.executable, "sample_scaling_experiment.py", str(level_file)]
    if not synthetic:
        cmd_parts.extend(["--displacement", str(displacement)])
    cmd_parts.extend(["--seeds"] + [str(s) for s in seeds])
    cmd_parts.extend(
        ["--min-samples", str(min_samples), "--max-samples", str(max_samples)]
    )
    cmd_parts.extend(
        ["--max-steps", str(max_steps), "--settling-steps", str(settling_steps)]
    )
    if synthetic:
        cmd_parts.append("--synthetic")
        if synthetic_graph:
            cmd_parts.extend(["--synthetic-graph", synthetic_graph])
        if synthetic_probs:
            cmd_parts.extend(["--synthetic-probs", synthetic_probs])
    if evaluate_baselines:
        cmd_parts.append("--evaluate-baselines")
    if skip_pc:
        cmd_parts.append("--skip-pc")
    if success_threshold != 0.95:
        cmd_parts.extend(["--success-threshold", str(success_threshold)])

    print(f"\nCommand: {' '.join(cmd_parts)}\n")

    print(f"Mode: {'SYNTHETIC' if synthetic else 'PHYSICS'}")
    print(f"Environment: {env_name}")
    if synthetic:
        print(f"Synthetic graph: {synthetic_graph}")
        print(f"Synthetic probs: {synthetic_probs}")
    else:
        print(f"Displacement: {displacement}")
    print(f"Seeds: {len(seeds)}")
    print(f"Sample range: {min_samples} to {max_samples}")
    if evaluate_baselines:
        print(f"Baseline evaluation: ENABLED (threshold: {success_threshold*100:.0f}%)")
        if skip_pc:
            print(f"  - PC algorithm: SKIPPED")
    print(f"{'='*70}\n")

    # Get number of objects/variables
    if synthetic:
        # For synthetic mode, determine num_vars from the graph file
        causal_graph = []
        with open(level_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and "->" in line:
                    parent, child = line.split("->")
                    causal_graph.append((int(parent), int(child)))

        all_nodes = set()
        for parent, child in causal_graph:
            all_nodes.add(parent)
            all_nodes.add(child)
        num_objects = max(all_nodes) if all_nodes else 0
        print(f"Number of variables: {num_objects}\n")
    else:
        # For physics mode, parse the level file
        parser = LevelParser()
        level = parser.parse_file(str(level_path))
        # Exclude ramps
        obj_list = [obj_def for obj_def in level.objects if obj_def["type"] != "ramp"]
        num_objects = len(obj_list)
        print(f"Number of objects: {num_objects}\n")

    # Create temp directory
    temp_dir = output_dir / f"temp_{env_name}_d{displacement}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Check if we should skip scaling and go straight to evaluation
    skip_scaling = (min_samples == max_samples) and evaluate_baselines

    if skip_scaling:
        print(f"  Skipping scaling experiment (min_samples == max_samples)")
        print(f"   Running baseline evaluation only at M={min_samples}\n")
        min_samples_per_action = min_samples
        results = []
        scaling_file = None
    elif min_samples == max_samples and not evaluate_baselines:
        print(
            f"   Warning: min_samples == max_samples but --evaluate-baselines not set"
        )
        print(f"   Will run scaling for single point only\n")
        # Fall through to normal scaling loop

    if not skip_scaling:
        # Collect data points
        results = []
        consecutive_perfect = 0
        min_samples_per_action = None  # Track first M that achieves threshold

        for samples_per_object in range(min_samples, max_samples + 1):
            print(f"Testing {samples_per_object} samples per object...")

            success_prob, avg_skeleton_shd = test_samples_per_object(
                level_file=str(level_path),
                seeds=seeds,
                displacement=displacement,
                samples_per_object=samples_per_object,
                num_objects=num_objects,
                temp_dir=temp_dir,
                max_steps=max_steps,
                settling_steps=settling_steps,
                synthetic=synthetic,
                synthetic_graph=synthetic_graph,
                synthetic_probs=synthetic_probs,
            )

            print(
                f"  Success: {success_prob*100:.1f}%, Avg Skeleton SHD: {avg_skeleton_shd:.2f}"
            )

            results.append(
                {
                    "samples_per_object": samples_per_object,
                    "success_probability": success_prob,
                    "avg_skeleton_shd": avg_skeleton_shd,
                }
            )

            # Track first M that achieves threshold (for baseline comparison)
            if min_samples_per_action is None and success_prob >= success_threshold:
                min_samples_per_action = samples_per_object
                print(
                    f"  → First to achieve {success_threshold*100:.0f}% success threshold!"
                )

            # Check stopping criterion (for plotting completeness)
            if success_prob >= 1.0:
                consecutive_perfect += 1
                if consecutive_perfect >= 2:
                    print(
                        f"\n✓ Stopping: Achieved 100% success for 2 consecutive sample sizes"
                    )
                    break
            else:
                consecutive_perfect = 0

        # Save scaling results (for plotting)
        timestamp = int(time.time())
        scaling_file = (
            output_dir / f"{env_name}_d{displacement}_scaling_{timestamp}.csv"
        )
        df = pd.DataFrame(results)
        df.to_csv(scaling_file, index=False)

        print(f"\n✓ Scaling results saved to: {scaling_file}")
        print(f"  Total data points: {len(results)}")

    # Evaluate baselines if requested and we found a minimum
    comparison_result = None
    if evaluate_baselines and min_samples_per_action is not None:
        print(f"\n--- Baseline Comparison (at M={min_samples_per_action}) ---")

        avg_metrics, avg_times, our_method_metrics, our_method_times = (
            evaluate_baselines_at_min_samples(
                level_file=str(level_path),
                seeds=seeds,
                displacement=displacement,
                min_samples_per_action=min_samples_per_action,
                num_vars=num_objects,
                temp_dir=temp_dir,
                max_steps=max_steps,
                settling_steps=settling_steps,
                synthetic=synthetic,
                synthetic_graph=synthetic_graph,
                synthetic_probs=synthetic_probs,
                skip_pc=skip_pc,
            )
        )

        # Build comparison result
        total_samples = min_samples_per_action * num_objects
        comparison_result = {
            "environment": env_name,
            "displacement": displacement,
            "num_variables": num_objects,
            "min_samples_per_action": min_samples_per_action,
            "total_samples": total_samples,
            "seeds_tested": len(seeds),
            "mode": "synthetic" if synthetic else "physics",
        }

        # Add our method metrics
        if our_method_metrics:
            comparison_result["our_precision"] = our_method_metrics["precision"]
            comparison_result["our_recall"] = our_method_metrics["recall"]
            comparison_result["our_f1"] = our_method_metrics["f1"]
            comparison_result["our_shd"] = our_method_metrics["shd"]
            comparison_result["our_skeleton_shd"] = our_method_metrics["skeleton_shd"]
            comparison_result["our_time"] = our_method_times.get("time", np.nan)
        else:
            comparison_result["our_precision"] = np.nan
            comparison_result["our_recall"] = np.nan
            comparison_result["our_f1"] = np.nan
            comparison_result["our_shd"] = np.nan
            comparison_result["our_skeleton_shd"] = np.nan
            comparison_result["our_time"] = np.nan

        # Add PC metrics
        if avg_metrics.get("pc"):
            comparison_result["pc_precision"] = avg_metrics["pc"]["precision"]
            comparison_result["pc_recall"] = avg_metrics["pc"]["recall"]
            comparison_result["pc_f1"] = avg_metrics["pc"]["f1"]
            comparison_result["pc_shd"] = avg_metrics["pc"]["shd"]
            comparison_result["pc_skeleton_shd"] = avg_metrics["pc"]["skeleton_shd"]
            comparison_result["pc_time"] = avg_times.get("pc", np.nan)
        else:
            comparison_result["pc_precision"] = np.nan
            comparison_result["pc_recall"] = np.nan
            comparison_result["pc_f1"] = np.nan
            comparison_result["pc_shd"] = np.nan
            comparison_result["pc_skeleton_shd"] = np.nan
            comparison_result["pc_time"] = np.nan

        # Add ad hoc metrics
        for method_name, prefix in [
            ("simple", "ad_hoc_simple_"),
            ("time", "ad_hoc_time_"),
        ]:
            if avg_metrics.get(method_name):
                comparison_result[f"{prefix}precision"] = avg_metrics[method_name][
                    "precision"
                ]
                comparison_result[f"{prefix}recall"] = avg_metrics[method_name][
                    "recall"
                ]
                comparison_result[f"{prefix}f1"] = avg_metrics[method_name]["f1"]
                comparison_result[f"{prefix}shd"] = avg_metrics[method_name]["shd"]
                comparison_result[f"{prefix}skeleton_shd"] = avg_metrics[method_name][
                    "skeleton_shd"
                ]
                comparison_result[f"{prefix}time"] = avg_times.get(method_name, np.nan)
            else:
                comparison_result[f"{prefix}precision"] = np.nan
                comparison_result[f"{prefix}recall"] = np.nan
                comparison_result[f"{prefix}f1"] = np.nan
                comparison_result[f"{prefix}shd"] = np.nan
                comparison_result[f"{prefix}skeleton_shd"] = np.nan
                comparison_result[f"{prefix}time"] = np.nan

        # Print LaTeX table
        print("\n--- LaTeX Table ---")
        print_latex_table(comparison_result, synthetic)

    elif evaluate_baselines and min_samples_per_action is None:
        print(
            f"\n⚠ Could not evaluate baselines: no sample count achieved {success_threshold*100:.0f}% threshold"
        )

    return str(scaling_file) if scaling_file else None, comparison_result


def main():
    parser = argparse.ArgumentParser(
        description="Sample scaling experiment: test how success varies with samples per object",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "level_file",
        help="Level file (e.g., t0.yaml)",
    )
    parser.add_argument(
        "--displacement",
        type=float,
        default=0.1,
        help="Displacement value (default: 0.1)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Random seeds to test (default: 42 43 44 45 46)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: 'scaling_experiments/' for physics, 'synthetic_scaling/' for synthetic)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum samples per object to start testing (default: 1)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum samples per object to test (default: 100)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum simulation steps per episode (default: 1000)",
    )
    parser.add_argument(
        "--settling-steps",
        type=int,
        default=50,
        help="Number of physics steps to let objects settle after spawn (default: 50)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data generation instead of physics simulation",
    )
    parser.add_argument(
        "--synthetic-graph",
        type=str,
        help="Path to causal graph file for synthetic mode (format: 'parent->child' per line, 1-indexed)",
    )
    parser.add_argument(
        "--synthetic-probs",
        type=str,
        default="0.1",
        help="Failure probabilities for synthetic mode: either a single float (e.g., '0.1') or comma-separated list (e.g., '0.1,0.2,0.15')",
    )

    # Baseline evaluation
    parser.add_argument(
        "--evaluate-baselines",
        action="store_true",
        help="Evaluate PC and ad hoc baseline methods at minimum sample count",
    )
    parser.add_argument(
        "--skip-pc",
        action="store_true",
        help="Skip PC algorithm baseline (useful for very large datasets)",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.95,
        help="Success threshold for baseline comparison (default: 0.95)",
    )

    args = parser.parse_args()

    # Set default output directory based on mode
    if args.output_dir is None:
        args.output_dir = (
            "synthetic_scaling" if args.synthetic else "physical_scaling_experiments"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scaling_file, comparison_result = run_scaling_experiment(
        level_file=args.level_file,
        displacement=args.displacement,
        seeds=args.seeds,
        output_dir=output_dir,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        max_steps=args.max_steps,
        settling_steps=args.settling_steps,
        synthetic=args.synthetic,
        synthetic_graph=args.synthetic_graph,
        synthetic_probs=args.synthetic_probs,
        evaluate_baselines=args.evaluate_baselines,
        skip_pc=args.skip_pc,
        success_threshold=args.success_threshold,
    )

    if scaling_file:
        print(f"\nTo plot results, use:")
        print(f'  python plot_scaling_curves.py "Method 1" {scaling_file}')

    if comparison_result:
        # Save single comparison result
        df = pd.DataFrame([comparison_result])
        comparison_csv = (
            output_dir
            / f"comparison_{Path(args.level_file).stem}_d{args.displacement}_{int(time.time())}.csv"
        )
        df.to_csv(comparison_csv, index=False)
        print(f"\nComparison result saved to: {comparison_csv}")


if __name__ == "__main__":
    main()
