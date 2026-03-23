from typing import List, Tuple, Dict, Any
import argparse
import math
import os
import json

Edge = Tuple[int, int]


def evaluate_graphs(true_edges: List[Edge], pred_edges: List[Edge]) -> Dict[str, Any]:
    """
    Evaluate a learned directed graph against a true directed graph.

    Args:
        true_edges: list of (u, v) edges in the ground-truth graph G.
        pred_edges: list of (u, v) edges in the learned graph G^.

    Returns:
        Dictionary with metrics and components (TP, FP, FN, SHD, skeleton_SHD, etc.)
        and prints them in a table.
    """
    true_set = set(true_edges)
    pred_set = set(pred_edges)

    # Precision / Recall / F1 (directed, parent-child edges)
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if tp + fp > 0 else math.nan
    recall = tp / (tp + fn) if tp + fn > 0 else math.nan
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        f1 = math.nan
    else:
        f1 = 2 * precision * recall / (precision + recall)

    # SHD (directed) with reversals counted as 1 operation
    # Edges present only in one graph
    true_only = true_set - pred_set
    pred_only = pred_set - true_set

    # Reversals: (u, v) in true, (v, u) in pred
    reversal_pairs = set()
    for u, v in true_set:
        if (v, u) in pred_set:
            reversal_pairs.add(frozenset((u, v)))

    num_reversals = len(reversal_pairs)
    # Each reversal contributes one edge to true_only and one to pred_only.
    # If we count reversals as a single operation, SHD = |true_only| + |pred_only| - |reversals|
    shd = len(true_only) + len(pred_only) - num_reversals

    # Skeleton SHD (undirected)
    def to_skeleton(edges: set[Edge]):
        skel = set()
        for u, v in edges:
            if u == v:
                continue
            skel.add(frozenset((u, v)))
        return skel

    true_skel = to_skeleton(true_set)
    pred_skel = to_skeleton(pred_set)

    skel_fp = len(pred_skel - true_skel)
    skel_fn = len(true_skel - pred_skel)
    skeleton_shd = skel_fp + skel_fn

    # Collect results
    results = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "shd": shd,
        "num_reversals": num_reversals,
        "skeleton_shd": skeleton_shd,
        "skeleton_fp": skel_fp,
        "skeleton_fn": skel_fn,
    }

    # Pretty print as a table
    def fmt(x):
        if isinstance(x, float):
            if math.isnan(x):
                return "nan"
            return f"{x:.3f}"
        return str(x)

    rows = [
        ("TP", results["tp"], ""),
        ("FP", results["fp"], ""),
        ("FN", results["fn"], ""),
        ("Precision", fmt(results["precision"]), "TP / (TP + FP)"),
        ("Recall", fmt(results["recall"]), "TP / (TP + FN)"),
        ("F1", fmt(results["f1"]), "2PR / (P + R)"),
        ("SHD", results["shd"], "additions + deletions + reversals (1 op)"),
        ("Reversals", results["num_reversals"], "counted as 1 operation each"),
        ("Skeleton SHD", results["skeleton_shd"], "FP_skel + FN_skel"),
        ("Skeleton FP", results["skeleton_fp"], "edges only in learned skeleton"),
        ("Skeleton FN", results["skeleton_fn"], "edges only in true skeleton"),
    ]

    col1 = "Metric"
    col2 = "Value"
    col3 = "Definition"
    width1 = max(len(col1), max(len(r[0]) for r in rows))
    width2 = max(len(col2), max(len(str(r[1])) for r in rows))
    width3 = max(len(col3), max(len(r[2]) for r in rows))

    header = f"{col1:<{width1}}  {col2:>{width2}}  {col3:<{width3}}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, val, desc in rows:
        print(f"{name:<{width1}}  {str(val):>{width2}}  {desc:<{width3}}")
    print(sep)

    return results


def load_edges_from_file(filepath: str) -> List[Edge]:
    """
    Load edges from a text file.

    Expected format: one edge per line as "src->dst" (1-based indices)
    Example: "3->6" means edge from node 3 to node 6

    Args:
        filepath: Path to edge file

    Returns:
        List of (src, dst) tuples with 1-based indices
    """
    edges = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "->" not in line:
                continue
            src, dst = line.split("->")
            edges.append((int(src), int(dst)))
    return edges


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predicted causal graph against ground truth graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "true_graph",
        help="Path to true/ground truth graph file (format: src->dst per line)",
    )
    parser.add_argument(
        "pred_graph", help="Path to predicted graph file (format: src->dst per line)"
    )

    args = parser.parse_args()

    # Load graphs
    print(f"Loading true graph from: {args.true_graph}")
    true_edges = load_edges_from_file(args.true_graph)
    print(f"  - {len(true_edges)} edges")

    print(f"\nLoading predicted graph from: {args.pred_graph}")
    pred_edges = load_edges_from_file(args.pred_graph)
    print(f"  - {len(pred_edges)} edges")

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70 + "\n")

    results = evaluate_graphs(true_edges, pred_edges)

    # Save results to file in JSON format
    pred_graph_path = args.pred_graph
    output_path = os.path.splitext(pred_graph_path)[0] + ".eval"

    output_data = {
        "true_graph": args.true_graph,
        "pred_graph": args.pred_graph,
        "true_edges_count": len(true_edges),
        "pred_edges_count": len(pred_edges),
        "metrics": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
