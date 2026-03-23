"""PC algorithm baseline for causal discovery evaluation"""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np

# Import causal-learn to use PC
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz


def load_edges_from_file(filepath: str):
    """Load edges from file in format: parent->child"""
    edges = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "->" not in line:
                continue
            src, dst = line.split("->")
            edges.append((int(src), int(dst)))
    return edges


def run_pc_algorithm(
    observational_csv: Path,
    true_graph: Path,
    alpha: float = 0.05,
) -> Optional[Dict]:
    """
    Run PC algorithm on observational data and evaluate against true graph.

    PC returns a CPDAG (completed partially directed acyclic graph) which may
    have both directed and undirected edges.

    Args:
        observational_csv: Path to observational samples CSV
        true_graph: Path to true graph file
        alpha: Significance level for conditional independence tests

    Returns:
        Dictionary with evaluation metrics (skeleton-based since PC returns CPDAG)
    """
    if not observational_csv.exists():
        return None

    # Load observational data
    df = pd.read_csv(observational_csv)
    data = df.to_numpy()

    # Run PC algorithm
    # Use fisherz for continuous independence test
    # (even though our data is binary, PC expects continuous by default)
    try:
        cg = pc(data, alpha=alpha, indep_test=fisherz)

        # Extract graph structure from PC result
        # cg.G.graph[i,j] = -1 means i --> j
        # cg.G.graph[i,j] = 1 means i <-- j  (same as j --> i)
        # cg.G.graph[i,j] = -1 and cg.G.graph[j,i] = -1 means i --- j (undirected)
        # cg.G.graph[i,j] = 0 means no edge

        G = cg.G.graph
        N = G.shape[0]

        # Extract edges (both directed and undirected)
        # For skeleton comparison, we just care about presence of edges
        pred_skeleton = set()
        pred_directed_edges = []

        for i in range(N):
            for j in range(i + 1, N):
                if G[i, j] == -1 or G[j, i] == -1:
                    # There is some edge between i and j
                    pred_skeleton.add(frozenset((i + 1, j + 1)))  # Convert to 1-indexed

                    # Check if directed
                    if G[i, j] == -1 and G[j, i] == 0:
                        # i --> j
                        pred_directed_edges.append((i + 1, j + 1))
                    elif G[j, i] == -1 and G[i, j] == 0:
                        # j --> i
                        pred_directed_edges.append((j + 1, i + 1))
                    # If both -1, it's undirected, don't add to directed edges

        # Load true graph
        true_edges = load_edges_from_file(str(true_graph))
        true_set = set(true_edges)

        # Compute skeleton metrics
        def to_skeleton(edges):
            skel = set()
            for u, v in edges:
                if u != v:
                    skel.add(frozenset((u, v)))
            return skel

        true_skel = to_skeleton(true_set)

        # Skeleton metrics
        skeleton_tp = len(true_skel & pred_skeleton)
        skeleton_fp = len(pred_skeleton - true_skel)
        skeleton_fn = len(true_skel - pred_skeleton)

        skeleton_precision = (
            skeleton_tp / (skeleton_tp + skeleton_fp)
            if skeleton_tp + skeleton_fp > 0
            else 0.0
        )
        skeleton_recall = (
            skeleton_tp / (skeleton_tp + skeleton_fn)
            if skeleton_tp + skeleton_fn > 0
            else 0.0
        )
        skeleton_f1 = (
            2
            * skeleton_precision
            * skeleton_recall
            / (skeleton_precision + skeleton_recall)
            if skeleton_precision + skeleton_recall > 0
            else 0.0
        )
        skeleton_shd = skeleton_fp + skeleton_fn

        # Also compute directed edge metrics (for edges that PC oriented)
        pred_directed_set = set(pred_directed_edges)
        directed_tp = len(true_set & pred_directed_set)
        directed_fp = len(pred_directed_set - true_set)
        directed_fn = len(true_set - pred_directed_set)

        directed_precision = (
            directed_tp / (directed_tp + directed_fp)
            if directed_tp + directed_fp > 0
            else 0.0
        )
        directed_recall = (
            directed_tp / (directed_tp + directed_fn)
            if directed_tp + directed_fn > 0
            else 0.0
        )
        directed_f1 = (
            2
            * directed_precision
            * directed_recall
            / (directed_precision + directed_recall)
            if directed_precision + directed_recall > 0
            else 0.0
        )

        # Compute SHD (Structural Hamming Distance) for directed edges
        # This counts: missing edges + extra edges + reversed edges
        true_only = true_set - pred_directed_set
        pred_only = pred_directed_set - true_set

        # Find reversed edges
        reversal_pairs = set()
        for u, v in true_set:
            if (v, u) in pred_directed_set:
                reversal_pairs.add(frozenset((u, v)))
        num_reversals = len(reversal_pairs)

        # SHD = |true \ pred| + |pred \ true| - |reversals|
        # (reversals are counted in both true_only and pred_only, so we subtract once)
        shd = len(true_only) + len(pred_only) - num_reversals

        return {
            # Skeleton metrics (most relevant for CPDAG)
            "precision": skeleton_precision,
            "recall": skeleton_recall,
            "f1": skeleton_f1,
            "skeleton_shd": skeleton_shd,
            # Directed edge metrics (only for edges PC oriented)
            "directed_precision": directed_precision,
            "directed_recall": directed_recall,
            "directed_f1": directed_f1,
            "shd": shd,  # Regular SHD for directed edges
            # Counts for debugging
            "num_edges": len(pred_skeleton),
            "num_directed": len(pred_directed_edges),
            "num_undirected": len(pred_skeleton) - len(pred_directed_edges),
        }

    except Exception as e:
        print(f"    PC algorithm failed: {e}")
        return None
