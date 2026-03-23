import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from eval import evaluate_graphs

import numpy as np
import pandas as pd


def load_intervention_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load interventional data from CSV.

    Expected format:
        intervened_variable,object_1,object_2,...,object_N

    Args:
        csv_path: Path to intervention CSV file

    Returns:
        I: Array of shape (M,) with 0-based intervention indices
        X: Array of shape (M, N) with binary observations
    """
    df = pd.read_csv(csv_path)

    if df.shape[1] < 2:
        raise ValueError(
            "Interventional CSV must have at least 2 columns: "
            "[intervened_variable, object_1, ...]"
        )

    # Extract intervention indices (1-based in CSV, convert to 0-based)
    I_raw = df.iloc[:, 0].to_numpy()
    if not np.issubdtype(I_raw.dtype, np.integer):
        raise ValueError("intervened_variable column must contain integers 1..N")

    I = I_raw.astype(int) - 1

    # Extract binary observations
    X = df.iloc[:, 1:].to_numpy()
    N = X.shape[1]

    # Validate intervention indices
    if np.any(I < 0) or np.any(I >= N):
        raise ValueError(
            f"intervened_variable must be between 1 and N "
            f"(got values in [{I.min()+1}, {I.max()+1}], N={N})"
        )

    return I, X


def load_observational_data(csv_path: str, expected_N: int) -> np.ndarray:
    """
    Load observational data from CSV.

    Expected format:
        object_1,object_2,...,object_N

    Args:
        csv_path: Path to observational CSV file
        expected_N: Expected number of variables

    Returns:
        X_obs: Array of shape (M_obs, N) with binary observations
    """
    df = pd.read_csv(csv_path)

    if df.shape[1] != expected_N:
        raise ValueError(
            f"Observational CSV must have {expected_N} columns, got {df.shape[1]}"
        )

    return df.to_numpy()


def estimate_intervention_probabilities(
    I: np.ndarray, X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate p_ij = Pr(X_j=1 | do(X_i=0)) for all pairs (i,j).

    This is:
        p_hat_ij = (1/n_i) * sum_{e: I_e=i} 1{X_j^(e) = 1}
        (proportion of times where X_j is 1 when we intervene X_i to 0)

    Args:
        I: Intervention indices, shape (M,)
        X: Binary observations, shape (M, N)

    Returns:
        p_hat: Matrix of shape (N, N) with estimated probabilities
        n_i: Array of shape (N,) with intervention counts per node
    """
    M, N = X.shape
    p_hat = np.zeros((N, N), dtype=float)
    n_i = np.zeros(N, dtype=int)

    for i in range(N):
        # Get all samples where we intervened on node i
        mask = I == i
        n = mask.sum()
        n_i[i] = n

        if n == 0:
            # No interventions on node i, cannot estimate this row
            continue

        # Empirical probability: fraction of times X_j=1 when intervening on i
        X_sub = X[mask]  # shape (n, N)
        p_hat[i, :] = X_sub.mean(axis=0)

        # Node is never its own descendant (probably not strictly needed)
        p_hat[i, i] = 0.0

    return p_hat, n_i


def build_ancestor_matrix(p_hat: np.ndarray, n_i: np.ndarray) -> np.ndarray:
    """
    Build ancestor matrix A where A[i,j]=1 if j is a descendant of i.

    Uses threshold τ_ij = 1/n_i:
        A(i,j) = 1{p_hat_ij < 1/n_i}
    or:
        A(i,j) = 1{p_hat_ij == 0}
    These should be equivalent.

    Rationale: If X_j = 1 at least once among n_i samples under do(X_i=0),
    then p_hat_ij ≥ 1/n_i, which proves j is NOT a descendant (by deterministic
    cascade property). We use strict < to implement "never observed X_j=1".

    Under the monotone-AND model, true descendants have p_ij = 0 (forced to 0).

    Args:
        p_hat: Estimated intervention probabilities, shape (N, N)
        n_i: Intervention counts per node, shape (N,)

    Returns:
        A: Ancestor matrix, shape (N, N)
    """
    N = p_hat.shape[0]
    A = np.zeros((N, N), dtype=int)
    A2 = np.zeros((N, N), dtype=int)

    for i in range(N):
        if n_i[i] == 0:
            continue

        # Threshold: if p_hat_ij < 1/n_i, classify j as descendant
        # Note: We use strict < because seeing X_j=1 even once when knocking
        # out i proves j is NOT a descendant (deterministic cascade property)
        tau_i = 1.0 / n_i[i]
        A[i, :] = (p_hat[i, :] < tau_i).astype(int)
        A2[i, :] = (p_hat[i, :] == 0).astype(int)

        # Enforce zero diagonal (node is not its own ancestor)
        A[i, i] = 0
        A2[i, i] = 0

    # Verify that both thresholdings produce identical results
    # (Since p_hat values are multiples of 1/n_i, the only value < 1/n_i is 0)
    # TODO: delete before release
    np.testing.assert_array_equal(
        A, A2, err_msg="A and A2 should be equal since p_hat takes discrete values"
    )

    return A


# Possible when i->j and in all the samples where j = 0, it so happened that i = 0 too.
def resolve_contradictions(
    A: np.ndarray, p_hat: np.ndarray, X_obs: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Resolve contradictory pairs where both A[i,j]=1 and A[j,i]=1.

    Strategy:
    1. If observational data available: check for co-occurrence patterns
       - If X_j=1 and X_i=0 co-occur, then j is NOT descendant of i
    2. Otherwise: keep direction with smaller p_hat (closer to true 0)
    3. Tie-breaking: prefer smaller index as ancestor (deterministic)

    Args:
        A: Ancestor matrix, shape (N, N)
        p_hat: Estimated intervention probabilities, shape (N, N)
        X_obs: Optional observational data, shape (M_obs, N)

    Returns:
        A_resolved: Resolved ancestor matrix, shape (N, N)
    """
    A_resolved = A.copy()
    N = A.shape[0]

    def resolve_pair(i: int, j: int) -> None:
        """Resolve contradictory pair (i,j) where both claim ancestry."""
        nonlocal A_resolved

        # Strategy 1: Use observational data if available
        if X_obs is not None:
            Xi = X_obs[:, i]
            Xj = X_obs[:, j]

            # Check if j can be active when i is not (j not descendant of i)
            j_not_desc_of_i = np.any((Xi == 0) & (Xj == 1))

            # Check if i can be active when j is not (i not descendant of j)
            i_not_desc_of_j = np.any((Xj == 0) & (Xi == 1))

            if j_not_desc_of_i and not i_not_desc_of_j:
                A_resolved[i, j] = 0  # j is NOT descendant of i
                return
            if i_not_desc_of_j and not j_not_desc_of_i:
                A_resolved[j, i] = 0  # i is NOT descendant of j
                return
            # If both or neither observed, fall through to p_hat comparison

        # Strategy 2: Compare p_hat values (keep smaller, closer to true 0)
        p_ij = p_hat[i, j]
        p_ji = p_hat[j, i]

        if p_ij < p_ji:
            A_resolved[j, i] = 0  # Keep i->j, remove j->i
        elif p_ji < p_ij:
            A_resolved[i, j] = 0  # Keep j->i, remove i->j
        else:
            # Strategy 3: Tie-break deterministically (smaller index wins)
            A_resolved[j, i] = 0  # Prefer i->j when i < j

    # Check all pairs for contradictions
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] == 1 and A[j, i] == 1:
                resolve_pair(i, j)

    return A_resolved


def transitive_reduction(
    A: np.ndarray, p_hat: np.ndarray, force_tree: bool = False, verbose: bool = True
) -> np.ndarray:
    """
    Compute transitive reduction of ancestor matrix to get direct parent edges.

    The idea is to remove transitive edges and keep direct parents:
    For each i->j edge, check if there exists k such that i->k->j. If so,
    remove i->j since it's transitive.

    Note: Transitive reduction does NOT guarantee a tree structure - nodes may
    have multiple parents. If force_tree=True, we enforce the tree constraint
    by keeping only the parent with smallest p_hat (most confident relationship).

    Args:
        A: Ancestor matrix where A[i,j]=1 means i is ancestor of j
        p_hat: Estimated intervention probabilities, shape (N, N)
        force_tree: If True, enforce tree constraint (at most one parent per node)
        verbose: If True, report when multiple parents are found

    Returns:
        P: Parent matrix where P[i,j]=1 means i is parent of j
    """
    N = A.shape[0]
    P = np.zeros((N, N), dtype=int)

    # Step 1: Remove transitive edges
    for i in range(N):
        for j in range(N):
            if i == j or A[i, j] == 0:
                continue

            # Check if there's an intermediate node k with i->k->j
            has_intermediate = False
            for k in range(N):
                if k == i or k == j:
                    continue
                if A[i, k] == 1 and A[k, j] == 1:
                    has_intermediate = True
                    break

            # If no intermediate path, this is a direct parent edge
            if not has_intermediate:
                P[i, j] = 1

    # Step 2: Check for multiple parents (tree constraint violation)
    multi_parent_nodes = []
    for j in range(N):
        parents = np.where(P[:, j] == 1)[0]

        if len(parents) > 1:
            multi_parent_nodes.append((j, parents))

            if verbose:
                p_vals = [p_hat[p, j] for p in parents]
                print(
                    f"  WARNING: object_{j+1} has {len(parents)} candidate parents: "
                    f"{[f'object_{p+1}' for p in parents]}"
                )
                print(f"           p_hat values: {[f'{pv:.4f}' for pv in p_vals]}")

            if force_tree:
                # Keep parent with smallest p_hat (most confident about descendancy)
                best_idx = int(np.argmin([p_hat[p, j] for p in parents]))
                best_parent = parents[best_idx]

                if verbose:
                    print(
                        f"           Keeping object_{best_parent+1} (smallest p_hat={p_hat[best_parent, j]:.4f})"
                    )

                # Remove all other parents
                for p in parents:
                    if p != best_parent:
                        P[p, j] = 0

    if multi_parent_nodes and verbose:
        if force_tree:
            print(
                f"  Enforced tree constraint for {len(multi_parent_nodes)} nodes with multiple parents"
            )
        else:
            print(
                f"  Data inconsistent with tree structure: {len(multi_parent_nodes)} nodes have multiple parents"
            )
            print("  Use --force-tree to resolve ambiguities")

    return P


def reconstruct_tree(
    intervention_csv: str,
    observational_csv: Optional[str] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    force_tree: bool = False,
    true_graph: List[tuple] = None,
):
    """
    Main algorithm:
    1. Load interventional (and optional observational) data
    2. Estimate p_ij
    3. Build ancestor matrix
    4. Resolve contradictions
    5. Compute transitive reduction to get parent edges
    6. Save and return results

    Args:
        intervention_csv: Path to intervention data CSV
        observational_csv: Optional path to observational data CSV
        output_dir: Optional output directory (defaults to intervention CSV dir)
        verbose: Whether to print progress information
        force_tree: If True, enforce tree constraint when multiple parents exist
    """
    if verbose:
        print("=" * 70)
        print("Knock-Out Ancestral Tree Reconstruction")
        print("=" * 70)

    # Step 0: Load data
    if verbose:
        print(f"\nLoading intervention data from: {intervention_csv}")
    I, X_int = load_intervention_data(intervention_csv)
    M, N = X_int.shape

    if verbose:
        print(f"  - {M} intervention samples")
        print(f"  - {N} nodes")

    X_obs = None
    if observational_csv and Path(observational_csv).exists():
        if verbose:
            print(f"\nLoading observational data from: {observational_csv}")
        X_obs = load_observational_data(observational_csv, N)
        if verbose:
            print(f"  - {len(X_obs)} observational samples")

    # Step 1: Estimate p_ij
    if verbose:
        print("\nStep 1: Estimating intervention probabilities p_ij...")
    p_hat, n_i = estimate_intervention_probabilities(I, X_int)

    if verbose:
        print(f"  - Intervention counts per node: {dict(enumerate(n_i))}")
        nodes_without_interventions = np.where(n_i == 0)[0]
        if len(nodes_without_interventions) > 0:
            print(
                f"  - WARNING: No interventions for nodes: {nodes_without_interventions.tolist()}"
            )

    # Step 2: Build ancestor matrix A
    if verbose:
        print("\nStep 2: Building ancestor matrix A using threshold τ_ij = 1/n_i...")
    A = build_ancestor_matrix(p_hat, n_i)

    if verbose:
        n_ancestor_pairs = A.sum()
        print(f"  - {n_ancestor_pairs} ancestor-descendant pairs detected")
        print(f"  - Ancestor matrix A:\n{A}")

    # Step 3: Resolve contradictions
    if verbose:
        print("\nStep 3: Resolving contradictory pairs...")
    A_resolved = resolve_contradictions(A, p_hat, X_obs)

    n_contradictions = 0
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] == 1 and A[j, i] == 1:
                n_contradictions += 1

    if verbose:
        print(f"  - {n_contradictions} contradictory pairs found and resolved")
        if n_contradictions > 0:
            print(f"  - Resolved ancestor matrix:\n{A_resolved}")

    # Step 4: Transitive reduction
    if verbose:
        print("\nStep 4: Computing transitive reduction to obtain parent edges...")
    P = transitive_reduction(A_resolved, p_hat, force_tree=force_tree, verbose=verbose)

    if verbose:
        n_edges = P.sum()
        structure_type = "tree" if force_tree else "DAG"
        print(f"  - {n_edges} parent-child edges in reconstructed {structure_type}")
        print(f"  - Parent matrix P:\n{P}")

    # Extract edge list
    edges = []
    for i in range(N):
        for j in range(N):
            if P[i, j] == 1:
                edges.append((i + 1, j + 1))  # Convert to 1-based for output

    if verbose:
        print(f"\nReconstructed tree edges:")
        for parent, child in edges:
            print(f"  - object_{parent} -> object_{child}")

    # Compute empirical q_min (for validation/analysis)
    q_vals = []
    for i in range(N):
        for j in range(N):
            if i != j and A_resolved[i, j] == 0:
                # (i,j) is a non-descendant pair
                q_vals.append(p_hat[i, j])
    q_hat_min = float(np.min(q_vals)) if q_vals else None

    if verbose and q_hat_min is not None:
        print(
            f"\nEmpirical q_min (smallest non-descendant activation): {q_hat_min:.4f}"
        )
        if q_hat_min > 0:
            print(
                f"  (Sample complexity scales as 1/q_min * log(N^2) = {np.log(N**2)/q_hat_min:.1f})"
            )
        else:
            print(
                "  (Note: q_min = 0 suggests possible descendant misclassification or insufficient data)"
            )

    # Save results
    if output_dir is None:
        output_dir = Path(intervention_csv).parent / "cd"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Main output: edge_predictions.txt
    pred_file = output_dir / "edge_predictions.txt"
    with open(pred_file, "w") as f:
        for parent, child in edges:
            f.write(f"{parent}->{child}\n")

    if verbose:
        print(f"\n{'='*70}")
        print(f"Saved edge predictions to: {pred_file}")

    # Detailed output subdirectory
    detail_dir = output_dir / "causal_discovery_output"
    detail_dir.mkdir(exist_ok=True)

    # Save matrices
    np.savetxt(detail_dir / "ancestor_matrix.txt", A, fmt="%d")
    np.savetxt(detail_dir / "resolved_ancestor_matrix.txt", A_resolved, fmt="%d")
    np.savetxt(detail_dir / "parent_matrix.txt", P, fmt="%d")

    # Save p_ij values
    with open(detail_dir / "intervention_probabilities.txt", "w") as f:
        f.write("i,j,p_ij,threshold_tau_ij\n")
        for i in range(N):
            tau_i = 1.0 / n_i[i] if n_i[i] > 0 else np.inf
            for j in range(N):
                if i != j:
                    f.write(f"{i+1},{j+1},{p_hat[i,j]:.6f},{tau_i:.6f}\n")

    # Save intervention counts
    with open(detail_dir / "intervention_counts.txt", "w") as f:
        f.write("node,n_interventions\n")
        for i in range(N):
            f.write(f"object_{i+1},{n_i[i]}\n")

    # Save edges in multiple formats
    with open(detail_dir / "edges.txt", "w") as f:
        for parent, child in edges:
            f.write(f"object_{parent}->object_{child}\n")

    with open(detail_dir / "adjacency_list.txt", "w") as f:
        for i in range(1, N + 1):
            children = [j for (p, j) in edges if p == i]
            children_str = ", ".join([f"object_{c}" for c in children])
            f.write(f"object_{i}: {children_str if children_str else '(leaf)'}\n")

    # Save summary statistics
    with open(detail_dir / "summary.txt", "w") as f:
        f.write("Tree Reconstruction Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of nodes: {N}\n")
        f.write(f"Intervention samples: {M}\n")
        f.write(f"Observational samples: {len(X_obs) if X_obs is not None else 0}\n\n")
        f.write(f"Intervention counts: {dict(enumerate(n_i))}\n\n")
        f.write(f"Ancestor pairs detected: {A.sum()}\n")
        f.write(f"Contradictions resolved: {n_contradictions}\n")
        f.write(f"Final tree edges: {len(edges)}\n\n")
        if q_hat_min is not None:
            f.write(f"Empirical q_min: {q_hat_min:.6f}\n")
            if q_hat_min > 0:
                f.write(
                    f"Theoretical sample complexity: {np.log(N**2)/q_hat_min:.1f} per node\n\n"
                )
            else:
                f.write("Theoretical sample complexity: undefined (q_min = 0)\n\n")
        f.write("Reconstructed edges:\n")
        for parent, child in edges:
            f.write(f"  object_{parent} -> object_{child}\n")

    if verbose:
        print(f"Saved detailed output to: {detail_dir}")
        print("=" * 70)

    if true_graph:
        evaluate_graphs(true_graph, edges)


def main():
    parser = argparse.ArgumentParser(
        description="Causal Discovery using Monotone-AND Structural Models\n"
        "Implements Algorithm 1 from method.sty",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "intervention_csv",
        help="Path to intervention samples CSV file\n"
        "Format: intervened_variable,object_1,object_2,...,object_N",
    )
    parser.add_argument(
        "--observational-csv",
        help="Optional path to observational samples CSV file\n"
        "Format: object_1,object_2,...,object_N",
    )
    parser.add_argument(
        "--true-graph",
        help="Optional path to true graph (to compute evaluation metrics at the end)",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory (defaults to intervention CSV directory)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument(
        "--force-tree",
        action="store_true",
        help="Enforce tree constraint: resolve multiple parents by keeping smallest p_hat",
    )

    args = parser.parse_args()

    true_graph = None
    if args.true_graph:
        true_graph = []
        with open(args.true_graph) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                src, dst = line.split("->")
                true_graph.append((int(src), int(dst)))

    reconstruct_tree(
        args.intervention_csv,
        args.observational_csv,
        args.output_dir,
        verbose=not args.quiet,
        force_tree=args.force_tree,
        true_graph=true_graph,
    )


if __name__ == "__main__":
    main()
