#!/usr/bin/env python3
"""Main script to run Rube Goldberg simulations"""
import argparse
import time
from pathlib import Path
import random
import numpy as np

from rube.synthetic_data_generator import SyntheticDataGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Run Rube Goldberg physics simulation or synthetic data generation"
    )
    parser.add_argument(
        "level_file", help="Path to level YAML file (physics mode only)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum simulation steps per episode (physics mode only)",
    )
    parser.add_argument(
        "--displacement",
        type=float,
        default=0.1,
        help="Random displacement range for objects (physics mode only)",
    )
    parser.add_argument(
        "--settling-steps",
        type=int,
        default=0,
        help="Number of physics steps to let objects settle after spawn (physics mode only)",
    )
    parser.add_argument(
        "-M",
        "--intervention-samples",
        type=int,
        default=10,
        help="Number of intervention samples (total if random, per action if --per-action)",
    )
    parser.add_argument(
        "--per-action",
        action="store_true",
        help="Collect M samples for each action instead of M total with random actions",
    )
    parser.add_argument(
        "-K",
        "--no-intervention-samples",
        type=int,
        default=10,
        help="Number of no-intervention samples",
    )
    parser.add_argument(
        "--output-dir", default="runs", help="Output directory (default: runs/)"
    )
    parser.add_argument("--no-gif", action="store_true", help="Disable GIF generation")

    # Synthetic data generation options
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data generation instead of physics simulation",
    )
    parser.add_argument(
        "--synthetic-graph",
        type=str,
        help="Path to causal graph file (format: 'parent->child' per line, 1-indexed)",
    )
    parser.add_argument(
        "--synthetic-probs",
        type=str,
        default="0.1",
        help="Failure probabilities: either a single float (e.g., '0.1') or comma-separated list (e.g., '0.1,0.2,0.15')",
    )

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Create output directory with timestamp and level filename
    timestamp = int(time.time())
    output_dir = Path(args.output_dir) / f"{timestamp}_{Path(args.level_file).stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Choose mode: synthetic or physics
    if args.synthetic:
        print("=" * 60)
        print("SYNTHETIC DATA GENERATION MODE")
        print("=" * 60)

        # Load causal graph
        if not args.synthetic_graph:
            print("ERROR: --synthetic-graph is required in synthetic mode")
            return

        graph_path = Path(args.synthetic_graph)
        if not graph_path.exists():
            print(f"ERROR: Graph file not found: {graph_path}")
            return

        # Parse graph file (format: parent->child per line)
        causal_graph = []
        with open(graph_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and "->" in line:
                    parent, child = line.split("->")
                    causal_graph.append((int(parent), int(child)))

        # Determine number of variables
        all_nodes = set()
        for parent, child in causal_graph:
            all_nodes.add(parent)
            all_nodes.add(child)
        num_vars = max(all_nodes) if all_nodes else 0

        print(f"Loaded causal graph: {len(causal_graph)} edges")
        print(f"Number of variables: {num_vars}")

        # Parse failure probabilities
        prob_str = args.synthetic_probs
        if "," in prob_str:
            # Multiple probabilities
            failure_probs = [float(p.strip()) for p in prob_str.split(",")]
            if len(failure_probs) != num_vars:
                print(
                    f"ERROR: Expected {num_vars} probabilities, got {len(failure_probs)}"
                )
                return
            print(f"Failure probabilities: {failure_probs}")
        else:
            # Single probability
            failure_probs = float(prob_str)
            print(f"Failure probability (all variables): {failure_probs}")

        # Create synthetic data generator
        collector = SyntheticDataGenerator(
            num_vars=num_vars,
            causal_graph=causal_graph,
            failure_probs=failure_probs,
            seed=args.seed,
        )

        # Save the true causal graph as the "solution"
        collector.save_causal_graph(output_dir / "solution.txt")
        print(f"Saved true causal graph to solution.txt")

    else:
        print("=" * 60)
        print("PHYSICS SIMULATION MODE")
        print("=" * 60)

        # Import physics-related modules (only when needed)
        from rube.level_parser import LevelParser
        from rube.physics_env import PhysicsEnvironment
        from rube.data_collector import DataCollector

        # Parse level
        print(f"Loading level: {args.level_file}")
        parser_obj = LevelParser()
        level = parser_obj.parse_file(args.level_file)
        print(f"Level ID: {level.id}")
        obj_list = [obj_def for obj_def in level.objects if obj_def["type"] != "ramp"]
        print(f"Number of objects: {len(obj_list)}")

        # Create environment
        env = PhysicsEnvironment(
            level,
            displacement=args.displacement,
            seed=args.seed,
            settling_steps=args.settling_steps,
        )

        # Create data collector
        collector = DataCollector(env, max_steps=args.max_steps)

    # Save object info
    print("\nSaving object info...")
    collector.save_object_info(output_dir / "object_info.csv")

    # Create motion screenshot (physics mode only)
    if not args.synthetic:
        print("Creating motion screenshot...")
        collector.create_motion_screenshot(output_dir / "initial_motion.png")

    # Collect intervention samples (M samples)
    if args.per_action:
        print(f"\nCollecting {args.intervention_samples} samples per action...")
        intervention_samples, intervention_actions = (
            collector.collect_intervention_samples_per_action(args.intervention_samples)
        )
    else:
        print(f"\nCollecting {args.intervention_samples} intervention samples...")
        intervention_samples, intervention_actions = (
            collector.collect_intervention_samples(args.intervention_samples)
        )

    # Save intervention samples
    collector.save_samples_csv(
        intervention_samples,
        output_dir / "intervention_samples.csv",
        intervention_actions,
    )
    collector.save_samples_npy(
        intervention_samples,
        output_dir / "intervention_samples.npy",
        intervention_actions,
    )

    print(f"  Saved to intervention_samples.csv and .npy")

    if not args.no_gif and not args.synthetic:
        # Generate intervention GIFs (physics mode only)
        print("\nGenerating intervention GIF...")
        # action = random.randint(1, env.num_objects)
        for action in range(0):  # range(1, env.num_objects + 1):
            _, _, frames = collector.run_episode(action, record_frames=True)
            if frames:
                collector.renderer.save_gif(
                    frames,
                    output_dir / f"intervention_action_{action}.gif",
                    duration=50,
                )
                print(f"  Saved to intervention_action_{action}.gif")

    # Collect no-intervention samples (K samples)
    print(f"\nCollecting {args.no_intervention_samples} no-intervention samples...")
    no_intervention_samples, all_collisions = collector.collect_no_intervention_samples(
        args.no_intervention_samples
    )

    # Save no-intervention samples
    collector.save_samples_csv(
        no_intervention_samples, output_dir / "no_intervention_samples.csv"
    )
    collector.save_samples_npy(
        no_intervention_samples, output_dir / "no_intervention_samples.npy"
    )

    print(f"  Saved to no_intervention_samples.csv and .npy")

    # Generate no-intervention GIFs (physics mode only)
    if not args.no_gif and not args.synthetic:
        print("\nGenerating no-intervention GIFs...")
        for run_idx in range(0):
            binary_states, _, frames = collector.run_episode(
                action=0, record_frames=True
            )
            print(binary_states)
            if frames:
                collector.renderer.save_gif(
                    frames, output_dir / f"no_intervention_{run_idx}.gif", duration=50
                )
                print(f"  Saved to no_intervention_{run_idx}.gif")

    # Generate collision graphs (physics mode only)
    if not args.synthetic:
        print("\nGenerating collision graphs...")

        # Simple collision graph (ad_hoc_collisions.txt)
        collector.generate_collision_graph(
            all_collisions, output_dir / "ad_hoc_collisions.txt"
        )
        print("  Saved to ad_hoc_collisions.txt")

        # Collision graph with temporal info
        # Re-run to get collision data per sample with timestamps
        print("  Collecting temporal collision data...")
        samples_with_collisions = []
        for i in range(args.no_intervention_samples):
            binary_states, collisions_with_time, _ = collector.run_episode_with_time(
                action=0
            )
            samples_with_collisions.append((binary_states, collisions_with_time))

        collector.generate_collision_graph_with_time(
            samples_with_collisions, output_dir / "ad_hoc_collisions_with_time.txt"
        )
        print("  Saved to ad_hoc_collisions_with_time.txt")

    # Generate solution file if solution exists in level (physics mode only)
    if not args.synthetic and level.solution and "edges" in level.solution:
        print("\nTranslating solution to object IDs...")
        solution_edges = level.solution["edges"]
        translated_edges = []

        for edge in solution_edges:
            from_pos = tuple(edge[0])  # [row, col] -> (row, col)
            to_pos = tuple(edge[1])

            from_id = env.get_object_id_at_position(from_pos)
            to_id = env.get_object_id_at_position(to_pos)

            if from_id is not None and to_id is not None:
                translated_edges.append((from_id, to_id))
            else:
                print(
                    f"  Warning: Could not find object at position {from_pos} or {to_pos}"
                )

        # Save solution to file
        solution_file = output_dir / "solution.txt"
        with open(solution_file, "w") as f:
            for from_id, to_id in translated_edges:
                f.write(f"{from_id}->{to_id}\n")

        print(f"  Saved solution to solution.txt ({len(translated_edges)} edges)")

    print(f"\n✓ All outputs saved to: {output_dir}")

    if args.synthetic:
        print("\nSynthetic data generation complete!")
        print(f"  True causal graph: {output_dir}/solution.txt")
    else:
        print("\nTo plot graphs, use:")
        print(
            f"  python plot_graph.py {output_dir}/ad_hoc_collisions.txt --info {output_dir}/object_info.csv --level {level.id}"
        )


if __name__ == "__main__":
    main()
