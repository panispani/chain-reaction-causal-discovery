#!/usr/bin/env python3
"""
Plot scaling curves from multiple experiment files.

Generates 4 plots:
- Success probability vs samples per object (linear scale)
- Success probability vs samples per object (log scale)
- Skeleton SHD vs samples per object (linear scale)
- Skeleton SHD vs samples per object (log scale)
"""
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np


def plot_scaling_curves(data_files, labels, output_prefix="scaling_plot"):
    """
    Plot scaling curves from multiple data files.

    Args:
        data_files: List of paths to CSV files
        labels: List of labels for each file (for legend)
        output_prefix: Prefix for output plot files
    """
    if len(data_files) != len(labels):
        raise ValueError("Number of data files must match number of labels")

    # Load all data
    all_data = []
    for file_path, label in zip(data_files, labels):
        df = pd.read_csv(file_path)
        all_data.append((df, label))
        print(f"Loaded {file_path}: {len(df)} data points")

    # Create 4 plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sample Scaling Analysis", fontsize=16, fontweight="bold")

    # Colors for different curves
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))

    # Plot 1: Success probability vs samples (linear scale)
    ax = axes[0, 0]
    for (df, label), color in zip(all_data, colors):
        ax.plot(
            df["samples_per_object"],
            df["success_probability"] * 100,
            marker="o",
            label=label,
            color=color,
            linewidth=2,
            markersize=6,
        )
    ax.set_xlabel("Samples per Object", fontsize=12)
    ax.set_ylabel("Success Probability (%)", fontsize=12)
    ax.set_title("Success Probability vs Samples", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim(-5, 105)

    # Plot 2: Success probability vs samples (log scale)
    ax = axes[0, 1]
    for (df, label), color in zip(all_data, colors):
        ax.plot(
            df["samples_per_object"],
            df["success_probability"] * 100,
            marker="o",
            label=label,
            color=color,
            linewidth=2,
            markersize=6,
        )
    ax.set_xlabel("Samples per Object (log scale)", fontsize=12)
    ax.set_ylabel("Success Probability (%)", fontsize=12)
    ax.set_title("Success Probability vs Samples (Log Scale)", fontsize=13)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10)
    ax.set_ylim(-5, 105)

    # Plot 3: Skeleton SHD vs samples (linear scale)
    ax = axes[1, 0]
    for (df, label), color in zip(all_data, colors):
        ax.plot(
            df["samples_per_object"],
            df["avg_skeleton_shd"],
            marker="o",
            label=label,
            color=color,
            linewidth=2,
            markersize=6,
        )
    ax.set_xlabel("Samples per Object", fontsize=12)
    ax.set_ylabel("Average Skeleton SHD", fontsize=12)
    ax.set_title("Skeleton SHD vs Samples", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Plot 4: Skeleton SHD vs samples (log scale)
    ax = axes[1, 1]
    for (df, label), color in zip(all_data, colors):
        ax.plot(
            df["samples_per_object"],
            df["avg_skeleton_shd"],
            marker="o",
            label=label,
            color=color,
            linewidth=2,
            markersize=6,
        )
    ax.set_xlabel("Samples per Object (log scale)", fontsize=12)
    ax.set_ylabel("Average Skeleton SHD", fontsize=12)
    ax.set_title("Skeleton SHD vs Samples (Log Scale)", fontsize=13)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10)

    plt.tight_layout()

    # Save the plot
    output_file = f"{output_prefix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n✓ Plots saved to: {output_file}")

    # Also save as PDF
    output_pdf = f"{output_prefix}.pdf"
    plt.savefig(output_pdf, bbox_inches="tight")
    print(f"✓ PDF saved to: {output_pdf}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot scaling curves from multiple experiment files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python plot_scaling_curves.py "Method A" results_a.csv "Method B" results_b.csv

  python plot_scaling_curves.py \\
    "Displacement 0.1" t0_d0.1_1234567890.csv \\
    "Displacement 0.2" t0_d0.2_1234567891.csv \\
    --output scaling_comparison
        """,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Alternating labels and file paths: label1 file1 label2 file2 ...",
    )
    parser.add_argument(
        "--output",
        default="scaling_plot",
        help="Output file prefix (default: scaling_plot)",
    )

    args = parser.parse_args()

    # Parse alternating labels and files
    if len(args.inputs) % 2 != 0:
        parser.error(
            "Must provide an even number of arguments (label file pairs). "
            "Got: " + " ".join(args.inputs)
        )

    labels = []
    files = []
    for i in range(0, len(args.inputs), 2):
        labels.append(args.inputs[i])
        files.append(args.inputs[i + 1])

    # Verify all files exist
    for file_path in files:
        if not Path(file_path).exists():
            parser.error(f"File not found: {file_path}")

    print(f"Plotting {len(files)} curves:")
    for label, file_path in zip(labels, files):
        print(f"  - {label}: {file_path}")
    print()

    plot_scaling_curves(files, labels, output_prefix=args.output)


if __name__ == "__main__":
    main()
