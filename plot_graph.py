#!/usr/bin/env python3
"""Script to plot graphs from edge specifications"""
import argparse
import csv
import time
from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np


def parse_edges(edge_file: str):
    """
    Parse edge file with format:
    1->2
    2->3

    Returns list of (source, target) tuples
    """
    edges = []
    with open(edge_file, "r") as f:
        for line in f:
            line = line.strip()
            if "->" in line:
                source, target = line.split("->")
                edges.append((int(source), int(target)))
    return edges


def load_object_info(info_file: str):
    """Load object info (id, type, color) from CSV"""
    obj_info = {}
    with open(info_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj_id = int(row["id"])
            obj_type = row["type"]
            color_r = int(row["color_r"]) / 255.0
            color_g = int(row["color_g"]) / 255.0
            color_b = int(row["color_b"]) / 255.0
            obj_info[obj_id] = {"type": obj_type, "color": (color_r, color_g, color_b)}
    return obj_info


def draw_ball(ax, x, y, radius, color):
    """Draw a ball with stripes like in the simulation"""
    # Draw circle
    circle = patches.Circle(
        (x, y), radius, facecolor=color, edgecolor="black", linewidth=2
    )
    ax.add_patch(circle)

    # Add perpendicular diameter lines (stripes)
    # Horizontal diameter
    ax.plot([x - radius * 0.85, x + radius * 0.85], [y, y], "k-", linewidth=2)
    # Vertical diameter
    ax.plot([x, x], [y - radius * 0.85, y + radius * 0.85], "k-", linewidth=2)


def draw_domino(ax, x, y, width, height, color):
    """Draw a domino as a rectangle"""
    rect = patches.Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        facecolor=color,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(rect)


def draw_button(ax, x, y, width, height, color):
    """Draw a button as a rectangle with a border"""
    rect = patches.Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        facecolor=color,
        edgecolor="black",
        linewidth=2.5,
    )
    ax.add_patch(rect)


def draw_label_background(ax, x, y, text, fontsize):
    """Draw a simple white rectangle background for text like in renderer"""
    # Estimate text size (approximate based on fontsize)
    # Average character is about 0.6 times fontsize in width
    text_width = len(text) * fontsize * 0.006  # Scale factor for data coordinates
    text_height = fontsize * 0.01  # Scale factor for data coordinates

    # Add small padding
    pad = 0.005

    # Draw white rectangle
    rect = patches.Rectangle(
        (x - text_width / 2 - pad, y - text_height / 2 - pad),
        text_width + 2 * pad,
        text_height + 2 * pad,
        facecolor="white",
        edgecolor="none",
        zorder=9,
    )
    ax.add_patch(rect)


def plot_graph(edges, obj_info, level_name: str, output_file: str):
    """
    Plot directed graph with custom shapes for each object type

    Args:
        edges: List of (source, target) tuples
        obj_info: Dictionary of object info
        level_name: Name of the level
        output_file: Output filename
    """
    # Create directed graph
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Add isolated nodes (objects that don't appear in edges)
    for obj_id in obj_info.keys():
        if obj_id not in G:
            G.add_node(obj_id)

    # Use hierarchical layout for compact representation
    # Check if graph is a DAG (directed acyclic graph)
    if nx.is_directed_acyclic_graph(G):
        # Use layered layout for DAGs
        for layer, nodes in enumerate(nx.topological_generations(G)):
            for node in nodes:
                G.nodes[node]["layer"] = layer
        try:
            pos = nx.multipartite_layout(G, subset_key="layer", align="vertical")
        except:
            # Fall back to shell layout
            pos = nx.shell_layout(G)
    else:
        # For cyclic graphs, use shell layout for compact circular arrangement
        pos = nx.shell_layout(G)

    # Create figure with appropriate size
    num_nodes = len(G.nodes())
    fig_width = max(8, min(16, num_nodes * 1.5))
    fig_height = max(6, min(12, num_nodes * 1.0))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Draw edges with arrows stopping well before the center/labels
    for edge in G.edges():
        x_start, y_start = pos[edge[0]]
        x_end, y_end = pos[edge[1]]

        # Make arrows much shorter so they don't overlap with labels
        # Draw arrow
        ax.annotate(
            "",
            xy=(x_end, y_end),
            xytext=(x_start, y_start),
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3,rad=0.1",
                lw=2,
                color="black",
                shrinkA=35,  # Shrink in points (pixels)
                shrinkB=35,
            ),
        )

    # Draw nodes with custom shapes
    for node in G.nodes():
        x, y = pos[node]

        if node in obj_info:
            obj_type = obj_info[node]["type"]
            color = obj_info[node]["color"]

            if obj_type == "ball":
                draw_ball(ax, x, y, radius=0.12, color=color)
            elif obj_type == "domino":
                draw_domino(ax, x, y, width=0.09, height=0.27, color=color)
            elif obj_type == "button":
                draw_button(ax, x, y, width=0.18, height=0.12, color=color)
            else:
                # Default circle for unknown types
                circle = patches.Circle(
                    (x, y), 0.12, facecolor=color, edgecolor="black", linewidth=2
                )
                ax.add_patch(circle)
        else:
            # Gray circle for unknown objects
            circle = patches.Circle(
                (x, y), 0.12, facecolor=(0.5, 0.5, 0.5), edgecolor="black", linewidth=2
            )
            ax.add_patch(circle)

        # Draw label with white rectangle background (like renderer)
        draw_label_background(ax, x, y, str(node), fontsize=10)
        ax.text(
            x,
            y,
            str(node),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            zorder=10,
        )

    ax.set_title(f"Causal Graph - {level_name}", fontsize=16, fontweight="bold", pad=20)
    ax.axis("equal")
    ax.axis("off")

    # Tight layout to minimize space
    plt.tight_layout()

    # Save
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Graph saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot graph from edge specification")
    parser.add_argument("edge_file", help="File with edge specifications (e.g., 1->2)")
    parser.add_argument("--info", required=True, help="Object info CSV file")
    parser.add_argument("--level", default="unknown", help="Level name")
    parser.add_argument("--output", help="Output filename (default: auto-generated)")

    args = parser.parse_args()

    # Parse edges
    edges = parse_edges(args.edge_file)

    # Load object info
    obj_info = load_object_info(args.info)

    # Generate output filename if not provided
    if args.output is None:
        timestamp = int(time.time())
        args.output = f"graph_{args.level}_{timestamp}.png"

    # Plot
    plot_graph(edges, obj_info, args.level, args.output)


if __name__ == "__main__":
    main()
