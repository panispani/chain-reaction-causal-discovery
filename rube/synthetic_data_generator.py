"""Synthetic data generation following monotone cascade SCM"""

import numpy as np
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional
import random


class SyntheticDataGenerator:
    """Generates synthetic data following the monotone cascade SCM"""

    def __init__(
        self,
        num_vars: int,
        causal_graph: List[Tuple[int, int]],
        failure_probs: Union[float, List[float]],
        seed: Optional[int] = None,
    ):
        """
        Initialize synthetic data generator

        Args:
            num_vars: Number of variables (nodes in the causal graph)
            causal_graph: List of edges [(from, to), ...] representing the causal structure
                         Node IDs should be 1-indexed (1 to num_vars)
            failure_probs: Either a single probability (applied to all variables)
                          or a list of N probabilities (one per variable)
            seed: Random seed for reproducibility
        """
        self.num_vars = num_vars
        self.causal_graph = causal_graph
        self.seed = seed

        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Process failure probabilities
        if isinstance(failure_probs, (int, float)):
            # Single probability for all variables
            self.failure_probs = [float(failure_probs)] * num_vars
        else:
            # List of probabilities
            if len(failure_probs) != num_vars:
                raise ValueError(
                    f"failure_probs must have length {num_vars}, got {len(failure_probs)}"
                )
            self.failure_probs = list(failure_probs)

        # Build parent mapping: child -> list of parents
        self.parents = {i: [] for i in range(1, num_vars + 1)}
        for parent, child in causal_graph:
            self.parents[child].append(parent)

        # Find root nodes (nodes with no parents)
        self.root_nodes = [i for i in range(1, num_vars + 1) if not self.parents[i]]

        # Build children mapping: parent -> list of children
        self.children = {i: [] for i in range(1, num_vars + 1)}
        for parent, child in causal_graph:
            self.children[parent].append(child)

    def _sample_cascade(self, intervention: Optional[int] = None) -> List[int]:
        """
        Sample from the monotone cascade SCM

        Args:
            intervention: If provided, intervene to block this variable (1-indexed)

        Returns:
            Binary activation states for all variables (0-indexed list)
        """
        # Initialize all variables to inactive
        X = [0] * self.num_vars

        # If intervention, the intervened variable stays at 0
        # We'll track which nodes are blocked
        blocked = set()
        if intervention is not None:
            blocked.add(intervention)

        # Process nodes in topological order using BFS from root nodes
        queue = list(self.root_nodes)
        visited = set()

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            # Check if this node is blocked by intervention
            if node in blocked:
                X[node - 1] = 0  # Convert to 0-indexed
            else:
                # Check if all parents are active
                parents = self.parents[node]
                if not parents:
                    # Root node - sample from Bernoulli
                    p_fail = self.failure_probs[node - 1]
                    X[node - 1] = np.random.binomial(1, 1 - p_fail)
                else:
                    # Check if any parent is inactive
                    if any(X[p - 1] == 0 for p in parents):
                        # If any parent is inactive, this node stays inactive
                        X[node - 1] = 0
                    else:
                        # All parents are active - sample from Bernoulli
                        p_fail = self.failure_probs[node - 1]
                        X[node - 1] = np.random.binomial(1, 1 - p_fail)

            # If this node is inactive (either blocked or failed), block all descendants
            if X[node - 1] == 0:
                # TODO: this is redundant, because we initialized them all to 0, one can just not process the dependants at all.
                blocked.add(node)
                # Add all descendants to blocked set (transitive closure)
                descendants = self._get_descendants(node)
                blocked.update(descendants)

            # Add children to queue
            for child in self.children[node]:
                if child not in visited:
                    queue.append(child)

        return X

    def _get_descendants(self, node: int) -> set:
        """Get all descendants of a node (transitive closure)"""
        descendants = set()
        queue = [node]
        while queue:
            current = queue.pop(0)
            for child in self.children[current]:
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)
        return descendants

    def collect_intervention_samples(
        self, num_samples: int
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Collect samples with random interventions

        Args:
            num_samples: Number of samples to collect

        Returns:
            Tuple of (samples, actions) where samples is list of binary states
        """
        samples = []
        actions = []

        for _ in range(num_samples):
            # Choose random intervention target from 1..N
            action = random.randint(1, self.num_vars)
            binary_states = self._sample_cascade(intervention=action)

            samples.append(binary_states)
            actions.append(action)

        return samples, actions

    def collect_intervention_samples_per_action(
        self, num_samples_per_action: int
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Collect samples with interventions, collecting num_samples_per_action for each action

        Args:
            num_samples_per_action: Number of samples to collect per action

        Returns:
            Tuple of (samples, actions) where samples is list of binary states
        """
        samples = []
        actions = []

        for action in range(1, self.num_vars + 1):
            for _ in range(num_samples_per_action):
                binary_states = self._sample_cascade(intervention=action)

                samples.append(binary_states)
                actions.append(action)

        return samples, actions

    def collect_no_intervention_samples(
        self, num_samples: int
    ) -> Tuple[List[List[int]], List]:
        """
        Collect samples without intervention

        Args:
            num_samples: Number of samples to collect

        Returns:
            Tuple of (samples, empty_list) for compatibility with DataCollector interface
        """
        samples = []

        for _ in range(num_samples):
            binary_states = self._sample_cascade(intervention=None)
            samples.append(binary_states)

        # Return empty list for compatibility (no collision data in synthetic mode)
        return samples, []

    def save_samples_csv(
        self, samples: List[List[int]], filename: str, actions: List[int] = None
    ):
        """
        Save samples to CSV file

        Args:
            samples: List of binary state samples
            filename: Output filename
            actions: Optional list of actions (intervened variable IDs)
        """
        if not samples:
            return
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            headers = [f"object_{i+1}" for i in range(len(samples[0]))]
            if actions is not None:
                headers.insert(0, "intervened_variable")
            writer.writerow(headers)

            # Data
            for i, sample in enumerate(samples):
                row = list(sample)
                if actions is not None:
                    row.insert(0, actions[i])
                writer.writerow(row)

    def save_samples_npy(
        self, samples: List[List[int]], filename: str, actions: List[int] = None
    ):
        """
        Save samples to NPY file

        Args:
            samples: List of binary state samples
            filename: Output filename
            actions: Optional list of actions (intervened variable IDs)
        """
        if actions is not None:
            # Add actions as first column
            samples_array = np.array(samples)
            actions_array = np.array(actions).reshape(-1, 1)
            arr = np.hstack([actions_array, samples_array])
        else:
            arr = np.array(samples)
        np.save(filename, arr)

    def save_object_info(self, filename: str):
        """
        Save object info to file (simplified for synthetic data)

        Args:
            filename: Output filename
        """
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "type", "color_r", "color_g", "color_b"])

            # Generate dummy object info (all circles with default color)
            for obj_id in range(1, self.num_vars + 1):
                writer.writerow([obj_id, "circle", 200, 200, 200])

    def save_causal_graph(self, filename: str):
        """
        Save the true causal graph to file

        Args:
            filename: Output filename
        """
        with open(filename, "w") as f:
            for parent, child in sorted(self.causal_graph):
                f.write(f"{parent}->{child}\n")
