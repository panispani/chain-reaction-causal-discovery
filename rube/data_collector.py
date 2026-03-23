"""Data collection and export for physics simulations"""

import numpy as np
import csv
from pathlib import Path
from typing import List, Tuple, Dict
import random
from rube.physics_env import PhysicsEnvironment
from rube.renderer import Renderer
from PIL import Image


class DataCollector:
    """Collects data from physics simulations"""

    def __init__(self, env: PhysicsEnvironment, max_steps: int = 1000):
        """
        Initialize data collector

        Args:
            env: Physics environment
            max_steps: Maximum number of simulation steps per episode
        """
        self.env = env
        self.max_steps = max_steps
        self.renderer = Renderer(env)

    def run_episode(
        self, action: int = 0, record_frames: bool = False
    ) -> Tuple[List[int], List[Tuple[int, int]], List[Image.Image]]:
        """
        Run a single episode

        Args:
            action: Action to take (0 for no-op, 1..N to hold object)
            record_frames: Whether to record frames for GIF

        Returns:
            Tuple of (binary_states, collisions, frames)
        """
        # This cleans any actions
        self.env.reset()

        # Let objects settle to resolve any overlaps from displacement
        self.env.settle()

        # Get initial positions (after settling)
        initial_positions = self.env.get_initial_object_positions()

        # Apply action
        self.env.apply_action(action)

        # Record frames if needed
        frames = []
        if record_frames:
            # Capture frame every N steps for reasonable GIF size
            frame_interval = max(1, self.max_steps // 100)
            for step in range(self.max_steps):
                if step % frame_interval == 0:
                    frames.append(self.renderer.render())
                self.env.step(1)
        else:
            # Just run simulation
            self.env.step(self.max_steps)

        # Get final states
        binary_states = self.env.get_object_moved_states(initial_positions)
        collisions = self.env.get_collisions()

        return binary_states, collisions, frames

    def run_episode_with_time(
        self, action: int = 0, record_frames: bool = False
    ) -> Tuple[List[int], List[Tuple[int, int, int]], List[Image.Image]]:
        """
        Run a single episode and return collisions with timestamps

        Args:
            action: Action to take (0 for no-op, 1..N to hold object)
            record_frames: Whether to record frames for GIF

        Returns:
            Tuple of (binary_states, collisions_with_time, frames)
            where collisions_with_time is List[Tuple[int, int, int]] = (obj1, obj2, timestep)
        """
        # This cleans any actions
        self.env.reset()

        # Let objects settle to resolve any overlaps from displacement
        self.env.settle()

        # Get initial positions (after settling)
        initial_positions = self.env.get_initial_object_positions()

        # Apply action
        self.env.apply_action(action)

        # Record frames if needed
        frames = []
        if record_frames:
            # Capture frame every N steps for reasonable GIF size
            frame_interval = max(1, self.max_steps // 100)
            for step in range(self.max_steps):
                if step % frame_interval == 0:
                    frames.append(self.renderer.render())
                self.env.step(1)
        else:
            # Just run simulation
            self.env.step(self.max_steps)

        # Get final states
        binary_states = self.env.get_object_moved_states(initial_positions)
        collisions_with_time = self.env.get_collisions_with_time()

        return binary_states, collisions_with_time, frames

    def run_with_action_check(
        self, max_checks: int = 5
    ) -> Tuple[int, List[int], List[Image.Image]]:
        """
        Run episode and check initial motion to determine action

        Args:
            max_checks: Maximum number of steps to check for initial motion

        Returns:
            Tuple of (action_taken, binary_states, frames)
        """
        self.env.reset()
        self.env.settle()
        initial_positions = self.env.get_initial_object_positions()

        # Run a few steps to see what moves
        self.env.step(max_checks)

        # Check which object moved (after 0.1s as per spec)
        # We'll use 5 steps = 0.1s with TIME_STEP=0.02
        check_states = self.env.get_object_moved_states(
            initial_positions, threshold=5.0
        )

        # Find objects that moved
        moved_objects = [i + 1 for i, state in enumerate(check_states) if state == 1]

        # Choose a random action from objects that didn't move (or no-op if all moved)
        non_moved_objects = [
            i + 1 for i, state in enumerate(check_states) if state == 0
        ]

        if non_moved_objects:
            action = random.choice(non_moved_objects)
        else:
            action = 0  # No-op if all objects moved

        # Reset and run with the chosen action
        return action, *self.run_episode(action, record_frames=True)

    def collect_intervention_samples(
        self, num_samples: int
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Collect samples with random interventions (actions 1..N)

        Args:
            num_samples: Number of samples to collect

        Returns:
            Tuple of (samples, actions) where samples is list of binary states
        """
        samples = []
        actions = []

        for _ in range(num_samples):
            # Choose random action from 1..N
            action = random.randint(1, self.env.num_objects)
            binary_states, _, _ = self.run_episode(action)

            # Check if intervened variable incorrectly shows as moved
            # (action is 1-indexed, binary_states is 0-indexed)
            if binary_states[action - 1] == 1:
                print(
                    f"Warning: Intervened variable (object {action}) shows as moved. "
                    f"Forcing to 0. This may be due to initial collision from displacement."
                )
                binary_states[action - 1] = 0

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

        for action in range(1, self.env.num_objects + 1):
            for _ in range(num_samples_per_action):
                binary_states, _, _ = self.run_episode(action)

                # Check if intervened variable incorrectly shows as moved
                # (action is 1-indexed, binary_states is 0-indexed)
                if binary_states[action - 1] == 1:
                    print(
                        f"Warning: Intervened variable (object {action}) shows as moved. "
                        f"Forcing to 0. This may be due to initial collision from displacement."
                    )
                    binary_states[action - 1] = 0

                samples.append(binary_states)
                actions.append(action)

        return samples, actions

    def collect_no_intervention_samples(
        self, num_samples: int
    ) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
        """
        Collect samples without intervention (action 0)

        Args:
            num_samples: Number of samples to collect

        Returns:
            Tuple of (samples, all_collisions)
        """
        samples = []
        all_collisions = []

        for _ in range(num_samples):
            binary_states, collisions, _ = self.run_episode(action=0)

            samples.append(binary_states)
            all_collisions.extend(collisions)

        return samples, all_collisions

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
        Save object info (id, type, color) to file

        Args:
            filename: Output filename
        """
        info = self.env.get_object_info()

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "type", "color_r", "color_g", "color_b"])

            for obj_id, obj_type, color in info:
                writer.writerow([obj_id, obj_type, color[0], color[1], color[2]])

    def generate_collision_graph(
        self, collisions: List[Tuple[int, int]], filename: str
    ):
        """
        Generate collision graph (edges between objects that collided)

        Args:
            collisions: List of collision pairs
            filename: Output filename
        """
        # Keep first collision direction between each pair
        edges = []
        seen_pairs = set()
        for obj1, obj2 in collisions:
            pair = frozenset([obj1, obj2])
            if pair not in seen_pairs:
                edges.append((obj1, obj2))
                seen_pairs.add(pair)

        with open(filename, "w") as f:
            for obj1, obj2 in sorted(edges):
                f.write(f"{obj1}->{obj2}\n")

    def generate_collision_graph_with_time(
        self,
        samples_with_collisions: List[Tuple[List[int], List[Tuple[int, int, int]]]],
        filename: str,
    ):
        """
        Generate collision graph with temporal information

        This tracks collisions, and for objects that moved without collision,
        adds edges from objects involved in the last collision (temporally)

        Args:
            samples_with_collisions: List of (binary_states, collisions_with_time) tuples
                where collisions_with_time is List[Tuple[int, int, int]] = (obj1, obj2, timestep)
            filename: Output filename
        """
        edges = []
        seen_pairs = set()

        for binary_states, collisions_with_time in samples_with_collisions:
            # Track collisions (keep first direction seen for each pair)
            for obj1, obj2, timestep in collisions_with_time:
                pair = frozenset([obj1, obj2])
                if pair not in seen_pairs:
                    edges.append((obj1, obj2))
                    seen_pairs.add(pair)

            # For objects that moved without direct collision,
            # connect to objects involved in the last collision (by time)
            moved_objects = [
                i + 1 for i, state in enumerate(binary_states) if state == 1
            ]

            if collisions_with_time:
                # Get all objects involved in collisions
                collision_objects = set()
                for obj1, obj2, timestep in collisions_with_time:
                    collision_objects.add(obj1)
                    collision_objects.add(obj2)

                # For moved objects not in collisions, add edges from objects in the
                # most recent collision (by timestep, not by object ID)
                # Find the last collision by timestep
                last_collision = max(collisions_with_time, key=lambda x: x[2])
                last_collision_objects = {last_collision[0], last_collision[1]}

                for obj in moved_objects:
                    if obj not in collision_objects:
                        # Connect to both objects in the last collision
                        for last_obj in last_collision_objects:
                            pair = frozenset([last_obj, obj])
                            if pair not in seen_pairs:
                                edges.append((last_obj, obj))
                                seen_pairs.add(pair)

        with open(filename, "w") as f:
            for obj1, obj2 in sorted(edges):
                f.write(f"{obj1}->{obj2}\n")

    def create_motion_screenshot(self, filename: str):
        """
        Create screenshot with motion arrows showing initial motion

        Args:
            filename: Output filename
        """
        self.env.reset()
        self.env.settle()
        initial_positions = self.env.get_initial_object_positions()

        # Run for ~0.1s (5 steps with TIME_STEP=0.02)
        self.env.step(5)

        # Render with motion arrows
        img = self.renderer.render_with_motion_arrows(initial_positions)
        img.save(filename)
