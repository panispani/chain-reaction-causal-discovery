"""Level parser for Rube Goldberg environment"""

import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple


@dataclass
class LevelConfig:
    """Configuration for a level"""

    id: str
    cell_size: float
    layout: List[List[str]]
    legend: Dict[str, Any]
    objects: List[Dict[str, Any]] = field(default_factory=list)
    walls: List[Tuple[int, int]] = field(default_factory=list)
    solution: Dict[str, Any] = field(default_factory=dict)
    settle_positions: List[Tuple[int, int]] = field(default_factory=list)


class LevelParser:
    """Parser for level configuration files"""

    def parse(self, yaml_string: str) -> LevelConfig:
        """Parse a YAML string into the LevelConfig dataclass"""
        data = yaml.safe_load(yaml_string)
        return self._parse_data(data)

    def parse_file(self, filepath: str) -> LevelConfig:
        """Parse a YAML file into a LevelConfig"""
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return self._parse_data(data)

    def _parse_data(self, data: Dict[str, Any]) -> LevelConfig:
        """Parse the loaded YAML data into a LevelConfig"""
        level_id = data["id"]
        cell_size = data["cell_size"]
        layout_str = data["layout"].strip()
        legend = data["legend"]
        solution = data.get("solution", {})
        settle_positions = [tuple(pos) for pos in data.get("settle", [])]

        # Parse layout into 2D array
        layout = []
        for line in layout_str.split("\n"):
            if line.strip():
                layout.append(list(line))

        # Extract objects and walls
        objects = []
        walls = []

        for row_idx, row in enumerate(layout):
            for col_idx, char in enumerate(row):
                if char in legend:
                    obj_def = legend[char]
                    obj_type = obj_def.get("type")

                    if obj_type == "wall_tile":
                        walls.append((row_idx, col_idx))
                    elif obj_type != "empty":
                        # Create an object
                        obj = {
                            "type": obj_type,
                            "position": (row_idx, col_idx),
                            "char": char,
                        }

                        # Copy all other properties
                        for key, value in obj_def.items():
                            if key != "type":
                                obj[key] = value

                        objects.append(obj)

        return LevelConfig(
            id=level_id,
            cell_size=cell_size,
            layout=layout,
            legend=legend,
            objects=objects,
            walls=walls,
            solution=solution,
            settle_positions=settle_positions,
        )
