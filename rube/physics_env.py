"""Physics environment using pymunk"""

import pymunk
import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from rube.level_parser import LevelConfig


@dataclass
class PhysicsObject:
    """Represents a physics object in the simulation"""

    body: pymunk.Body
    shape: pymunk.Shape
    obj_type: str
    obj_id: int
    color: Tuple[int, int, int]
    pressed: bool = False  # For buttons
    triggers: Optional[List[Dict[str, Any]]] = None  # For buttons with triggers


class PhysicsEnvironment:
    """Physics environment using pymunk"""

    # Physics constants
    GRAVITY = (0, 980)  # pixels/s^2 (downwards)
    TIME_STEP = 0.02  # 20ms per step
    FRICTION = 0.5

    def __init__(
        self,
        level: LevelConfig,
        displacement: float = 0.0,
        seed: Optional[int] = None,
        settling_steps: int = 0,
    ):
        """
        Initialize physics environment

        Args:
            level: Level configuration
            displacement: Random displacement range for objects
            seed: Random seed
            settling_steps: Number of physics steps to run after reset to let objects settle (resolves overlaps from displacement)
        """
        self.level = level
        self.displacement = displacement
        self.seed = seed
        self.settling_steps = settling_steps

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Create space
        self.space = pymunk.Space()
        self.space.gravity = self.GRAVITY

        # Create objects
        self.objects: List[PhysicsObject] = []
        self.walls: List[pymunk.Shape] = []
        self.wall_map: Dict[Tuple[int, int], pymunk.Shape] = (
            {}
        )  # Map (row, col) -> shape
        self.ramps: List[Dict[str, Any]] = []  # Store ramp definitions for rendering
        self.removed_walls: set = set()  # Track removed wall positions (row, col)
        self.num_objects = 0
        self.held_object_id: Optional[int] = None
        self.hold_constraints: List[pymunk.Constraint] = []

        # Collision tracking
        self.collisions: List[Tuple[int, int]] = []
        self.collisions_with_time: List[Tuple[int, int, int]] = (
            []
        )  # (obj1, obj2, timestep)
        self.current_timestep: int = 0
        # We'll add collision handlers for specific collision types later
        self.collision_handlers = []

        self._setup_environment()

    def _setup_environment(self):
        """Setup the physics environment"""
        # Create walls
        self._create_walls()

        # Filter out ramps before assigning IDs (ramps are created as static walls, not objects)
        obj_list = [
            obj_def for obj_def in self.level.objects if obj_def["type"] != "ramp"
        ]
        random.shuffle(obj_list)

        # Assign IDs and colors
        for idx, obj_def in enumerate(obj_list):
            obj_id = idx + 1
            color = self._get_color_for_type(obj_def["type"])

            # Create physics object
            phys_obj = self._create_object(obj_def, obj_id, color)
            if phys_obj:
                self.objects.append(phys_obj)

        self.num_objects = len(self.objects)

        # Store initial object definitions for reset
        self.object_definitions = obj_list
        self.initial_positions = (
            {}
        )  # Store base (x, y) position for each object (before displacement)
        # Not very clean, ideally the obj_def would carry this information
        for obj, obj_def in zip(self.objects, self.object_definitions):
            row, col = obj_def["position"]
            base_x = (col + 0.5) * self.level.cell_size * 100
            base_y = (
                obj.body.position.y
            )  # Use actual y (accounts for button positioning)
            self.initial_positions[obj.obj_id] = (base_x, base_y)

        # Setup collision handlers after objects are created
        self._setup_collision_handlers()

    def _setup_collision_handlers(self):
        """Setup collision handlers for tracking collisions"""
        # Use the on_collision for pymunk 7.x with wildcard (None matches all)
        self.space.on_collision(
            collision_type_a=None, collision_type_b=None, begin=self._collision_callback
        )

    def _create_walls(self):
        """Create static walls"""
        static_body = self.space.static_body

        for row, col in self.level.walls:
            # Convert grid to physics coordinates
            x = (col + 0.5) * self.level.cell_size * 100  # Scale up for better physics
            y = (row + 0.5) * self.level.cell_size * 100

            size = self.level.cell_size * 100 / 2

            # Create box for wall
            points = [(-size, -size), (size, -size), (size, size), (-size, size)]
            shape = pymunk.Poly(
                static_body, points, transform=pymunk.Transform(tx=x, ty=y)
            )
            shape.friction = self.FRICTION
            shape.collision_type = 0  # Walls don't have collision tracking
            self.space.add(shape)
            self.walls.append(shape)
            self.wall_map[(row, col)] = shape  # Track by position

        # Create ramps
        for obj_def in self.level.objects:
            if obj_def["type"] == "ramp":
                self._create_ramp(obj_def)

    def _create_ramp(self, obj_def: Dict[str, Any]):
        """Create a static ramp"""
        row, col = obj_def["position"]
        x = (col + 0.5) * self.level.cell_size * 100
        y = (row + 0.5) * self.level.cell_size * 100

        size = self.level.cell_size * 100 / 2

        # Create triangle for ramp
        direction = obj_def.get("direction", "up")
        if direction == "up":
            points = [(-size, size), (size, size), (size, -size)]
        else:  # down
            points = [(-size, -size), (-size, size), (size, size)]

        static_body = self.space.static_body
        shape = pymunk.Poly(static_body, points, transform=pymunk.Transform(tx=x, ty=y))
        shape.friction = self.FRICTION
        shape.collision_type = 0  # Static environment
        self.space.add(shape)
        self.walls.append(shape)

        # Store ramp info for rendering
        self.ramps.append(
            {
                "position": (x, y),
                "direction": direction,
                "size": size,
                "color": self._get_color_for_type("ramp"),
            }
        )

    def _create_object(
        self, obj_def: Dict[str, Any], obj_id: int, color: Tuple[int, int, int]
    ) -> Optional[PhysicsObject]:
        """Create a physics object"""
        obj_type = obj_def["type"]

        if obj_type == "ramp":
            return None  # Ramps are created as walls

        row, col = obj_def["position"]

        # Apply displacement to x
        dx = np.random.uniform(-self.displacement, self.displacement) * 100
        x = (col + 0.5) * self.level.cell_size * 100 + dx

        # Calculate y position based on object type
        # Objects should be positioned at the bottom of their cell
        floor_y = (row + 1) * self.level.cell_size * 100  # Bottom of the cell

        if obj_type == "ball":
            # Position ball so it sits on the floor
            radius = obj_def.get("radius", 0.3) * 100
            y = floor_y - radius
            return self._create_ball(obj_def, x, y, obj_id, color)
        elif obj_type == "domino":
            # Position domino so it sits on the floor
            height = obj_def.get("height", 1.0) * 100
            y = floor_y - height / 2
            return self._create_domino(obj_def, x, y, obj_id, color)
        elif obj_type == "button":
            # Positioning logic is in _create_button (TODO: refactor)
            y = (row + 0.5) * self.level.cell_size * 100
            return self._create_button(obj_def, x, y, obj_id, color)

        return None

    def _create_ball(
        self,
        obj_def: Dict[str, Any],
        x: float,
        y: float,
        obj_id: int,
        color: Tuple[int, int, int],
    ) -> PhysicsObject:
        """Create a ball"""
        radius = obj_def.get("radius", 0.3) * 100
        mass = obj_def.get("mass", 1.0)
        elasticity = obj_def.get("elasticity", 0.7)
        initial_velocity = obj_def.get("initial_velocity", [0, 0])

        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = x, y
        body.velocity = (initial_velocity[0] * 100, initial_velocity[1] * 100)

        shape = pymunk.Circle(body, radius)
        shape.elasticity = elasticity
        shape.friction = self.FRICTION
        shape.collision_type = obj_id

        self.space.add(body, shape)

        return PhysicsObject(
            body=body, shape=shape, obj_type="ball", obj_id=obj_id, color=color
        )

    def _create_domino(
        self,
        obj_def: Dict[str, Any],
        x: float,
        y: float,
        obj_id: int,
        color: Tuple[int, int, int],
    ) -> PhysicsObject:
        """Create a domino (can topple)"""
        width = obj_def.get("width", 0.2) * 100
        height = obj_def.get("height", 1.0) * 100
        mass = obj_def.get("mass", 0.5)

        moment = pymunk.moment_for_box(mass, (width, height))
        body = pymunk.Body(mass, moment)
        body.position = x, y

        shape = pymunk.Poly.create_box(body, (width, height))
        shape.elasticity = 0.3
        shape.friction = self.FRICTION
        shape.collision_type = obj_id

        self.space.add(body, shape)

        return PhysicsObject(
            body=body, shape=shape, obj_type="domino", obj_id=obj_id, color=color
        )

    def _create_button(
        self,
        obj_def: Dict[str, Any],
        x: float,
        y: float,
        obj_id: int,
        color: Tuple[int, int, int],
    ) -> PhysicsObject:
        """Create a button (rectangle stuck to ground at bottom of cell)"""
        # Button dimensions: wide but short
        width = obj_def.get("width", 0.6) * 100
        height = obj_def.get("height", 0.15) * 100

        # Buttons are kinematic (don't move but can detect collisions)
        # Position button so it sits on the ground (top of the cell below)
        row, col = obj_def["position"]
        # The floor/wall below is at row+1, centered at (row + 1 + 0.5) * 100
        # with half-size of 50, so its top surface is at:
        wall_center_y = (row + 1.0 + 0.5) * self.level.cell_size * 100
        wall_half_size = self.level.cell_size * 100 / 2
        y_floor_top = wall_center_y - wall_half_size  # Top surface of the floor
        # Position button so its bottom sits on the floor
        button_y = y_floor_top - height / 2

        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = x, button_y

        # Create rectangular button
        shape = pymunk.Poly.create_box(body, (width, height))
        shape.collision_type = obj_id
        shape.sensor = True  # Sensor doesn't create physical response

        self.space.add(body, shape)

        # Get triggers from object definition
        triggers = obj_def.get("triggers", [])

        return PhysicsObject(
            body=body,
            shape=shape,
            obj_type="button",
            obj_id=obj_id,
            color=color,
            pressed=False,
            triggers=triggers,
        )

    def _get_color_for_type(self, obj_type: str) -> Tuple[int, int, int]:
        """Get the color for an object type"""
        if obj_type == "ball":
            return (218, 45, 71)  # Red: #DA2D47
        elif obj_type == "domino":
            return (57, 81, 172)  # Blue: #3951AC
        elif obj_type == "button":
            return (81, 172, 57)  # Green: #51AC39
        elif obj_type == "ramp":
            return (0, 0, 0)  # Black (same as walls)
        else:
            return (128, 128, 128)  # Default gray

    def _collision_callback(
        self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: Dict
    ) -> bool:
        """Callback for collision detection"""
        shapes = arbiter.shapes

        # Get collision types (which are object IDs)
        id1 = shapes[0].collision_type
        id2 = shapes[1].collision_type

        # Only track collisions between objects (not walls)
        if id1 > 0 and id2 > 0:
            self.collisions.append((id1, id2))
            self.collisions_with_time.append((id1, id2, self.current_timestep))

            # Check if button was pressed
            for obj in self.objects:
                if obj.obj_id == id1 or obj.obj_id == id2:
                    if obj.obj_type == "button" and not obj.pressed:
                        # Don't press button if it's being held (intervened on)
                        if self.held_object_id == obj.obj_id:
                            continue
                        # Check if collision is from above
                        # contact_normal = arbiter.contact_point_set.normal
                        # if contact_normal.y < -0.5:  # Pressed from above
                        obj.pressed = True
                        # Execute button triggers
                        self._execute_button_triggers(obj)

        return True

    def _execute_button_triggers(self, button: PhysicsObject):
        """Execute triggers when button is pressed"""
        if not button.triggers:
            return

        for trigger in button.triggers:
            action = trigger.get("action")
            if action == "open_walls":
                wall_cells = trigger.get("wall_cells", [])
                self._remove_walls(wall_cells)

    def _remove_walls(self, wall_cells: List[Tuple[int, int]]):
        """Remove walls at specified grid coordinates"""
        for row, col in wall_cells:
            # Check if wall exists at this position
            if (row, col) in self.wall_map and (row, col) not in self.removed_walls:
                # Add to removed walls set (for rendering)
                self.removed_walls.add((row, col))

                # Remove from physics space
                wall_shape = self.wall_map[(row, col)]
                self.space.remove(wall_shape)
                self.walls.remove(wall_shape)

    def settle(self):
        """
        Run settling steps to let objects resolve overlaps from displacement.
        Should be called after reset() and before recording initial positions.

        Objects at settle_positions (from level config) are held in place during settling,
        then released and restored to their initial velocities.
        """
        if self.settling_steps > 0:
            # Find objects at settle positions and hold them
            settle_object_ids = []
            settle_object_velocities = {}  # Store intended initial velocities

            for settle_pos in self.level.settle_positions:
                obj_id = self.get_object_id_at_position(settle_pos)
                if obj_id is not None:
                    settle_object_ids.append(obj_id)

                    # Find the object and its definition to get initial velocity
                    for obj, obj_def in zip(self.objects, self.object_definitions):
                        if obj.obj_id == obj_id:
                            # Store intended initial velocity
                            if "initial_velocity" in obj_def:
                                settle_object_velocities[obj_id] = obj_def[
                                    "initial_velocity"
                                ]
                            break

            # Hold all settle objects in place
            for obj_id in settle_object_ids:
                for obj in self.objects:
                    if obj.obj_id == obj_id:
                        if obj.obj_type == "button":
                            # For buttons, just mark as held
                            self.held_object_id = obj_id
                        elif obj.body.body_type == pymunk.Body.DYNAMIC:
                            # Stop the object
                            obj.body.velocity = (0, 0)
                            obj.body.angular_velocity = 0

                            # Lock position
                            pivot = pymunk.PivotJoint(
                                self.space.static_body, obj.body, obj.body.position
                            )
                            pivot.max_force = float("inf")

                            # Lock rotation
                            gear = pymunk.GearJoint(
                                self.space.static_body, obj.body, 0.0, 1.0
                            )
                            gear.max_force = float("inf")

                            self.space.add(pivot, gear)
                            self.hold_constraints.extend([pivot, gear])
                        break

            # Run settling with objects held
            self.step(self.settling_steps)

            # Release held objects
            for constraint in self.hold_constraints:
                self.space.remove(constraint)
            self.hold_constraints = []
            self.held_object_id = None

            # Restore initial velocities to settle objects
            for obj_id, initial_velocity in settle_object_velocities.items():
                for obj in self.objects:
                    if obj.obj_id == obj_id:
                        obj.body.velocity = (
                            initial_velocity[0] * 100,
                            initial_velocity[1] * 100,
                        )
                        break

            # Reset collision tracking after settling (we don't care about settling collisions)
            self.collisions = []
            self.collisions_with_time = []
            self.current_timestep = 0

    def reset(self):
        """Reset the environment for a new episode with new displacement"""
        self.collisions = []
        self.collisions_with_time = []
        self.current_timestep = 0

        # Re-add any removed walls
        for row, col in self.removed_walls:
            if (row, col) in self.wall_map:
                wall_shape = self.wall_map[(row, col)]
                self.space.add(wall_shape)
                if wall_shape not in self.walls:
                    self.walls.append(wall_shape)

        self.removed_walls = set()  # Clear removed walls set

        # Release held object by removing constraints
        for constraint in self.hold_constraints:
            self.space.remove(constraint)
        self.hold_constraints = []
        self.held_object_id = None

        # Reset each object to its original position (with new displacement)
        for obj, obj_def in zip(self.objects, self.object_definitions):
            # Get stored base position (without displacement)
            base_x, base_y = self.initial_positions[obj.obj_id]

            # Apply new displacement to x only
            dx = np.random.uniform(-self.displacement, self.displacement) * 100
            x = base_x + dx
            y = base_y

            # Reset position
            obj.body.position = (x, y)
            obj.body.velocity = (0, 0)
            obj.body.angular_velocity = 0
            obj.body.angle = 0

            # Reset initial velocity for balls
            if obj.obj_type == "ball" and "initial_velocity" in obj_def:
                initial_velocity = obj_def["initial_velocity"]
                obj.body.velocity = (
                    initial_velocity[0] * 100,
                    initial_velocity[1] * 100,
                )

            # Reset button pressed state
            if obj.obj_type == "button":
                obj.pressed = False

    def apply_action(self, action: int):
        """
        Apply an action

        Args:
            action: 0 for no-op, 1..N to hold object 1..N in place
        """
        # Release previously held object if needed
        if self.held_object_id is not None and action != self.held_object_id:
            # Remove all hold constraints
            for constraint in self.hold_constraints:
                self.space.remove(constraint)
            self.hold_constraints = []
            self.held_object_id = None

        # Now apply the new action
        if action == 0:
            # Already released above, nothing more to do
            pass
        else:
            # Hold object by pinning it in place with constraints
            for obj in self.objects:
                if obj.obj_id == action:
                    if obj.obj_type == "button":
                        # For buttons, just mark as held (prevents pressing)
                        self.held_object_id = action
                    elif obj.body.body_type == pymunk.Body.DYNAMIC:
                        # For dynamic objects, add constraints
                        # Stop the object
                        obj.body.velocity = (0, 0)
                        obj.body.angular_velocity = 0

                        # Lock position
                        pivot = pymunk.PivotJoint(
                            self.space.static_body, obj.body, obj.body.position
                        )
                        pivot.max_force = float("inf")  # Infinite holding force

                        # Lock rotation
                        gear = pymunk.GearJoint(
                            self.space.static_body, obj.body, 0.0, 1.0
                        )
                        gear.max_force = float("inf")  # Infinite rotational force

                        self.space.add(pivot, gear)
                        self.hold_constraints.extend([pivot, gear])
                        self.held_object_id = action
                    break

    def step(self, num_steps: int = 1):
        """
        Step the simulation

        Args:
            num_steps: Number of physics steps to run
        """
        for _ in range(num_steps):
            self.space.step(self.TIME_STEP)
            self.current_timestep += 1

    def get_initial_object_positions(self) -> Dict[int, Tuple[float, float, float]]:
        """Get initial positions of all objects"""
        positions = {}
        for obj in self.objects:
            positions[obj.obj_id] = (
                obj.body.position.x,
                obj.body.position.y,
                obj.body.angle,
            )
        return positions

    def get_object_moved_states(
        self,
        initial_positions: Dict[int, Tuple[float, float, float]],
        threshold: float = 1.0,
    ) -> List[int]:
        """
        Get binary states indicating whether each object moved

        Args:
            initial_positions: Initial positions from get_initial_object_positions()
            threshold: Movement threshold in pixels

        Returns:
            List of binary values (1 if moved, 0 if not) in object ID order
        """
        states = []

        for obj in sorted(self.objects, key=lambda o: o.obj_id):
            if obj.obj_type == "button":
                # Button state is whether it was pressed
                states.append(1 if obj.pressed else 0)
            else:
                # Check if position changed
                initial = initial_positions[obj.obj_id]
                current = (obj.body.position.x, obj.body.position.y, obj.body.angle)

                dx = current[0] - initial[0]
                dy = current[1] - initial[1]
                dangle = abs(current[2] - initial[2])

                moved = (dx**2 + dy**2) ** 0.5 > threshold or dangle > 0.1
                states.append(1 if moved else 0)

        return states

    def get_collisions(self) -> List[Tuple[int, int]]:
        """Get all collisions that occurred"""
        return list(set(self.collisions))  # Remove duplicates

    def get_collisions_with_time(self) -> List[Tuple[int, int, int]]:
        """Get all collisions with timesteps: (obj1, obj2, timestep)"""
        return self.collisions_with_time

    def get_object_info(self) -> List[Tuple[int, str, Tuple[int, int, int]]]:
        """Get object info: (id, type, color)"""
        info = []
        for obj in sorted(self.objects, key=lambda o: o.obj_id):
            info.append((obj.obj_id, obj.obj_type, obj.color))
        return info

    def get_object_id_at_position(self, position: Tuple[int, int]) -> Optional[int]:
        """
        Get the object ID at a given grid position (row, col)

        Args:
            position: Tuple of (row, col)

        Returns:
            Object ID if found, None otherwise
        """
        for obj, obj_def in zip(self.objects, self.object_definitions):
            if obj_def["position"] == position:
                return obj.obj_id
        return None
