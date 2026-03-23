"""Renderer for physics environment"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Tuple, Optional
import math
from rube.physics_env import PhysicsEnvironment


class Renderer:
    """Renders physics environment to images"""

    def __init__(self, env: PhysicsEnvironment, width: int = 800, height: int = 600):
        """
        Initialize renderer

        Args:
            env: Physics environment to render
            width: Image width in pixels
            height: Image height in pixels
        """
        self.env = env
        self.width = width
        self.height = height

        # Calculate scale based on level size
        max_x = max(col for row, col in env.level.walls) + 2
        max_y = max(row for row, col in env.level.walls) + 2

        self.scale_x = width / (max_x * env.level.cell_size * 100)
        self.scale_y = height / (max_y * env.level.cell_size * 100)
        self.scale = min(self.scale_x, self.scale_y)

    def render(self, show_labels: bool = True) -> Image.Image:
        """
        Render current state of environment

        Args:
            show_labels: Whether to show object IDs

        Returns:
            PIL Image
        """
        # Create white background
        img = Image.new("RGB", (self.width, self.height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw walls
        self._draw_walls(draw)

        # Draw ramps
        self._draw_ramps(draw)

        # Draw objects
        for obj in self.env.objects:
            self._draw_object(draw, obj, show_labels)

        return img

    def _draw_walls(self, draw: ImageDraw.ImageDraw):
        """Draw walls"""
        for row, col in self.env.level.walls:
            # Skip removed walls
            if (row, col) in self.env.removed_walls:
                continue

            x = col * self.env.level.cell_size * 100 * self.scale
            y = row * self.env.level.cell_size * 100 * self.scale
            size = self.env.level.cell_size * 100 * self.scale

            # Draw filled rectangle for wall
            draw.rectangle(
                [x, y, x + size, y + size], fill=(0, 0, 0), outline=(0, 0, 0)
            )

    def _draw_ramps(self, draw: ImageDraw.ImageDraw):
        """Draw ramps"""
        for ramp in self.env.ramps:
            x = ramp["position"][0] * self.scale
            y = ramp["position"][1] * self.scale
            size = ramp["size"] * self.scale
            direction = ramp["direction"]
            color = ramp["color"]

            # Create triangle points based on direction
            if direction == "up":
                points = [
                    (x - size, y + size),
                    (x + size, y + size),
                    (x + size, y - size),
                ]
            else:  # down
                points = [
                    (x - size, y - size),
                    (x - size, y + size),
                    (x + size, y + size),
                ]

            # Draw triangle
            draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)

    def _draw_object(self, draw: ImageDraw.ImageDraw, obj, show_labels: bool):
        """Draw a physics object"""
        x = obj.body.position.x * self.scale
        y = obj.body.position.y * self.scale

        if obj.obj_type == "ball":
            radius = obj.shape.radius * self.scale

            # Draw circle
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=obj.color,
                outline=(0, 0, 0),
                width=2,
            )

            # Add two perpendicular diameters to show rotation
            angle = obj.body.angle

            # First diameter
            d1_start_x = x - radius * 0.85 * math.cos(angle)
            d1_start_y = y - radius * 0.85 * math.sin(angle)
            d1_end_x = x + radius * 0.85 * math.cos(angle)
            d1_end_y = y + radius * 0.85 * math.sin(angle)
            draw.line(
                [(d1_start_x, d1_start_y), (d1_end_x, d1_end_y)],
                fill=(0, 0, 0),
                width=2,
            )

            # Second diameter (perpendicular)
            angle2 = angle + math.pi / 2
            d2_start_x = x - radius * 0.85 * math.cos(angle2)
            d2_start_y = y - radius * 0.85 * math.sin(angle2)
            d2_end_x = x + radius * 0.85 * math.cos(angle2)
            d2_end_y = y + radius * 0.85 * math.sin(angle2)
            draw.line(
                [(d2_start_x, d2_start_y), (d2_end_x, d2_end_y)],
                fill=(0, 0, 0),
                width=2,
            )

        elif obj.obj_type == "domino":
            # Get vertices and transform to world coordinates
            vertices = obj.shape.get_vertices()
            # Vertices are in body-local coords, transform to world coords
            world_vertices = []
            for v in vertices:
                # Transform by body position and rotation
                world_v = obj.body.local_to_world(v)
                world_vertices.append(world_v)

            # Scale to screen coordinates
            points = [(v.x * self.scale, v.y * self.scale) for v in world_vertices]

            # Draw polygon
            draw.polygon(points, fill=obj.color, outline=(0, 0, 0), width=2)

        elif obj.obj_type == "button":
            # Get vertices and transform to world coordinates
            vertices = obj.shape.get_vertices()
            # Vertices are in body-local coords, transform to world coords
            world_vertices = []
            for v in vertices:
                # Transform by body position and rotation
                world_v = obj.body.local_to_world(v)
                world_vertices.append(world_v)

            # Scale to screen coordinates
            points = [(v.x * self.scale, v.y * self.scale) for v in world_vertices]

            # Draw rectangle with different appearance if pressed
            fill_color = (150, 250, 150) if obj.pressed else obj.color

            draw.polygon(points, fill=fill_color, outline=(0, 0, 0), width=2)

        # Draw label
        if show_labels:
            try:
                font = ImageFont.truetype(
                    "/System/Library/Fonts/Helvetica.ttc", size=20
                )
            except:
                font = ImageFont.load_default()

            label = str(obj.obj_id)

            # Get text bbox
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Draw label with white background
            draw.rectangle(
                [
                    x - text_width / 2 - 2,
                    y - text_height / 2 - 2,
                    x + text_width / 2 + 2,
                    y + text_height / 2 + 2,
                ],
                fill=(255, 255, 255),
            )
            draw.text(
                (x - text_width / 2, y - text_height / 2),
                label,
                fill=(0, 0, 0),
                font=font,
            )

    def render_with_motion_arrows(self, initial_positions: dict) -> Image.Image:
        """
        Render with arrows showing initial motion

        Args:
            initial_positions: Initial positions from environment

        Returns:
            PIL Image with motion arrows
        """
        img = self.render(show_labels=True)
        draw = ImageDraw.Draw(img)

        # Check which objects moved in first 0.1s
        for obj in self.env.objects:
            initial = initial_positions.get(obj.obj_id)
            if initial is None:
                continue

            current = (obj.body.position.x, obj.body.position.y, obj.body.angle)

            dx = current[0] - initial[0]
            dy = current[1] - initial[1]
            dangle = current[2] - initial[2]

            # Check if moved significantly
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 1.0 or abs(dangle) > 0.05:
                # Draw arrow showing motion direction
                # Add x-axis displacement to place arrow next to object
                x_offset = 30  # pixels to the right of object
                start_x = initial[0] * self.scale + x_offset
                start_y = initial[1] * self.scale

                # Normalize and scale arrow
                if dist > 0.01:
                    arrow_len = min(100, dist * self.scale)
                    end_x = start_x + (dx / dist) * arrow_len
                    end_y = start_y + (dy / dist) * arrow_len

                    # Draw line (stop 10px before end for arrowhead)
                    angle = math.atan2(dy, dx)
                    line_end_x = end_x - 10 * math.cos(angle)
                    line_end_y = end_y - 10 * math.sin(angle)
                    draw.line(
                        [(start_x, start_y), (line_end_x, line_end_y)],
                        fill=(218, 45, 71),
                        width=3,
                    )

                    # Draw arrowhead
                    arrow_size = 12
                    left_x = end_x - arrow_size * math.cos(angle + math.pi / 6)
                    left_y = end_y - arrow_size * math.sin(angle + math.pi / 6)
                    right_x = end_x - arrow_size * math.cos(angle - math.pi / 6)
                    right_y = end_y - arrow_size * math.sin(angle - math.pi / 6)
                    draw.polygon(
                        [(end_x, end_y), (left_x, left_y), (right_x, right_y)],
                        fill=(218, 45, 71),
                    )

        return img

    def save_gif(self, frames: List[Image.Image], filename: str, duration: int = 50):
        """
        Save frames as animated GIF

        Args:
            frames: List of PIL Images
            filename: Output filename
            duration: Duration of each frame in milliseconds
        """
        if frames:
            frames[0].save(
                filename,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
            )
