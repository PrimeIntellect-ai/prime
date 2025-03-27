import random
import time

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Profiler:
    """Profiler that tracks nested sessions and prints their durations in a tree structure."""

    def __init__(self):
        # List of top-level sessions (roots)
        self.root_sessions = []
        # Stack of currently open sessions
        self.session_stack = []
        self.prev_time = 0

    def session(self, name: str):
        """Returns a context manager for timing a named session."""
        return _SessionContextManager(self, name)

    def start_session(self, name: str):
        new_node = SessionNode(name)
        new_node.start_time = max(time.perf_counter(), self.prev_time)
        self.prev_time = new_node.start_time

        if self.session_stack:
            parent_node = self.session_stack[-1]
            parent_node.children.append(new_node)
            new_node.parent = parent_node
        else:
            self.root_sessions.append(new_node)

        self.session_stack.append(new_node)

    def end_session(self):
        if not self.session_stack:
            raise RuntimeError("No session is currently open to end.")
        node = self.session_stack.pop()
        node.end_time = time.perf_counter()

    def print_report(self):
        """Prints a tree-structured timing report of all recorded sessions."""
        for session in self.root_sessions:
            self._print_session(session, level=0)

    def _print_session(self, session_node: "SessionNode", level: int):
        indent = '  ' * level
        print(f"{indent}- {session_node.name}: {session_node.duration:.6f} seconds")
        for child in session_node.children:
            self._print_session(child, level + 1)

    def export_timeline(
            self,
            filename=None,
            width=2400,
            row_height=40,
            return_image=False
    ):
        """
        Draw a timeline image of this Profiler's data.
        If return_image=True, return a PIL Image object instead of saving to a file.
        If filename is provided and return_image=False, save the image to disk as filename.
        """

        # Gather nodes
        all_nodes = []

        def collect_nodes(node, depth=0):
            all_nodes.append((node, depth))
            for c in node.children:
                collect_nodes(c, depth + 1)

        for root in self.root_sessions:
            collect_nodes(root)

        if not all_nodes:
            print("No recorded sessions.")
            return None

        min_start = min(n.start_time for n, _ in all_nodes)
        max_end = max(n.end_time for n, _ in all_nodes)
        total_duration = max_end - min_start
        if total_duration <= 0:
            print("Profiler data has no measurable duration.")
            return None

        # Layout
        max_depth = max(d for _, d in all_nodes)
        PADDING_X = 50
        PADDING_Y = 30
        TIMELINE_AXIS_HEIGHT = 40
        BARS_OFFSET = 15
        effective_width = width - 2 * PADDING_X
        height = TIMELINE_AXIS_HEIGHT + BARS_OFFSET + (max_depth + 1) * row_height + PADDING_Y

        # Create image
        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Font
        try:
            font = ImageFont.truetype("Arial.ttf", int(row_height / 3))
        except OSError:
            font = ImageFont.load_default()

        # Make color generation deterministic
        random.seed(1234)

        def pastel_color():
            r = 150 + random.randint(0, 105)
            g = 150 + random.randint(0, 105)
            b = 150 + random.randint(0, 105)
            return (r, g, b)

        # Draw top timeline axis
        axis_y = TIMELINE_AXIS_HEIGHT // 2
        draw.line([(PADDING_X, axis_y), (PADDING_X + effective_width, axis_y)], fill=(0, 0, 0), width=1)

        # Ticks
        num_ticks = 10
        step = total_duration / num_ticks
        for i in range(num_ticks + 1):
            t = i * step
            x_tick = PADDING_X + int((t / total_duration) * effective_width)

            # Long gray line downward
            draw.line([(x_tick, axis_y), (x_tick, height)], fill=(180, 180, 180), width=1)

            # Short black tick
            tick_size = 5
            draw.line([(x_tick, axis_y - tick_size), (x_tick, axis_y + tick_size)], fill=(0, 0, 0), width=1)

            # Label
            label_text = f"{t:.2f}s"
            bbox = draw.textbbox((0, 0), label_text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            draw.text((x_tick - w // 2, axis_y + tick_size + 2), label_text, fill=(0, 0, 0), font=font)

        # Helper to truncate text if bar is too short
        def truncate_text_to_fit(original_text, max_width):
            bbox = draw.textbbox((0, 0), original_text, font=font)
            text_w = bbox[2] - bbox[0]
            if text_w <= max_width:
                return original_text

            base = original_text
            while base:
                trial = base + "..."
                trial_w = draw.textbbox((0, 0), trial, font=font)
                if (trial_w[2] - trial_w[0]) <= max_width:
                    return trial
                base = base[:-1]
            return "..."

        # Draw session bars
        outline_color = (120, 120, 120)
        for node, depth in all_nodes:
            start_off = node.start_time - min_start
            end_off = node.end_time - min_start
            x1 = PADDING_X + int((start_off / total_duration) * effective_width)
            x2 = PADDING_X + int((end_off / total_duration) * effective_width)
            y1 = TIMELINE_AXIS_HEIGHT + BARS_OFFSET + depth * row_height
            y2 = y1 + row_height - 10

            rect_color = pastel_color()
            draw.rectangle([(x1, y1), (x2, y2)], fill=rect_color, outline=outline_color)

            # Label
            raw_text = f"{node.name} ({node.duration:.4f}s)"
            bar_width = (x2 - x1) - 10
            label_text = truncate_text_to_fit(raw_text, bar_width)
            draw.text((x1 + 5, y1 + 5), label_text, fill=(0, 0, 0), font=font)

        # Return or save
        if return_image:
            return img
        else:
            if filename:
                img.save(filename)
                print(f"Timeline exported to {filename}")
            return None


class SessionNode:
    """Tree node holding data about an individual session."""

    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.children = []
        self.parent = None

    @property
    def duration(self):
        """Returns the duration of the session (in seconds)."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return 0


class _SessionContextManager:
    """Context manager used by Profiler to start/end sessions."""

    def __init__(self, profiler, name):
        self.profiler = profiler
        self.name = name

    def __enter__(self):
        self.profiler.start_session(self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_session()
        # Returning False so that any exception is still raised
        return False


class ProfilerCollection:
    """
    Collects multiple Profiler instances and defers rendering them
    until `render_as_video(...)` is called. Each Profiler is converted
    to a PIL image (via profiler.export_timeline(return_image=True)) exactly
    once (lazy rendering). If you add new profilers later and call
    `render_as_video` again, it will only render the new ones.
    """

    def __init__(self):
        """
        We'll store a list of dicts with keys:
         - 'profiler': the Profiler instance
         - 'label': an optional string label
         - 'image': a PIL Image cache (None until we actually render)
        """
        self.frames = []

    def add_profiler(self, profiler, label=None):
        """
        Store this profiler for future rendering. We do NOT call export_timeline here,
        so it's fully deferred. We only do the actual rendering on render_as_video().
        """
        if label is None:
            label = f"Frame {len(self.frames)}"
        entry = {
            "profiler": profiler,
            "label": label,
            "image": None  # will be filled in when we do lazy rendering
        }
        self.frames.append(entry)

    def render_as_video(self, out_filename="profiler_video.mp4", fps=2):
        """
        Render all frames to a video. For each stored Profiler that hasn't been
        rendered yet (image=None), we call export_timeline(return_image=True),
        cache the result in memory, and then use that to build the final video.
        If you call this multiple times, only newly added profilers get rendered.
        """
        if not self.frames:
            print("ProfilerCollection is empty; nothing to render.")
            return

        # Open an imageio writer for the final video
        with imageio.get_writer(out_filename, fps=fps) as writer:
            # Go through each frame
            for i, entry in enumerate(self.frames):
                if entry["image"] is None:
                    # We haven't rendered this profiler's timeline yet => do it now
                    profiler = entry["profiler"]
                    img = profiler.export_timeline(return_image=True)
                    if img is None:
                        # e.g. no sessions or 0 duration
                        continue
                    entry["image"] = img  # cache the PIL image in memory

                # We now have a cached PIL image
                image_pil = entry["image"]

                # Convert to a numpy array for imageio
                frame_array = np.array(image_pil)
                writer.append_data(frame_array)

        print(f"Video rendered with {len(self.frames)} frames at {fps} FPS => {out_filename}")
