"""
Unified viewers for Malmo agent runs.

`full`:
- single realtime panel with Minecraft video frame + agent/environment metrics

`terminal`:
- minimal single-line updates
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None
    PYGAME_AVAILABLE = False


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _collect_scalars(data: Any, prefix: str = "", out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if out is None:
        out = {}

    if isinstance(data, dict):
        for key, value in data.items():
            if key in {"_frame", "frame", "video"}:
                continue
            nested = f"{prefix}.{key}" if prefix else str(key)
            _collect_scalars(value, nested, out)
        return out

    if isinstance(data, (list, tuple)):
        if len(data) <= 6 and all(isinstance(v, (int, float, bool, str)) for v in data):
            out[prefix] = "[" + ", ".join(_format_value(v) for v in data) + "]"
        return out

    if isinstance(data, (int, float, bool, str)) and prefix:
        out[prefix] = data
    return out


class TerminalMinimalViewer:
    """
    Minimal terminal viewer: one compact line per step.
    """

    def __init__(self) -> None:
        self.episode = 0
        self.total_episodes = 0
        self.max_steps = 0
        self.last_pos = (0.0, 0.0, 0.0)
        self.last_life = 20.0

    def start(self, total_episodes: int, max_steps: int, policy: str) -> None:
        self.total_episodes = total_episodes
        self.max_steps = max_steps
        print(f"[viewer:terminal] episodes={total_episodes} max_steps={max_steps} policy={policy}")

    def start_episode(self, episode: int) -> None:
        self.episode = episode
        print(f"\n[episode {episode}/{self.total_episodes}]")

    def update_step(
        self,
        step: int,
        action: Dict[str, float],
        observation: Dict[str, Any],
        reward: float,
        total_reward: float,
        agent_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        x = _as_optional_float(observation.get("XPos"))
        y = _as_optional_float(observation.get("YPos"))
        z = _as_optional_float(observation.get("ZPos"))
        life = _as_optional_float(observation.get("Life"))

        if x is not None and y is not None and z is not None:
            self.last_pos = (x, y, z)
        if life is not None:
            self.last_life = life

        x, y, z = self.last_pos
        life = self.last_life

        move = _as_float(action.get("move"))
        turn = _as_float(action.get("turn"))
        jump = _as_float(action.get("jump"))

        dead_flag = ""
        if agent_metrics and agent_metrics.get("is_dead"):
            dead_flag = " DEAD"

        print(
            f"s={step:03d}/{self.max_steps:03d} "
            f"pos=({x:7.2f},{y:7.2f},{z:7.2f}) "
            f"life={life:5.1f} "
            f"a[m={move:+.2f},t={turn:+.2f},j={jump:.0f}] "
            f"r={reward:+.2f} R={total_reward:+.2f}{dead_flag}"
        )

    def end_episode(self) -> None:
        pass

    def close(self) -> None:
        pass

    def is_running(self) -> bool:
        return True


class Unified3DViewer:
    """
    Single realtime panel with game video + metrics/charts.
    """

    def __init__(
        self,
        width: int = 1360,
        height: int = 780,
        fps_limit: int = 30,
        history_size: int = 300,
    ) -> None:
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is not installed. Install with: pip install pygame")

        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Malmo Agent Viewer - Full")

        self.width = width
        self.height = height
        self.fps_limit = fps_limit
        self.history_size = history_size

        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Segoe UI", 16)
        self.font_small = pygame.font.SysFont("Consolas", 14)
        self.font_title = pygame.font.SysFont("Segoe UI Semibold", 28, bold=True)

        self.colors = {
            "bg_top": (9, 14, 22),
            "bg_bottom": (16, 25, 38),
            "panel_bg": (19, 30, 46),
            "panel_border": (64, 94, 132),
            "panel_header": (143, 191, 255),
            "text_main": (223, 236, 252),
            "text_muted": (153, 178, 209),
            "accent_blue": (86, 169, 255),
            "accent_orange": (238, 178, 96),
            "accent_green": (93, 209, 143),
            "accent_red": (222, 83, 83),
        }

        self.running = True
        self.total_episodes = 0
        self.max_steps = 0
        self.policy = ""
        self.episode = 0
        self.current_step = 0
        self.current_reward = 0.0
        self.total_reward = 0.0
        self.avg_reward = 0.0
        self.life = 20.0
        self.current_action: Dict[str, float] = {}
        self.current_pos = (0.0, 0.0, 0.0)
        self.has_valid_pose = False
        self.agent_metrics: Dict[str, Any] = {}
        self.scalar_obs: Dict[str, Any] = {}
        self.frame: Optional[np.ndarray] = None
        self.start_time = time.time()
        self.episode_start = time.time()
        self._scalar_page = 0
        self._last_page_switch = time.time()

        self.reward_history: deque[float] = deque(maxlen=history_size)
        self.height_history: deque[float] = deque(maxlen=history_size)
        self.trajectory: deque[Tuple[float, float]] = deque(maxlen=history_size)

        self._layout = self._build_layout()

    def _build_layout(self) -> Dict[str, "pygame.Rect"]:
        pad = 12
        header_h = 40
        video_h = int((self.height - header_h - 4 * pad) * 0.72)
        left_w = int(self.width * 0.68)

        header = pygame.Rect(pad, pad, self.width - 2 * pad, header_h)
        video = pygame.Rect(pad, pad * 2 + header_h, left_w - 2 * pad, video_h)
        bottom_y = video.bottom + pad
        bottom_h = self.height - bottom_y - pad
        chart = pygame.Rect(pad, bottom_y, int(video.width * 0.58), bottom_h)
        map_rect = pygame.Rect(chart.right + pad, bottom_y, video.width - chart.width - pad, bottom_h)
        stats = pygame.Rect(left_w, pad * 2 + header_h, self.width - left_w - pad, self.height - header_h - 3 * pad)

        return {
            "header": header,
            "video": video,
            "chart": chart,
            "map": map_rect,
            "stats": stats,
        }

    def _process_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False

    def _draw_background(self) -> None:
        for y in range(self.height):
            t = y / max(1, self.height - 1)
            r = int(self.colors["bg_top"][0] * (1 - t) + self.colors["bg_bottom"][0] * t)
            g = int(self.colors["bg_top"][1] * (1 - t) + self.colors["bg_bottom"][1] * t)
            b = int(self.colors["bg_top"][2] * (1 - t) + self.colors["bg_bottom"][2] * t)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.width, y))

    def start(self, total_episodes: int, max_steps: int, policy: str) -> None:
        self.total_episodes = total_episodes
        self.max_steps = max_steps
        self.policy = policy
        self.start_time = time.time()

    def start_episode(self, episode: int) -> None:
        self.episode = episode
        self.current_step = 0
        self.current_reward = 0.0
        self.total_reward = 0.0
        self.avg_reward = 0.0
        self.episode_start = time.time()
        self.reward_history.clear()
        self.height_history.clear()
        self.trajectory.clear()

    def update_step(
        self,
        step: int,
        action: Dict[str, float],
        observation: Dict[str, Any],
        reward: float,
        total_reward: float,
        agent_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._process_events()
        if not self.running:
            return

        self.current_step = step
        self.current_action = {
            "move": _as_float(action.get("move")),
            "turn": _as_float(action.get("turn")),
            "jump": _as_float(action.get("jump")),
        }
        self.current_reward = reward
        self.total_reward = total_reward
        self.avg_reward = total_reward / max(step, 1)

        self.frame = self._extract_frame(observation)
        self.scalar_obs = _collect_scalars(observation)
        self.agent_metrics = dict(agent_metrics or {})

        x_obs = _as_optional_float(observation.get("XPos"))
        y_obs = _as_optional_float(observation.get("YPos"))
        z_obs = _as_optional_float(observation.get("ZPos"))
        life_obs = _as_optional_float(observation.get("Life"))

        if x_obs is not None and y_obs is not None and z_obs is not None:
            self.current_pos = (x_obs, y_obs, z_obs)
            self.has_valid_pose = True
        if life_obs is not None:
            self.life = life_obs

        self.reward_history.append(self.total_reward)
        if self.has_valid_pose:
            self.height_history.append(self.current_pos[1])
            self.trajectory.append((self.current_pos[0], self.current_pos[2]))

        self.render()

    def _extract_frame(self, observation: Dict[str, Any]) -> Optional[np.ndarray]:
        frame = observation.get("_frame")
        if frame is None:
            frame = observation.get("frame")
        if frame is None and hasattr(observation, "shape"):
            frame = observation
        if frame is None:
            return None

        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3:
            return None
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.shape[2] != 3:
            return None
        # Malmo frames are bottom-up for this pipeline; flip vertically for correct orientation.
        return np.flipud(arr)

    def render(self) -> None:
        self._process_events()
        if not self.running:
            return

        self._draw_background()

        self._draw_header(self._layout["header"])
        self._draw_video(self._layout["video"])
        self._draw_chart(self._layout["chart"])
        self._draw_map(self._layout["map"])
        self._draw_stats(self._layout["stats"])

        pygame.display.flip()
        self.clock.tick(self.fps_limit)

    def _draw_panel(self, rect: "pygame.Rect", title: str) -> None:
        pygame.draw.rect(self.screen, self.colors["panel_bg"], rect, border_radius=8)
        pygame.draw.rect(self.screen, self.colors["panel_border"], rect, width=1, border_radius=8)
        title_surf = self.font.render(title, True, self.colors["panel_header"])
        self.screen.blit(title_surf, (rect.x + 10, rect.y + 8))
        pygame.draw.line(
            self.screen,
            self.colors["panel_border"],
            (rect.x + 10, rect.y + 32),
            (rect.right - 10, rect.y + 32),
            1,
        )

    def _draw_header(self, rect: "pygame.Rect") -> None:
        pygame.draw.rect(self.screen, (10, 18, 28), rect, border_radius=8)
        pygame.draw.rect(self.screen, self.colors["panel_border"], rect, width=1, border_radius=8)

        title = self.font_title.render("MALMO AGENT", True, self.colors["text_main"])
        self.screen.blit(title, (rect.x + 10, rect.y + 8))
        subtitle = self.font_small.render("single panel | video + telemetry", True, self.colors["text_muted"])
        self.screen.blit(subtitle, (rect.x + 250, rect.y + 18))

        elapsed = time.time() - self.start_time
        steps_per_sec = self.current_step / max(0.001, time.time() - self.episode_start)
        right = (
            f"ep {self.episode}/{self.total_episodes} | "
            f"step {self.current_step}/{self.max_steps} | "
            f"policy {self.policy} | "
            f"sps {steps_per_sec:.2f} | "
            f"t {elapsed:0.1f}s"
        )
        right_surf = self.font_small.render(right, True, self.colors["text_muted"])
        self.screen.blit(right_surf, (rect.right - right_surf.get_width() - 10, rect.y + 13))

    def _draw_video(self, rect: "pygame.Rect") -> None:
        self._draw_panel(rect, "Minecraft Video")
        body = rect.inflate(-20, -40)
        body.y += 20

        pygame.draw.rect(self.screen, (6, 10, 15), body, border_radius=6)
        pygame.draw.rect(self.screen, (49, 62, 80), body, width=1, border_radius=6)

        if self.frame is None:
            msg = self.font.render("Waiting for first frame from Malmo...", True, self.colors["text_muted"])
            self.screen.blit(
                msg,
                (body.centerx - msg.get_width() // 2, body.centery - msg.get_height() // 2),
            )
            return

        frame_surface = pygame.surfarray.make_surface(np.swapaxes(self.frame, 0, 1))
        frame_h, frame_w = self.frame.shape[:2]
        scale = min(body.width / max(1, frame_w), body.height / max(1, frame_h))
        render_w = max(1, int(frame_w * scale))
        render_h = max(1, int(frame_h * scale))

        scaled = pygame.transform.smoothscale(frame_surface, (render_w, render_h))
        target = pygame.Rect(
            body.x + (body.width - render_w) // 2,
            body.y + (body.height - render_h) // 2,
            render_w,
            render_h,
        )
        self.screen.blit(scaled, target.topleft)
        pygame.draw.rect(self.screen, self.colors["accent_blue"], target, width=1, border_radius=2)

        info = self.font_small.render(f"frame {frame_w}x{frame_h}", True, self.colors["text_muted"])
        self.screen.blit(info, (body.x + 10, body.y + 8))

    def _draw_chart(self, rect: "pygame.Rect") -> None:
        self._draw_panel(rect, "Reward + Height")
        area = rect.inflate(-20, -40)
        area.y += 20

        if not self.reward_history:
            return

        reward_rect = pygame.Rect(area.x, area.y, area.width, int(area.height * 0.52))
        height_rect = pygame.Rect(area.x, reward_rect.bottom + 8, area.width, area.height - reward_rect.height - 8)

        self._draw_line_series(
            reward_rect,
            list(self.reward_history),
            line_color=self.colors["accent_green"],
            title="total_reward",
        )
        self._draw_line_series(
            height_rect,
            list(self.height_history),
            line_color=self.colors["accent_blue"],
            title="height_y",
        )

    def _draw_line_series(
        self,
        rect: "pygame.Rect",
        values: List[float],
        line_color: Tuple[int, int, int],
        title: str,
    ) -> None:
        pygame.draw.rect(self.screen, (11, 18, 27), rect, border_radius=6)
        pygame.draw.rect(self.screen, self.colors["panel_border"], rect, width=1, border_radius=6)

        if not values:
            return

        lo = min(values)
        hi = max(values)
        span = hi - lo

        if len(values) == 1 or span < 1e-6:
            y_mid = rect.y + rect.height // 2
            pygame.draw.line(
                self.screen,
                line_color,
                (rect.x + 1, y_mid),
                (rect.right - 2, y_mid),
                2,
            )
        else:
            points: List[Tuple[int, int]] = []
            for idx, value in enumerate(values):
                x = rect.x + int(idx / (len(values) - 1) * (rect.width - 1))
                y_norm = (value - lo) / span
                y = rect.bottom - 1 - int(y_norm * (rect.height - 1))
                points.append((x, y))

            if len(points) >= 2:
                pygame.draw.lines(self.screen, line_color, False, points, 2)

        title_text = self.font_small.render(
            f"{title}: min={lo:.2f} max={hi:.2f} last={values[-1]:.2f}",
            True,
            self.colors["text_muted"],
        )
        self.screen.blit(title_text, (rect.x + 8, rect.y + 6))

    def _draw_map(self, rect: "pygame.Rect") -> None:
        self._draw_panel(rect, "Trajectory (X,Z)")
        area = rect.inflate(-20, -40)
        area.y += 20

        pygame.draw.rect(self.screen, (11, 18, 27), area, border_radius=6)
        pygame.draw.rect(self.screen, self.colors["panel_border"], area, width=1, border_radius=6)

        if len(self.trajectory) < 2:
            return

        xs = [p[0] for p in self.trajectory]
        zs = [p[1] for p in self.trajectory]
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        span_x = max(max_x - min_x, 1e-6)
        span_z = max(max_z - min_z, 1e-6)

        points: List[Tuple[int, int]] = []
        for x, z in self.trajectory:
            px = area.x + int((x - min_x) / span_x * (area.width - 1))
            pz = area.bottom - 1 - int((z - min_z) / span_z * (area.height - 1))
            points.append((px, pz))

        if len(points) >= 2:
            pygame.draw.lines(self.screen, self.colors["accent_orange"], False, points, 2)
        pygame.draw.circle(self.screen, (255, 232, 163), points[-1], 5)

        label = self.font_small.render(
            f"range x={min_x:.2f}..{max_x:.2f} z={min_z:.2f}..{max_z:.2f}",
            True,
            self.colors["text_muted"],
        )
        self.screen.blit(label, (area.x + 8, area.y + 6))

    def _draw_stats(self, rect: "pygame.Rect") -> None:
        self._draw_panel(rect, "Live Metrics")
        area = rect.inflate(-20, -40)
        area.y += 20

        x = area.x
        y = area.y

        state_title = self.font.render("Agent State", True, self.colors["panel_header"])
        self.screen.blit(state_title, (x, y))
        y += 20

        px, py, pz = self.current_pos
        lines = [
            f"agent.policy: {self.agent_metrics.get('policy', self.policy)}",
            f"agent.step_count: {self.agent_metrics.get('step_count', self.current_step)}",
            f"agent.total_reward: {self.total_reward:.3f}",
            f"agent.avg_reward: {self.avg_reward:.3f}",
            f"agent.life: {self.agent_metrics.get('life', self.life)}",
            f"agent.is_dead: {self.agent_metrics.get('is_dead', False)}",
            f"agent.target_height: {self.agent_metrics.get('target_height', '-')}",
            f"agent.max_height: {self.agent_metrics.get('max_height', '-')}",
            f"pos: x={px:.2f} y={py:.2f} z={pz:.2f}",
        ]
        for line in lines:
            surf = self.font_small.render(line, True, self.colors["text_main"])
            self.screen.blit(surf, (x, y))
            y += 18

        y += 6
        self._draw_life_bar(x, y, area.width, 12, self.life, 20.0)
        y += 24

        controls_title = self.font.render("Controls", True, self.colors["panel_header"])
        self.screen.blit(controls_title, (x, y))
        y += 20

        self._draw_action_bar(x, y, area.width, "move", self.current_action.get("move", 0.0), -1.0, 1.0)
        y += 26
        self._draw_action_bar(x, y, area.width, "turn", self.current_action.get("turn", 0.0), -1.0, 1.0)
        y += 26
        self._draw_action_bar(x, y, area.width, "jump", self.current_action.get("jump", 0.0), 0.0, 1.0)
        y += 30

        obs_title = self.font.render("Environment Scalars", True, self.colors["panel_header"])
        self.screen.blit(obs_title, (x, y))
        y += 20

        remaining_h = area.bottom - y - 4
        row_h = 16
        max_rows = max(1, remaining_h // row_h)
        items = sorted(self.scalar_obs.items(), key=lambda kv: kv[0])

        total_pages = max(1, math.ceil(len(items) / max_rows))
        now = time.time()
        if total_pages > 1 and now - self._last_page_switch > 2.0:
            self._scalar_page = (self._scalar_page + 1) % total_pages
            self._last_page_switch = now

        start = self._scalar_page * max_rows
        end = start + max_rows
        page_items = items[start:end]

        for key, value in page_items:
            text = f"{key}: {_format_value(value)}"
            surf = self.font_small.render(text, True, self.colors["text_muted"])
            self.screen.blit(surf, (x, y))
            y += row_h

        if total_pages > 1:
            footer = self.font_small.render(
                f"page {self._scalar_page + 1}/{total_pages}",
                True,
                self.colors["text_muted"],
            )
            self.screen.blit(footer, (area.right - footer.get_width(), area.bottom - 16))

    def _draw_life_bar(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        current: float,
        max_value: float,
    ) -> None:
        ratio = max(0.0, min(1.0, current / max(max_value, 1e-6)))
        pygame.draw.rect(self.screen, (66, 28, 35), (x, y, width, height), border_radius=4)
        pygame.draw.rect(
            self.screen,
            self.colors["accent_red"],
            (x, y, int(width * ratio), height),
            border_radius=4,
        )
        label = self.font_small.render(f"life: {current:.1f}/{max_value:.1f}", True, self.colors["text_main"])
        self.screen.blit(label, (x, y - 14))

    def _draw_action_bar(
        self,
        x: int,
        y: int,
        width: int,
        name: str,
        value: float,
        min_value: float,
        max_value: float,
    ) -> None:
        label_y = y - 14
        pygame.draw.rect(self.screen, (18, 26, 37), (x, y, width, 10), border_radius=4)
        span = max(max_value - min_value, 1e-6)
        ratio = max(0.0, min(1.0, (value - min_value) / span))
        fill = int(width * ratio)
        pygame.draw.rect(self.screen, self.colors["accent_blue"], (x, y, fill, 10), border_radius=4)
        label = self.font_small.render(f"{name}: {value:+.3f}", True, self.colors["text_main"])
        self.screen.blit(label, (x, label_y))

    def end_episode(self) -> None:
        pass

    def close(self) -> None:
        self.running = False
        if PYGAME_AVAILABLE:
            pygame.quit()

    def is_running(self) -> bool:
        return self.running


def create_unified_viewer(mode: str = "full") -> Optional[Any]:
    """
    Factory for viewer modes supported by the new pipeline.
    """
    if mode == "none" or mode is None:
        return None
    if mode == "terminal":
        return TerminalMinimalViewer()
    if mode == "full":
        if PYGAME_AVAILABLE:
            return Unified3DViewer()
        return TerminalMinimalViewer()
    return None
