"""
Unified viewers for Malmo agent runs.

`full`:
- compact realtime panel focused on video + RL-critical telemetry

`terminal`:
- minimal single-line updates
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

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
        action: Dict[str, Any],
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

        action_text = "-"
        if isinstance(action, dict) and "action_command" in action:
            idx = action.get("action_index")
            action_text = f"id={idx} cmd={action.get('action_command')}"
        elif isinstance(action, dict):
            move = _as_float(action.get("move"))
            turn = _as_float(action.get("turn"))
            jump = _as_float(action.get("jump"))
            action_text = f"m={move:+.2f},t={turn:+.2f},j={jump:.0f}"
        elif action is not None:
            action_text = str(action)

        dead_flag = ""
        if agent_metrics and agent_metrics.get("is_dead"):
            dead_flag = " DEAD"

        print(
            f"s={step:03d}/{self.max_steps:03d} "
            f"pos=({x:7.2f},{y:7.2f},{z:7.2f}) "
            f"life={life:5.1f} "
            f"a[{action_text}] "
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
    Compact realtime panel with game video + relevant RL telemetry.
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
        self.current_action_index: Optional[int] = None
        self.current_action_command: str = "-"
        self.current_pos = (0.0, 0.0, 0.0)
        self.has_valid_pose = False
        self.agent_metrics: Dict[str, Any] = {}
        self.frame: Optional[np.ndarray] = None
        self.start_time = time.time()
        self.episode_start = time.time()

        self._layout = self._build_layout()

    def _build_layout(self) -> Dict[str, "pygame.Rect"]:
        pad = 12
        header_h = 46
        content_top = pad * 2 + header_h
        content_h = self.height - content_top - pad
        left_w = int(self.width * 0.68)

        header = pygame.Rect(pad, pad, self.width - 2 * pad, header_h)
        video = pygame.Rect(pad, content_top, left_w - 2 * pad, content_h)
        right_x = left_w
        right_w = self.width - left_w - pad
        vitals_h = int(content_h * 0.40)
        vitals = pygame.Rect(right_x, content_top, right_w, vitals_h)
        learning = pygame.Rect(right_x, vitals.bottom + pad, right_w, content_h - vitals_h - pad)

        return {
            "header": header,
            "video": video,
            "vitals": vitals,
            "learning": learning,
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
        self.current_action_index = None
        self.current_action_command = "-"

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
        self.current_action = {}
        self.current_action_index = None
        self.current_action_command = "-"
        if isinstance(action, dict):
            if "action_index" in action:
                try:
                    self.current_action_index = int(action.get("action_index"))
                except (TypeError, ValueError):
                    self.current_action_index = None
            if "action_command" in action:
                self.current_action_command = str(action.get("action_command"))
            else:
                # Fallback for old continuous-style actions.
                move = _as_float(action.get("move"))
                turn = _as_float(action.get("turn"))
                jump = _as_float(action.get("jump"))
                self.current_action = {"move": move, "turn": turn, "jump": jump}
                self.current_action_command = f"m={move:+.2f} t={turn:+.2f} j={jump:.0f}"
        elif action is not None:
            self.current_action_command = str(action)

        self.current_reward = reward
        self.total_reward = total_reward
        self.avg_reward = total_reward / max(step, 1)

        self.frame = self._extract_frame(observation)
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
        self._draw_vitals(self._layout["vitals"])
        self._draw_learning(self._layout["learning"])

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
        subtitle = self.font_small.render("full viewer | focused telemetry", True, self.colors["text_muted"])
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

    def _draw_vitals(self, rect: "pygame.Rect") -> None:
        self._draw_panel(rect, "Vitals")
        area = rect.inflate(-20, -40)
        area.y += 20

        x = area.x
        y = area.y
        px, py, pz = self.current_pos

        lines = [
            f"reward_now: {self.current_reward:+.3f}",
            f"reward_total: {self.total_reward:+.3f}",
            f"reward_avg: {self.avg_reward:+.3f}",
            f"life: {self.life:.1f}",
            f"pos: x={px:.2f} y={py:.2f} z={pz:.2f}",
            f"is_dead: {self.agent_metrics.get('is_dead', False)}",
        ]
        for line in lines:
            surf = self.font_small.render(line, True, self.colors["text_main"])
            self.screen.blit(surf, (x, y))
            y += 20

        y += 6
        self._draw_life_bar(x, y, area.width, 12, self.life, 20.0)

    def _draw_learning(self, rect: "pygame.Rect") -> None:
        self._draw_panel(rect, "Saved State + Learning")
        area = rect.inflate(-20, -40)
        area.y += 20

        x = area.x
        y = area.y
        lines = [
            f"action_idx: {self.current_action_index if self.current_action_index is not None else '-'}",
            f"action_cmd: {self.current_action_command}",
            f"last_state: {self.agent_metrics.get('last_state', '-')}",
            f"q_states: {self.agent_metrics.get('q_states', '-')}",
            f"q_table: {self.agent_metrics.get('q_table_path', '-')}",
            f"epsilon: {self.agent_metrics.get('epsilon', '-')}",
            f"alpha: {self.agent_metrics.get('alpha', '-')}",
            f"gamma: {self.agent_metrics.get('gamma', '-')}",
            f"policy: {self.agent_metrics.get('policy', self.policy)}",
            f"step: {self.current_step}/{self.max_steps}",
        ]

        max_chars = max(24, area.width // 8)
        for line in lines:
            text = str(line)
            if len(text) > max_chars:
                text = text[: max_chars - 3] + "..."
            surf = self.font_small.render(text, True, self.colors["text_main"])
            self.screen.blit(surf, (x, y))
            y += 20

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
