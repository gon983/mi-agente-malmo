"""
Basic agent for Microsoft Malmo.
"""

from __future__ import annotations

import copy
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


DEFAULT_WORLD_RULES: Dict[str, Any] = {
    "objective": {
        "target_height": 0.0,
    }
}

DEFAULT_AGENT_PARAMS: Dict[str, Any] = {
    "run": {
        "episodes": 5,
        "max_steps": 50,
    },
    "behavior": {
        "policy": "random",
        "action_delay": 0.05,
        "verbose": True,
    },
    "actions": {
        "random": {
            "move_min": -1.0,
            "move_max": 1.0,
            "turn_min": -1.0,
            "turn_max": 1.0,
            "jump_probability": 0.25,
        },
        "forward": {
            "move": 1.0,
            "turn": 0.0,
            "jump": 0.0,
        },
        "explore": {
            "turn_interval": 10,
            "turn_choices": [-1.0, 1.0],
            "move": 1.0,
            "turn_jitter_min": -0.2,
            "turn_jitter_max": 0.2,
            "jump": 0.0,
        },
    },
    "termination": {
        "death_on_zero_life": True,
        "min_life": 0.0,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(config_path: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load YAML config from disk, returning defaults if file is missing.
    """
    default_dict = copy.deepcopy(default or {})
    path = Path(config_path)

    if not path.exists():
        return default_dict

    if yaml is None:
        logger.warning("PyYAML is not installed. Using defaults for %s", config_path)
        return default_dict

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        logger.warning("YAML root in %s is not a mapping. Using defaults.", config_path)
        return default_dict

    return _deep_merge(default_dict, loaded)


class BasicAgent:
    """
    Minimal Malmo agent with configurable policy/action/termination knobs.
    """

    def __init__(
        self,
        policy: Optional[str] = None,
        action_delay: Optional[float] = None,
        verbose: Optional[bool] = None,
        agent_params: Optional[Dict[str, Any]] = None,
        world_rules: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.agent_params = _deep_merge(DEFAULT_AGENT_PARAMS, agent_params or {})
        self.world_rules = _deep_merge(DEFAULT_WORLD_RULES, world_rules or {})

        behavior_cfg = self.agent_params.get("behavior", {})
        termination_cfg = self.agent_params.get("termination", {})

        self.policy = policy if policy is not None else str(behavior_cfg.get("policy", "random"))
        self.action_delay = float(
            action_delay if action_delay is not None else behavior_cfg.get("action_delay", 0.05)
        )
        self.verbose = bool(verbose if verbose is not None else behavior_cfg.get("verbose", True))

        self.death_on_zero_life = bool(termination_cfg.get("death_on_zero_life", True))
        self.min_life = float(termination_cfg.get("min_life", 0.0))
        self.target_height = float(
            self.world_rules.get("objective", {}).get("target_height", 0.0)
        )

        self.step_count = 0
        self.total_reward = 0.0
        self.max_height = float("-inf")
        self.last_life: Optional[float] = None
        self.last_observation: Dict[str, Any] = {}
        self._is_dead = False

    def get_action(self, observation: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        self.step_count += 1

        if self.policy == "random":
            return self._random_action()
        if self.policy == "forward":
            return self._forward_action()
        if self.policy == "explore":
            return self._explore_action(observation)
        return self._random_action()

    def _random_action(self) -> Dict[str, float]:
        cfg = self.agent_params.get("actions", {}).get("random", {})
        move = random.uniform(float(cfg.get("move_min", -1.0)), float(cfg.get("move_max", 1.0)))
        turn = random.uniform(float(cfg.get("turn_min", -1.0)), float(cfg.get("turn_max", 1.0)))
        jump_probability = float(cfg.get("jump_probability", 0.25))
        jump = 1.0 if random.random() < jump_probability else 0.0
        return {"move": move, "turn": turn, "jump": jump}

    def _resolve_jump(self, cfg: Dict[str, Any]) -> float:
        jump_probability = cfg.get("jump_probability")
        if jump_probability is not None:
            try:
                probability = max(0.0, min(1.0, float(jump_probability)))
            except (TypeError, ValueError):
                probability = 0.0
            return 1.0 if random.random() < probability else 0.0
        return float(cfg.get("jump", 0.0))

    def _forward_action(self) -> Dict[str, float]:
        cfg = self.agent_params.get("actions", {}).get("forward", {})
        return {
            "move": float(cfg.get("move", 1.0)),
            "turn": float(cfg.get("turn", 0.0)),
            "jump": self._resolve_jump(cfg),
        }

    def _explore_action(self, observation: Optional[Dict[str, Any]]) -> Dict[str, float]:
        cfg = self.agent_params.get("actions", {}).get("explore", {})
        interval = max(1, int(cfg.get("turn_interval", 10)))
        turn_choices = cfg.get("turn_choices", [-1.0, 1.0]) or [-1.0, 1.0]

        if self.step_count % interval == 0:
            return {
                "move": 0.0,
                "turn": float(random.choice(turn_choices)),
                "jump": self._resolve_jump(cfg),
            }

        return {
            "move": float(cfg.get("move", 1.0)),
            "turn": random.uniform(
                float(cfg.get("turn_jitter_min", -0.2)),
                float(cfg.get("turn_jitter_max", 0.2)),
            ),
            "jump": self._resolve_jump(cfg),
        }

    def process_observation(self, observation: Optional[Dict[str, Any]]) -> None:
        obs = observation or {}
        self.last_observation = obs

        y = _as_float(obs.get("YPos"))
        if y is not None:
            self.max_height = max(self.max_height, y)

        life = _as_float(obs.get("Life"))
        if life is not None:
            self.last_life = life
            if self.death_on_zero_life and life <= self.min_life:
                self._is_dead = True

        if self.verbose and obs:
            x = _as_float(obs.get("XPos"), 0.0)
            y_print = _as_float(obs.get("YPos"), 0.0)
            z = _as_float(obs.get("ZPos"), 0.0)
            life_print = _as_float(obs.get("Life"), 0.0)
            print(
                f"[Step {self.step_count}] Pos: ({x:.2f}, {y_print:.2f}, {z:.2f}) | "
                f"Life: {life_print:.2f}"
            )

    def should_terminate(self, observation: Optional[Dict[str, Any]] = None) -> bool:
        if observation is not None:
            self.process_observation(observation)
        return self._is_dead

    def process_reward(self, reward: float) -> None:
        self.total_reward += reward
        if reward != 0 and self.verbose:
            print(f"  -> Reward: {reward:+.2f} | Total: {self.total_reward:.2f}")

    def reset(self) -> None:
        self.step_count = 0
        self.total_reward = 0.0
        self.max_height = float("-inf")
        self.last_life = None
        self.last_observation = {}
        self._is_dead = False
        if self.verbose:
            print("\n" + "=" * 48)
            print("NEW EPISODE")
            print("=" * 48)

    def episode_summary(self) -> Dict[str, Any]:
        max_height = self.max_height if self.max_height != float("-inf") else None
        return {
            "steps": self.step_count,
            "total_reward": self.total_reward,
            "avg_reward": self.total_reward / max(1, self.step_count),
            "max_height": max_height,
            "target_height": self.target_height,
            "died": self._is_dead,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "life": self.last_life,
            "is_dead": self._is_dead,
            "death_on_zero_life": self.death_on_zero_life,
            "target_height": self.target_height,
            "max_height": self.max_height if self.max_height != float("-inf") else None,
        }


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
