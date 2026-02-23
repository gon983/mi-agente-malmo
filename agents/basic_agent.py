"""
Basic tabular Q-learning agent for Microsoft Malmo.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


State = Tuple[int, ...]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(
    config_path: str,
    default: Optional[Dict[str, Any]] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Load YAML config from disk.

    - strict=False: returns merged defaults when available.
    - strict=True : file must exist, root must be a mapping, no default merge.
    """
    path = Path(config_path)
    default_dict = copy.deepcopy(default or {})

    if strict and not path.exists():
        raise FileNotFoundError(f"Required config file not found: {path}")
    if not strict and not path.exists():
        return default_dict

    if yaml is None:
        if strict:
            raise RuntimeError(f"PyYAML is required to load strict config: {config_path}")
        logger.warning("PyYAML is not installed. Using defaults for %s", config_path)
        return default_dict

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        if strict:
            raise ValueError(f"YAML root in {config_path} must be a mapping.")
        logger.warning("YAML root in %s is not a mapping. Using defaults.", config_path)
        return default_dict

    if strict:
        return loaded

    return _deep_merge(default_dict, loaded)


def _require_mapping(data: Dict[str, Any], key: str, context: str) -> Dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid mapping '{context}.{key}'")
    return value


def _require_list(data: Dict[str, Any], key: str, context: str) -> List[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Missing or invalid list '{context}.{key}'")
    return value


def _require_bool(data: Dict[str, Any], key: str, context: str) -> bool:
    value = data.get(key)
    if isinstance(value, bool):
        return value
    raise ValueError(f"Missing or invalid bool '{context}.{key}'")


def _require_str(data: Dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid string '{context}.{key}'")
    return value.strip()


def _require_int(data: Dict[str, Any], key: str, context: str, min_value: Optional[int] = None) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Missing or invalid int '{context}.{key}'")
    if min_value is not None and value < min_value:
        raise ValueError(f"'{context}.{key}' must be >= {min_value}")
    return value


def _require_float(data: Dict[str, Any], key: str, context: str, min_value: Optional[float] = None) -> float:
    value = data.get(key)
    if not isinstance(value, (float, int)):
        raise ValueError(f"Missing or invalid float '{context}.{key}'")
    casted = float(value)
    if min_value is not None and casted < min_value:
        raise ValueError(f"'{context}.{key}' must be >= {min_value}")
    return casted


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class BasicAgent:
    """
    Tabular Q-learning agent restricted to an explore policy.
    """

    def __init__(
        self,
        policy: Optional[str] = None,
        action_delay: Optional[float] = None,
        verbose: Optional[bool] = None,
        agent_params: Optional[Dict[str, Any]] = None,
        world_rules: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(agent_params, dict):
            raise ValueError("agent_params must be provided as a mapping (strict mode).")
        if not isinstance(world_rules, dict):
            raise ValueError("world_rules must be provided as a mapping (strict mode).")

        self.agent_params = copy.deepcopy(agent_params)
        self.world_rules = copy.deepcopy(world_rules)

        behavior_cfg = _require_mapping(self.agent_params, "behavior", "agent_params")
        termination_cfg = _require_mapping(self.agent_params, "termination", "agent_params")
        actions_cfg = _require_mapping(self.agent_params, "actions", "agent_params")
        state_cfg = _require_mapping(self.agent_params, "state", "agent_params")
        learning_cfg = _require_mapping(self.agent_params, "learning", "agent_params")
        objective_cfg = _require_mapping(self.world_rules, "objective", "world_rules")

        selected_policy = policy if policy is not None else _require_str(behavior_cfg, "policy", "agent_params.behavior")
        if selected_policy != "explore":
            raise ValueError("Only 'explore' policy is supported in this agent version.")
        self.policy = selected_policy

        cfg_action_delay = _require_float(behavior_cfg, "action_delay", "agent_params.behavior", min_value=0.0)
        self.action_delay = float(action_delay) if action_delay is not None else cfg_action_delay
        self.verbose = bool(verbose) if verbose is not None else _require_bool(behavior_cfg, "verbose", "agent_params.behavior")

        self.death_on_zero_life = _require_bool(termination_cfg, "death_on_zero_life", "agent_params.termination")
        self.min_life = _require_float(termination_cfg, "min_life", "agent_params.termination")
        self.stop_on_target_height = _require_bool(
            termination_cfg,
            "stop_on_target_height",
            "agent_params.termination",
        )
        self.no_height_gain_patience = _require_int(
            termination_cfg,
            "no_height_gain_patience",
            "agent_params.termination",
            min_value=0,
        )
        self.height_gain_epsilon = _require_float(
            termination_cfg,
            "height_gain_epsilon",
            "agent_params.termination",
            min_value=0.0,
        )
        self.target_height = _require_float(objective_cfg, "target_height", "world_rules.objective")

        self.position_bin_size = _require_float(state_cfg, "position_bin_size", "agent_params.state", min_value=0.0001)
        self.y_bin_size = _require_float(state_cfg, "y_bin_size", "agent_params.state", min_value=0.0001)
        self.yaw_bins = _require_int(state_cfg, "yaw_bins", "agent_params.state", min_value=1)
        self.floor_grid_key = _require_str(state_cfg, "floor_grid_key", "agent_params.state")
        self.include_floor_grid = _require_bool(state_cfg, "include_floor_grid", "agent_params.state")
        self.include_line_of_sight = _require_bool(state_cfg, "include_line_of_sight", "agent_params.state")

        discrete_actions = _require_list(actions_cfg, "discrete", "agent_params.actions")
        cleaned_actions: List[str] = []
        for index, command in enumerate(discrete_actions):
            if not isinstance(command, str) or not command.strip():
                raise ValueError(f"Invalid action command at agent_params.actions.discrete[{index}]")
            cleaned_actions.append(command.strip())
        if len(cleaned_actions) != 5:
            raise ValueError("agent_params.actions.discrete must contain exactly 5 commands.")
        self.actions = cleaned_actions
        self.num_actions = len(self.actions)

        self.alpha = _require_float(learning_cfg, "alpha", "agent_params.learning", min_value=0.0)
        self.gamma = _require_float(learning_cfg, "gamma", "agent_params.learning", min_value=0.0)
        self.epsilon = _require_float(learning_cfg, "epsilon", "agent_params.learning", min_value=0.0)
        if self.alpha > 1.0:
            raise ValueError("agent_params.learning.alpha must be <= 1.0")
        if self.gamma > 1.0:
            raise ValueError("agent_params.learning.gamma must be <= 1.0")
        if self.epsilon > 1.0:
            raise ValueError("agent_params.learning.epsilon must be <= 1.0")

        self.q_table_path = Path(_require_str(learning_cfg, "q_table_path", "agent_params.learning"))
        self.autosave_each_episode = _require_bool(
            learning_cfg,
            "autosave_each_episode",
            "agent_params.learning",
        )

        self.step_count = 0
        self.total_reward = 0.0
        self.max_height = float("-inf")
        self.steps_without_height_gain = 0
        self.last_life: Optional[float] = None
        self.last_observation: Dict[str, Any] = {}
        self._is_dead = False
        self._termination_reason: Optional[str] = None

        self.q_table: Dict[State, np.ndarray] = {}
        self._last_state: Optional[State] = None
        self._last_action: Optional[int] = None
        self._last_action_command: Optional[str] = None

        self.load_q_table()

    def _encode_block(self, block: Any) -> int:
        name = str(block).strip().lower()
        if "lava" in name:
            return 2
        if "air" in name:
            return 0
        if "gold" in name:
            return 1
        if name in {"stone", "dirt", "grass", "cobblestone"}:
            return 1
        return 3

    def _bucket_count(self, value: int) -> int:
        if value <= 0:
            return 0
        if value <= 2:
            return 1
        return 2

    def _encode_floor_grid(self, obs: Dict[str, Any]) -> Tuple[int, int, int]:
        if not self.include_floor_grid:
            return (-1, -1, -1)

        raw_grid = obs.get(self.floor_grid_key)
        if raw_grid is None:
            raw_grid = self.last_observation.get(self.floor_grid_key, [])
        if not isinstance(raw_grid, list):
            return (-2, -2, -2)

        encoded = [self._encode_block(item) for item in raw_grid]
        if not encoded:
            return (-2, -2, -2)

        center_code = encoded[len(encoded) // 2]
        lava_cells = sum(1 for item in encoded if item == 2)
        air_cells = sum(1 for item in encoded if item == 0)
        return (center_code, self._bucket_count(lava_cells), self._bucket_count(air_cells))

    def _encode_line_of_sight(self, obs: Dict[str, Any]) -> int:
        if not self.include_line_of_sight:
            return -1

        los = obs.get("LineOfSight")
        if los is None:
            los = self.last_observation.get("LineOfSight")
        if not isinstance(los, dict):
            return -2

        block_type = str(los.get("type", "")).strip().lower()
        if not block_type:
            return -2
        return self._encode_block(block_type)

    def _state_from_observation(self, observation: Optional[Dict[str, Any]]) -> State:
        obs = observation or {}
        fallback = self.last_observation

        x = _as_float(obs.get("XPos"), _as_float(fallback.get("XPos"), 0.0)) or 0.0
        y = _as_float(obs.get("YPos"), _as_float(fallback.get("YPos"), 0.0)) or 0.0
        z = _as_float(obs.get("ZPos"), _as_float(fallback.get("ZPos"), 0.0)) or 0.0
        yaw = _as_float(obs.get("Yaw"), _as_float(fallback.get("Yaw"), 0.0)) or 0.0

        x_bin = int(math.floor(x / self.position_bin_size))
        y_bin = int(math.floor(y / self.y_bin_size))
        z_bin = int(math.floor(z / self.position_bin_size))
        yaw_norm = yaw % 360.0
        yaw_bin = int(math.floor((yaw_norm / 360.0) * self.yaw_bins))
        if yaw_bin >= self.yaw_bins:
            yaw_bin = self.yaw_bins - 1

        center_code, lava_bucket, air_bucket = self._encode_floor_grid(obs)
        los_code = self._encode_line_of_sight(obs)
        return (x_bin, y_bin, z_bin, yaw_bin, center_code, lava_bucket, air_bucket, los_code)

    def _ensure_state(self, state: State) -> None:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions, dtype=float)

    def _state_to_key(self, state: State) -> str:
        return json.dumps(state, separators=(",", ":"))

    def _key_to_state(self, key: str) -> State:
        raw = json.loads(key)
        if not isinstance(raw, list) or len(raw) == 0:
            raise ValueError(f"Invalid state key in q-table file: {key}")
        try:
            return tuple(int(item) for item in raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid state key in q-table file: {key}") from exc

    def load_q_table(self) -> None:
        if not self.q_table_path.exists():
            return
        try:
            loaded = json.loads(self.q_table_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Could not load q-table file {self.q_table_path}: {exc}") from exc

        if not isinstance(loaded, dict):
            raise ValueError(f"Q-table file must contain a mapping: {self.q_table_path}")

        for key, values in loaded.items():
            state = self._key_to_state(str(key))
            vector = np.asarray(values, dtype=float)
            if vector.shape != (self.num_actions,):
                raise ValueError(
                    f"Invalid q-vector size for state {state}. "
                    f"Expected {self.num_actions}, got {vector.shape}."
                )
            self.q_table[state] = vector

        if self.verbose:
            print(f"Loaded Q-table states: {len(self.q_table)} from {self.q_table_path}")

    def save_q_table(self) -> None:
        payload = {self._state_to_key(state): values.tolist() for state, values in self.q_table.items()}
        self.q_table_path.parent.mkdir(parents=True, exist_ok=True)
        self.q_table_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def get_action(self, observation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.step_count += 1
        state = self._state_from_observation(observation)
        self._ensure_state(state)

        if random.random() < self.epsilon:
            action_id = random.randrange(self.num_actions)
        else:
            action_id = int(np.argmax(self.q_table[state]))

        action_command = self.actions[action_id]

        self._last_state = state
        self._last_action = action_id
        self._last_action_command = action_command

        return {
            "action_index": action_id,
            "action_command": action_command,
        }

    def process_observation(self, observation: Optional[Dict[str, Any]]) -> None:
        obs = observation or {}
        self.last_observation = dict(obs)

        y = _as_float(obs.get("YPos"))
        if y is not None:
            if self.max_height == float("-inf") or y > (self.max_height + self.height_gain_epsilon):
                self.max_height = y
                self.steps_without_height_gain = 0
            else:
                self.max_height = max(self.max_height, y)
                self.steps_without_height_gain += 1

        life = _as_float(obs.get("Life"))
        if life is not None:
            self.last_life = life
            if self.death_on_zero_life and life <= self.min_life:
                self._is_dead = True
                self._termination_reason = "death"

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
        if self._is_dead:
            if self._termination_reason is None:
                self._termination_reason = "death"
            return True

        if self.stop_on_target_height and self.max_height != float("-inf") and self.max_height >= self.target_height:
            self._termination_reason = "target_height"
            return True

        if self.no_height_gain_patience > 0 and self.steps_without_height_gain >= self.no_height_gain_patience:
            self._termination_reason = "no_height_gain"
            return True

        return False

    def process_reward(
        self,
        reward: float,
        next_observation: Optional[Dict[str, Any]] = None,
        done: bool = False,
    ) -> None:
        self.total_reward += reward

        if self._last_state is None or self._last_action is None:
            if reward != 0 and self.verbose:
                print(f"  -> Reward: {reward:+.2f} | Total: {self.total_reward:.2f}")
            return

        next_state = self._state_from_observation(next_observation)
        self._ensure_state(next_state)

        current_q = float(self.q_table[self._last_state][self._last_action])
        next_best = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = float(reward) + self.gamma * next_best
        updated_q = current_q + self.alpha * (td_target - current_q)
        self.q_table[self._last_state][self._last_action] = updated_q

        if reward != 0 and self.verbose:
            print(
                f"  -> Reward: {reward:+.2f} | Total: {self.total_reward:.2f} | "
                f"Q[{self._last_action_command}]={updated_q:.4f}"
            )

    def reset(self) -> None:
        self.step_count = 0
        self.total_reward = 0.0
        self.max_height = float("-inf")
        self.steps_without_height_gain = 0
        self.last_life = None
        self.last_observation = {}
        self._is_dead = False
        self._termination_reason = None
        self._last_state = None
        self._last_action = None
        self._last_action_command = None

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
            "steps_without_height_gain": self.steps_without_height_gain,
            "termination_reason": self._termination_reason,
            "q_states": len(self.q_table),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "life": self.last_life,
            "is_dead": self._is_dead,
            "termination_reason": self._termination_reason,
            "death_on_zero_life": self.death_on_zero_life,
            "target_height": self.target_height,
            "max_height": self.max_height if self.max_height != float("-inf") else None,
            "steps_without_height_gain": self.steps_without_height_gain,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "q_states": len(self.q_table),
            "q_table_path": str(self.q_table_path),
            "last_action": self._last_action_command,
            "last_action_index": self._last_action,
            "last_state": list(self._last_state) if self._last_state is not None else None,
        }
