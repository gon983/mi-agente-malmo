#!/usr/bin/env python
"""
Main script to run the Malmo agent.

Viewer modes:
- none: no live viewer
- terminal: minimal terminal line output
- full: single realtime panel with video + live variables
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add current project root to import path
sys.path.insert(0, str(Path(__file__).parent))

from agents.basic_agent import BasicAgent, load_yaml_config  # noqa: E402
from utils.malmo_connector import MalmoConnector  # noqa: E402
from utils.viewer_3d import create_unified_viewer  # noqa: E402


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


def setup_logging(config: Dict[str, Any]) -> None:
    log_cfg = _require_mapping(config, "logging", "runtime config")
    log_dir = Path(_require_str(log_cfg, "log_dir", "runtime config.logging"))
    level_name = _require_str(log_cfg, "level", "runtime config.logging").upper()
    log_level = getattr(logging, level_name, None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid logging level: {level_name}")

    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def _parse_info_payload(info: Any) -> Dict[str, Any]:
    if info is None:
        return {}
    if isinstance(info, dict):
        return dict(info)
    if isinstance(info, str):
        stripped = info.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
            return {"info_list": parsed}
        except json.JSONDecodeError:
            return {"info_raw": stripped}
    return {"info_raw": str(info)}


def parse_observation(
    obs: Any,
    info: Any = None,
    frame_shape: Optional[tuple[int, ...]] = None,
) -> Dict[str, Any]:
    """
    Parse Malmo observation and info into a single dictionary.
    """
    parsed: Dict[str, Any] = {}

    if isinstance(obs, dict):
        parsed.update(obs)
    elif hasattr(obs, "shape"):
        frame = obs
        if (
            isinstance(obs, np.ndarray)
            and obs.ndim == 1
            and frame_shape is not None
            and int(np.prod(frame_shape)) == int(obs.size)
        ):
            frame = obs.reshape(frame_shape)
        parsed["_frame"] = frame
    elif obs is not None:
        parsed["raw_obs"] = obs

    parsed.update(_parse_info_payload(info))
    return parsed


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_training_reward(
    reward_mode: str,
    env_reward: float,
    next_obs: Dict[str, Any],
    height_gain_scale: float,
    min_height_delta: float,
    best_y: Optional[float],
) -> tuple[float, Optional[float]]:
    if reward_mode == "malmo":
        return float(env_reward), best_y

    if reward_mode == "height_gain_only":
        next_y = _to_float(next_obs.get("YPos"))
        if next_y is None:
            return 0.0, best_y

        if best_y is None:
            return 0.0, next_y

        delta = next_y - best_y
        if delta <= min_height_delta:
            return 0.0, best_y
        return delta * height_gain_scale, next_y

    raise ValueError(f"Unsupported reward mode: {reward_mode}")


def run_episode(
    connector: MalmoConnector,
    agent: BasicAgent,
    max_steps: int,
    reward_mode: str,
    height_gain_scale: float,
    min_height_delta: float,
    viewer: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run one episode and return summary stats.
    """
    agent.reset()
    obs = connector.reset()
    done = False
    step = 0
    frame_shape = getattr(getattr(connector, "observation_space", None), "shape", None)
    last_known: Dict[str, Any] = {}
    best_y: Optional[float] = None

    tracked_keys = [
        "XPos",
        "YPos",
        "ZPos",
        "Life",
        "Yaw",
        "Pitch",
        "WorldTime",
        "TotalTime",
    ]
    if agent.include_floor_grid:
        tracked_keys.append(agent.floor_grid_key)
    if agent.include_line_of_sight:
        tracked_keys.append("LineOfSight")

    while not done and step < max_steps:
        obs_dict = parse_observation(obs, frame_shape=frame_shape)
        for key, value in last_known.items():
            obs_dict.setdefault(key, value)

        action = agent.get_action(obs_dict)

        obs, reward, done, info = connector.step(action)
        next_obs = parse_observation(obs, info, frame_shape=frame_shape)

        # Some ticks can arrive without payload; keep last known state features.
        for key, value in last_known.items():
            next_obs.setdefault(key, value)
        for key in tracked_keys:
            if key in next_obs:
                last_known[key] = next_obs[key]

        agent.process_observation(next_obs)
        if agent.should_terminate():
            done = True

        reward_value, best_y = _compute_training_reward(
            reward_mode=reward_mode,
            env_reward=float(reward),
            next_obs=next_obs,
            height_gain_scale=height_gain_scale,
            min_height_delta=min_height_delta,
            best_y=best_y,
        )
        agent.process_reward(reward_value, next_observation=next_obs, done=done)

        step += 1

        if viewer:
            viewer.update_step(
                step=step,
                action=action,
                observation=next_obs,
                reward=reward_value,
                total_reward=agent.total_reward,
                agent_metrics=agent.get_status(),
            )
            if hasattr(viewer, "is_running") and not viewer.is_running():
                print("\nViewer closed by user.")
                break

        time.sleep(agent.action_delay)

    return agent.episode_summary()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Malmo agent runner")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Runtime config file")
    parser.add_argument("--world-rules", type=str, default=None, help="World rules YAML")
    parser.add_argument("--agent-params", type=str, default=None, help="Agent params YAML")
    parser.add_argument("--mission", type=str, default=None, help="Mission XML path")
    parser.add_argument("--port", type=int, default=None, help="Malmo port")
    parser.add_argument("--episodes", type=int, default=None, help="Episode count override")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps override")
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        choices=["explore"],
        help="Policy override (only explore supported)",
    )
    parser.add_argument(
        "--viewer",
        type=str,
        default="none",
        choices=["none", "terminal", "full"],
        help="Viewer mode: none, terminal, full",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    repo_root = Path(__file__).resolve().parent

    runtime_cfg = load_yaml_config(args.config, strict=True)
    paths_cfg = _require_mapping(runtime_cfg, "paths", "runtime config")
    world_rules_path = args.world_rules or _require_str(paths_cfg, "world_rules", "runtime config.paths")
    agent_params_path = args.agent_params or _require_str(paths_cfg, "agent_params", "runtime config.paths")

    world_rules = load_yaml_config(world_rules_path, strict=True)
    agent_params = load_yaml_config(agent_params_path, strict=True)

    world_cfg = _require_mapping(world_rules, "world", "world_rules")
    world_server = _require_mapping(world_cfg, "server", "world_rules.world")
    behavior_cfg = _require_mapping(agent_params, "behavior", "agent_params")
    run_cfg = _require_mapping(agent_params, "run", "agent_params")
    actions_cfg = _require_mapping(agent_params, "actions", "agent_params")
    reward_cfg = _require_mapping(agent_params, "reward", "agent_params")
    discrete_actions = _require_list(actions_cfg, "discrete", "agent_params.actions")

    if args.port is not None:
        world_server["port"] = args.port
    if args.mission:
        world_cfg["mission_file"] = args.mission
    if args.policy:
        behavior_cfg["policy"] = args.policy
    if args.episodes is not None:
        run_cfg["episodes"] = args.episodes
    if args.max_steps is not None:
        run_cfg["max_steps"] = args.max_steps

    setup_logging(runtime_cfg)

    mission_rel_path = _require_str(world_cfg, "mission_file", "world_rules.world")
    mission_path = Path(mission_rel_path)
    if not mission_path.is_absolute():
        mission_path = repo_root / mission_rel_path
    mission_path = mission_path.resolve()

    if not mission_path.exists():
        print(f"Mission file not found: {mission_path}")
        sys.exit(1)

    episodes = _require_int(run_cfg, "episodes", "agent_params.run", min_value=1)
    max_steps = _require_int(run_cfg, "max_steps", "agent_params.run", min_value=1)
    policy = _require_str(behavior_cfg, "policy", "agent_params.behavior")
    if policy != "explore":
        raise ValueError("Only 'explore' policy is supported.")
    reward_mode = _require_str(reward_cfg, "mode", "agent_params.reward")
    if reward_mode not in {"malmo", "height_gain_only"}:
        raise ValueError("agent_params.reward.mode must be 'malmo' or 'height_gain_only'")
    height_gain_scale = _require_float(reward_cfg, "height_gain_scale", "agent_params.reward", min_value=0.0)
    min_height_delta = _require_float(reward_cfg, "min_height_delta", "agent_params.reward", min_value=0.0)

    host = _require_str(world_server, "host", "world_rules.world.server")
    port = _require_int(world_server, "port", "world_rules.world.server", min_value=1)
    role = _require_int(world_server, "role", "world_rules.world.server", min_value=0)
    experiment_id = _require_str(world_server, "experiment_id", "world_rules.world.server")

    action_commands = []
    for index, command in enumerate(discrete_actions):
        if not isinstance(command, str) or not command.strip():
            raise ValueError(f"Invalid command in agent_params.actions.discrete[{index}]")
        action_commands.append(command.strip())
    action_filter = {item.split()[0] for item in action_commands}
    if not action_filter:
        raise ValueError("No valid action roots found in agent_params.actions.discrete")

    print("=" * 72)
    print("MALMO AGENT")
    print("=" * 72)
    print(f"mission       : {mission_path}")
    print(f"server        : {host}:{port}")
    print(f"policy        : {policy}")
    print(f"episodes      : {episodes}")
    print(f"max_steps     : {max_steps}")
    print(f"reward_mode   : {reward_mode}")
    print(f"viewer        : {args.viewer}")
    print(f"world_rules   : {world_rules_path}")
    print(f"agent_params  : {agent_params_path}")
    print("=" * 72)

    connector = MalmoConnector(
        mission_xml=str(mission_path),
        port=port,
        server=host,
        role=role,
        experiment_id=experiment_id,
        action_filter=action_filter,
    )

    if not connector.connect():
        print("\nCould not connect to Malmo.")
        print("Start Minecraft Malmo first, for example:")
        print(f"  cd c:\\Users\\gonza\\malmo\\Minecraft")
        print(f"  launchClient.bat -port {port} -env")
        sys.exit(1)

    actions_list = getattr(connector.action_space, "actions", None)
    if not isinstance(actions_list, list) or not actions_list:
        raise ValueError(
            "Expected a discrete Malmo action space with explicit commands. "
            f"Got: {type(actions_list).__name__}"
        )
    missing_commands = [command for command in action_commands if command not in actions_list]
    if missing_commands:
        raise ValueError(
            "Invalid agent_params.actions.discrete: command(s) not available in current mission "
            f"action space. Missing={missing_commands} available={actions_list}"
        )

    viewer = create_unified_viewer(args.viewer)
    agent = BasicAgent(
        policy=policy,
        action_delay=_require_float(behavior_cfg, "action_delay", "agent_params.behavior", min_value=0.0),
        verbose=_require_bool(behavior_cfg, "verbose", "agent_params.behavior") and args.viewer == "none",
        agent_params=agent_params,
        world_rules=world_rules,
    )

    if viewer:
        viewer.start(total_episodes=episodes, max_steps=max_steps, policy=policy)

    all_stats = []
    try:
        for episode in range(1, episodes + 1):
            if viewer:
                viewer.start_episode(episode)
            else:
                print(f"\nEpisode {episode}/{episodes}")

            stats = run_episode(
                connector,
                agent,
                max_steps=max_steps,
                reward_mode=reward_mode,
                height_gain_scale=height_gain_scale,
                min_height_delta=min_height_delta,
                viewer=viewer,
            )
            all_stats.append(stats)

            if agent.autosave_each_episode:
                agent.save_q_table()

            if viewer:
                viewer.end_episode()
            else:
                print(
                    f"steps={stats['steps']} total_reward={stats['total_reward']:.2f} "
                    f"avg_reward={stats['avg_reward']:.4f} died={stats.get('died')} "
                    f"q_states={stats.get('q_states')}"
                )
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        # Always persist latest values once more at shutdown.
        try:
            agent.save_q_table()
        except Exception as exc:  # pragma: no cover - defensive shutdown path
            print(f"Warning: could not save q-table: {exc}")

        if viewer:
            viewer.close()
        connector.close()

    if all_stats:
        total_reward = sum(item["total_reward"] for item in all_stats)
        avg_reward = total_reward / max(1, len(all_stats))
        deaths = sum(1 for item in all_stats if item.get("died"))
        max_height = max((item.get("max_height") or 0.0) for item in all_stats)

        print("\n" + "=" * 72)
        print("FINAL SUMMARY")
        print("=" * 72)
        print(f"episodes     : {len(all_stats)}")
        print(f"total_reward : {total_reward:.2f}")
        print(f"avg_reward   : {avg_reward:.2f}")
        print(f"deaths       : {deaths}")
        print(f"max_height   : {max_height:.2f}")
        print(f"q_states     : {len(agent.q_table)}")
        print(f"q_table_path : {agent.q_table_path}")
        print("=" * 72)


if __name__ == "__main__":
    main()
