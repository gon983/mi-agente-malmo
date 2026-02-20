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
from typing import Any, Dict, Optional

import numpy as np

# Add current project root to import path
sys.path.insert(0, str(Path(__file__).parent))

from agents.basic_agent import (  # noqa: E402
    DEFAULT_AGENT_PARAMS,
    DEFAULT_WORLD_RULES,
    BasicAgent,
    load_yaml_config,
)
from utils.malmo_connector import MalmoConnector  # noqa: E402
from utils.viewer_3d import create_unified_viewer  # noqa: E402


DEFAULT_RUNTIME_CONFIG: Dict[str, Any] = {
    "paths": {
        "world_rules": "config/world_rules.yaml",
        "agent_params": "config/agent_params.yaml",
    },
    "malmo": {
        "port": 9000,
        "server": "127.0.0.1",
        "role": 0,
        "experiment_id": "experiment",
        "mission": "missions/simple_test.xml",
    },
    "logging": {
        "level": "INFO",
        "log_dir": "logs",
    },
}


def setup_logging(config: Dict[str, Any]) -> None:
    log_cfg = config.get("logging", {})
    log_dir = Path(log_cfg.get("log_dir", "logs"))
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO),
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


def run_episode(
    connector: MalmoConnector,
    agent: BasicAgent,
    max_steps: int = 100,
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
    last_scalars: Dict[str, Any] = {}

    while not done and step < max_steps:
        obs_dict = parse_observation(obs, frame_shape=frame_shape)
        action = agent.get_action(obs_dict)

        obs, reward, done, info = connector.step(action)
        next_obs = parse_observation(obs, info, frame_shape=frame_shape)

        # Some ticks can arrive without info payload; keep last known scalar state.
        for key, value in last_scalars.items():
            next_obs.setdefault(key, value)
        for key in ("XPos", "YPos", "ZPos", "Life", "Yaw", "Pitch", "WorldTime", "TotalTime"):
            if key in next_obs:
                last_scalars[key] = next_obs[key]

        agent.process_observation(next_obs)
        if agent.should_terminate():
            done = True

        reward_value = float(reward)
        agent.process_reward(reward_value)

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
        choices=["random", "forward", "explore"],
        help="Policy override",
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

    runtime_cfg = load_yaml_config(args.config, DEFAULT_RUNTIME_CONFIG)
    world_rules_path = args.world_rules or runtime_cfg.get("paths", {}).get("world_rules", "config/world_rules.yaml")
    agent_params_path = args.agent_params or runtime_cfg.get("paths", {}).get("agent_params", "config/agent_params.yaml")

    world_rules = load_yaml_config(world_rules_path, DEFAULT_WORLD_RULES)
    agent_params = load_yaml_config(agent_params_path, DEFAULT_AGENT_PARAMS)

    world_cfg = world_rules.setdefault("world", {})
    world_server = world_cfg.setdefault("server", {})
    runtime_malmo = runtime_cfg.get("malmo", {})
    behavior_cfg = agent_params.setdefault("behavior", {})
    run_cfg = agent_params.setdefault("run", {})

    # Keep backward compatibility with config/config.yaml defaults.
    world_server.setdefault("host", runtime_malmo.get("server", "127.0.0.1"))
    world_server.setdefault("port", runtime_malmo.get("port", 9000))
    world_server.setdefault("role", runtime_malmo.get("role", 0))
    world_server.setdefault("experiment_id", runtime_malmo.get("experiment_id", "experiment"))
    world_cfg.setdefault("mission_file", runtime_malmo.get("mission", "missions/simple_test.xml"))

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

    mission_rel_path = str(world_cfg.get("mission_file", "missions/simple_test.xml"))
    mission_path = Path(mission_rel_path)
    if not mission_path.is_absolute():
        mission_path = repo_root / mission_rel_path
    mission_path = mission_path.resolve()

    if not mission_path.exists():
        print(f"Mission file not found: {mission_path}")
        sys.exit(1)

    episodes = int(run_cfg.get("episodes", 5))
    max_steps = int(run_cfg.get("max_steps", 50))
    policy = str(behavior_cfg.get("policy", "explore"))

    print("=" * 72)
    print("MALMO AGENT")
    print("=" * 72)
    print(f"mission       : {mission_path}")
    print(f"server        : {world_server.get('host')}:{world_server.get('port')}")
    print(f"policy        : {policy}")
    print(f"episodes      : {episodes}")
    print(f"max_steps     : {max_steps}")
    print(f"viewer        : {args.viewer}")
    print(f"world_rules   : {world_rules_path}")
    print(f"agent_params  : {agent_params_path}")
    print("=" * 72)

    connector = MalmoConnector(
        mission_xml=str(mission_path),
        port=int(world_server.get("port", 9000)),
        server=str(world_server.get("host", "127.0.0.1")),
        role=int(world_server.get("role", 0)),
        experiment_id=str(world_server.get("experiment_id", "experiment")),
    )

    if not connector.connect():
        print("\nCould not connect to Malmo.")
        print("Start Minecraft Malmo first, for example:")
        print(f"  cd c:\\Users\\gonza\\malmo\\Minecraft")
        print(f"  launchClient.bat -port {world_server.get('port', 9000)} -env")
        sys.exit(1)

    viewer = create_unified_viewer(args.viewer)
    agent = BasicAgent(
        policy=policy,
        action_delay=float(behavior_cfg.get("action_delay", 0.05)),
        verbose=bool(behavior_cfg.get("verbose", True)) and args.viewer == "none",
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

            stats = run_episode(connector, agent, max_steps=max_steps, viewer=viewer)
            all_stats.append(stats)

            if viewer:
                viewer.end_episode()
            else:
                print(
                    f"steps={stats['steps']} total_reward={stats['total_reward']:.2f} "
                    f"avg_reward={stats['avg_reward']:.4f} died={stats.get('died')}"
                )
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
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
        print("=" * 72)


if __name__ == "__main__":
    main()
