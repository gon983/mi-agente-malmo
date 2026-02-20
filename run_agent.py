#!/usr/bin/env python
"""
Script principal para ejecutar el agente de Malmo.

Uso:
    python run_agent.py
    python run_agent.py --episodes 10 --policy explore
    python run_agent.py --mission missions/simple_test.xml --port 9000
    python run_agent.py --viewer terminal
    python run_agent.py --viewer grafico
    python run_agent.py --viewer full
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Agregar directorio actual al path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import yaml
except ImportError:
    yaml = None
    print("Warning: PyYAML no está instalado. Usando configuración por defecto.")

from agents.basic_agent import BasicAgent
from utils.malmo_connector import MalmoConnector
from utils.agent_viewer import create_viewer, AgentViewer, SimpleViewer
from utils.graphical_viewer import create_graphical_viewer, GraphicalViewer


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Carga la configuración desde archivo YAML.
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Diccionario con configuración
    """
    default_config = {
        'malmo': {
            'port': 9000,
            'server': '127.0.0.1',
            'mission': 'missions/simple_test.xml',
            'role': 0,
            'experiment_id': 'experiment'
        },
        'agent': {
            'type': 'basic',
            'policy': 'explore',
            'episodes': 5,
            'max_steps': 50,
            'action_delay': 0.05
        },
        'logging': {
            'level': 'INFO',
            'save_rewards': True,
            'log_dir': 'logs'
        }
    }
    
    if yaml and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            # Merge con defaults
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]
            return config
    
    return default_config


def setup_logging(config: dict) -> None:
    """Configura el sistema de logging."""
    log_dir = Path(config['logging'].get('log_dir', 'logs'))
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'run_{timestamp}.log'
    
    logging.basicConfig(
        level=getattr(logging, config['logging'].get('level', 'INFO')),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def parse_observation(obs) -> dict:
    """
    Parsea la observación del entorno a un diccionario.
    
    Args:
        obs: Observación del entorno (puede ser array, dict, etc.)
        
    Returns:
        Diccionario con datos de observación
    """
    if obs is None:
        return {}
    
    # Si ya es un diccionario
    if isinstance(obs, dict):
        return obs
    
    # Si es un array numpy (frame)
    if hasattr(obs, 'shape'):
        # Intentar extraer información adicional si está disponible
        return {'frame_shape': obs.shape, 'raw_obs': obs}
    
    # Si tiene atributos
    if hasattr(obs, '__dict__'):
        return obs.__dict__
    
    return {'raw': obs}


def run_episode(connector: MalmoConnector, 
                agent: BasicAgent, 
                max_steps: int = 100,
                viewer = None) -> dict:
    """
    Ejecuta un episodio del agente con visualización.
    
    Args:
        connector: Conector Malmo
        agent: Agente a ejecutar
        max_steps: Máximo número de pasos
        viewer: Visualizador (opcional)
        
    Returns:
        Diccionario con estadísticas del episodio
    """
    agent.reset()
    
    # Reiniciar entorno
    obs = connector.reset()
    done = False
    step = 0
    
    while not done and step < max_steps:
        # Parsear observación
        obs_dict = parse_observation(obs)
        
        # Obtener acción del agente
        action = agent.get_action(obs_dict)
        
        # Ejecutar acción
        obs, reward, done, info = connector.step(action)
        
        # Parsear nueva observación
        new_obs_dict = parse_observation(obs)
        
        # Procesar resultados
        agent.process_reward(reward)
        
        # Actualizar visualización
        if viewer:
            viewer.update_step(
                step=step + 1,
                action=action,
                observation=new_obs_dict,
                reward=reward,
                total_reward=agent.total_reward
            )
            
            # Verificar si el viewer sigue activo (para Pygame)
            if hasattr(viewer, 'is_running') and not viewer.is_running():
                print("\n⚠️  Visualización cerrada por el usuario.")
                break
        
        step += 1
        time.sleep(agent.action_delay)
    
    return agent.episode_summary()


def create_viewer_by_type(viewer_type: str):
    """
    Crea el visualizador según el tipo especificado.
    
    Args:
        viewer_type: 'terminal', 'grafico', 'full', o 'none'
        
    Returns:
        Instancia del visualizador o None
    """
    if viewer_type == 'none' or viewer_type is None:
        return None
    
    if viewer_type == 'terminal':
        return create_viewer(use_rich=True)
    
    if viewer_type == 'grafico':
        return create_graphical_viewer(pygame=True, matplotlib=True)
    
    if viewer_type == 'full':
        # Combinado: terminal Rich + gráficos
        class CombinedViewer:
            """Visualizador combinado terminal + gráfico."""
            
            def __init__(self):
                self.terminal = create_viewer(use_rich=True)
                self.graphic = create_graphical_viewer(pygame=True, matplotlib=False)
                
            def start(self, total_episodes, max_steps, policy):
                self.terminal.start(total_episodes, max_steps, policy)
                self.graphic.start(total_episodes, max_steps, policy)
                
            def start_episode(self, episode):
                self.terminal.start_episode(episode)
                self.graphic.start_episode(episode)
                
            def update_step(self, step, action, observation, reward, total_reward):
                self.terminal.update_step(step, action, observation, reward, total_reward)
                self.graphic.update_step(step, action, observation, reward, total_reward)
                
            def end_episode(self):
                self.terminal.end_episode()
                self.graphic.end_episode()
                
            def close(self):
                self.graphic.close()
                
            def is_running(self):
                return self.graphic.is_running()
        
        return CombinedViewer()
    
    return None


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Agente de Malmo')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Archivo de configuración')
    parser.add_argument('--mission', type=str, default=None,
                        help='Archivo de misión XML')
    parser.add_argument('--port', type=int, default=None,
                        help='Puerto de Malmo')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Número de episodios')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Máximo pasos por episodio')
    parser.add_argument('--policy', type=str, default=None,
                        choices=['random', 'forward', 'explore'],
                        help='Política del agente')
    parser.add_argument('--viewer', type=str, default='none',
                        choices=['none', 'terminal', 'grafico', 'full'],
                        help='Tipo de visualización: none, terminal (Rich), grafico (Pygame+Matplotlib), full (ambos)')
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Sobrescribir con argumentos de línea de comandos
    if args.mission:
        config['malmo']['mission'] = args.mission
    if args.port:
        config['malmo']['port'] = args.port
    if args.episodes:
        config['agent']['episodes'] = args.episodes
    if args.max_steps:
        config['agent']['max_steps'] = args.max_steps
    if args.policy:
        config['agent']['policy'] = args.policy
    
    # Configurar logging
    setup_logging(config)
    
    print("="*60)
    print("🎮 AGENTE DE MALMO")
    print("="*60)
    print(f"Misión: {config['malmo']['mission']}")
    print(f"Puerto: {config['malmo']['port']}")
    print(f"Política: {config['agent']['policy']}")
    print(f"Episodios: {config['agent']['episodes']}")
    print(f"Max pasos: {config['agent']['max_steps']}")
    print(f"Visualización: {args.viewer}")
    print("="*60)
    
    # Verificar que Minecraft esté corriendo
    print("\n⚠️  Asegúrate de que Minecraft Malmo esté corriendo:")
    print(f"   cd c:\\Users\\gonza\\malmo\\Minecraft")
    print(f"   launchClient.bat -port {config['malmo']['port']} -env")
    print()
    
    # Crear conector
    mission_path = Path(__file__).parent / config['malmo']['mission']
    connector = MalmoConnector(
        mission_xml=str(mission_path),
        port=config['malmo']['port'],
        server=config['malmo']['server'],
        role=config['malmo']['role'],
        experiment_id=config['malmo']['experiment_id']
    )
    
    # Conectar
    if not connector.connect():
        print("\n❌ No se pudo conectar a Malmo.")
        print("Verifica que Minecraft esté corriendo con el mod Malmo.")
        sys.exit(1)
    
    # Crear agente
    agent = BasicAgent(
        policy=config['agent']['policy'],
        action_delay=config['agent']['action_delay'],
        verbose=(args.viewer == 'none')  # Solo verbose si no hay viewer
    )
    
    # Crear visualizador
    viewer = create_viewer_by_type(args.viewer)
    
    if viewer:
        viewer.start(
            total_episodes=config['agent']['episodes'],
            max_steps=config['agent']['max_steps'],
            policy=config['agent']['policy']
        )
    
    # Ejecutar episodios
    all_stats = []
    
    try:
        for episode in range(config['agent']['episodes']):
            # Iniciar episodio en viewer
            if viewer:
                viewer.start_episode(episode + 1)
            else:
                print(f"\n{'='*60}")
                print(f"EPISODIO {episode + 1}/{config['agent']['episodes']}")
                print('='*60)
            
            stats = run_episode(
                connector, 
                agent, 
                max_steps=config['agent']['max_steps'],
                viewer=viewer
            )
            all_stats.append(stats)
            
            # Finalizar episodio en viewer
            if viewer:
                viewer.end_episode()
            
            # Mostrar resumen del episodio
            if args.viewer == 'none':
                print(f"\n📊 Resumen del episodio:")
                print(f"   Pasos: {stats['steps']}")
                print(f"   Recompensa total: {stats['total_reward']:.2f}")
                print(f"   Recompensa promedio: {stats['avg_reward']:.4f}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Detenido por el usuario.")
    
    finally:
        # Cerrar visualizador
        if viewer:
            viewer.close()
        connector.close()
    
    # Resumen final
    if all_stats:
        print("\n" + "="*60)
        print("📈 RESUMEN FINAL")
        print("="*60)
        total_reward = sum(s['total_reward'] for s in all_stats)
        avg_reward = total_reward / len(all_stats)
        print(f"Total episodios: {len(all_stats)}")
        print(f"Recompensa total: {total_reward:.2f}")
        print(f"Recompensa promedio por episodio: {avg_reward:.2f}")
        print("="*60)


if __name__ == '__main__':
    main()