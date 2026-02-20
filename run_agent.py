#!/usr/bin/env python
"""
Script principal para ejecutar el agente de Malmo.

Uso:
    python run_agent.py
    python run_agent.py --episodes 10 --policy explore
    python run_agent.py --mission missions/simple_test.xml --port 9000
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


def run_episode(connector: MalmoConnector, 
                agent: BasicAgent, 
                max_steps: int = 100) -> dict:
    """
    Ejecuta un episodio del agente.
    
    Args:
        connector: Conector Malmo
        agent: Agente a ejecutar
        max_steps: Máximo número de pasos
        
    Returns:
        Diccionario con estadísticas del episodio
    """
    agent.reset()
    
    # Reiniciar entorno
    obs = connector.reset()
    done = False
    step = 0
    
    while not done and step < max_steps:
        # Obtener acción del agente
        action = agent.get_action(obs)
        
        # Ejecutar acción
        obs, reward, done, info = connector.step(action)
        
        # Procesar resultados
        agent.process_reward(reward)
        
        step += 1
        time.sleep(agent.action_delay)
    
    return agent.episode_summary()


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
        verbose=True
    )
    
    # Ejecutar episodios
    all_stats = []
    
    try:
        for episode in range(config['agent']['episodes']):
            print(f"\n{'='*60}")
            print(f"EPISODIO {episode + 1}/{config['agent']['episodes']}")
            print('='*60)
            
            stats = run_episode(
                connector, 
                agent, 
                max_steps=config['agent']['max_steps']
            )
            all_stats.append(stats)
            
            print(f"\n📊 Resumen del episodio:")
            print(f"   Pasos: {stats['steps']}")
            print(f"   Recompensa total: {stats['total_reward']:.2f}")
            print(f"   Recompensa promedio: {stats['avg_reward']:.4f}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Detenido por el usuario.")
    
    finally:
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