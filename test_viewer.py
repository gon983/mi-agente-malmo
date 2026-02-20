#!/usr/bin/env python
"""
Script de prueba para verificar la visualización del agente.

Prueba los viewers sin necesidad de conexión a Minecraft.
Uso:
    python test_viewer.py --viewer terminal
    python test_viewer.py --viewer grafico
    python test_viewer.py --viewer full
"""

import sys
import time
import random
import argparse
from pathlib import Path

# Agregar directorio actual al path
sys.path.insert(0, str(Path(__file__).parent))

from utils.agent_viewer import create_viewer
from utils.graphical_viewer import create_graphical_viewer


def simulate_observation(step: int, base_x: float = 0, base_z: float = 0):
    """Simula una observación del entorno."""
    # Simular movimiento
    x = base_x + step * 0.5 + random.uniform(-0.2, 0.2)
    z = base_z + random.uniform(-1, 1) * 0.3
    y = 64.0 + (1.0 if step % 10 in [5, 6] else 0.0)  # Saltar cada 10 pasos
    
    return {
        'XPos': x,
        'YPos': y,
        'ZPos': z,
        'Life': 20.0 - step * 0.05,  # Pierde vida gradualmente
        'Yaw': random.uniform(0, 360),
        'Pitch': 0.0
    }


def simulate_action():
    """Simula una acción del agente."""
    return {
        'move': random.choice([0.0, 0.5, 1.0]),
        'turn': random.uniform(-0.5, 0.5),
        'jump': 1.0 if random.random() < 0.1 else 0.0
    }


def simulate_reward(step: int):
    """Simula una recompensa."""
    if step % 10 == 0:
        return 1.0  # Recompensa cada 10 pasos
    elif random.random() < 0.1:
        return random.uniform(-0.5, 0.5)
    return 0.0


def create_viewer_by_type(viewer_type: str):
    """Crea el visualizador según el tipo."""
    if viewer_type == 'terminal':
        return create_viewer(use_rich=True)
    
    if viewer_type == 'grafico':
        return create_graphical_viewer(pygame=True, matplotlib=True)
    
    if viewer_type == 'full':
        class CombinedViewer:
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


def test_viewer(viewer_type: str, episodes: int = 2, max_steps: int = 50):
    """Prueba el visualizador con datos simulados."""
    print(f"Probando visualizador: {viewer_type}")
    print(f"Episodios: {episodes}, Pasos por episodio: {max_steps}")
    print("-" * 50)
    
    viewer = create_viewer_by_type(viewer_type)
    
    if viewer is None:
        print("No se pudo crear el visualizador")
        return
    
    viewer.start(
        total_episodes=episodes,
        max_steps=max_steps,
        policy='test'
    )
    
    try:
        for episode in range(1, episodes + 1):
            viewer.start_episode(episode)
            
            total_reward = 0.0
            
            for step in range(1, max_steps + 1):
                # Simular datos
                observation = simulate_observation(step)
                action = simulate_action()
                reward = simulate_reward(step)
                total_reward += reward
                
                # Actualizar viewer
                viewer.update_step(
                    step=step,
                    action=action,
                    observation=observation,
                    reward=reward,
                    total_reward=total_reward
                )
                
                # Verificar si el viewer sigue activo
                if hasattr(viewer, 'is_running') and not viewer.is_running():
                    print("\nVisualización cerrada por el usuario")
                    return
                
                # Pausa para visualización
                time.sleep(0.05)
            
            viewer.end_episode()
            print(f"\nEpisodio {episode} completado. Reward total: {total_reward:.2f}")
    
    except KeyboardInterrupt:
        print("\n\nPrueba interrumpida por el usuario")
    
    finally:
        viewer.close()
    
    print("\nPrueba completada exitosamente!")


def main():
    parser = argparse.ArgumentParser(description='Prueba de visualización del agente')
    parser.add_argument('--viewer', type=str, default='grafico',
                        choices=['terminal', 'grafico', 'full'],
                        help='Tipo de visualización')
    parser.add_argument('--episodes', type=int, default=2,
                        help='Número de episodios a simular')
    parser.add_argument('--max-steps', type=int, default=50,
                        help='Pasos por episodio')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🧪 TEST DE VISUALIZACIÓN DEL AGENTE")
    print("="*60)
    
    test_viewer(args.viewer, args.episodes, args.max_steps)


if __name__ == '__main__':
    main()