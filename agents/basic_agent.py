"""
Agente Básico para Microsoft Malmo

Un agente minimalista que:
- Se conecta a Minecraft/Malmo
- Toma acciones aleatorias o simples
- Muestra observaciones y recompensas
- Útil como base para agentes más complejos
"""

import random
import time
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class BasicAgent:
    """
    Agente básico que interactúa con Malmo.
    
    Este agente implementa una política simple que puede ser:
    - 'random': acciones completamente aleatorias
    - 'forward': siempre mover hacia adelante
    - 'explore': exploración simple alternando movimiento y giro
    """
    
    def __init__(self, 
                 policy: str = 'random',
                 action_delay: float = 0.05,
                 verbose: bool = True):
        """
        Inicializa el agente básico.
        
        Args:
            policy: Política a usar ('random', 'forward', 'explore')
            action_delay: Segundos a esperar entre acciones
            verbose: Si True, imprime información de cada paso
        """
        self.policy = policy
        self.action_delay = action_delay
        self.verbose = verbose
        self.step_count = 0
        self.total_reward = 0.0
        
    def get_action(self, observation: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Retorna la siguiente acción basada en la política.
        
        Args:
            observation: Observación actual del entorno (opcional)
            
        Returns:
            Diccionario con las acciones (ej: {'move': 1.0, 'turn': 0.0})
        """
        self.step_count += 1
        
        if self.policy == 'random':
            return self._random_action()
        elif self.policy == 'forward':
            return self._forward_action()
        elif self.policy == 'explore':
            return self._explore_action(observation)
        else:
            return self._random_action()
    
    def _random_action(self) -> Dict[str, float]:
        """Acción completamente aleatoria."""
        return {
            'move': random.uniform(-1, 1),
            'turn': random.uniform(-1, 1),
            'jump': random.choice([0, 0, 0, 1])  # Salta con 25% probabilidad
        }
    
    def _forward_action(self) -> Dict[str, float]:
        """Siempre mover hacia adelante."""
        return {
            'move': 1.0,
            'turn': 0.0,
            'jump': 0.0
        }
    
    def _explore_action(self, observation: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """
        Exploración simple: cada cierto número de pasos, gira aleatoriamente.
        """
        # Cada 10 pasos, girar para explorar
        if self.step_count % 10 == 0:
            return {
                'move': 0.0,
                'turn': random.choice([-1.0, 1.0]),
                'jump': 0.0
            }
        else:
            return {
                'move': 1.0,
                'turn': random.uniform(-0.2, 0.2),
                'jump': 0.0
            }
    
    def process_observation(self, observation: Dict[str, Any]) -> None:
        """
        Procesa la observación recibida del entorno.
        
        Args:
            observation: Diccionario con datos de observación
        """
        if self.verbose and observation:
            # Extraer información útil
            x = observation.get('XPos', 0)
            y = observation.get('YPos', 0)
            z = observation.get('ZPos', 0)
            life = observation.get('Life', 0)
            
            print(f"[Paso {self.step_count}] Pos: ({x:.1f}, {y:.1f}, {z:.1f}) | Vida: {life}")
    
    def process_reward(self, reward: float) -> None:
        """
        Procesa la recompensa recibida.
        
        Args:
            reward: Valor de recompensa
        """
        self.total_reward += reward
        if reward != 0 and self.verbose:
            print(f"  → Recompensa: {reward:.2f} | Total: {self.total_reward:.2f}")
    
    def reset(self) -> None:
        """Reinicia el estado del agente para un nuevo episodio."""
        self.step_count = 0
        self.total_reward = 0.0
        if self.verbose:
            print("\n" + "="*50)
            print("NUEVO EPISODIO")
            print("="*50)
    
    def episode_summary(self) -> Dict[str, Any]:
        """
        Retorna un resumen del episodio completado.
        
        Returns:
            Diccionario con estadísticas del episodio
        """
        return {
            'steps': self.step_count,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(1, self.step_count)
        }