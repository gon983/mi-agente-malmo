"""
Conector para Microsoft Malmo

Maneja la comunicación entre el agente y Minecraft via malmoenv.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

try:
    import malmoenv
except ImportError:
    print("ERROR: malmoenv no está instalado.")
    print("Instálalo con: pip install malmoenv")
    raise

logger = logging.getLogger(__name__)


class MalmoConnector:
    """
    Conector para comunicarse con Minecraft/Malmo.
    
    Usa malmoenv para crear un entorno tipo OpenAI Gym.
    """
    
    def __init__(self,
                 mission_xml: str,
                 port: int = 9000,
                 server: str = "127.0.0.1",
                 role: int = 0,
                 experiment_id: str = "experiment",
                 episode: int = 0,
                 resync: int = 0):
        """
        Inicializa el conector Malmo.
        
        Args:
            mission_xml: Contenido XML de la misión o ruta al archivo
            port: Puerto donde escucha Minecraft MalmoMod
            server: Dirección del servidor
            role: Rol del agente (para multi-agente)
            experiment_id: ID único del experimento
            episode: Número de episodio inicial
            resync: Resincronizar cada N episodios (0 = nunca)
        """
        self.port = port
        self.server = server
        self.role = role
        self.experiment_id = experiment_id
        self.episode = episode
        self.resync = resync
        
        # Cargar misión XML
        self.mission_xml = self._load_mission(mission_xml)
        
        # Entorno malmoenv
        self.env = None
        self.connected = False
        
    def _load_mission(self, mission_xml: str) -> str:
        """
        Carga la misión XML desde archivo o string.
        
        Args:
            mission_xml: Ruta al archivo XML o contenido XML
            
        Returns:
            Contenido XML de la misión
        """
        # Si es una ruta a archivo
        if mission_xml.endswith('.xml'):
            path = Path(mission_xml)
            if path.exists():
                return path.read_text(encoding='utf-8')
            else:
                raise FileNotFoundError(f"Archivo de misión no encontrado: {mission_xml}")
        
        # Si ya es contenido XML
        return mission_xml
    
    def connect(self) -> bool:
        """
        Conecta con Minecraft/Malmo.
        
        Returns:
            True si la conexión fue exitosa
        """
        try:
            print(f"Conectando a Malmo en {self.server}:{self.port}...")
            
            # Crear entorno
            self.env = malmoenv.make()
            
            # Inicializar conexión
            self.env.init(
                self.mission_xml,
                self.port,
                server=self.server,
                server2=self.server,
                port2=self.port,
                role=self.role,
                exp_uid=self.experiment_id,
                episode=self.episode,
                resync=self.resync
            )
            
            self.connected = True
            print("✓ Conexión exitosa!")
            return True
            
        except Exception as e:
            print(f"✗ Error al conectar: {e}")
            self.connected = False
            return False
    
    def reset(self) -> Any:
        """
        Reinicia el entorno para un nuevo episodio.
        
        Returns:
            Observación inicial
        """
        if not self.connected:
            raise RuntimeError("No conectado. Llama a connect() primero.")
        
        return self.env.reset()
    
    def step(self, action: Dict[str, float]) -> Tuple[Any, float, bool, Dict]:
        """
        Ejecuta una acción en el entorno.
        
        Args:
            action: Diccionario con las acciones (move, turn, jump, etc.)
            
        Returns:
            Tupla (observación, recompensa, done, info)
        """
        if not self.connected:
            raise RuntimeError("No conectado. Llama a connect() primero.")
        
        # malmoenv espera acciones como array numpy
        # Convertir diccionario a array según el action space
        action_array = self._dict_to_action_array(action)
        
        obs, reward, done, info = self.env.step(action_array)
        
        return obs, reward, done, info
    
    def _dict_to_action_array(self, action_dict: Dict[str, float]) -> Any:
        """
        Convierte diccionario de acciones a array para malmoenv.
        
        Args:
            action_dict: Diccionario con acciones
            
        Returns:
            Array de acciones compatible con el entorno
        """
        # malmoenv usa action_space.sample() para obtener el formato
        # Por ahora usamos action_space para determinar el formato
        if hasattr(self.env, 'action_space'):
            # Crear acción base
            action = self.env.action_space.sample()
            
            # Asignar valores del diccionario si es posible
            if hasattr(action, '__len__'):
                # Array de acciones continuas: [move, turn, jump, ...]
                action_list = list(action)
                action_list[0] = action_dict.get('move', 0.0)
                if len(action_list) > 1:
                    action_list[1] = action_dict.get('turn', 0.0)
                if len(action_list) > 2:
                    action_list[2] = action_dict.get('jump', 0.0)
                return action_list
            
            return action
        
        # Fallback: retornar array simple
        return [action_dict.get('move', 0.0), 
                action_dict.get('turn', 0.0),
                action_dict.get('jump', 0.0)]
    
    def close(self) -> None:
        """Cierra la conexión con Malmo."""
        if self.env:
            try:
                self.env.close()
            except:
                pass
        self.connected = False
        print("Conexión cerrada.")
    
    def get_observation_info(self, obs: Any) -> Dict[str, Any]:
        """
        Extrae información útil de la observación.
        
        Args:
            obs: Observación del entorno
            
        Returns:
            Diccionario con datos parseados
        """
        info = {}
        
        # La observación puede ser un array (frame) o dict
        if isinstance(obs, dict):
            info = obs
        elif hasattr(obs, 'shape'):
            # Es un frame (imagen)
            info['frame_shape'] = obs.shape
        else:
            info['raw'] = obs
        
        return info
    
    @property
    def action_space(self):
        """Retorna el espacio de acciones."""
        if self.env:
            return self.env.action_space
        return None
    
    @property
    def observation_space(self):
        """Retorna el espacio de observaciones."""
        if self.env:
            return self.env.observation_space
        return None