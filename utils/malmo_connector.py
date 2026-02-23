"""
Conector para Microsoft Malmo

Maneja la comunicación entre el agente y Minecraft via malmoenv.
"""

import logging
from pathlib import Path
from numbers import Integral
from typing import Dict, Any, Optional, Tuple, List, Set

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
                 resync: int = 0,
                 reshape: bool = True,
                 action_filter: Optional[Set[str]] = None):
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
        self.reshape = reshape
        self.action_filter = set(action_filter) if action_filter else {
            "move",
            "turn",
            "jump",
            "use",
            "attack",
        }
        
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
            print(f"Connecting to Malmo at {self.server}:{self.port}...")
            
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
                resync=self.resync,
                action_filter=self.action_filter,
                reshape=self.reshape
            )

            actions = getattr(getattr(self.env, "action_space", None), "actions", None)
            if isinstance(actions, list):
                preview = ", ".join(actions[:10])
                if len(actions) > 10:
                    preview += ", ..."
                print(f"Action space ({len(actions)}): {preview}")
            
            self.connected = True
            print("Connection established.")
            return True
            
        except Exception as e:
            print(f"Connection error: {e}")
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
    
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """
        Ejecuta una acción en el entorno.
        
        Args:
            action: Accion del agente (indice discreto, comando o diccionario)
            
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
    
    def _dict_to_action_array(self, action: Any) -> Any:
        """
        Convierte diccionario de acciones a array para malmoenv.
        
        Args:
            action: Accion del agente
            
        Returns:
            Array de acciones compatible con el entorno
        """
        if not hasattr(self.env, "action_space"):
            raise RuntimeError("Environment has no action_space; cannot map agent action.")

        action_space = self.env.action_space
        actions = getattr(action_space, "actions", None)
        if not isinstance(actions, list) or len(actions) == 0:
            raise ValueError(
                "Expected discrete Malmo action space with explicit command list. "
                f"Got action_space={type(action_space).__name__} actions={actions!r}"
            )

        return self._dict_to_discrete_index(action, actions)

    def _dict_to_discrete_index(self, action: Any, actions: List[str]) -> int:
        """
        Convert agent action into a discrete Malmo command index.
        """
        index_by_cmd = {cmd: idx for idx, cmd in enumerate(actions)}

        if isinstance(action, Integral):
            candidate = int(action)
            if 0 <= candidate < len(actions):
                return candidate
            raise ValueError(
                f"Invalid discrete action index {candidate}. "
                f"Valid range is [0, {len(actions) - 1}]"
            )

        if isinstance(action, str):
            command = " ".join(action.strip().split())
            if command in index_by_cmd:
                return index_by_cmd[command]
            raise ValueError(
                f"Action command '{command}' not available in action space. "
                f"Available: {actions}"
            )

        action_dict = action if isinstance(action, dict) else {}
        if "action_command" in action_dict:
            command = " ".join(str(action_dict["action_command"]).strip().split())
            if command in index_by_cmd:
                return index_by_cmd[command]
            raise ValueError(
                f"Action command '{command}' not available in action space. "
                f"Available: {actions}"
            )

        if "action_index" in action_dict:
            # Fallback only: this index is assumed to already match env.action_space.actions.
            try:
                candidate = int(action_dict["action_index"])
                if 0 <= candidate < len(actions):
                    return candidate
            except (TypeError, ValueError):
                raise ValueError(f"Invalid action_index value: {action_dict['action_index']!r}")
            raise ValueError(
                f"Action index {candidate} out of range for action space size {len(actions)}"
            )

        if any(key in action_dict for key in ("move", "turn", "jump")):
            move = float(action_dict.get("move", 0.0))
            turn = float(action_dict.get("turn", 0.0))
            jump = float(action_dict.get("jump", 0.0))

            if jump > 0.5:
                command = "jump 1"
            elif move > 0.1:
                command = "move 1"
            elif move < -0.1:
                command = "move -1"
            elif turn > 0.1:
                command = "turn 1"
            elif turn < -0.1:
                command = "turn -1"
            else:
                raise ValueError(
                    "Legacy move/turn/jump action has no active command above threshold. "
                    "Use action_command/action_index with a valid discrete command."
                )

            if command in index_by_cmd:
                return index_by_cmd[command]
            raise ValueError(
                f"Legacy command '{command}' not available in action space. "
                f"Available: {actions}"
            )

        raise ValueError(
            "Invalid action payload. Expected one of: "
            "discrete index (int), action command (str), "
            "{action_command: str}, {action_index: int}, "
            "or legacy {move/turn/jump} with one active command."
        )
    
    def close(self) -> None:
        """Cierra la conexión con Malmo."""
        if self.env:
            try:
                self.env.close()
            except:
                pass
        self.connected = False
        print("Connection closed.")
    
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
