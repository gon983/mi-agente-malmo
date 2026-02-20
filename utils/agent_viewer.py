"""
Visualización del agente en terminal usando Rich.

Muestra en tiempo real:
- Estado del episodio y paso actual
- Posición y vida del agente
- Acción tomada en cada paso
- Recompensas y estadísticas
- Historial de acciones recientes
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.style import Style
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Rich no está instalado. Instala con: pip install rich")


@dataclass
class StepInfo:
    """Información de un paso del agente."""
    step: int
    action: Dict[str, float]
    position: tuple  # (x, y, z)
    life: float
    reward: float
    total_reward: float
    timestamp: float = field(default_factory=time.time)


class AgentViewer:
    """
    Visualizador del agente en terminal usando Rich.
    
    Uso:
        viewer = AgentViewer()
        viewer.start_episode(1, 5)
        for step in range(max_steps):
            viewer.update_step(step_info)
        viewer.end_episode()
    """
    
    def __init__(self, 
                 max_history: int = 10,
                 refresh_rate: float = 0.05):
        """
        Inicializa el visualizador.
        
        Args:
            max_history: Máximo número de pasos en el historial
            refresh_rate: Tasa de refresco en segundos
        """
        if not RICH_AVAILABLE:
            raise ImportError("Rich no está instalado. Instala con: pip install rich")
        
        self.console = Console()
        self.max_history = max_history
        self.refresh_rate = refresh_rate
        
        # Estado
        self.episode = 0
        self.total_episodes = 0
        self.current_step = 0
        self.max_steps = 0
        self.policy = ""
        
        # Datos
        self.history: deque = deque(maxlen=max_history)
        self.step_info: Optional[StepInfo] = None
        self.episode_start_time: float = 0
        self.total_start_time: float = 0
        
        # Live display
        self.live: Optional[Live] = None
        
    def start(self, total_episodes: int, max_steps: int, policy: str) -> None:
        """Inicia la visualización general."""
        self.total_episodes = total_episodes
        self.max_steps = max_steps
        self.policy = policy
        self.total_start_time = time.time()
        
    def start_episode(self, episode: int) -> None:
        """Inicia un nuevo episodio."""
        self.episode = episode
        self.current_step = 0
        self.history.clear()
        self.episode_start_time = time.time()
        
    def update_step(self, 
                    step: int,
                    action: Dict[str, float],
                    observation: Dict[str, Any],
                    reward: float,
                    total_reward: float) -> None:
        """
        Actualiza la información del paso actual.
        
        Args:
            step: Número de paso
            action: Acción tomada
            observation: Observación del entorno
            reward: Recompensa del paso
            total_reward: Recompensa acumulada
        """
        # Extraer posición
        x = observation.get('XPos', 0)
        y = observation.get('YPos', 0)
        z = observation.get('ZPos', 0)
        life = observation.get('Life', 20)
        
        self.step_info = StepInfo(
            step=step,
            action=action,
            position=(x, y, z),
            life=life,
            reward=reward,
            total_reward=total_reward
        )
        
        self.history.append(self.step_info)
        self.current_step = step
        
    def end_episode(self) -> None:
        """Finaliza el episodio actual."""
        pass
        
    def render(self) -> Panel:
        """Genera el panel de visualización."""
        # Crear layout principal
        layout = Layout()
        
        # Header
        header = self._render_header()
        
        # Estado actual
        status = self._render_status()
        
        # Historial
        history = self._render_history()
        
        # Stats
        stats = self._render_stats()
        
        # Combinar en tabla
        table = Table(show_header=False, box=None, expand=True)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        
        table.add_row(status, stats)
        table.add_row(history, "")
        
        return Panel(
            table,
            title=f"[bold cyan]🎮 AGENTE MALMO - VISUALIZACIÓN EN TIEMPO REAL[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        )
        
    def _render_header(self) -> Text:
        """Renderiza el header con episodio y paso."""
        text = Text()
        text.append(f"Episodio: ", style="bold")
        text.append(f"{self.episode}/{self.total_episodes}", style="cyan")
        text.append("    ")
        text.append(f"Paso: ", style="bold")
        text.append(f"{self.current_step}/{self.max_steps}", style="green")
        text.append("    ")
        text.append(f"Política: ", style="bold")
        text.append(f"{self.policy}", style="yellow")
        return text
        
    def _render_status(self) -> Panel:
        """Renderiza el estado actual."""
        if self.step_info is None:
            return Panel("Esperando datos...", title="📍 ESTADO")
        
        info = self.step_info
        
        # Posición
        pos_text = Text()
        pos_text.append(f"X: {info.position[0]:7.2f}\n", style="green")
        pos_text.append(f"Y: {info.position[1]:7.2f}", style="yellow")
        y_indicator = self._get_y_indicator(info.position[1])
        pos_text.append(f" {y_indicator}\n")
        pos_text.append(f"Z: {info.position[2]:7.2f}", style="green")
        
        # Vida
        life_bar = self._render_life_bar(info.life)
        
        # Acción
        action_text = self._render_action(info.action)
        
        # Crear tabla para estado
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(justify="left")
        table.add_column(justify="left")
        
        table.add_row("📍 POSICIÓN", "🎮 ACCIÓN")
        table.add_row(pos_text, action_text)
        table.add_row("", "")
        table.add_row("❤️ VIDA", "🏆 RECOMPENSA")
        table.add_row(life_bar, f"Total: [bold green]{info.total_reward:.2f}[/bold green]\n"
                                f"Paso: {'[green]' if info.reward >= 0 else '[red]'}{info.reward:+.2f}[/]")
        
        return Panel(table, title="[bold]ESTADO ACTUAL[/bold]", border_style="blue")
        
    def _render_action(self, action: Dict[str, float]) -> Text:
        """Renderiza la acción actual."""
        text = Text()
        
        move = action.get('move', 0)
        turn = action.get('turn', 0)
        jump = action.get('jump', 0)
        
        # Move
        move_dir = "⬆️ adelante" if move > 0.3 else "⬇️ atrás" if move < -0.3 else "➡️ quieto"
        text.append(f"move: {move:5.2f} {move_dir}\n")
        
        # Turn
        turn_dir = "➡️ derecha" if turn > 0.1 else "⬅️ izquierda" if turn < -0.1 else "⬆️ recto"
        text.append(f"turn: {turn:5.2f} {turn_dir}\n")
        
        # Jump
        jump_status = "🦘 SALTANDO" if jump > 0 else "🚶 en suelo"
        text.append(f"jump: {jump:5.2f} {jump_status}")
        
        return text
        
    def _render_life_bar(self, life: float, max_life: float = 20.0) -> Text:
        """Renderiza barra de vida."""
        filled = int(life / max_life * 10)
        empty = 10 - filled
        
        bar = "❤️" * filled + "🖤" * empty
        text = Text()
        text.append(f"{bar} ")
        text.append(f"{life:.0f}/{max_life:.0f}", style="red")
        return text
        
    def _get_y_indicator(self, y: float) -> str:
        """Obtiene indicador de altura."""
        if not hasattr(self, '_base_y'):
            self._base_y = y
            return "🏠 suelo"
        
        diff = y - self._base_y
        if diff < 0.5:
            return "🏠 suelo"
        elif diff < 1.5:
            return "🦘 saltando"
        elif diff < 3:
            return "⬆️ elevado"
        else:
            return "🏔️ alto"
            
    def _render_history(self) -> Panel:
        """Renderiza historial de pasos."""
        if not self.history:
            return Panel("Sin historial...", title="📊 HISTORIAL")
        
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("#", style="dim", width=4)
        table.add_column("Acción", width=20)
        table.add_column("Reward", width=8)
        table.add_column("Posición", width=20)
        
        for info in list(self.history)[-5:]:  # Últimos 5
            action_str = f"m:{info.action.get('move', 0):.1f} t:{info.action.get('turn', 0):.1f}"
            reward_str = f"{info.reward:+.2f}"
            pos_str = f"({info.position[0]:.1f}, {info.position[2]:.1f})"
            
            table.add_row(
                str(info.step),
                action_str,
                f"[green]{reward_str}[/green]" if info.reward >= 0 else f"[red]{reward_str}[/red]",
                pos_str
            )
        
        return Panel(table, title="[bold]📊 HISTORIAL (últimos 5)[/bold]", border_style="yellow")
        
    def _render_stats(self) -> Panel:
        """Renderiza estadísticas."""
        if not self.history:
            return Panel("Sin datos...", title="📈 STATS")
        
        # Calcular estadísticas
        elapsed = time.time() - self.episode_start_time
        steps_per_sec = len(self.history) / max(1, elapsed)
        
        avg_reward = sum(s.reward for s in self.history) / max(1, len(self.history))
        
        total_elapsed = time.time() - self.total_start_time if self.total_start_time else 0
        
        text = Text()
        text.append(f"⏱️ Velocidad: [cyan]{steps_per_sec:.1f}[/cyan] pasos/seg\n\n")
        text.append(f"📊 Recompensa promedio:\n   [green]{avg_reward:.4f}[/green]\n\n")
        text.append(f"⏰ Tiempo episodio: [yellow]{self._format_time(elapsed)}[/yellow]\n\n")
        text.append(f"🕐 Tiempo total: [yellow]{self._format_time(total_elapsed)}[/yellow]")
        
        return Panel(text, title="[bold]📈 ESTADÍSTICAS[/bold]", border_style="green")
        
    def _format_time(self, seconds: float) -> str:
        """Formatea tiempo en mm:ss."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
        
    def close(self) -> None:
        """Cierra el visualizador."""
        pass
        
    def is_running(self) -> bool:
        """Retorna si el viewer sigue activo."""
        return True
        
    def print_summary(self, stats: Dict[str, Any]) -> None:
        """Imprime resumen final."""
        self.console.print()
        self.console.rule("[bold cyan]📈 RESUMEN FINAL[/bold cyan]")
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        
        table.add_row("Total episodios", str(stats.get('total_episodes', 0)))
        table.add_row("Recompensa total", f"{stats.get('total_reward', 0):.2f}")
        table.add_row("Promedio por episodio", f"{stats.get('avg_reward', 0):.2f}")
        
        self.console.print(table)
        self.console.rule()


class SimpleViewer:
    """
    Visualizador simple sin Rich (fallback).
    Usa print básico con formateo.
    """
    
    def __init__(self):
        self.step_count = 0
        self.total_reward = 0.0
        self.episode = 0
        self.total_episodes = 0
        
    def start(self, total_episodes: int, max_steps: int, policy: str) -> None:
        """Inicia la visualización."""
        self.total_episodes = total_episodes
        
    def start_episode(self, episode: int) -> None:
        """Inicia un episodio."""
        self.episode = episode
        print(f"\n{'='*60}")
        print(f"EPISODIO {episode}/{self.total_episodes}")
        print('='*60)
        
    def update_step(self,
                    step: int,
                    action: Dict[str, float],
                    observation: Dict[str, Any],
                    reward: float,
                    total_reward: float) -> None:
        x = observation.get('XPos', 0)
        y = observation.get('YPos', 0)
        z = observation.get('ZPos', 0)
        
        move = action.get('move', 0)
        turn = action.get('turn', 0)
        jump = action.get('jump', 0)
        
        print(f"[{step:3d}] Pos:({x:6.1f}, {y:5.1f}, {z:6.1f}) | "
              f"Action: m={move:.1f} t={turn:.1f} j={jump:.0f} | "
              f"R: {reward:+.2f} (Total: {total_reward:.2f})")
        
    def end_episode(self) -> None:
        """Finaliza el episodio."""
        pass
        
    def close(self) -> None:
        """Cierra el visualizador."""
        pass
        
    def is_running(self) -> bool:
        """Retorna si el viewer sigue activo."""
        return True


def create_viewer(use_rich: bool = True) -> Any:
    """
    Factory para crear visualizador.
    
    Args:
        use_rich: Si True, usa Rich si está disponible
        
    Returns:
        Instancia de visualizador
    """
    if use_rich and RICH_AVAILABLE:
        return AgentViewer()
    else:
        return SimpleViewer()