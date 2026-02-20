"""
Visualización gráfica del agente usando Pygame y Matplotlib.

Proporciona:
- Mini-mapa 2D con trayectoria del agente (X,Z)
- Codificación de altura Y mediante colores
- Gráficos de recompensa y estadísticas en tiempo real
"""

import time
import threading
import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: Pygame no está instalado. Instala con: pip install pygame")

try:
    import matplotlib
    matplotlib.use('TkAgg')  # Backend para ventanas separadas
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib no está instalado. Instala con: pip install matplotlib")

import numpy as np


# Colores para altura Y
HEIGHT_COLORS = {
    'ground': (34, 139, 34),      # Verde bosque - suelo
    'low': (144, 238, 144),       # Verde claro - bajo
    'jumping': (255, 215, 0),     # Oro - saltando
    'elevated': (255, 165, 0),    # Naranja - elevado
    'high': (255, 69, 0),         # Rojo-naranja - alto
    'falling': (30, 144, 255),    # Azul - cayendo
}


@dataclass
class TrajectoryPoint:
    """Punto en la trayectoria del agente."""
    x: float
    y: float
    z: float
    step: int
    reward: float
    action: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


class PygameViewer:
    """
    Visualizador gráfico con Pygame.
    
    Muestra un mini-mapa 2D con:
    - Posición actual del agente
    - Trayectoria histórica
    - Color codificado por altura Y
    - Panel de estadísticas
    """
    
    def __init__(self,
                 width: int = 800,
                 height: int = 600,
                 max_trail: int = 500,
                 grid_size: float = 1.0):
        """
        Inicializa el visualizador Pygame.
        
        Args:
            width: Ancho de la ventana
            height: Alto de la ventana
            max_trail: Máximo puntos en la trayectoria
            grid_size: Tamaño del grid para el mapa
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("Pygame no está instalado. Instala con: pip install pygame")
        
        self.width = width
        self.height = height
        self.max_trail = max_trail
        self.grid_size = grid_size
        
        # Inicializar pygame
        pygame.init()
        pygame.font.init()
        
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("🎮 Agente Malmo - Visualizador")
        
        self.font = pygame.font.SysFont('Arial', 14)
        self.font_large = pygame.font.SysFont('Arial', 18, bold=True)
        
        # Colores
        self.BG_COLOR = (20, 20, 30)
        self.GRID_COLOR = (40, 40, 50)
        self.TEXT_COLOR = (200, 200, 200)
        self.AGENT_COLOR = (255, 100, 100)
        self.TRAIL_COLOR = (100, 100, 255)
        
        # Estado
        self.trajectory: deque = deque(maxlen=max_trail)
        self.current_pos = (0.0, 64.0, 0.0)  # (x, y, z)
        self.current_action = {}
        self.current_reward = 0.0
        self.total_reward = 0.0
        self.life = 20.0
        self.step = 0
        self.episode = 0
        self.total_episodes = 0
        
        # Centro del mapa (se actualiza dinámicamente)
        self.center_x = 0.0
        self.center_z = 0.0
        self.base_y = 64.0  # Y de referencia (suelo)
        
        # Escala (pixels por bloque)
        self.scale = 20.0
        self.auto_scale = True
        
        # Tiempos
        self.start_time = time.time()
        self.running = True
        
        # Layout
        self.map_rect = pygame.Rect(10, 50, width - 220, height - 60)
        self.stats_rect = pygame.Rect(width - 200, 50, 190, height - 60)
        
    def world_to_screen(self, x: float, z: float) -> Tuple[int, int]:
        """Convierte coordenadas del mundo a pantalla."""
        # Centro del mapa
        cx = self.map_rect.centerx
        cz = self.map_rect.centery
        
        # Convertir a pixels
        px = cx + (x - self.center_x) * self.scale
        pz = cz + (z - self.center_z) * self.scale
        
        return int(px), int(pz)
        
    def get_height_color(self, y: float) -> Tuple[int, int, int]:
        """Obtiene color basado en la altura Y."""
        diff = y - self.base_y
        
        if diff < -0.5:
            return HEIGHT_COLORS['falling']
        elif diff < 0.5:
            return HEIGHT_COLORS['ground']
        elif diff < 1.5:
            return HEIGHT_COLORS['jumping']
        elif diff < 3.0:
            return HEIGHT_COLORS['elevated']
        else:
            return HEIGHT_COLORS['high']
            
    def start(self, total_episodes: int, max_steps: int, policy: str) -> None:
        """Inicia la visualización."""
        self.total_episodes = total_episodes
        self.start_time = time.time()
        
    def start_episode(self, episode: int) -> None:
        """Inicia un nuevo episodio."""
        self.episode = episode
        self.step = 0
        self.total_reward = 0.0
        
    def update_step(self,
                    step: int,
                    action: Dict[str, float],
                    observation: Dict[str, Any],
                    reward: float,
                    total_reward: float) -> None:
        """Actualiza el estado del paso actual."""
        x = observation.get('XPos', 0)
        y = observation.get('YPos', 0)
        z = observation.get('ZPos', 0)
        life = observation.get('Life', 20)
        
        # Actualizar Y base si es el primer paso
        if step == 1:
            self.base_y = y
            self.center_x = x
            self.center_z = z
            
        self.current_pos = (x, y, z)
        self.current_action = action
        self.current_reward = reward
        self.total_reward = total_reward
        self.life = life
        self.step = step
        
        # Agregar a trayectoria
        point = TrajectoryPoint(x=x, y=y, z=z, step=step, reward=reward, action=action)
        self.trajectory.append(point)
        
        # Auto-centrar y auto-escalar
        if self.auto_scale and len(self.trajectory) > 1:
            self._update_view()
            
    def _update_view(self) -> None:
        """Actualiza el centro y escala de la vista."""
        if not self.trajectory:
            return
            
        # Calcular bounds
        xs = [p.x for p in self.trajectory]
        zs = [p.z for p in self.trajectory]
        
        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        
        # Centro
        self.center_x = (min_x + max_x) / 2
        self.center_z = (min_z + max_z) / 2
        
        # Escala para que quepa
        dx = max_x - min_x + 2
        dz = max_z - min_z + 2
        
        scale_x = (self.map_rect.width - 20) / max(dx, 1)
        scale_z = (self.map_rect.height - 20) / max(dz, 1)
        
        self.scale = min(scale_x, scale_z, 30)  # Max 30 pixels por bloque
        
    def render(self) -> None:
        """Renderiza la visualización."""
        # Procesar eventos pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    # Reset view
                    self.trajectory.clear()
                    
        # Limpiar pantalla
        self.screen.fill(self.BG_COLOR)
        
        # Dibujar header
        self._render_header()
        
        # Dibujar mapa
        self._render_map()
        
        # Dibujar panel de stats
        self._render_stats_panel()
        
        # Actualizar display
        pygame.display.flip()
        
    def _render_header(self) -> None:
        """Renderiza el header."""
        # Título
        title = self.font_large.render("🎮 AGENTE MALMO - MINI-MAPA", True, (100, 200, 255))
        self.screen.blit(title, (10, 10))
        
        # Info episodio
        ep_text = self.font.render(f"Episodio: {self.episode}/{self.total_episodes} | Paso: {self.step}", True, self.TEXT_COLOR)
        self.screen.blit(ep_text, (300, 15))
        
    def _render_map(self) -> None:
        """Renderiza el mini-mapa."""
        # Fondo del mapa
        pygame.draw.rect(self.screen, (30, 30, 40), self.map_rect)
        pygame.draw.rect(self.screen, (60, 60, 80), self.map_rect, 2)
        
        # Grid
        self._render_grid()
        
        # Trayectoria
        self._render_trajectory()
        
        # Agente actual
        self._render_agent()
        
        # Brújula
        self._render_compass()
        
        # Leyenda de altura
        self._render_height_legend()
        
    def _render_grid(self) -> None:
        """Renderiza grid del mapa."""
        # Líneas cada ciertos bloques
        grid_step = max(1, int(5 * self.scale / 20))  # Ajustar según escala
        
        cx, cz = self.map_rect.centerx, self.map_rect.centery
        
        for i in range(-20, 21, grid_step):
            # Líneas verticales
            x = cx + i * self.scale
            if self.map_rect.left < x < self.map_rect.right:
                pygame.draw.line(self.screen, self.GRID_COLOR,
                               (x, self.map_rect.top),
                               (x, self.map_rect.bottom))
            
            # Líneas horizontales
            z = cz + i * self.scale
            if self.map_rect.top < z < self.map_rect.bottom:
                pygame.draw.line(self.screen, self.GRID_COLOR,
                               (self.map_rect.left, z),
                               (self.map_rect.right, z))
                               
    def _render_trajectory(self) -> None:
        """Renderiza la trayectoria del agente."""
        if len(self.trajectory) < 2:
            return
            
        points = list(self.trajectory)
        
        # Dibujar trail con colores por altura
        for i in range(1, len(points)):
            p1 = points[i-1]
            p2 = points[i]
            
            # Coordenadas de pantalla
            x1, z1 = self.world_to_screen(p1.x, p1.z)
            x2, z2 = self.world_to_screen(p2.x, p2.z)
            
            # Solo dibujar si está en el mapa
            if (self.map_rect.collidepoint(x1, z1) or 
                self.map_rect.collidepoint(x2, z2)):
                
                # Color por altura
                color = self.get_height_color(p2.y)
                
                # Grosor por reward
                thickness = 1 if p2.reward == 0 else 2
                
                pygame.draw.line(self.screen, color, (x1, z1), (x2, z2), thickness)
                
    def _render_agent(self) -> None:
        """Renderiza la posición actual del agente."""
        x, y, z = self.current_pos
        px, pz = self.world_to_screen(x, z)
        
        if not self.map_rect.collidepoint(px, pz):
            return
            
        # Color por altura
        color = self.get_height_color(y)
        
        # Tamaño por si está saltando
        radius = 8 if abs(y - self.base_y) < 0.5 else 10
        
        # Círculo exterior (dirección)
        if self.current_action:
            move = self.current_action.get('move', 0)
            turn = self.current_action.get('turn', 0)
            
            # Indicador de dirección
            angle = turn * math.pi / 2  # Convertir turn a ángulo
            dx = math.sin(angle) * 15
            dy = -move * 10
            
            # Flecha de dirección
            end_x = px + dx
            end_y = pz + dy
            pygame.draw.line(self.screen, (255, 255, 255), (px, pz), (end_x, end_y), 2)
        
        # Punto central del agente
        pygame.draw.circle(self.screen, color, (px, pz), radius)
        pygame.draw.circle(self.screen, (255, 255, 255), (px, pz), radius, 2)
        
        # Indicador de salto
        if y > self.base_y + 0.5:
            jump_text = self.font.render("🦘", True, (255, 215, 0))
            self.screen.blit(jump_text, (px + 10, pz - 10))
            
    def _render_compass(self) -> None:
        """Renderiza la brújula."""
        # Posición en esquina superior izquierda del mapa
        cx = self.map_rect.left + 30
        cy = self.map_rect.top + 30
        
        # Círculo de fondo
        pygame.draw.circle(self.screen, (40, 40, 50), (cx, cy), 20)
        pygame.draw.circle(self.screen, (80, 80, 100), (cx, cy), 20, 1)
        
        # Norte
        pygame.draw.line(self.screen, (255, 100, 100), (cx, cy), (cx, cy - 15), 2)
        n_text = self.font.render("N", True, (255, 100, 100))
        self.screen.blit(n_text, (cx - 5, cy - 35))
        
    def _render_height_legend(self) -> None:
        """Renderiza leyenda de altura."""
        # Posición en esquina inferior izquierda del mapa
        x = self.map_rect.left + 10
        y = self.map_rect.bottom - 100
        
        texts = [
            ("🟢 Suelo", HEIGHT_COLORS['ground']),
            ("🟡 Saltando", HEIGHT_COLORS['jumping']),
            ("🟠 Elevado", HEIGHT_COLORS['elevated']),
            ("🔵 Cayendo", HEIGHT_COLORS['falling']),
        ]
        
        for i, (text, color) in enumerate(texts):
            pygame.draw.rect(self.screen, color, (x, y + i * 20, 12, 12))
            label = self.font.render(text, True, self.TEXT_COLOR)
            self.screen.blit(label, (x + 18, y + i * 20))
            
    def _render_stats_panel(self) -> None:
        """Renderiza panel de estadísticas."""
        # Fondo
        pygame.draw.rect(self.screen, (30, 30, 40), self.stats_rect)
        pygame.draw.rect(self.screen, (60, 60, 80), self.stats_rect, 2)
        
        y = self.stats_rect.top + 10
        x = self.stats_rect.left + 10
        
        # Título
        title = self.font_large.render("📊 ESTADÍSTICAS", True, (100, 200, 255))
        self.screen.blit(title, (x, y))
        y += 30
        
        # Posición
        x_pos, y_pos, z_pos = self.current_pos
        pos_text = [
            f"📍 Posición:",
            f"  X: {x_pos:.2f}",
            f"  Y: {y_pos:.2f} ({self._get_y_status(y_pos)})",
            f"  Z: {z_pos:.2f}",
        ]
        
        for text in pos_text:
            label = self.font.render(text, True, self.TEXT_COLOR)
            self.screen.blit(label, (x, y))
            y += 18
            
        y += 10
        
        # Vida
        life_text = f"❤️ Vida: {self.life:.0f}/20"
        label = self.font.render(life_text, True, (255, 100, 100))
        self.screen.blit(label, (x, y))
        y += 20
        
        # Barra de vida
        life_width = int((self.life / 20) * (self.stats_rect.width - 30))
        pygame.draw.rect(self.screen, (80, 20, 20), (x, y, self.stats_rect.width - 30, 10))
        pygame.draw.rect(self.screen, (200, 50, 50), (x, y, life_width, 10))
        y += 20
        
        # Recompensa
        reward_color = (100, 255, 100) if self.current_reward >= 0 else (255, 100, 100)
        reward_text = f"🏆 Reward paso: {self.current_reward:+.2f}"
        label = self.font.render(reward_text, True, reward_color)
        self.screen.blit(label, (x, y))
        y += 20
        
        total_text = f"   Total: {self.total_reward:.2f}"
        label = self.font.render(total_text, True, (100, 255, 100))
        self.screen.blit(label, (x, y))
        y += 25
        
        # Acción actual
        action_title = self.font.render("🎮 Acción:", True, (200, 200, 100))
        self.screen.blit(action_title, (x, y))
        y += 20
        
        if self.current_action:
            move = self.current_action.get('move', 0)
            turn = self.current_action.get('turn', 0)
            jump = self.current_action.get('jump', 0)
            
            move_text = f"  move: {move:.2f} {self._get_move_dir(move)}"
            turn_text = f"  turn: {turn:.2f} {self._get_turn_dir(turn)}"
            jump_text = f"  jump: {jump:.0f} {'🦘' if jump > 0 else ''}"
            
            for text in [move_text, turn_text, jump_text]:
                label = self.font.render(text, True, self.TEXT_COLOR)
                self.screen.blit(label, (x, y))
                y += 18
                
        y += 10
        
        # Tiempo
        elapsed = time.time() - self.start_time
        time_text = f"⏱️ Tiempo: {self._format_time(elapsed)}"
        label = self.font.render(time_text, True, (200, 200, 100))
        self.screen.blit(label, (x, y))
        y += 25
        
        # Velocidad
        if elapsed > 0:
            speed = self.step / elapsed
            speed_text = f"⚡ Velocidad: {speed:.1f} pasos/seg"
            label = self.font.render(speed_text, True, self.TEXT_COLOR)
            self.screen.blit(label, (x, y))
            
    def _get_y_status(self, y: float) -> str:
        """Obtiene descripción del estado de altura."""
        diff = y - self.base_y
        if diff < -0.5:
            return "⬇️ cayendo"
        elif diff < 0.5:
            return "🏠 suelo"
        elif diff < 1.5:
            return "🦘 saltando"
        else:
            return "⬆️ elevado"
            
    def _get_move_dir(self, move: float) -> str:
        """Obtiene dirección de movimiento."""
        if move > 0.3:
            return "⬆️"
        elif move < -0.3:
            return "⬇️"
        return "➡️"
    
    def _get_turn_dir(self, turn: float) -> str:
        """Obtiene dirección de giro."""
        if turn > 0.1:
            return "➡️"
        elif turn < -0.1:
            return "⬅️"
        return "⬆️"
        
    def _format_time(self, seconds: float) -> str:
        """Formatea tiempo en mm:ss."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
        
    def end_episode(self) -> None:
        """Finaliza el episodio."""
        pass
        
    def close(self) -> None:
        """Cierra la ventana."""
        self.running = False
        pygame.quit()
        
    def is_running(self) -> bool:
        """Retorna si la ventana sigue abierta."""
        return self.running


class MatplotlibViewer:
    """
    Visualizador con gráficos Matplotlib en tiempo real.
    
    Muestra:
    - Gráfico de recompensa acumulada vs pasos
    - Gráfico de altura Y vs pasos
    - Mapa de calor de posición (opcional)
    """
    
    def __init__(self, update_interval: int = 1000):
        """
        Inicializa el visualizador Matplotlib.
        
        Args:
            update_interval: Intervalo de actualización en ms
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib no está instalado. Instala con: pip install matplotlib")
        
        self.update_interval = update_interval
        
        # Datos
        self.steps: List[int] = []
        self.rewards: List[float] = []
        self.cumulative_rewards: List[float] = []
        self.heights: List[float] = []
        self.positions: List[Tuple[float, float]] = []
        
        # Estado
        self.total_reward = 0.0
        self.current_step = 0
        self.episode = 0
        
        # Figura
        self.fig = None
        self.axes = None
        self.initialized = False
        
    def _init_figure(self) -> None:
        """Inicializa la figura de matplotlib."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('🎮 Agente Malmo - Visualización', fontsize=14, fontweight='bold')
        
        # Configurar subplots
        self.ax_reward = self.axes[0, 0]
        self.ax_height = self.axes[0, 1]
        self.ax_trajectory = self.axes[1, 0]
        self.ax_heatmap = self.axes[1, 1]
        
        # Reward plot
        self.ax_reward.set_title('Recompensa Acumulada')
        self.ax_reward.set_xlabel('Pasos')
        self.ax_reward.set_ylabel('Reward Total')
        self.ax_reward.grid(True, alpha=0.3)
        
        # Height plot
        self.ax_height.set_title('Altura Y')
        self.ax_height.set_xlabel('Pasos')
        self.ax_height.set_ylabel('Y')
        self.ax_height.grid(True, alpha=0.3)
        
        # Trajectory plot
        self.ax_trajectory.set_title('Trayectoria (X, Z)')
        self.ax_trajectory.set_xlabel('X')
        self.ax_trajectory.set_ylabel('Z')
        self.ax_trajectory.grid(True, alpha=0.3)
        self.ax_trajectory.set_aspect('equal')
        
        # Heatmap
        self.ax_heatmap.set_title('Mapa de Calor de Posición')
        self.ax_heatmap.set_xlabel('X')
        self.ax_heatmap.set_ylabel('Z')
        
        plt.tight_layout()
        self.initialized = True
        
    def start(self, total_episodes: int, max_steps: int, policy: str) -> None:
        """Inicia la visualización."""
        self._init_figure()
        plt.show(block=False)
        
    def start_episode(self, episode: int) -> None:
        """Inicia un nuevo episodio."""
        self.episode = episode
        
    def update_step(self,
                    step: int,
                    action: Dict[str, float],
                    observation: Dict[str, Any],
                    reward: float,
                    total_reward: float) -> None:
        """Actualiza los datos."""
        x = observation.get('XPos', 0)
        y = observation.get('YPos', 0)
        z = observation.get('ZPos', 0)
        
        self.steps.append(step)
        self.rewards.append(reward)
        self.cumulative_rewards.append(total_reward)
        self.heights.append(y)
        self.positions.append((x, z))
        
        self.total_reward = total_reward
        self.current_step = step
        
        # Actualizar cada N pasos
        if step % 5 == 0:
            self._update_plots()
            
    def _update_plots(self) -> None:
        """Actualiza los gráficos."""
        if not self.initialized:
            return
            
        # Limpiar axes
        for ax in [self.ax_reward, self.ax_height, self.ax_trajectory, self.ax_heatmap]:
            ax.clear()
        
        # Reward
        self.ax_reward.plot(self.steps, self.cumulative_rewards, 'b-', linewidth=2)
        self.ax_reward.fill_between(self.steps, self.cumulative_rewards, alpha=0.3)
        self.ax_reward.set_title('Recompensa Acumulada')
        self.ax_reward.set_xlabel('Pasos')
        self.ax_reward.set_ylabel('Reward Total')
        self.ax_reward.grid(True, alpha=0.3)
        
        # Height
        colors = ['green' if h > 64 else 'blue' for h in self.heights]
        self.ax_height.scatter(self.steps, self.heights, c=colors, s=2)
        self.ax_height.axhline(y=64, color='brown', linestyle='--', alpha=0.5, label='Suelo')
        self.ax_height.set_title('Altura Y')
        self.ax_height.set_xlabel('Pasos')
        self.ax_height.set_ylabel('Y')
        self.ax_height.grid(True, alpha=0.3)
        self.ax_height.legend()
        
        # Trajectory
        if self.positions:
            xs = [p[0] for p in self.positions]
            zs = [p[1] for p in self.positions]
            
            # Color por índice (más reciente = más brillante)
            colors = plt.cm.viridis(np.linspace(0, 1, len(xs)))
            
            for i in range(1, len(xs)):
                self.ax_trajectory.plot([xs[i-1], xs[i]], [zs[i-1], zs[i]], 
                                       color=colors[i], linewidth=1)
            
            # Punto actual
            self.ax_trajectory.scatter([xs[-1]], [zs[-1]], c='red', s=100, marker='*', zorder=5)
            
            self.ax_trajectory.set_title('Trayectoria (X, Z)')
            self.ax_trajectory.set_xlabel('X')
            self.ax_trajectory.set_ylabel('Z')
            self.ax_trajectory.grid(True, alpha=0.3)
            
        # Heatmap
        if len(self.positions) > 10:
            xs = np.array([p[0] for p in self.positions])
            zs = np.array([p[1] for p in self.positions])
            
            # Crear grid para heatmap
            x_bins = np.linspace(xs.min() - 1, xs.max() + 1, 20)
            z_bins = np.linspace(zs.min() - 1, zs.max() + 1, 20)
            
            heatmap, xedges, yedges = np.histogram2d(xs, zs, bins=[x_bins, z_bins])
            
            self.ax_heatmap.imshow(heatmap.T, origin='lower', aspect='auto',
                                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                   cmap='hot', interpolation='nearest')
            self.ax_heatmap.set_title('Mapa de Calor')
            self.ax_heatmap.set_xlabel('X')
            self.ax_heatmap.set_ylabel('Z')
        
        plt.tight_layout()
        plt.pause(0.001)
        
    def end_episode(self) -> None:
        """Finaliza el episodio."""
        self._update_plots()
        
    def close(self) -> None:
        """Cierra la figura."""
        if self.fig:
            plt.close(self.fig)


class GraphicalViewer:
    """
    Visualizador gráfico combinado.
    
    Combina Pygame (mini-mapa) y Matplotlib (gráficos).
    """
    
    def __init__(self, use_pygame: bool = True, use_matplotlib: bool = True):
        """
        Inicializa el visualizador combinado.
        
        Args:
            use_pygame: Si True, usa Pygame para mini-mapa
            use_matplotlib: Si True, usa Matplotlib para gráficos
        """
        self.pygame_viewer = None
        self.matplotlib_viewer = None
        
        if use_pygame and PYGAME_AVAILABLE:
            self.pygame_viewer = PygameViewer()
            
        if use_matplotlib and MATPLOTLIB_AVAILABLE:
            self.matplotlib_viewer = MatplotlibViewer()
            
    def start(self, total_episodes: int, max_steps: int, policy: str) -> None:
        """Inicia la visualización."""
        if self.pygame_viewer:
            self.pygame_viewer.start(total_episodes, max_steps, policy)
        if self.matplotlib_viewer:
            self.matplotlib_viewer.start(total_episodes, max_steps, policy)
            
    def start_episode(self, episode: int) -> None:
        """Inicia un nuevo episodio."""
        if self.pygame_viewer:
            self.pygame_viewer.start_episode(episode)
        if self.matplotlib_viewer:
            self.matplotlib_viewer.start_episode(episode)
            
    def update_step(self,
                    step: int,
                    action: Dict[str, float],
                    observation: Dict[str, Any],
                    reward: float,
                    total_reward: float) -> None:
        """Actualiza la visualización."""
        if self.pygame_viewer:
            self.pygame_viewer.update_step(step, action, observation, reward, total_reward)
            self.pygame_viewer.render()
            
        if self.matplotlib_viewer:
            self.matplotlib_viewer.update_step(step, action, observation, reward, total_reward)
            
    def end_episode(self) -> None:
        """Finaliza el episodio."""
        if self.pygame_viewer:
            self.pygame_viewer.end_episode()
        if self.matplotlib_viewer:
            self.matplotlib_viewer.end_episode()
            
    def close(self) -> None:
        """Cierra la visualización."""
        if self.pygame_viewer:
            self.pygame_viewer.close()
        if self.matplotlib_viewer:
            self.matplotlib_viewer.close()
            
    def is_running(self) -> bool:
        """Retorna si las ventanas siguen abiertas."""
        if self.pygame_viewer and not self.pygame_viewer.is_running():
            return False
        return True


def create_graphical_viewer(pygame: bool = True, matplotlib: bool = True) -> Any:
    """
    Factory para crear visualizador gráfico.
    
    Args:
        pygame: Si True, incluye mini-mapa Pygame
        matplotlib: Si True, incluye gráficos Matplotlib
        
    Returns:
        Instancia de visualizador gráfico
    """
    return GraphicalViewer(use_pygame=pygame, use_matplotlib=matplotlib)