# 🎮 Agente de Malmo

Proyecto para entrenar agentes de inteligencia artificial utilizando Microsoft Malmo y Minecraft.

---

## 📋 Requisitos Previos Instalados

### Software del Sistema

| Software | Versión | Ubicación | Propósito |
|----------|---------|-----------|-----------|
| **Java JDK 8** | 8.0.482.8-hotspot | `C:\Program Files\Eclipse Adoptium\jdk-8.0.482.8-hotspot` | Minecraft/Forge |
| **FFmpeg** | 8.0.1 (Gyan) | Instalado via winget | Codificación de video |
| **Python** | 3.13 | Sistema | Lenguaje principal |
| **malmoenv** | 0.0.8 | `C:\Users\gonza\AppData\Roaming\Python\Python313\site-packages` | Librería Malmo para Python |
| **gymnasium** | 1.2.3 | `C:\Users\gonza\AppData\Roaming\Python\Python313\site-packages` | Reemplazo moderno de OpenAI Gym |

### Variables de Entorno Configuradas

| Variable | Valor |
|----------|-------|
| `JAVA_HOME` | `C:\Program Files\Eclipse Adoptium\jdk-8.0.482.8-hotspot` |
| `MALMO_XSD_PATH` | `c:\Users\gonza\malmo\Schemas` |
| `PATH` | Incluye FFmpeg y Java |

### Proyecto Malmo Base

| Componente | Ubicación |
|------------|-----------|
| Repositorio Malmo | `c:\Users\gonza\malmo` |
| Schemas XML | `c:\Users\gonza\malmo\Schemas\` |
| Misiones de ejemplo | `c:\Users\gonza\malmo\MalmoEnv\missions\` |
| Minecraft con MalmoMod | `c:\Users\gonza\malmo\Minecraft\` |

---

## 🚀 Cómo Usar Este Proyecto

### Paso 1: Iniciar Minecraft con MalmoMod

Abre una terminal y ejecuta:

```bash
cd c:\Users\gonza\malmo\Minecraft
launchClient.bat -port 9000 -env
```

Esto abrirá Minecraft con el mod de Malmo escuchando en el puerto 9000.

### Paso 2: Ejecutar el Agente

En otra terminal:

```bash
cd C:\Users\gonza\agentedemalmo
python run_agent.py
```

---

## 📁 Estructura del Proyecto

```
agentedemalmo/
├── README.md                 # Esta documentación
├── requirements.txt          # Dependencias Python
├── config/
│   └── config.yaml          # Configuración del agente
├── agents/
│   ├── __init__.py
│   └── basic_agent.py       # Agente minimalista
├── missions/
│   └── simple_test.xml      # Misiones personalizadas
├── utils/
│   ├── __init__.py
│   └── malmo_connector.py   # Conector con Malmo
├── logs/                    # Logs de entrenamiento
└── models/                  # Modelos guardados
```

### Descripción de Carpetas

| Carpeta | Propósito |
|---------|-----------|
| `config/` | Archivos de configuración YAML para el agente |
| `agents/` | Implementaciones de agentes (RL, básicos, etc.) |
| `missions/` | Archivos XML con definiciones de misiones |
| `utils/` | Utilidades y conectores |
| `logs/` | Logs generados durante entrenamiento |
| `models/` | Modelos entrenados guardados |

---

## 🔧 Configuración del Agente

El archivo `config/config.yaml` contiene todos los parámetros configurables:

```yaml
malmo:
  port: 9000
  server: "127.0.0.1"
  mission: "missions/simple_test.xml"

agent:
  type: "basic"  # basic, dqn, ppo, etc.
  episodes: 10
  max_steps: 100

logging:
  level: "INFO"
  save_rewards: true
```

---

## 🤖 Tipos de Agentes

### Agente Básico (`basic_agent.py`)

Agente minimalista que:
- Se conecta a Minecraft/Malmo
- Toma acciones aleatorias
- Muestra observaciones y recompensas
- Útil como base para agentes más complejos

### Para crear un nuevo agente:

1. Crear un nuevo archivo en `agents/` (ej: `mi_agente.py`)
2. Heredar de la clase base o implementar interfaz
3. Agregar configuración en `config.yaml`
4. Importar en `run_agent.py`

---

## 📊 Observaciones y Acciones

### Espacio de Observaciones

Malmo proporciona:
- **Frame**: Imagen RGB del juego (240x320 píxeles por defecto)
- **Life**: Vida del jugador
- **XPos, YPos, ZPos**: Posición en el mundo
- **Yaw, Pitch**: Orientación de la cámara

### Espacio de Acciones

Acciones disponibles (dependen de la misión):
- `move` (1/-1/0): Mover adelante/atrás/quieto
- `turn` (1/-1/0): Girar derecha/izquierda/recto
- `attack` (0/1): Atacar
- `use` (0/1): Usar objeto
- `jump` (0/1): Saltar

---

## 🎯 Misiones

Las misiones se definen en archivos XML. Ver ejemplos en:
- `missions/simple_test.xml` - Misión minimalista
- `c:\Users\gonza\malmo\MalmoEnv\missions\` - Misiones de ejemplo

### Crear una Misión

```xml
<?xml version="1.0" encoding="utf-8"?>
<Mission xmlns="http://ProjectMalmo.microsoft.com">
  <About>
    <Summary>Mi misión</Summary>
  </About>
  <ModSettings>
    <MsPerTick>50</MsPerTick>
  </ModSettings>
  <ServerSection>
    <ServerInitialConditions>
      <Time>
        <StartTime>6000</StartTime>
      </Time>
    </ServerInitialConditions>
    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1"/>
      <ServerQuitFromTimeUp timeLimitMs="30000"/>
    </ServerHandlers>
  </ServerSection>
  <AgentSection mode="Survival">
    <Name>Agente</Name>
    <AgentHandlers>
      <ObservationFromFullStats/>
      <ContinuousMovementCommands turnSpeedDegs="180">
        <ModifierList type="allow">
          <command>move</command>
          <command>turn</command>
        </ModifierList>
      </ContinuousMovementCommands>
    </AgentHandlers>
  </AgentSection>
</Mission>
```

---

## 📈 Próximos Pasos

1. **Agente DQN**: Implementar Deep Q-Network
2. **Agente PPO**: Implementar Proximal Policy Optimization
3. **Misiones complejas**: Crear misiones con objetivos específicos
4. **Logging avanzado**: TensorBoard para visualización
5. **Hyperparameter tuning**: Optimización de parámetros

---

## 🔗 Recursos Útiles

- [Documentación oficial de Malmo](https://microsoft.github.io/malmo/)
- [Repositorio de Malmo](https://github.com/microsoft/malmo)
- [OpenAI Gym (malmoenv)](https://github.com/microsoft/malmo/tree/master/MalmoEnv)
- [Ejemplos de misiones](https://github.com/microsoft/malmo/tree/master/sample_missions)

---

## ❓ Solución de Problemas

### Minecraft no inicia
- Verificar que JAVA_HOME esté configurado correctamente
- Asegurar que el puerto 9000 no esté en uso

### El agente no se conecta
- Verificar que Minecraft esté corriendo con `-env` flag
- Comprobar que el puerto coincida (default: 9000)

### Error de schemas
- Verificar MALMO_XSD_PATH apunte a la carpeta correcta

---

*Creado para entrenamiento de agentes con Microsoft Malmo*