# Malmo Agent

Proyecto para correr un agente base en Microsoft Malmo con configuracion desacoplada y visualizacion en tiempo real.

## Estructura

```
agentedemalmo/
|-- agents/
|   `-- basic_agent.py
|-- config/
|   |-- config.yaml
|   |-- world_rules.yaml
|   `-- agent_params.yaml
|-- missions/
|   |-- simple_test.xml
|   `-- climb_challenge.xml
|-- utils/
|   |-- malmo_connector.py
|   `-- viewer_3d.py
`-- run_agent.py
```

## Configuracion

- `config/world_rules.yaml`: variables del entorno/mundo (mision, objetivo, terreno, recompensas).
- `config/agent_params.yaml`: variables del agente (politica, acciones, termination, run).
- `config/config.yaml`: runtime (rutas de YAML, host/port Malmo, logging).

## Ejecutar

1. Iniciar Minecraft Malmo:

```bash
cd c:\Users\gonza\malmo\Minecraft
launchClient.bat -port 9000 -env
```

2. Correr el agente:

```bash
cd c:\Users\gonza\agentedemalmo
python run_agent.py
```

## Viewer modes

- `--viewer none`: sin visualizacion.
- `--viewer terminal`: salida minimalista en terminal.
- `--viewer full`: panel unico con video de Malmo + metricas/agente/entorno + graficos de reward/altura + trayectoria.

Ejemplos:

```bash
python run_agent.py --viewer terminal
python run_agent.py --viewer full
python run_agent.py --policy explore --episodes 3 --max-steps 80
```
