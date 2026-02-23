# Referencia completa de `agent_params.yaml` (modo Q-learning)

Este documento describe el esquema actual del agente:

1. Sin defaults silenciosos para `agent_params`.
2. Politica unica: `explore`.
3. Estado discreto para tabla Q.
4. Acciones discretas explicitas.
5. Aprendizaje tabular con `alpha`, `gamma`, `epsilon`.

## Regla principal: modo estricto

El runner carga `agent_params.yaml` en modo estricto:

1. Si el archivo no existe, falla.
2. Si falta una clave requerida, falla.
3. Si un tipo/rango no coincide, falla.

No hay fallback automatico en `agent_params`.
Tampoco hay fallback automatico en el mapeo de acciones del conector: si una accion no matchea, se lanza error.

## Esquema global

```yaml
run:
  episodes: <int >= 1>
  max_steps: <int >= 1>

behavior:
  policy: "explore"
  action_delay: <float >= 0>
  verbose: <bool>

state:
  position_bin_size: <float > 0>
  y_bin_size: <float > 0>
  yaw_bins: <int >= 1>
  floor_grid_key: <string>
  include_floor_grid: <bool>
  include_line_of_sight: <bool>

actions:
  discrete: <list[string] exacto de 5 comandos>

learning:
  alpha: <float [0,1]>
  gamma: <float [0,1]>
  epsilon: <float [0,1]>
  q_table_path: <string path>
  autosave_each_episode: <bool>

reward:
  mode: <"malmo" | "height_gain_only">
  height_gain_scale: <float >= 0>
  min_height_delta: <float >= 0>

termination:
  death_on_zero_life: <bool>
  min_life: <float>
  stop_on_target_height: <bool>
  no_height_gain_patience: <int >= 0>
  height_gain_epsilon: <float >= 0>
```

## Bloque `run`

| Variable | Tipo | Efecto |
|---|---|---|
| `run.episodes` | `int >= 1` | Cantidad de episodios totales. |
| `run.max_steps` | `int >= 1` | Maximo de pasos por episodio. |

## Bloque `behavior`

| Variable | Tipo | Efecto |
|---|---|---|
| `behavior.policy` | Solo `"explore"` | Politica obligatoria en esta version. |
| `behavior.action_delay` | `float >= 0` | Pausa entre pasos (`time.sleep`). |
| `behavior.verbose` | `bool` | Logs de estado/reward en consola. |

## Bloque `state`

Define como se construye el estado discreto para la Q-table.

Estado usado por el agente:

1. `x_bin`: `XPos` cuantizado por `position_bin_size`.
2. `y_bin`: `YPos` cuantizado por `y_bin_size` (progreso de escalada).
3. `z_bin`: `ZPos` cuantizado por `position_bin_size`.
4. `yaw_bin`: `Yaw` cuantizado en `yaw_bins`.
5. `center_code`: tipo del bloque central del `floor5x5` (aire/solido/lava/desconocido).
6. `lava_bucket`: bucket de cantidad de celdas de lava en el grid (`0`, `1-2`, `3+`).
7. `air_bucket`: bucket de cantidad de celdas de aire en el grid (`0`, `1-2`, `3+`).
8. `los_code`: tipo de bloque visto por `LineOfSight` (si `include_line_of_sight=true`).

| Variable | Tipo | Efecto |
|---|---|---|
| `state.position_bin_size` | `float > 0` | Granularidad espacial en X/Z. Menor valor = mas estados. |
| `state.y_bin_size` | `float > 0` | Granularidad vertical. Menor valor = mas estados por altura. |
| `state.yaw_bins` | `int >= 1` | Resolucion angular. Mas bins = mas estados. |
| `state.floor_grid_key` | `string` | Clave de observacion del grid (ejemplo: `floor5x5`). |
| `state.include_floor_grid` | `bool` | Activa/desactiva el uso de buckets del grid en el estado. |
| `state.include_line_of_sight` | `bool` | Activa/desactiva `LineOfSight` en el estado. |

## Bloque `actions`

Define el espacio discreto de acciones. En esta version se exige exactamente 5 comandos.

Ejemplo recomendado:

1. `move 1`
2. `turn -1`
3. `turn 1`
4. `jump 1`
5. `jump 0`

| Variable | Tipo | Efecto |
|---|---|---|
| `actions.discrete` | `list[string]` de longitud `5` | Define `len(ACTIONS)` y el mapeo `action_id -> comando`. |

## Bloque `learning`

Configura el aprendizaje Q-learning.

Actualizacion por paso:

```text
Q[s,a] += alpha * (reward + gamma * max(Q[s_next]) - Q[s,a])
```

Seleccion de accion (`epsilon`-greedy):

1. Con probabilidad `epsilon`: explora (accion aleatoria).
2. Si no: explota (`argmax Q[s]`).

| Variable | Tipo | Efecto |
|---|---|---|
| `learning.alpha` | `float [0,1]` | Tasa de aprendizaje. |
| `learning.gamma` | `float [0,1]` | Peso del valor futuro. |
| `learning.epsilon` | `float [0,1]` | Probabilidad de exploracion. |
| `learning.q_table_path` | `string path` | Archivo donde se carga/guarda la tabla Q. |
| `learning.autosave_each_episode` | `bool` | Si `true`, guarda al final de cada episodio. |

## Bloque `reward`

Define la fuente de recompensa usada por Q-learning en `run_agent.py`.

| Variable | Tipo | Efecto |
|---|---|---|
| `reward.mode` | `"malmo"` o `"height_gain_only"` | Selecciona entre recompensa nativa de mision o recompensa por subida de altura. |
| `reward.height_gain_scale` | `float >= 0` | Multiplicador para ganancia de `Y` sobre el mejor `Y` previo del episodio. |
| `reward.min_height_delta` | `float >= 0` | Umbral minimo de ganancia vertical para considerar recompensa (evita ruido). |

## Bloque `termination`

| Variable | Tipo | Efecto |
|---|---|---|
| `termination.death_on_zero_life` | `bool` | Si `true`, termina cuando `Life <= min_life`. |
| `termination.min_life` | `float` | Umbral de vida para terminar. |
| `termination.stop_on_target_height` | `bool` | Si `true`, termina cuando `max_height >= world_rules.objective.target_height`. |
| `termination.no_height_gain_patience` | `int >= 0` | Corta episodio tras N steps sin mejorar altura. `0` desactiva este criterio. |
| `termination.height_gain_epsilon` | `float >= 0` | Delta minimo para considerar que realmente hubo mejora de altura. |

## Relacion con Malmo

`agent_params.yaml` define politica/aprendizaje, pero depende de la mision:

1. La mision debe emitir observaciones necesarias (`XPos`, `ZPos`, `Yaw`, `floor5x5`, `Life`).
2. La mision debe habilitar comandos coherentes con `actions.discrete`.
3. El conector filtra acciones por raiz de comando (`move`, `turn`, `jump`, etc.).

## Checklist rapido para iterar

1. Ajustar `state.position_bin_size` y `state.yaw_bins` para controlar tamano de estado.
2. Mantener acciones discretas pequeñas al inicio (5 esta bien).
3. Verificar `epsilon` alto para explorar al principio.
4. Revisar `q_table_path` para no reutilizar una tabla incompatible de un estado viejo.
5. Revisar `q_states` en el resumen final para detectar explosion de estados.
