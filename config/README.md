# Referencia completa de `agent_params.yaml`

Este documento describe todo lo que el agente soporta hoy en `config/agent_params.yaml`, sin depender de los valores actuales.

## Alcance y precedencia

El comportamiento final sale de esta precedencia (de menor a mayor prioridad):

1. Defaults en codigo (`agents/basic_agent.py`, `DEFAULT_AGENT_PARAMS`).
2. Valores del YAML (`config/agent_params.yaml`).
3. Overrides por CLI en `run_agent.py` (`--policy`, `--episodes`, `--max-steps`).

Si falta una clave en el YAML, se usa el default del codigo.

## Esquema global

```yaml
run:
  episodes: <int>
  max_steps: <int>

behavior:
  policy: <string>
  action_delay: <float>
  verbose: <bool>

actions:
  random:
    move_min: <float>
    move_max: <float>
    turn_min: <float>
    turn_max: <float>
    jump_probability: <float>
  forward:
    move: <float>
    turn: <float>
    jump: <float>
    jump_probability: <float opcional>
  explore:
    turn_interval: <int>
    turn_choices: <list[float]>
    move: <float>
    turn_jitter_min: <float>
    turn_jitter_max: <float>
    jump: <float>
    jump_probability: <float opcional>

termination:
  death_on_zero_life: <bool>
  min_life: <float>
```

## Bloque `run`

Controla cuanto dura la ejecucion global.

| Variable | Tipo / opciones | Que cambia al modificarla | Notas |
|---|---|---|---|
| `run.episodes` | `int` (esperado >= 1) | Sube o baja la cantidad de episodios totales. | Se puede overridear con `--episodes`. |
| `run.max_steps` | `int` (esperado >= 1) | Sube o baja el maximo de pasos por episodio. | El episodio termina por `done`, muerte o este limite, lo que ocurra primero. Tambien existe limite de tiempo de mision en XML. |

## Bloque `behavior`

Define la politica activa y la cadencia del loop.

| Variable | Tipo / opciones | Que cambia al modificarla | Notas |
|---|---|---|---|
| `behavior.policy` | `random`, `forward`, `explore` | Cambia la forma de elegir acciones. | Si pones otra cadena no soportada, el agente cae en fallback a `random`. Se puede overridear con `--policy`. |
| `behavior.action_delay` | `float` segundos (esperado >= 0) | Controla la pausa entre pasos (`time.sleep`). | Valores altos hacen ejecucion mas lenta. Valores 0 o chicos aceleran, pero pueden volver menos legible el debug visual. |
| `behavior.verbose` | `true` / `false` | Activa o desactiva prints de estado/recompensas. | Si usas `--viewer terminal` o `--viewer full`, el runner fuerza `verbose` efectivo a `false` para evitar ruido duplicado. |

## Bloque `actions`

Este bloque contiene parametros por politica. Solo se usa el sub-bloque de la politica elegida.

### `actions.random`

Politica de exploracion pura por muestreo aleatorio independiente en cada paso.

| Variable | Tipo / opciones | Que cambia al modificarla | Notas |
|---|---|---|---|
| `move_min`, `move_max` | `float` | Amplian o reducen rango de avance/retroceso aleatorio. | Se usa `random.uniform(min, max)`. |
| `turn_min`, `turn_max` | `float` | Amplian o reducen rango de giro aleatorio. | Se usa `random.uniform(min, max)`. |
| `jump_probability` | `float` recomendado en `[0,1]` | A mayor valor, mas pasos con salto (`jump=1`). | Es un Bernoulli por paso. |

Que significa `random` como estrategia:

- No usa observacion para decidir.
- Cada paso se decide por azar.
- Sirve para baseline/exploracion, no para control inteligente.

### `actions.forward`

Politica determinista: siempre devuelve la misma accion (salvo si usas `jump_probability` opcional).

| Variable | Tipo / opciones | Que cambia al modificarla | Notas |
|---|---|---|---|
| `move` | `float` | Velocidad de avance/retroceso constante. | En comandos continuos de Malmo, magnitud y signo importan. |
| `turn` | `float` | Giro constante. | `0` mantiene rumbo, positivos/negativos giran en sentidos opuestos. |
| `jump` | `float` usual `0` o `1` | Salto fijo por paso. | Solo se usa si `jump_probability` no esta definido. |
| `jump_probability` (opcional) | `float` recomendado en `[0,1]` | Convierte el salto en probabilistico. | Si existe, tiene prioridad sobre `jump`. |

Que significa `forward` como estrategia:

- Es una politica scriptada fija.
- Util para pruebas controladas o para verificar conectividad.

### `actions.explore`

Politica hibrida: avance con ruido + giros periodicos fuertes.

Mecanica exacta por paso:

1. Si `step_count % turn_interval == 0`: no avanza (`move=0`) y gira fuerte (`turn` tomado de `turn_choices`).
2. En el resto de pasos: avanza con `move` fijo y agrega jitter de giro uniforme entre `turn_jitter_min` y `turn_jitter_max`.
3. El salto se resuelve con `_resolve_jump`: si hay `jump_probability`, manda; si no, usa `jump`.

| Variable | Tipo / opciones | Que cambia al modificarla | Notas |
|---|---|---|---|
| `turn_interval` | `int` | Cada cuantos pasos ocurre giro fuerte. | El codigo aplica `max(1, int(valor))`. Si pones `0` o negativo, queda `1`. |
| `turn_choices` | `list[float]` | Define giros fuertes posibles. | Si lista vacia/falsy, usa fallback `[-1.0, 1.0]`. |
| `move` | `float` | Avance en pasos normales. | En pasos de giro fuerte no se usa (se fuerza `move=0`). |
| `turn_jitter_min`, `turn_jitter_max` | `float` | Ancho del ruido de giro continuo. | Se usa `random.uniform(min, max)` en pasos normales. |
| `jump` | `float` usual `0` o `1` | Salto fijo. | Solo aplica si no hay `jump_probability`. |
| `jump_probability` (opcional) | `float` recomendado en `[0,1]` | Salto probabilistico por paso. | Si existe, reemplaza a `jump`. |

Que significa `explore` como estrategia:

- No planifica con observacion.
- Recorre espacio con patron de avance y giros para cubrir terreno.
- Es util para exploracion simple sin entrenamiento.

## Como se interpreta `jump` y por que existe `jump_probability`

En este proyecto, el agente es mayormente scriptado. `jump_probability` no es una "decision inteligente", sino un mecanismo de exploracion estocastica.

Regla de prioridad en codigo:

1. Si existe `jump_probability`, se clampa a `[0,1]` y el salto se decide por azar.
2. Si no existe, se usa `jump` fijo.

Esto aplica en `forward` y `explore`. En `random`, siempre es probabilistico.

## Bloque `termination`

Define corte adicional por estado del agente (ademas de `done` del entorno y `max_steps`).

| Variable | Tipo / opciones | Que cambia al modificarla | Notas |
|---|---|---|---|
| `death_on_zero_life` | `true` / `false` | Activa o desactiva terminar por vida baja. | Si esta en `false`, el agente no corta por vida aunque caiga. |
| `min_life` | `float` | Umbral de vida para considerar muerte (`Life <= min_life`). | Solo aplica si `death_on_zero_life` esta activo y la observacion trae campo `Life`. |

## Relacion con Malmo (limites reales del "universo de posibilidades")

`agent_params.yaml` no controla todo Malmo. Controla la politica local del agente dentro de lo que la mision habilita.

Puntos importantes:

1. Los comandos disponibles dependen de la mision XML (`ContinuousMovementCommands`, etc.) y del `action_filter` del conector.
2. Este conector usa por defecto `move`, `turn`, `jump`, `use`, `attack` en `action_filter`, pero `BasicAgent` solo emite `move`, `turn`, `jump`.
3. Si el `action_space` es discreto, el conector transforma `{move,turn,jump}` a un comando Malmo cercano. Hay umbrales (`move/turn > 0.1`, `jump > 0.5`) y prioridad de seleccion.
4. Si quieres nuevas capacidades (inventario, uso de items, ataque, navegacion por observacion), debes extender `BasicAgent` y/o crear nuevas politicas.

## Checklist de diseno de politica

Usa este checklist para ubicarse antes de cambiar parametros:

1. Objetivo: explorar, llegar a un punto, evitar riesgos, testear conectividad.
2. Politica base: `random`, `forward`, `explore` o nueva politica en codigo.
3. Duracion: `episodes` y `max_steps`.
4. Cadencia: `action_delay` segun velocidad deseada.
5. Salto: fijo (`jump`) o estocastico (`jump_probability`).
6. Terminacion: por vida (`death_on_zero_life`, `min_life`) si la mision reporta `Life`.
7. Compatibilidad: verificar que la mision XML realmente habilite los comandos requeridos.
