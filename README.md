# 🏥 Sistema Inteligente de Asignación de Agenda Quirúrgica

> Optimización de bloques operatorios semanales mediante un enfoque híbrido **Algoritmo Genético + Programación Entera Mixta (MIP)**, con reconstrucción horaria por heurística de trenes.

---

## Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Estructura de Archivos](#estructura-de-archivos)
4. [Modelos de Datos](#modelos-de-datos)
5. [Flujo de Ejecución](#flujo-de-ejecución)
6. [Nivel 1 — Algoritmo Genético](#nivel-1--algoritmo-genético)
7. [Nivel 2 — Optimización MIP por Turno](#nivel-2--optimización-mip-por-turno)
8. [Nivel 3 — Heurística de Trenes](#nivel-3--heurística-de-trenes)
9. [Función de Fitness](#función-de-fitness)
10. [Restricciones del Modelo](#restricciones-del-modelo)
11. [Configuración](#configuración)
12. [Salida del Sistema](#salida-del-sistema)
13. [Cómo Ejecutar](#cómo-ejecutar)
14. [Dependencias](#dependencias)

---

## Descripción General

Este sistema resuelve el problema de **planificación de agenda quirúrgica semanal** en un hospital, un problema NP-difícil que requiere balancear simultáneamente:

- La **prioridad clínica** de los pacientes en lista de espera.
- La **disponibilidad horaria** de cirujanos por día y turno.
- La **compatibilidad** entre especialidades y tipos de quirófano.
- Las **cuotas mínimas y máximas** de bloques garantizadas por especialidad.
- Las **restricciones de médico forzado** (pacientes que solo pueden ser operados por un cirujano específico).

El enfoque adoptado es **híbrido en tres niveles**:

```
Nivel 1: Algoritmo Genético  →  decide QUÉ especialidad va en cada bloque (día/turno/quirófano)
Nivel 2: MIP por turno       →  decide QUÉ pacientes y QUIÉN los opera dentro de cada bloque
Nivel 3: Heurística trenes   →  construye el cronograma horario minuto a minuto
```

---

## Arquitectura del Sistema

```
┌──────────────────────────────────────────────────────────────┐
│                        main.py                               │
│  Orquesta datos, ejecuta el AG y reconstruye el cronograma   │
└─────────────────────────┬────────────────────────────────────┘
                          │
          ┌───────────────▼──────────────┐
          │     genetic_algorithm.py     │ 
          │  Algoritmo Genético (Nivel 1)│
          │  Cromosoma: [días × turnos × │
          │  quirófanos] = especialidad  │
          └───────────────┬──────────────┘
                          │ evalúa fitness llamando al MIP
          ┌───────────────▼──────────────┐
          │           mip.py             │
          │   Optimización MIP (Nivel 2) │
          │  Asigna pacientes y cirujanos│
          │  por turno completo (PuLP)   │
          └───────────────┬──────────────┘
                          │ resultado cacheado
          ┌───────────────▼──────────────┐
          │    Heurística de Trenes      │
          │  (dentro de main.py)         │
          │  Construye cronograma horario│
          │  estricto por quirófano      │
          └──────────────────────────────┘
```

---

## Estructura de Archivos

```
.
├── main.py                # Punto de entrada; orquesta todo el flujo
├── genetic_algorithm.py   # Clase GeneticAlgorithm e Individual
├── mip.py                 # Solver MIP por turno (PuLP / CBC)
├── models.py              # Dataclasses del dominio
└── agenda_resultado.json  # Salida generada tras la ejecución
```

---

## Modelos de Datos

Definidos en `models.py`:

### `OperatingRoom`
Representa un quirófano físico.

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | int | Identificador único |
| `name` | str | Nombre descriptivo |
| `or_type` | str | `alta_complejidad`, `media_complejidad` o `baja_complejidad` |
| `availability` | `List[List[bool]]` | Disponibilidad `[día][turno]` |

### `Specialty`
Representa una especialidad médica.

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | int | 0 = bloque libre |
| `compatible_or_types` | `List[str]` | Tipos de quirófano que puede usar |
| `min_blocks` | int | Mínimo de bloques semanales garantizados |
| `max_blocks` | int | Máximo de bloques semanales permitidos |

### `Patient`
Representa un paciente en lista de espera.

| Campo | Tipo | Descripción |
|---|---|---|
| `estimated_duration` | int | Duración estimada de la cirugía en minutos |
| `clinical_priority` | float | Prioridad clínica (1.0 – 10.0; casos críticos pueden llegar a 99) |
| `forced_surgeon_id` | `Optional[int]` | Si está presente, solo ese cirujano puede operarlo |

### `Staff`
Representa un profesional del equipo quirúrgico.

| Campo | Tipo | Descripción |
|---|---|---|
| `role` | str | `"cirujano"`, `"anestesista"`, etc. |
| `specialties_ids` | `List[int]` | Especialidades que puede atender |
| `availability_hours` | `Dict[int, Tuple[int,int]]` | Disponibilidad en minutos por día: `{día: (inicio, fin)}` |

El método `get_range_for_block(day, is_morning)` calcula el **solapamiento** entre la disponibilidad del médico y el turno del quirófano (mañana: 480–720 min / tarde: 780–1020 min).

### `GAConfig`
Agrupa todos los parámetros configurables del sistema (ver sección [Configuración](#configuración)).

---

## Flujo de Ejecución

```
main()
  │
  ├─ Construir datos: staff, quirófanos, especialidades, pacientes
  │
  ├─ GeneticAlgorithm.run()
  │     ├─ initialize_population()      # 50 individuos aleatorios válidos
  │     ├─ evaluate_fitness() × pop     # Llama al MIP para cada turno
  │     └─ Bucle generacional (máx 200):
  │           ├─ Selección por torneo
  │           ├─ Cruce (día o quirófano)
  │           ├─ Mutación (reasignación o intercambio)
  │           ├─ Reparación (corrige incompatibilidades)
  │           └─ Parada por convergencia (15 gen sin mejora)
  │
  ├─ ga.print_schedule(best)           # Imprime agenda semanal en consola
  │
  └─ reconstruct_agenda()
        ├─ get_schedule_details(best)   # Reutiliza el MIP cacheado
        ├─ Heurística de trenes         # Asigna horarios minuto a minuto
        └─ Guardar agenda_resultado.json
```

---

## Nivel 1 — Algoritmo Genético

### Representación del Cromosoma

El cromosoma es un array NumPy tridimensional de enteros:

```
cromosoma[día][turno][quirófano] = id_especialidad
```

Con 5 días, 2 turnos y 3 quirófanos, el cromosoma tiene **30 genes**. Cada gen indica qué especialidad médica ocupa ese bloque (`0` = libre).

### Inicialización

Cada individuo se genera aleatoriamente respetando desde el inicio:
- Que el quirófano esté disponible ese día/turno.
- Que el tipo de quirófano sea compatible con la especialidad.
- Que exista al menos un cirujano disponible para esa especialidad.
- Con un 15 % de probabilidad, se asigna bloque libre (`0`).

### Operadores Genéticos

**Selección:** torneo de tamaño configurable (`tournament_size = 5`). Se extraen candidatos al azar y gana el de mayor fitness.

**Cruce (crossover):** con probabilidad `crossover_rate = 0.85`, se aplica uno de dos tipos elegido al azar:
- **Cruce por días:** corte en un índice de día; cada hijo toma mitades de distintos padres.
- **Cruce por quirófanos:** corte en un índice de quirófano; intercambia las "columnas" del tensor.

**Mutación:** con probabilidad `mutation_rate = 0.10` por gen, se aplica uno de dos operadores:
- **Reasignación aleatoria:** reemplaza el gen por una especialidad válida aleatoria.
- **Intercambio:** swapea el gen con otro posición aleatoria del cromosoma.

**Reparación:** después de cruzar y mutar, se verifica cada gen y se corrigen:
- Asignaciones incompatibles (tipo de quirófano ≠ especialidad).
- Quirófanos deshabilitados ese día/turno.
- Déficits de cuota mínima (se fuerza la asignación de los bloques faltantes).

### Elitismo y Convergencia

Los `elite_count = 2` mejores individuos pasan intactos a la siguiente generación. El algoritmo se detiene si no hay mejora durante `convergence_patience = 15` generaciones consecutivas.

### Cache MIP

El AG puede evaluar miles de combinaciones de bloques idénticas en distintos individuos. Para evitar resolver el mismo problema MIP repetidamente, se mantiene un **diccionario de caché** cuya clave codifica exactamente los pacientes, cirujanos y especialidades de cada turno. En ejecuciones típicas se ahorra entre 60 % y 80 % de las llamadas al solver.

---

## Nivel 2 — Optimización MIP por Turno

El módulo `mip.py` resuelve, para cada combinación `(día, turno)`, un **Problema de Programación Entera Mixta** que decide:

- Qué pacientes entran a quirófano.
- Qué cirujano opera a cada paciente.
- En qué quirófano se realiza la operación.

### Variables de Decisión

```
x[p, s, q] ∈ {0, 1}   → 1 si el paciente p es operado por el cirujano s en el quirófano q
c[s, q]    ∈ {0, 1}   → 1 si el cirujano s opera al menos una vez en el quirófano q
                          (usada para el bonus de concentración)
```

### Función Objetivo

```
maximizar:  α · Σ(prioridad_clínica · x)
          + β · Σ(duración_estimada · x) / tiempo_total_disponible
          - γ · Σ(c[s, q])
```

Donde `α = 0.7`, `β = 0.3` y `γ = 0.05`. El término negativo **penaliza la dispersión** de un cirujano en múltiples quirófanos; se premia que concentre sus cirugías en un solo OR.

### Restricciones

| # | Descripción |
|---|---|
| R1 | Cada paciente se opera **como máximo una vez** en el turno (en cualquier quirófano). |
| R2 | Cada cirujano no supera su **capacidad total de minutos** en el turno (suma sobre todos sus quirófanos). |
| R3 | Cada quirófano no supera su **capacidad física** (`block_duration_min = 240 min`). |
| R4 | **Modelo híbrido**: si un paciente tiene `forced_surgeon_id`, solo ese médico puede operarlo. |
| R5 | **Capacidad efectiva por (cirujano, quirófano)**: dado que los cirujanos operan en serie dentro de un quirófano, se calcula cuánto tiempo real le queda al quirófano cuando le toca a cada médico (según hora de salida). Evita asignar más trabajo del que cabe físicamente. |
| R6/R7 | **Vinculación c ↔ x**: fuerzan coherencia entre la variable binaria de concentración y las asignaciones reales, impidiendo que el solver "trampe" el objetivo. |

El solver utilizado es **CBC** a través de la librería [PuLP](https://coin-or.github.io/pulp/).

---

## Nivel 3 — Heurística de Trenes

Una vez obtenido el mejor individuo, `reconstruct_agenda()` construye el cronograma horario detallado con la metáfora del **tren de cirugías**:

1. Para cada día/turno/quirófano, se recuperan las asignaciones del MIP cacheado.
2. Se ordenan los pacientes de cada quirófano priorizando: **primero los cirujanos que antes salen** (su hora límite de salida), y entre ellos, **mayor prioridad clínica primero**.
3. Se avanza un reloj por médico (`libre_staff`) y otro por quirófano (`libre_q`). La hora de inicio real de cada cirugía es el máximo de ambos relojes.
4. Si la hora de fin calculada excede el límite de salida del médico, la cirugía se **descarta sin tolerancia** y se registra el conflicto en consola.
5. El resultado final incluye `hora_inicio`, `hora_fin` y `duración` de cada intervención.

---

## Función de Fitness

El fitness global de un individuo combina el resultado del MIP con penalizaciones por cuotas:

```
fitness = Σ_turnos( z_MIP(turno) )
        - Σ_especialidades( penalización_cuota(especialidad) )
```

Donde la penalización por cuota es:

```
si bloques_asignados < min_blocks:
    penalización += penalty_below_min_quota × (min_blocks - bloques_asignados)

si bloques_asignados > max_blocks:
    penalización += penalty_above_max_quota × (bloques_asignados - max_blocks)
```

Con `penalty_below_min_quota = 50` y `penalty_above_max_quota = 20`.

---

## Restricciones del Modelo

### Resumen de restricciones duras (siempre se cumplen)

- Un quirófano solo recibe especialidades compatibles con su tipo.
- Un cirujano solo opera especialidades de su lista (`specialties_ids`).
- Un paciente con médico forzado no puede ser asignado a otro cirujano.
- Un quirófano no puede superar sus minutos físicos disponibles por turno.
- Un cirujano no puede operar más minutos de los que tiene disponibles.

### Restricciones blandas (optimizadas por el fitness)

- Respetar las cuotas mínimas y máximas de bloques por especialidad.
- Maximizar la prioridad clínica de los pacientes operados.
- Maximizar el porcentaje de utilización de los quirófanos.
- Concentrar las cirugías de cada médico en un único quirófano por turno.

---

## Configuración

Todos los parámetros se centralizan en la clase `GAConfig`:

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `population_size` | 50 | Tamaño de la población del AG |
| `max_generations` | 200 | Máximo de generaciones |
| `convergence_patience` | 15 | Generaciones sin mejora para detener |
| `mutation_rate` | 0.10 | Probabilidad de mutar cada gen |
| `crossover_rate` | 0.85 | Probabilidad de realizar cruce |
| `tournament_size` | 5 | Competidores en selección por torneo |
| `elite_count` | 2 | Individuos élite que pasan sin cambios |
| `n_days` | 5 | Días de la semana (Lun–Vie) |
| `n_shifts` | 2 | Turnos por día (Mañana / Tarde) |
| `block_duration_min` | 240 | Minutos por bloque quirúrgico |
| `alpha` | 0.7 | Peso de prioridad clínica en el MIP |
| `beta` | 0.3 | Peso de utilización de tiempo en el MIP |
| `penalty_below_min_quota` | 50.0 | Penalización por bloque faltante bajo mínimo |
| `penalty_above_max_quota` | 20.0 | Penalización por bloque extra sobre máximo |

---

## Salida del Sistema

El archivo `agenda_resultado.json` contiene la agenda completa con la siguiente estructura:

```json
{
  "hospital": "Hospital Centenario",
  "fitness_total": 142.3821,
  "duracion_segundos": 38.7,
  "dias": [
    {
      "nombre": "Lunes",
      "bloques": [
        {
          "quirofano": "Quirófano 1 (Alta)",
          "turno": "Mañana",
          "especialidad": "Traumatología",
          "utilizacion_porcentaje": 87.5,
          "cronograma": [
            {
              "paciente_id": 2000,
              "medico": "Dr. Pérez",
              "hora_inicio": "08:00",
              "hora_fin": "11:20",
              "duracion": 200
            }
          ]
        }
      ]
    }
  ]
}
```

---

## Cómo Ejecutar

```bash
# 1. Instalar dependencias
pip install numpy pulp

# 2. Ejecutar el sistema
python main.py
```

La ejecución imprime en consola:
- El progreso del AG generación a generación.
- El porcentaje de ahorro por caché MIP.
- La agenda semanal por quirófano, turno y especialidad.
- Los conflictos detectados por la heurística de trenes (si los hay).
- El total de pacientes asignados y el tiempo de ejecución.

---

## Dependencias

| Librería | Uso |
|---|---|
| `numpy` | Representación y manipulación del cromosoma |
| `pulp` | Modelado y resolución del MIP (solver CBC incluido) |
| `random` | Generación de población inicial y operadores genéticos |
| `json` | Serialización de la agenda resultado |

---

> **Nota sobre escalabilidad:** el cuello de botella del sistema es el solver MIP. El caché interno alivia significativamente este costo en poblaciones grandes. Para instancias con más quirófanos o más días, se recomienda aumentar `convergence_patience` y reducir `population_size` para balancear calidad vs. tiempo de cómputo.