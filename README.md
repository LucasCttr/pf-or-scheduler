# Optimizador de agendas quirúrgicas

Este proyecto genera una agenda semanal de quirófanos utilizando un
algoritmo genético y un decoder determinista. La salida principal es
`agenda_resultado.json`, un archivo que puede ser consumido por otro sistema
para mostrar o continuar procesando la agenda.

La idea central es separar el problema en dos niveles:

1. El algoritmo genético decide qué especialidad se asigna a cada bloque
	 semanal, donde un bloque es la combinación de un día y un quirófano.
2. El decoder toma esa distribución y selecciona, en forma determinista, los
	 pacientes que efectivamente pueden ser operados en cada bloque.

Esta separación reduce el espacio de búsqueda. El algoritmo genético explora
distribuciones de especialidades, mientras que el decoder se ocupa de las
restricciones concretas de pacientes, cirujanos, procedimientos y quirófanos.

## Problema que resuelve

El sistema intenta programar pacientes hasta rellenas el tiempo disponible de cada quirofano,
o bien, hasta que se agoten los pacientes seleccionables, se da prioridad a los casos 
con mayor prioridad clínica.

La semana está formada por los días definidos en `main.py` y por todos los
quirófanos cargados desde `data/rooms.csv`. Cada combinación día-quirofano
constituye un bloque independiente con una capacidad diaria en minutos.

El sistema trabaja con estas entidades:

- **Especialidad:** identifica el servicio médico y puede exigir una cantidad
	mínima de bloques semanales mediante `min_blocks`.
- **Quirófano:** tiene un nivel de complejidad (`room_type`) y una capacidad
	diaria en minutos.
- **Procedimiento:** pertenece a una especialidad, requiere un tipo mínimo de
	quirófano y tiene una duración estimada.
- **Cirujano:** pertenece a una especialidad, define sus días disponibles y
	sus horas contractuales semanales.
- **Paciente:** referencia una especialidad, un procedimiento y un cirujano,
	además de tener una prioridad clínica.

## Flujo general

```text
Archivos CSV
		|
		v
Carga de especialidades, quirófanos, procedimientos, cirujanos y pacientes
		|
		v
Algoritmo genético
		|
		|  cromosoma: {bloque (día, quirófano) -> especialidad}
		v
Decoder determinista
		|
		|  filtra candidatos, ordena por prioridad y llena cada bloque
		v
Agenda semanal + fitness
		|
		v
agenda_resultado.json
```

## Representación del cromosoma

Un cromosoma es un diccionario cuya clave es un `Block(day, room_id)` y cuyo
valor es el identificador de una especialidad.

Por ejemplo:

```text
(lunes, Q1)     -> GIN
(lunes, Q2)     -> CG
(martes, Q1)    -> TRA
```

El cromosoma no contiene pacientes ni horarios individuales. Solo representa
la distribución semanal de especialidades. Las cirugías y sus horarios son
construidos después por `decoder.py`.

## Algoritmo genético

### 1. Población inicial

Se generan cromosomas aleatorios asignando una especialidad a cada bloque.
Antes de evaluarlos se aplica una reparación que intenta garantizar que cada
especialidad tenga al menos la cantidad de bloques indicada por `min_blocks`.

### 2. Evaluación

Cada cromosoma se pasa al decoder. El decoder devuelve una agenda y esa agenda
se utiliza para calcular el fitness.

El fitness combina dos objetivos:

```text
fitness = alpha * prioridad_normalizada + beta * utilizacion
```

Donde:

- `prioridad_normalizada` es la suma de las prioridades clínicas al cuadrado
	de los pacientes programados, normalizada contra una cota máxima alcanzable.
	Elevar al cuadrado da más peso a los pacientes de prioridad alta.
- `utilizacion` es el tiempo utilizado dividido por el tiempo total disponible.
- `alpha` y `beta` controlan la importancia relativa de ambos objetivos.

La configuración utilizada por `main.py` es `alpha = 1.0` y `beta = 0.3`.

### 3. Selección

Se utiliza selección por torneo: se eligen varios individuos al azar y se
selecciona el de mayor fitness como progenitor.

### 4. Crossover

Con la probabilidad configurada en `crossover_rate`, se aplica crossover de un
punto. Cada hijo recibe una parte de los bloques de un progenitor y el resto
del otro.

### 5. Mutación

Cada bloque puede cambiar su especialidad con la probabilidad
`mutation_rate`. Luego de mutar, el cromosoma vuelve a pasar por la reparación
de mínimos.

### 6. Convergencia

El algoritmo conserva el mejor individuo encontrado. Se detiene cuando alcanza
el máximo de generaciones o cuando pasan `stagnation_limit` generaciones sin
una mejora significativa.

## Decoder y construcción de la agenda

`decoder.py` transforma cada cromosoma en una agenda concreta siguiendo estos
pasos:

1. Reinicia el estado temporal `scheduled` de todos los pacientes.
2. Agrupa los pacientes por especialidad.
3. Recorre los bloques del cromosoma.
4. Para cada bloque, filtra los pacientes que pueden ser candidatos.
5. Ordena los candidatos de mayor a menor prioridad clínica.
6. Agrega cirugías secuencialmente mientras haya capacidad suficiente.
7. Registra el intervalo de tiempo de cada cirugía.

Un paciente solo se agrega si se cumplen las condiciones necesarias:

- todavía no fue programado;
- existe el cirujano asociado;
- el cirujano está disponible ese día;
- la jornada adicional no supera sus horas contractuales semanales;
- el procedimiento existe;
- el quirófano soporta la complejidad requerida;
- la cirugía entra en los minutos restantes del bloque;
- el cirujano no queda operando simultáneamente en otro quirófano.

La última condición se controla comparando intervalos de tiempo relativos al
inicio de la jornada. Por eso un cirujano puede trabajar en dos quirófanos el
mismo día si los intervalos no se superponen, pero no puede hacerlo al mismo
tiempo.

El decoder es greedy: una vez que procesa un bloque y selecciona cirugías, no
retrocede para probar otra combinación. La exploración de alternativas ocurre
en el algoritmo genético, a través de distintos cromosomas.

## Datos de entrada

Los archivos CSV se encuentran en `data/` y deben conservar sus encabezados:

| Archivo | Contenido principal |
| --- | --- |
| `specialties.csv` | `id`, `name`, `min_blocks` |
| `rooms.csv`       | `id`, `name`, `room_type`, `daily_capacity_minutes` |
| `procedures.csv`  | `id`, `name`, `specialty_id`, `required_room_type`, `estimated_duration` |
| `surgeons.csv`    | `id`, `name`, `specialty_id`, `available_days`, `contract_hours_week` |
| `patients.csv`    | `id`, `specialty_id`, `procedure_id`, `surgeon_id`, `clinical_priority` |

En `surgeons.csv`, los días disponibles se separan con punto y coma. Los
identificadores usados entre archivos deben coincidir para que las referencias
se puedan resolver correctamente.

## Ejecución

Se recomienda utilizar el entorno virtual del proyecto:

```powershell
.venv\Scripts\Activate.ps1
python main.py
```

También puede ejecutarse directamente sin activar el entorno:

```powershell
.venv\Scripts\python main.py
```

El programa carga los CSV, ejecuta el algoritmo, imprime un resumen y genera o
reemplaza `agenda_resultado.json`.

Los parámetros principales se configuran en la creación de
`GeneticAlgorithm` dentro de `main.py`:

- `population_size`: cantidad de cromosomas por generación.
- `generations`: máximo de generaciones.
- `tournament_size`: cantidad de competidores por torneo.
- `crossover_rate`: probabilidad de crossover.
- `mutation_rate`: probabilidad de mutación por bloque.
- `stagnation_limit`: generaciones permitidas sin mejora.
- `alpha` y `beta`: pesos de prioridad y utilización.

## Formato de salida

El archivo `agenda_resultado.json` conserva cuatro grupos de información:

- `fitness`, `generaciones_ejecutadas` y `tiempo_ejecucion_segundos`: datos de
	la ejecución del algoritmo.
- `distribucion_semanal_especialidades`: especialidad asignada a cada
	quirófano por día.
- `agenda`: detalle de cada bloque, incluyendo capacidad, minutos usados y
	cirugías.
- `resumen`: cantidad total de pacientes, pacientes programados, pendientes y
	sus identificadores.
- `historial_fitness`: evolución del mejor fitness durante las generaciones.

Cada cirugía contiene:

```json
{
	"paciente_id": "P0177",
	"especialidad": "GIN",
	"duracion_min": 30,
	"hora_inicio_min": 0,
	"hora_fin_min": 30,
	"prioridad_clinica": 8.0
}
```

Las horas son minutos relativos al inicio de la jornada del quirófano. La
estructura del JSON está pensada para que otro sistema pueda convertir esos
minutos a horas visibles o realizar su propio procesamiento.

## Validación de factibilidad

La carpeta `validation/feasibility/` contiene validaciones independientes que
analizan la agenda generada sin modificarla. Comprueban, entre otros puntos:

- referencias a pacientes, procedimientos y quirófanos;
- duración de las cirugías;
- compatibilidad entre procedimiento y quirófano;
- capacidad de cada bloque;
- disponibilidad y horas contractuales de los cirujanos;
- solapamientos de un mismo cirujano;
- mínimos de bloques por especialidad;
- unicidad de pacientes;
- prioridades, utilización y balance de carga.

El script `runa_validation.py` ejecuta distintos escenarios y exporta los
resultados comparativos en CSV.

## Búsqueda de parámetros y comparación

La carpeta `tuning/` contiene herramientas auxiliares:

- `parameter_tuning.py` ejecuta un grid search de configuraciones del
	algoritmo genético y guarda los resultados en `grid_search_results.csv`.
- `plot_grid_search.py` genera gráficos a partir de ese CSV.

La carpeta `validation/performance/` incluye `brute_force.py`, que permite
comparar el algoritmo genético con una búsqueda exhaustiva en instancias
pequeñas. Ambos métodos reutilizan el mismo decoder y la misma evaluación para
que la comparación sea consistente.

## Estructura del proyecto

```text
.
├── data/                       # CSV de entrada
├── validation/
│   ├── feasibility/            # Validación de restricciones y escenarios
│   └── performance/            # Comparación con fuerza bruta
├── tuning/                     # Ajuste y visualización de parámetros
├── data_loader.py              # Lectura y conversión de CSV a modelos
├── models.py                   # Entidades del dominio
├── decoder.py                  # Cromosoma -> agenda factible
├── genetic_algorithm.py        # Evolución y evaluación de cromosomas
├── main.py                     # Punto de entrada y exportación JSON
└── agenda_resultado.json       # Resultado generado
```

## Consideraciones de diseño

- La solución es heurística: no garantiza encontrar la agenda óptima global.
- El resultado puede cambiar entre ejecuciones porque la población, el
	crossover y la mutación usan aleatoriedad.
- La calidad depende tanto de los parámetros del algoritmo como del orden de
	procesamiento y de las restricciones implementadas en el decoder.
- Si se necesita reproducibilidad exacta, debe fijarse la semilla del generador
	aleatorio antes de ejecutar el algoritmo.
- El algoritmo optimiza principalmente la distribución de especialidades; el
	decoder determina qué pacientes entran efectivamente en cada bloque.


## Contrato HTTP

- `POST /planning`: recibe el payload autocontenido del Back y devuelve `202` con `uuid` y estado `planning`.
- `GET /planning/{uuid}`: consulta `planning`, `completed` o `failed`.
- Al terminar se envía el resultado a `BACK_CALLBACK_URL` usando `X-Scheduler-Token`.

La salida conserva el contrato `dias[].bloques[].cronograma[]` utilizado por SurgiCare. Una cirugía sólo se agenda si su cirujano asignado, procedimiento y quirófano son compatibles y si cabe dentro de la intersección entre su disponibilidad y la jornada 08:00–13:00.

## Ejecución

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn api:app --host 127.0.0.1 --port 3020 --reload
```

Variables opcionales:

- `BACK_CALLBACK_URL`
- `SCHEDULER_CALLBACK_TOKEN` (por defecto `dev-scheduler-token`)

## Pruebas

```bash
.venv/bin/pytest
```

El modo CSV sigue disponible con `.venv/bin/python main.py`; usa los archivos de `data/` y genera `agenda_resultado.json`.
