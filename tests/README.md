# Tests

Esta carpeta contiene tests automatizados para validar el comportamiento del algoritmo genético.

## Archivos

- `builders.py`: arma escenarios controlados y determinísticos para las pruebas.
- `conftest.py`: define fixtures compartidas, como la seed fija.
- `test_end_to_end.py`: valida un caso de punta a punta y la reproducibilidad con la misma seed.
- `test_repair.py`: cubre el comportamiento de `repair()` sobre compatibilidad, disponibilidad y cuotas mínimas.

## Objetivo del escenario end-to-end

El escenario de prueba fuerza una solución única y fácil de inspeccionar:

- 2 días
- 1 turno por día
- 2 quirófanos
- 2 especialidades
- cuotas mínimas y máximas exactamente iguales

Con eso se puede verificar no solo la asignación de especialidades por bloque, sino también los pacientes elegidos dentro de cada bloque y la reproducibilidad del resultado.
