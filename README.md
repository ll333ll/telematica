# Telemática — Repositorio de Actividades y Extra de Machine Learning

Este repositorio contiene:

- Proyectos de prácticas de red: `sockets/` y `http/` (ver `Actividades.md`).
- Actividades de la entrega 1: `telemetria_robot/` (ver `Entrega1_Explicacion.md`).
- Extra de Machine Learning para Detección de Intrusiones: `CiberTelepatia/`.

## Actualizar subrepositorios Git

Los proyectos `http/` y `sockets/` son repositorios Git independientes con remoto configurado. Ya quedaron sincronizados con `origin/main`. Para actualizar en el futuro:

```
git -C http pull --ff-only
git -C sockets pull --ff-only
```

## Actividad opcional — Ejecución base

La guía de ejecución paso a paso y justificación técnica está en `Actividades.md`. Resumen:

- `sockets/`: ejemplos de cliente/servidor, simulación DNS para `PlugAndAd.com`.
- `http/`: backend Flask + frontend estático.

Pasos típicos (consulta `Actividades.md` para detalles y contexto):

- Backend: `cd http/backend && source venv/bin/activate && python3 back.py`
- Frontend (estático): `cd http/frontend && python3 -m http.server 8000`
- Acceso: `http://localhost:8000`

## Extra (Machine Learning): CiberTelepatia

Implementación completa de un IDS con:

- EDA visual (Plotly), 10+ modelos, 3 ensambles.
- Notebook y script CLI.

Instalación y ejecución:

```
pip install -r CiberTelepatia/requirements.txt
jupyter notebook # abrir CiberTelepatia/notebooks/IDS_Analysis_and_Models.ipynb
# o por CLI
python CiberTelepatia/run_ids_ml.py
```

Documentación pedagógica: `CiberTelepatia/Analisis_y_Modelado_ML.txt`.

## Notas

- Si cambias la versión de Python, reinstala dependencias (`requirements.txt`).
- Para problemas específicos de red/puertos/DNS, sigue las secciones dedicadas en `Actividades.md` y `Entrega1_Explicacion.md`.
