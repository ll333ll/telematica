# Telemática — Repositorio de Actividades y Extra de Machine Learning

Este repositorio contiene:

- Entrega 1 — Telemetría: `Proyectos/Entrega1_Telemetria/` (ver `Entrega1_Explicacion.md` y `telemetria_robot/`).
- Actividad Opcional — Sockets + HTTP + DNS/Docker: `Proyectos/ActividadOpcional_Web_DNS/`.
- Extra de Machine Learning (IDS): `CiberTelepatia/`.

## Actualizar subrepositorios Git

Los proyectos `http/` y `sockets/` dentro de `Proyectos/ActividadOpcional_Web_DNS/` son repos Git independientes con remoto configurado. Para actualizar:

```
git -C Proyectos/ActividadOpcional_Web_DNS/http pull --ff-only
git -C Proyectos/ActividadOpcional_Web_DNS/sockets pull --ff-only
```

## Actividad opcional — Ejecución base

La guía de ejecución paso a paso y justificación técnica está en `Actividades.md`. Resumen:

- `sockets/`: ejemplos de cliente/servidor, simulación DNS para `PlugAndAd.com`.
- `http/`: backend Flask + frontend estático (Docker opcional con Nginx + Gunicorn).

Pasos típicos (consulta `Actividades.md` para detalles y contexto):

- Backend: `cd Proyectos/ActividadOpcional_Web_DNS/http/backend && python3 back.py`
- Frontend (estático): `cd Proyectos/ActividadOpcional_Web_DNS/http/frontend && python3 -m http.server 8000`
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
