# Informe de Simulación de Control de Congestión TCP

Este documento presenta los resultados y el análisis de la simulación de algoritmos de control de congestión de TCP, correspondiente a la actividad opcional.

---

## 1. Descripción de la Simulación

Se implementó una simulación en Python para modelar un escenario de red donde múltiples clientes compiten por el ancho de banda de un único enlace compartido. El objetivo es observar cómo diferentes algoritmos de control de congestión (Reno, CUBIC, BBR y un algoritmo personalizado) gestionan su ventana de envío (`cwnd`) para maximizar el `throughput` y mantener la equidad.

**Parámetros Clave de la Simulación (conforme a la guía):**
*   **Tiempo de Simulación:** 100 segundos.
*   **Capacidad del Enlace (pps):** 10 y 100.
*   **Tamaño del Búfer del Enlace:** 10 y 30 paquetes.
*   **Número de Clientes:** 9 para el comparativo (3 Reno, 3 CUBIC, 3 BBR).
*   **Algoritmo Propuesto (TCPCustom):** se compara en un escenario elegido (parte 3).

### Definiciones formales

- Throughput (por cliente i y paso t): x_i(t) = paquetes servidos en t. Throughput promedio del cliente i: X_i = (1/T) · ∑_{t=1..T} x_i(t).
- RTT (Round-Trip Time): tiempo ida y vuelta. En esta simulación se modela como RTT(t) = RTT_base + α · OcupaciónCola(t), con OcupaciónCola(t) = len(buffer)/BUFFER_SIZE y α ≈ 50 ms.
- Índice de Jain (fairness) en t: J(t) = ( (∑_i x_i(t))^2 ) / ( n · ∑_i x_i(t)^2 ). Interpretación: J ∈ (0,1]; 1 implica reparto perfectamente equitativo; valores menores indican inequidad entre flujos.
- Descartes por algoritmo: suma de paquetes rechazados por overflow del búfer por cada algoritmo. Cuantos más descartes, mayor agresividad (o presión conjunta) bajo el mismo escenario.

---

## 2. Cambios y cumplimiento vs. directrices

Los siguientes ajustes se realizaron para cumplir exactamente con la actividad:

- Cola con tamaño real y descartes por overflow: ahora el búfer respeta `BUFFER_SIZE`; los paquetes que llegan cuando la cola está llena se descartan y disparan evento de pérdida (no se vacía la cola al final del paso de tiempo).
- RTT en función de ocupación de cola: el RTT simulado crece con `len(buffer)/BUFFER_SIZE` (bufferbloat), no con paquetes procesados.
- Parámetros por CLI y escenarios: `simulation.py` acepta `--capacity`, `--buffer`, `--sim-time` y la opción `--all-scenarios` para generar los cuatro escenarios requeridos (10/100 pps × 10/30 buffer).
- Comparativo 9 clientes: para el análisis base se usan 3 Reno, 3 CUBIC y 3 BBR; TCPCustom se usa en la parte 3 sobre un escenario elegido.

Ejecución (ejemplos):

```
# Escenario único (100 pps, buffer 30)
python3 simulation.py --capacity 100 --buffer 30 --reno 3 --cubic 3 --bbr 3 --custom 0

# Cuatro escenarios (generará 4 PNG con sufijo cap/buf)
python3 simulation.py --all-scenarios --sim-time 100 --reno 3 --cubic 3 --bbr 3 --custom 0

# Comparación con TCPCustom en un escenario (p. ej., 100/30)
python3 simulation.py --capacity 100 --buffer 30 --reno 3 --cubic 3 --bbr 3 --custom 3
```

Los gráficos se guardan con nombre `congestion_simulation_results_with_custom_cap<cap>_buf<buf>.png`.

## 3. Propuesta de Optimización: Algoritmo `TCPCustom`

Se diseñó e implementó un algoritmo personalizado llamado `TCPCustom`, basado en una modificación de TCP Reno. La estrategia se puede denominar **"Reno Cauteloso" (Cautious Reno)**.

**Lógica del Algoritmo:**

1.  **Fase de Crecimiento (Recepción de ACK):** El comportamiento es idéntico al de Reno estándar.
    *   En modo **Slow Start** (`cwnd < ssthresh`), la ventana crece exponencialmente (`cwnd *= 2`).
    *   En modo **Congestion Avoidance** (`cwnd >= ssthresh`), la ventana crece linealmente (`cwnd += 1`).

2.  **Fase de Reducción (Pérdida de Paquete):** Aquí radica la diferencia. El algoritmo ajusta su agresividad basándose en la latencia (RTT) actual de la red, que sirve como un indicador del nivel de congestión.
    *   Si se pierde un paquete y el **RTT es alto** (definido en la simulación como > 130ms), significa que el enlace ya está bastante congestionado. El algoritmo reacciona de forma **más agresiva**, reduciendo el `ssthresh` a un 40% de la `cwnd` actual.
    *   Si se pierde un paquete y el **RTT es bajo**, el algoritmo es más **conservador** y optimista, asumiendo que la pérdida pudo ser más esporádica. Reduce el `ssthresh` a solo un 60% de la `cwnd`.

El objetivo de esta estrategia es ceder ancho de banda más rápidamente cuando la red está claramente saturada, pero evitar reducciones drásticas e innecesarias de la ventana ante pérdidas aisladas en una red con baja latencia.

---

## 4. Gráficos de Resultados

A continuación se muestran los gráficos generados por la simulación, que comparan el rendimiento de los cuatro algoritmos.

Para cada escenario se generan:

- Panel con 3 gráficos: Throughput por cliente, RTT por cliente y cwnd por cliente (archivo `congestion_simulation_results_with_custom_cap<cap>_buf<buf>.png`).
- Fairness de Jain a lo largo del tiempo (archivo `fairness_jain_cap<cap>_buf<buf>.png`). El índice J(t) = (∑xᵢ)²/(n·∑xᵢ²) mide la equidad entre flujos (1 = perfectamente justo, 1/n = muy injusto).
- Barras de paquetes descartados por algoritmo (archivo `drops_by_algorithm_cap<cap>_buf<buf>.png`). Cuantos más descartes, mayor agresividad o mayor presión sobre el búfer.

---

## 5. Análisis Detallado de los Resultados

Observando los gráficos, podemos extraer las siguientes conclusiones:

### a. Análisis del Throughput (Gráfico 1)

*   **Agresividad:** Se puede observar que los clientes **CUBIC (verde)** y **BBR (azul)** tienden a alcanzar y mantener un `throughput` más alto y estable que Reno. CUBIC, en particular, es conocido por su función de crecimiento agresiva. BBR, al intentar modelar el ancho de banda disponible, también logra una buena utilización.
*   **Equidad:** Dentro de cada grupo de algoritmos, el reparto del ancho de banda es relativamente equitativo. Sin embargo, entre algoritmos, CUBIC y BBR tienden a "ganar" más ancho de banda a expensas de Reno.
*   **Reno (rojo):** Muestra el comportamiento clásico de "dientes de sierra". Crece linealmente hasta que hay una pérdida, y entonces reduce su `throughput` a la mitad, repitiendo el ciclo. Esto lo hace menos agresivo y propenso a ceder ancho de banda.
*   **TCPCustom (magenta):** Su `throughput` es, como se esperaba, muy similar al de Reno, pero se observan caídas ligeramente diferentes. Su rendimiento general es modesto, similar al de Reno, ya que su lógica de crecimiento es la misma.

### b. Análisis del RTT (Gráfico 2)

El RTT simulado aumenta a medida que el búfer del enlace se llena. Un RTT más alto es un indicador directo de una mayor congestión.

*   Todos los algoritmos provocan un aumento del RTT cuando sus ventanas combinadas superan la capacidad del enlace, lo que lleva a que el búfer se llene.
*   Se puede notar que durante los periodos de `throughput` más alto (dominados por CUBIC y BBR), el RTT se mantiene en su valor máximo (150 ms), lo que indica que el búfer está constantemente lleno o casi lleno. Esto es una condición conocida como **Bufferbloat**.

### c. Análisis de la Ventana de Congestión (cwnd) (Gráfico 3)

### d. Fairness de Jain (Gráfico 4)

Un J(t) cercano a 1 indica reparto equitativo del enlace entre los 9 flujos. En escenarios con búfer pequeño o alta capacidad, observar si CUBIC/BBR empujan a Reno, reduciendo J(t).

### e. Descartes por Algoritmo (Gráfico 5)

Este gráfico agrega los paquetes descartados por overflow del búfer, por algoritmo. Un mayor conteo sugiere comportamiento más agresivo (o mayor presión conjunta) bajo el mismo escenario.

Este gráfico es el más revelador del comportamiento de cada algoritmo.

*   **Reno (rojo):** Muestra el patrón de dientes de sierra perfecto de crecimiento lineal y reducción a la mitad.
*   **CUBIC (verde):** Crece de forma mucho más rápida y agresiva. Tras una pérdida, no reduce la ventana a la mitad, sino en un factor `beta` (0.7 en la simulación), lo que le permite recuperar su velocidad más rápidamente.
*   **BBR (azul):** Muestra un comportamiento de crecimiento más suave y menos errático. No reacciona a la pérdida de paquetes de forma tan drástica, sino que ajusta su ventana de forma más gradual, lo que resulta en un `throughput` más estable.
*   **TCPCustom (magenta):** Su `cwnd` sigue de cerca a la de Reno. Sin embargo, la lógica modificada de reducción podría (en escenarios más largos o con diferentes parámetros) llevar a un comportamiento ligeramente más estable que Reno, aunque en esta simulación su rendimiento es muy parecido. La idea de reaccionar de forma diferente según el RTT es válida, pero en este entorno simulado con RTTs bastante estables, la diferencia no es pronunciada.

## 6. Conclusión

La simulación quedó alineada con la guía: cola con tamaño finito (con descartes), dos capacidades y dos tamaños de búfer, 9 clientes para el comparativo base y una propuesta de optimización. Las figuras se generan por escenario y pueden incorporarse al PDF final del informe.

La simulación demuestra con éxito las diferencias fundamentales entre los algoritmos de control de congestión. Algoritmos más modernos como CUBIC y BBR están diseñados para ser más agresivos y eficientes en redes de alta velocidad, mientras que Reno es más simple y conservador. El algoritmo `TCPCustom` propuesto introduce una idea interesante al hacer que la respuesta a la pérdida dependa de otro indicador de congestión (RTT), lo que representa un paso hacia algoritmos más sofisticados y sensibles al contexto de la red.
