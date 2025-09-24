# Informe de la Actividad Opcional: Sistema de Detección de Intrusiones (IDS) con Machine Learning

Este informe detalla la implementación de un Sistema de Detección de Intrusiones (IDS) utilizando técnicas de Machine Learning, como parte de la actividad opcional de Telemática.

---

## 1. Introducción y Objetivo

En el ámbito de la ciberseguridad, la detección de intrusiones es crucial para proteger las redes de ataques maliciosos. Tradicionalmente, los IDS se basan en reglas predefinidas, pero los enfoques basados en Machine Learning ofrecen la capacidad de detectar amenazas nuevas y complejas al aprender patrones directamente de los datos.

El objetivo de este proyecto fue construir y evaluar un modelo de clasificación capaz de distinguir entre tráfico de red normal y tráfico de ataque, utilizando el dataset NSL-KDD y la librería `scikit-learn`.

---

## 2. Metodología

El proyecto siguió una metodología estándar de Machine Learning:

### 2.1. Adquisición y Carga de Datos

Se utilizó el dataset **NSL-KDD**, una versión refinada del popular KDD Cup 1999, ampliamente usado para la evaluación de IDS. Este dataset contiene registros de conexiones de red con diversas características y una etiqueta que indica si la conexión es normal o un tipo específico de ataque.

*   **Datos de Entrenamiento:** `https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt`
*   **Datos de Prueba:** `https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt`

Se utilizó la librería `pandas` para cargar los datos directamente desde las URLs, asignando los nombres de columna correspondientes.

### 2.2. Preprocesamiento de Datos

El preprocesamiento es fundamental para preparar los datos para el modelo:

1.  **Definición del Objetivo:** La columna `attack` (que contiene el tipo de ataque o 'normal') se transformó en una variable binaria `is_attack`: `0` para tráfico normal y `1` para cualquier tipo de ataque.
2.  **Manejo de Características Categóricas:** Las columnas como `protocol_type`, `service` y `flag` son categóricas (texto). Se aplicó **One-Hot Encoding** (`pd.get_dummies()`) para convertirlas en un formato numérico que el modelo pudiera procesar. Esto crea nuevas columnas binarias para cada categoría única.
3.  **Alineación de Conjuntos:** Se aseguró que los conjuntos de entrenamiento y prueba tuvieran exactamente las mismas columnas y en el mismo orden, rellenando con ceros las columnas que pudieran faltar en uno de los conjuntos (debido a la presencia/ausencia de ciertas categorías en los datos).

### 2.3. Entrenamiento del Modelo

Se seleccionó un **RandomForestClassifier** de `scikit-learn` para la tarea de clasificación. Este modelo es un algoritmo de ensamble que construye múltiples árboles de decisión y combina sus predicciones, lo que lo hace robusto y efectivo para datasets con características mixtas.

*   **Parámetros:** Se usaron 100 estimadores (`n_estimators=100`) y se configuró `n_jobs=-1` para aprovechar todos los núcleos de la CPU durante el entrenamiento.

### 2.4. Evaluación del Modelo

El modelo entrenado se evaluó utilizando el conjunto de datos de prueba, que el modelo no había visto durante el entrenamiento. Se calcularon las siguientes métricas:

*   **Exactitud (Accuracy):** Proporción de predicciones correctas sobre el total.
*   **Matriz de Confusión:** Tabla que muestra el número de verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos.
*   **Reporte de Clasificación:** Proporciona métricas detalladas por clase, incluyendo Precisión, Recall y F1-score.

---

## 3. Resultados y Análisis

Los resultados obtenidos demuestran un rendimiento excepcional del modelo:

```text
--- Resultados de la Evaluación ---
Exactitud (Accuracy): 0.9970

Matriz de Confusión:
[[ 9709    10]
 [   26  4902]]

Reporte de Clasificación:
              precision    recall  f1-score   support

      Normal       1.00      1.00      1.00      9719
      Ataque       1.00      0.99      0.99      4928

    accuracy                           1.00     14647
   macro avg       1.00      0.99      1.00     14647
weighted avg       1.00      1.00      1.00     14647
```

### 3.1. Interpretación de la Matriz de Confusión

*   **Verdaderos Negativos (TN): 9709**
    *   El modelo clasificó correctamente 9709 conexiones como normales.
*   **Falsos Positivos (FP): 10**
    *   El modelo clasificó erróneamente 10 conexiones normales como ataques (falsas alarmas). Este número es muy bajo, lo que indica que el modelo no genera muchas alertas innecesarias.
*   **Falsos Negativos (FN): 26**
    *   El modelo clasificó erróneamente 26 ataques como conexiones normales (ataques no detectados). Este es el tipo de error más crítico en un IDS, ya que representa una vulnerabilidad. Aunque bajo, es el área de mejora.
*   **Verdaderos Positivos (TP): 4902**
    *   El modelo clasificó correctamente 4902 ataques como tales.

### 3.2. Interpretación del Reporte de Clasificación

*   **Clase 'Normal' (0):**
    *   **Precisión (1.00):** De todas las conexiones que el modelo predijo como normales, el 100% fueron realmente normales. No hubo falsos negativos para esta clase.
    *   **Recall (1.00):** De todas las conexiones que eran realmente normales, el modelo identificó el 100% correctamente.
    *   **F1-score (1.00):** Indica un equilibrio perfecto entre precisión y recall para la clase normal.

*   **Clase 'Ataque' (1):**
    *   **Precisión (1.00):** De todas las conexiones que el modelo predijo como ataques, el 100% fueron realmente ataques. Esto significa que cuando el modelo dice que hay un ataque, es muy probable que sea cierto.
    *   **Recall (0.99):** De todas las conexiones que eran realmente ataques, el modelo detectó el 99% correctamente. Esto es un valor muy alto, indicando que el modelo es muy bueno detectando ataques.
    *   **F1-score (0.99):** Un valor muy alto, mostrando un excelente rendimiento general para la detección de ataques.

### 3.3. Exactitud General

La exactitud del **99.70%** es muy alta, lo que sugiere que el modelo es extremadamente efectivo en la tarea de clasificación binaria de tráfico de red.

---

## 4. Conclusión

Este proyecto demuestra la viabilidad y la alta eficacia de utilizar modelos de Machine Learning, específicamente `RandomForestClassifier`, para la detección de intrusiones en redes. El modelo desarrollado es capaz de clasificar el tráfico de red con una precisión y recall muy elevados, minimizando las falsas alarmas y detectando la gran mayoría de los ataques. Esto lo convierte en una herramienta prometedora para mejorar la seguridad de las infraestructuras de red.
