# Guía Educativa de Actividades de Telemática

Este documento sirve como un tutorial detallado de los proyectos desarrollados en las carpetas `sockets` y `http`. El objetivo es enseñar los conceptos de telemática desde cero, justificando cada decisión de diseño y cada comando ejecutado.

---

## 1. Proyecto `sockets`: Fundamentos de la Comunicación en Red

**Ubicación:** `/home/666/UNIVERSIDAD/Telematica/sockets/`

Este proyecto es una introducción práctica al nivel más fundamental de la programación de redes: los **sockets**.

### 1.1. Concepto Clave: ¿Qué es un Socket?

Imagina que dos personas quieren hablar por teléfono. El teléfono en sí mismo es el "socket". Es un punto final de comunicación que permite enviar y recibir datos. En redes, un socket es una abstracción que el sistema operativo le da a los programadores para que puedan usar la red de forma sencilla, sin tener que preocuparse por los detalles de bajo nivel de los protocolos TCP/IP.

Un socket se define por una **dirección IP** y un **puerto**, como una dirección postal con un número de apartamento específico. `127.0.0.1:8080` significa "la máquina local, en el apartamento (puerto) 8080".

### 1.2. Explorando los Ejemplos

*   **`python_sockets/` y `c_sockets_simple/`**: Estos directorios contienen la implementación más básica del **modelo cliente-servidor**.

    *   **El Servidor:** Es un programa que se queda "escuchando" en un socket específico (IP y puerto). Su trabajo es esperar a que un cliente toque la puerta.
    *   **El Cliente:** Es un programa que "se conecta" al socket del servidor para iniciar la comunicación.

    La mejora clave en el ejemplo de Python fue usar **variables de entorno**. En lugar de escribir `servidor.connect(("127.0.0.1", 8080))` en el código (lo que se conoce como "hardcodear"), se lee la configuración del sistema. Esto es profesional porque permite cambiar la configuración de la aplicación sin modificar el código fuente, algo crucial en entornos de producción.

### 1.3. La Simulación DNS: El Corazón de la Actividad

La carpeta **`dns_test/`** es la más importante. Su objetivo es simular un **Servidor de Nombres de Dominio (DNS)** para resolver un dominio inventado (`PlugAndAd.com`) dentro de nuestra propia máquina.

#### Concepto: ¿Qué es DNS?

DNS es la "agenda de contactos de Internet". Los humanos recordamos nombres (`google.com`), pero las computadoras se comunican usando direcciones IP (`142.250.184.142`). DNS es el sistema que traduce esos nombres a direcciones IP.

#### ¿Por qué simular un DNS local?

Para desarrollar una aplicación web (como tu proyecto `PlugAndAd.com`), quieres que tu entorno de desarrollo sea lo más parecido posible al entorno de producción real. En producción, accederás a tu app a través de un dominio, no de `localhost:8000`. Simular un DNS local te permite probar tu aplicación usando un dominio real, como `https://PlugAndAd.com`, pero haciendo que ese dominio apunte a tu propia máquina.

#### Paso a Paso de la Solución y Ejecución

1.  **Configurar el Servidor DNS (BIND9):**
    *   **Justificación:** BIND9 es el software estándar de la industria para servidores DNS. La actividad consistió en configurarlo para que se considerara a sí mismo la máxima autoridad (`master`) para el dominio `PlugAndAd.com`.
    *   **Acción:** Se crearon archivos de configuración (como los que se encuentran en `/etc/bind/`) que definen la "zona" `PlugAndAd.com` y le dicen a BIND dónde encontrar los registros para ese dominio.

2.  **Crear el Registro de Zona:**
    *   **Justificación:** El archivo de zona es donde se define el mapeo. Se creó un **registro A** (el tipo de registro más común, que mapea un nombre a una IP) para que `PlugAndAd.com` apuntara a `127.0.0.1` (localhost).

3.  **Configurar el Cliente DNS del Sistema:**
    *   **Justificación:** Por defecto, tu sistema operativo pregunta a un servidor DNS externo (el de tu proveedor de internet). Se tuvo que modificar la configuración del sistema (el archivo `/etc/resolv.conf`) para que, en su lugar, le preguntara primero al servidor DNS que acabábamos de configurar en nuestra propia máquina.

4.  **Verificación:**
    *   **Comando:** `dig @127.0.0.1 PlugAndAd.com`
    *   **Análisis del Comando:**
        *   `dig`: Es una herramienta de diagnóstico de DNS.
        *   `@127.0.0.1`: Le dice a `dig` que ignore la configuración del sistema y envíe la consulta directamente al servidor que se ejecuta en `127.0.0.1`.
        *   `PlugAndAd.com`: El dominio que queremos resolver.
    *   **Resultado Esperado:** La respuesta de `dig` confirmaría que `PlugAndAd.com` tiene la dirección IP `127.0.0.1`, validando que nuestra simulación de DNS funciona correctamente.

---

## 2. Proyecto `http`: Arquitectura de una Aplicación Web Moderna

**Ubicación:** `/home/666/UNIVERSIDAD/Telematica/http/`

Este proyecto demuestra cómo se construyen las aplicaciones web hoy en día, separando la lógica de la presentación.

### 2.1. Concepto Clave: Frontend vs. Backend

*   **Backend (El Cerebro):** Es el servidor, la lógica que se ejecuta en una máquina remota. No tiene interfaz gráfica. Su trabajo es gestionar los datos, procesar la lógica de negocio y exponer esta funcionalidad a través de una **API (Interfaz de Programación de Aplicaciones)**. En este proyecto, el backend es `back.py`, un servidor hecho con **Flask** (un micro-framework de Python para construir APIs web).

*   **Frontend (La Cara):** Es lo que el usuario ve y con lo que interactúa en su navegador. Es una aplicación en sí misma (en este caso, `index.html` con JavaScript) que se ejecuta completamente en la máquina del usuario. Su trabajo es pedir datos al backend (a través de la API) y presentarlos de una manera amigable.

Esta separación es poderosa porque permite que el frontend y el backend sean desarrollados y actualizados de forma independiente. Podrías tener una app móvil y una página web (dos frontends diferentes) usando el mismo backend.

### 2.2. Análisis de la Implementación

*   **`backend/venv/`**: El **entorno virtual**.
    *   **Justificación:** Los proyectos de Python pueden depender de diferentes versiones de las mismas librerías. Un entorno virtual crea una instalación de Python aislada solo para este proyecto. Así, las librerías que instalemos para el proyecto `http` no entrarán en conflicto con las de otros proyectos en tu sistema. Es una práctica **esencial** en el desarrollo con Python.

*   **`frontend/index.html`**: La interfaz.
    *   **Mejora:** Se usó **Bootstrap**, un framework de CSS, para estilizar la página. Esto demuestra cómo los frontends modernos raramente se construyen desde cero, sino que se apoyan en librerías de diseño para crear interfaces atractivas y responsivas rápidamente.

### 2.3. Guía de Ejecución y Justificación

Para que la aplicación funcione, tanto el "cerebro" como la "cara" deben estar en ejecución.

**Terminal 1: Ejecutar el Backend**

1.  **Navega a la carpeta:** `cd /home/666/UNIVERSIDAD/Telematica/http/backend`
2.  **Activa el entorno virtual:** `source venv/bin/activate`
    *   **Justificación:** Este comando modifica tu sesión de terminal actual para que el comando `python` y las librerías que se usen sean las que están dentro de la carpeta `venv`, no las globales del sistema.
3.  **Inicia el servidor:** `python3 back.py`
    *   **Justificación:** Esto ejecuta el script de Flask, que inicia un servidor web que escucha peticiones HTTP (normalmente en el puerto 5000) y responde con datos en formato JSON, según la lógica de la API.

**Terminal 2: Servir el Frontend**

1.  **Navega a la carpeta:** `cd /home/666/UNIVERSIDAD/Telematica/http/frontend`
2.  **Inicia un servidor web estático:** `python3 -m http.server 8000`
    *   **Justificación:** Un navegador no puede simplemente "abrir" un archivo `index.html` desde el disco si este necesita hacer peticiones a una API (por políticas de seguridad de los navegadores). El archivo debe ser "servido" por un servidor web. Este simple comando de Python convierte cualquier carpeta en un servidor web estático, perfecto para desarrollo.

**Paso Final: Acceder a la App**

Abre un navegador y ve a `http://localhost:8000`. El navegador descargará `index.html` desde el servidor del frontend. El código JavaScript en ese archivo hará entonces una petición HTTP (usando `fetch` o `AJAX`) al `http://localhost:5000/users` (el backend) para obtener los datos de los usuarios y mostrarlos en la página.
