# Protocolo de Comunicación - Robot de Telemetría

## 1. Visión General

Este protocolo define la comunicación entre un servidor (el robot de telemetría) y múltiples clientes. Permite a los usuarios recibir datos de telemetría en tiempo real y a un administrador controlar el robot.

- **Capa de Transporte:** Se utilizará TCP (Sockets de flujo `SOCK_STREAM`) para garantizar una comunicación fiable y ordenada.
- **Formato:** Todos los mensajes son texto plano (ASCII), terminados con un carácter de nueva línea (`\n`).

## 2. Flujo de Operación

1.  Un cliente establece una conexión TCP con el servidor.
2.  El cliente se autentica. Hay dos roles: `USER` (solo puede recibir datos) y `ADMIN` (puede recibir datos, mover el robot y ver otros usuarios).
3.  Una vez autenticado, el servidor comienza a enviar datos de telemetría al cliente cada 15 segundos.
4.  El cliente `ADMIN` puede enviar comandos de movimiento. El servidor responderá si el movimiento fue exitoso o si encontró un obstáculo.
5.  La conexión finaliza cuando el cliente envía un comando `LOGOUT` o se cierra el socket.

## 3. Formato de Mensajes

### Mensajes del Cliente al Servidor

| Comando           | Parámetros                                       | Descripción                                                                                                                                |
| ----------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `LOGIN`           | `<rol> <password>`                               | Autentica al cliente. El `<rol>` puede ser `USER` o `ADMIN`. La `password` solo es necesaria para el `ADMIN`. Para `USER`, se puede usar `-`. |
| `GET_DATA`        | `[variable]`                                     | Solicita una variable específica (ej. `TEMP`, `HUM`). Si no se especifica, el servidor enviará todas las variables.                           |
| `MOVE`            | `<direction>`                                    | Envía un comando de movimiento. `direction` puede ser `UP`, `DOWN`, `LEFT`, `RIGHT`.                                                       |
| `LIST_USERS`      | -                                                | Solicita la lista de usuarios conectados.                                                                                                  |
| `LOGOUT`          | -                                                | Cierra la sesión actual.                                                                                                                   |

### Mensajes del Servidor al Cliente

| Comando           | Parámetros                                           | Descripción                                                                                                                            |
| ----------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `LOGIN_SUCCESS`   | `<rol>`                                              | Notifica que la autenticación fue exitosa.                                                                                             |
| `LOGIN_FAIL`      | -                                                    | Notifica que la autenticación falló.                                                                                                   |
| `DATA`            | `<timestamp> <var1>=<val1>;<var2>=<val2>;...`         | Envía los datos de telemetría. El timestamp es el momento de la lectura.                                                               |
| `MOVE_SUCCESS`    | `<direction>`                                        | Confirma que el movimiento se realizó con éxito.                                                                                       |
| `MOVE_FAIL`       | `<direction> OBSTACLE`                               | Informa que el robot no pudo moverse en la dirección indicada debido a un obstáculo.                                                   |
| `USER_LIST`       | `<user1_ip:port>;<user2_ip:port>;...`                | Envía la lista de clientes conectados.                                                                                                 |
| `PONG`            | -                                                    | Respuesta a un `PING` del cliente para mantener la conexión activa.                                                                    |
| `ERROR`           | `<message>`                                          | Envía un mensaje de error si un comando no es válido o no se tienen los permisos.                                                     |

## 4. Ejemplo de Interacción

1.  **Cliente (USER):** `LOGIN USER -\n`
2.  **Servidor:** `LOGIN_SUCCESS USER\n`
3.  **Servidor (15s después):** `DATA 1663524890 TEMP=23.5;HUM=60.2;PRES=1012.5;WIND=15.3\n`
4.  **Cliente (ADMIN):** `LOGIN ADMIN mysecretpassword\n`
5.  **Servidor:** `LOGIN_SUCCESS ADMIN\n`
6.  **Cliente (ADMIN):** `MOVE UP\n`
7.  **Servidor:** `MOVE_SUCCESS UP\n`
8.  **Cliente (ADMIN):** `MOVE RIGHT\n`
9.  **Servidor:** `MOVE_FAIL RIGHT OBSTACLE\n`
10. **Cliente (ADMIN):** `LIST_USERS\n`
11. **Servidor:** `USER_LIST 192.168.1.10:54321;192.168.1.12:12345\n`
12. **Cliente (USER):** `LOGOUT\n`
