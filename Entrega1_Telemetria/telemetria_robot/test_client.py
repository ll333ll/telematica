import socket
import sys
import time

def main():
    if len(sys.argv) != 3:
        print(f"Uso: python3 {sys.argv[0]} <host> <puerto>")
        sys.exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2])

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            print("Cliente de prueba conectado.")

            # Login USER
            s.sendall(b"LOGIN USER -\n")
            response = s.recv(1024).decode('utf-8').strip()
            print(f"Respuesta del servidor: {response}")

            if "LOGIN_SUCCESS" not in response:
                print("Fallo en el login.")
                sys.exit(1)

            print("Login exitoso.")
            
            # PING/PONG
            s.sendall(b"PING\n")
            print("PING enviado")
            pong = s.recv(1024).decode('utf-8').strip()
            print(f"PONG recibido: {pong}")

            # GET_DATA
            s.sendall(b"GET_DATA TEMP\n")
            data = s.recv(1024).decode('utf-8').strip()
            print(f"GET_DATA respuesta: {data}")

            # LIST_USERS como USER (debería error)
            s.sendall(b"LIST_USERS\n")
            resp = s.recv(1024).decode('utf-8').strip()
            print(f"LIST_USERS como USER: {resp}")

            # Logout
            s.sendall(b"LOGOUT\n")
            print("Comando LOGOUT enviado.")

    except Exception as e:
        print(f"Ocurrió un error: {e}")
        sys.exit(1)

    print("Prueba finalizada exitosamente.")

if __name__ == "__main__":
    main()
