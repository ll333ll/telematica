import socket
import threading
import sys

def receive_messages(sock):
    while True:
        try:
            message = sock.recv(1024).decode('utf-8')
            if message:
                print(f"\n[Servidor] {message.strip()}\n> ", end="")
            else:
                break
        except:
            break

def main():
    if len(sys.argv) != 3:
        print(f"Uso: python3 {sys.argv[0]} <host> <puerto>")
        sys.exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2])

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            print(f"Conectado al servidor en {host}:{port}")

            receive_thread = threading.Thread(target=receive_messages, args=(s,))
            receive_thread.daemon = True
            receive_thread.start()

            while True:
                command = input("> ")
                if command.strip():
                    s.sendall(f"{command}\n".encode('utf-8'))
                if command.strip().upper() == "LOGOUT":
                    break

        except ConnectionRefusedError:
            print("No se pudo conectar al servidor.")
        except KeyboardInterrupt:
            print("\nCerrando conexión.")
        finally:
            s.close()

if __name__ == "__main__":
    main()
