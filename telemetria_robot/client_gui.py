import socket
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox

class ClientGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cliente de Telemetría")
        self.geometry("600x700")
        self.sock = None
        self.is_admin = False

        self.create_widgets()

    def create_widgets(self):
        # Frame de conexión
        conn_frame = tk.Frame(self, pady=5)
        conn_frame.pack(fill=tk.X)
        tk.Label(conn_frame, text="Host:").pack(side=tk.LEFT, padx=5)
        self.host_entry = tk.Entry(conn_frame)
        self.host_entry.pack(side=tk.LEFT)
        self.host_entry.insert(0, "127.0.0.1")
        tk.Label(conn_frame, text="Puerto:").pack(side=tk.LEFT, padx=5)
        self.port_entry = tk.Entry(conn_frame, width=10)
        self.port_entry.pack(side=tk.LEFT)
        self.port_entry.insert(0, "8080")
        self.connect_button = tk.Button(conn_frame, text="Conectar", command=self.connect_server)
        self.connect_button.pack(side=tk.LEFT, padx=5)

        # Frame de Login
        login_frame = tk.Frame(self, pady=5)
        login_frame.pack(fill=tk.X)
        tk.Label(login_frame, text="Rol:").pack(side=tk.LEFT, padx=5)
        self.role_var = tk.StringVar(value="USER")
        tk.Radiobutton(login_frame, text="USER", variable=self.role_var, value="USER").pack(side=tk.LEFT)
        tk.Radiobutton(login_frame, text="ADMIN", variable=self.role_var, value="ADMIN").pack(side=tk.LEFT)
        tk.Label(login_frame, text="Password:").pack(side=tk.LEFT, padx=5)
        self.password_entry = tk.Entry(login_frame, show="*")
        self.password_entry.pack(side=tk.LEFT)
        self.login_button = tk.Button(login_frame, text="Login", command=self.login, state=tk.DISABLED)
        self.login_button.pack(side=tk.LEFT, padx=5)

        # Área de texto para mensajes
        self.log_area = scrolledtext.ScrolledText(self, state=tk.DISABLED)
        self.log_area.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)

        # Frame de comandos de Admin
        self.admin_frame = tk.Frame(self, pady=5)
        self.admin_frame.pack(fill=tk.X)
        tk.Label(self.admin_frame, text="Controles Admin:").pack(side=tk.LEFT, padx=5)
        tk.Button(self.admin_frame, text="Arriba", command=lambda: self.send_command("MOVE UP")).pack(side=tk.LEFT)
        tk.Button(self.admin_frame, text="Abajo", command=lambda: self.send_command("MOVE DOWN")).pack(side=tk.LEFT)
        tk.Button(self.admin_frame, text="Izquierda", command=lambda: self.send_command("MOVE LEFT")).pack(side=tk.LEFT)
        tk.Button(self.admin_frame, text="Derecha", command=lambda: self.send_command("MOVE RIGHT")).pack(side=tk.LEFT)
        tk.Button(self.admin_frame, text="Listar Usuarios", command=lambda: self.send_command("LIST_USERS")).pack(side=tk.LEFT, padx=10)

        # Frame para enviar comandos
        cmd_frame = tk.Frame(self, pady=5)
        cmd_frame.pack(fill=tk.X)
        self.cmd_entry = tk.Entry(cmd_frame)
        self.cmd_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)
        self.send_button = tk.Button(cmd_frame, text="Enviar", command=self.send_custom_command, state=tk.DISABLED)
        self.send_button.pack(side=tk.LEFT, padx=5)
        self.logout_button = tk.Button(cmd_frame, text="Logout", command=self.logout, state=tk.DISABLED)
        self.logout_button.pack(side=tk.RIGHT, padx=10)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def connect_server(self):
        host = self.host_entry.get()
        port = int(self.port_entry.get())
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((host, port))
            self.log("Conectado al servidor.")
            self.connect_button.config(state=tk.DISABLED)
            self.login_button.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)
            self.logout_button.config(state=tk.NORMAL)
            threading.Thread(target=self.receive_messages, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error de Conexión", str(e))
            self.sock = None

    def receive_messages(self):
        while True:
            try:
                message = self.sock.recv(1024).decode('utf-8').strip()
                if message:
                    self.log(f"[Servidor] {message}")
                    if "LOGIN_SUCCESS ADMIN" in message:
                        self.is_admin = True
                        self.toggle_admin_controls(True)
                else:
                    break
            except:
                self.log("Se perdió la conexión con el servidor.")
                self.reset_ui()
                break

    def login(self):
        role = self.role_var.get()
        password = self.password_entry.get() if role == "ADMIN" else "-"
        self.send_command(f"LOGIN {role} {password}")

    def send_command(self, command):
        if self.sock:
            try:
                self.sock.sendall(f"{command}\n".encode('utf-8'))
            except:
                self.log("Error al enviar comando.")

    def send_custom_command(self):
        command = self.cmd_entry.get()
        if command:
            self.send_command(command)
            self.cmd_entry.delete(0, tk.END)

    def logout(self):
        self.send_command("LOGOUT")
        self.on_closing()

    def on_closing(self):
        if self.sock:
            self.sock.close()
        self.destroy()

    def log(self, message):
        self.log_area.config(state=tk.NORMAL)
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.config(state=tk.DISABLED)
        self.log_area.see(tk.END)

    def toggle_admin_controls(self, is_admin):
        for child in self.admin_frame.winfo_children():
            child.config(state=tk.NORMAL if is_admin else tk.DISABLED)

    def reset_ui(self):
        self.connect_button.config(state=tk.NORMAL)
        self.login_button.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)
        self.logout_button.config(state=tk.DISABLED)
        self.toggle_admin_controls(False)
        self.is_admin = False

if __name__ == "__main__":
    app = ClientGUI()
    app.mainloop()
