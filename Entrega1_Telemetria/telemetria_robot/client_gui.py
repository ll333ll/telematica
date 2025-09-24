import socket
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox
from tkinter import ttk
import queue

class ClientGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cliente de Telemetría")
        self.geometry("600x700")
        self.sock = None
        self.is_admin = False
        self.msg_q = queue.Queue()

        self.create_widgets()
        self.after(50, self.pump_queue)

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
        self.log_area.tag_config('DATA', foreground='#1f7a1f')
        self.log_area.tag_config('ERROR', foreground='#b30000')
        self.log_area.tag_config('MOVE', foreground='#0044cc')

        # Mini-gráfica de telemetría (barras de progreso)
        telem = tk.Frame(self, pady=5)
        telem.pack(fill=tk.X)
        tk.Label(telem, text="Temp (°C)").pack(side=tk.LEFT, padx=5)
        self.temp_var = tk.DoubleVar(value=0.0)
        self.temp_bar = ttk.Progressbar(telem, orient='horizontal', length=200, mode='determinate', maximum=50, variable=self.temp_var)
        self.temp_bar.pack(side=tk.LEFT, padx=5)
        tk.Label(telem, text="Hum (%)").pack(side=tk.LEFT, padx=10)
        self.hum_var = tk.DoubleVar(value=0.0)
        self.hum_bar = ttk.Progressbar(telem, orient='horizontal', length=200, mode='determinate', maximum=100, variable=self.hum_var)
        self.hum_bar.pack(side=tk.LEFT, padx=5)

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

        # Barra de estado
        status = tk.Frame(self)
        status.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_var = tk.StringVar(value="Desconectado")
        self.status_lbl = tk.Label(status, textvariable=self.status_var, anchor='w')
        self.status_lbl.pack(side=tk.LEFT, padx=8)

        # Atajos WASD para MOVE
        self.bind('<w>', lambda e: self.send_move('UP'))
        self.bind('<a>', lambda e: self.send_move('LEFT'))
        self.bind('<s>', lambda e: self.send_move('DOWN'))
        self.bind('<d>', lambda e: self.send_move('RIGHT'))

    def connect_server(self):
        host = self.host_entry.get()
        port = int(self.port_entry.get())
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((host, port))
            self.enqueue_log("Conectado al servidor.")
            self.connect_button.config(state=tk.DISABLED)
            self.login_button.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)
            self.logout_button.config(state=tk.NORMAL)
            self.status_var.set("Conectado (sin autenticar)")
            threading.Thread(target=self.receive_messages, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error de Conexión", str(e))
            self.sock = None

    def receive_messages(self):
        while True:
            try:
                message = self.sock.recv(1024).decode('utf-8').strip()
                if message:
                    self.enqueue_log(f"[Servidor] {message}")
                    if "LOGIN_SUCCESS ADMIN" in message:
                        self.is_admin = True
                        self.toggle_admin_controls(True)
                        self.status_var.set("Conectado (ADMIN)")
                    elif "LOGIN_SUCCESS USER" in message:
                        self.status_var.set("Conectado (USER)")
                else:
                    break
            except:
                self.enqueue_log("Se perdió la conexión con el servidor.")
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
                self.enqueue_log("Error al enviar comando.")

    def send_move(self, direction: str):
        self.send_command(f"MOVE {direction}")

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

    def enqueue_log(self, message: str):
        self.msg_q.put(message)

    def pump_queue(self):
        try:
            while True:
                msg = self.msg_q.get_nowait()
                tag = None
                if 'ERROR' in msg:
                    tag = 'ERROR'
                elif 'MOVE_' in msg or 'MOVE ' in msg:
                    tag = 'MOVE'
                elif 'DATA ' in msg:
                    tag = 'DATA'

                self.log_area.config(state=tk.NORMAL)
                if tag:
                    self.log_area.insert(tk.END, msg + "\n", tag)
                else:
                    self.log_area.insert(tk.END, msg + "\n")
                self.log_area.config(state=tk.DISABLED)
                self.log_area.see(tk.END)

                # Actualizar barras si viene DATA
                if tag == 'DATA':
                    # Msg ejemplo: [Servidor] DATA 1690000000 TEMP=23.5;HUM=60.2
                    try:
                        parts = msg.split('DATA', 1)[1].strip()
                        # parts: "169.. TEMP=..;HUM=.."
                        kvs = parts.split()
                        if len(kvs) >= 2:
                            data = kvs[1]
                            for kv in data.split(';'):
                                if kv.startswith('TEMP='):
                                    tval = float(kv.split('=')[1])
                                    self.temp_var.set(max(0.0, min(50.0, tval)))
                                if kv.startswith('HUM='):
                                    hval = float(kv.split('=')[1])
                                    self.hum_var.set(max(0.0, min(100.0, hval)))
                    except Exception:
                        pass
        except queue.Empty:
            pass
        self.after(50, self.pump_queue)

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
        self.status_var.set("Desconectado")

if __name__ == "__main__":
    app = ClientGUI()
    app.mainloop()
