"""
Simulación de control de congestión TCP en un enlace compartido (comentada línea por línea).

Objetivo (actividad opcional):
- Modelar un enlace con cola de tamaño finito (BUFFER_SIZE) y capacidad fija por paso (LINK_CAPACITY).
- Simular 9 clientes (3 Reno, 3 CUBIC, 3 BBR) y, opcionalmente, 3 TCPCustom.
- Medir throughput, RTT, cwnd; graficar y calcular fairness (Índice de Jain) y descartes por algoritmo.

Notas de lectura:
- Este archivo contiene comentarios extensos para facilitar la defensa de la actividad.
- Los comentarios siguen el flujo del código y justifican cada decisión de diseño.
"""

import random                 # RNG para reproducibilidad y pequeñas decisiones estocásticas
import argparse               # Parseo de argumentos por CLI para ejecutar diferentes escenarios
import matplotlib.pyplot as plt  # Librería de gráficos para visualización de resultados

# --- Parámetros de la Simulación (por defecto; se pueden sobreescribir por CLI) ---
SIMULATION_TIME = 100  # Tiempo total de simulación (en pasos discretos que interpretamos como segundos)
LINK_CAPACITY = 100    # Capacidad del enlace por paso (paquetes/segundo)
BUFFER_SIZE = 30       # Tamaño máximo de la cola del enlace (paquetes)
NUM_CLIENTS_RENO = 3   # Número de clientes con TCP Reno
NUM_CLIENTS_CUBIC = 3  # Número de clientes con TCP CUBIC
NUM_CLIENTS_BBR = 3    # Número de clientes con TCP BBR
NUM_CLIENTS_CUSTOM = 3 # Número de clientes con algoritmo propuesto (TCPCustom)

# --- Clases de Algoritmos de Control de Congestión ---

class CongestionControlAlgorithm:
    """Clase base para los algoritmos de control de congestión.

    - ssthresh: umbral que separa Slow Start y Congestion Avoidance.
    - cwnd: ventana de congestión (número de paquetes en vuelo permitidos).
    """
    def __init__(self, ssthresh=64, cwnd=1):
        self.cwnd = cwnd            # Ventana de congestión (arranca en 1 paquete)
        self.ssthresh = ssthresh    # Umbral de slow start (valor inicial representativo)

    def on_ack(self, rtt):
        """Hook invocado al recibir ACK (crecimiento de cwnd)."""
        raise NotImplementedError

    def on_packet_loss(self, rtt):
        """Hook invocado al detectar pérdida (reducción de cwnd)."""
        raise NotImplementedError

class Reno(CongestionControlAlgorithm):
    """Implementación de TCP Reno (simplificada)."""
    def on_ack(self, rtt):
        # Slow Start: crecimiento exponencial (aprox.)
        if self.cwnd < self.ssthresh:
            self.cwnd *= 2
        else:
            # Congestion Avoidance: crecimiento lineal
            self.cwnd += 1

    def on_packet_loss(self, rtt):
        # Pérdida: multiplicative decrease (reduce a la mitad)
        self.ssthresh = self.cwnd / 2
        self.cwnd = self.ssthresh
        if self.cwnd < 1:
            self.cwnd = 1

class Cubic(CongestionControlAlgorithm):
    """Implementación simplificada de TCP CUBIC.

    - beta < 1: reducción menos agresiva que Reno (recupera más rápido).
    """
    def __init__(self, c=0.4, beta=0.7, **kwargs):
        super().__init__(**kwargs)
        self.c = c                  # Parámetro formal (no usado en esta simplificación)
        self.beta = beta            # Factor de reducción en pérdidas
        self.w_max = self.cwnd      # Recuerda cwnd previa a la última pérdida

    def on_ack(self, rtt):
        # Aproximación: crecimiento algo más rápido que lineal
        self.cwnd += 0.5

    def on_packet_loss(self, rtt):
        self.w_max = self.cwnd
        self.cwnd *= self.beta      # Reducción multiplicativa (0.7 por defecto)
        if self.cwnd < 1:
            self.cwnd = 1
        self.ssthresh = self.cwnd

class BBR(CongestionControlAlgorithm):
    """Implementación conceptual de BBR (crecimiento estable)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bottleneck_bw = float('inf')  # No modelamos bw, pero dejamos el placeholder
        self.min_rtt = float('inf')        # RTT mínimo observado

    def on_ack(self, rtt):
        self.min_rtt = min(self.min_rtt, rtt)
        self.cwnd += 0.75     # Crecimiento moderado/estable

    def on_packet_loss(self, rtt):
        self.cwnd *= 0.85     # Reducción suave ante pérdida
        if self.cwnd < 1:
            self.cwnd = 1

class TCPCustom(CongestionControlAlgorithm):
    """Algoritmo propuesto: 'Cautious Reno' (reducción guiada por RTT)."""
    def on_ack(self, rtt):
        # Igual que Reno en crecimiento
        if self.cwnd < self.ssthresh:
            self.cwnd *= 2
        else:
            self.cwnd += 1

    def on_packet_loss(self, rtt):
        # Si la red está muy congestionada (RTT alto), se reduce más (40%)
        if rtt > 0.13:
            self.ssthresh = self.cwnd * 0.4
        else:
            # Si el RTT no es tan alto, reducción menos agresiva (60%)
            self.ssthresh = self.cwnd * 0.6
        self.cwnd = self.ssthresh
        if self.cwnd < 1:
            self.cwnd = 1

# --- Clases de la Simulación ---

class Client:
    """Representa un cliente/flujo asociado a un algoritmo de congestión."""
    def __init__(self, client_id, algorithm):
        self.id = client_id                 # Identificador amigable (p.ej., Reno_0)
        self.algorithm = algorithm          # Instancia del algoritmo (Reno/CUBIC/BBR/Custom)
        self.packets_to_send = 0            # Paquetes que intentará encolar en este paso

class SharedLink:
    """Simula el enlace compartido (cola FIFO de tamaño finito y capacidad por paso)."""
    def __init__(self, capacity, buffer_size):
        self.capacity = capacity            # Paquetes que el enlace “sirve” por paso
        self.buffer_size = buffer_size      # Máximo de paquetes en cola
        self.buffer = []                    # Cola FIFO: lista de tuplas (client, t)

    def enqueue(self, client, t):
        """Intenta encolar un paquete: True si entra, False si se descarta por overflow."""
        if len(self.buffer) >= self.buffer_size:
            return False                    # Overflow → pérdida
        self.buffer.append((client, t))     # Encolar al final (FIFO)
        return True

    def process_packets(self):
        """Procesa hasta 'capacity' paquetes y deja el resto en cola (backlog)."""
        processed_packets = self.buffer[:self.capacity]   # Toma el frente de la cola
        self.buffer = self.buffer[self.capacity:]         # Resto queda en espera
        return processed_packets

# --- Lógica Principal de la Simulación ---

def run_simulation(sim_time=SIMULATION_TIME, capacity=LINK_CAPACITY, buffer_size=BUFFER_SIZE,
                   n_reno=NUM_CLIENTS_RENO, n_cubic=NUM_CLIENTS_CUBIC, n_bbr=NUM_CLIENTS_BBR,
                   n_custom=NUM_CLIENTS_CUSTOM, seed=42):
    """Ejecuta la simulación principal con parámetros configurables por CLI.

    - sim_time: duración (pasos)
    - capacity: paquetes/segundo que sirve el enlace
    - buffer_size: tamaño de la cola del enlace
    - n_*: número de clientes por algoritmo
    - seed: semilla de aleatoriedad para reproducibilidad
    """
    clients = []
    client_id_counter = 0
    random.seed(seed)  # Fijar semilla: reproducibilidad

    for _ in range(n_reno):
        clients.append(Client(f"Reno_{client_id_counter}", Reno()))
        client_id_counter += 1
    for _ in range(n_cubic):
        clients.append(Client(f"Cubic_{client_id_counter}", Cubic()))
        client_id_counter += 1
    for _ in range(n_bbr):
        clients.append(Client(f"BBR_{client_id_counter}", BBR()))
        client_id_counter += 1
    for _ in range(n_custom):
        clients.append(Client(f"Custom_{client_id_counter}", TCPCustom()))
        client_id_counter += 1

    link = SharedLink(capacity, buffer_size)  # Enlace compartido
    # Estructura histórica por cliente: métricas por cada paso
    history = {client.id: {"throughput": [], "rtt": [], "cwnd": [], "drops": []} for client in clients}

    for t in range(sim_time):              # Avanza el tiempo en pasos discretos
        # ROTACIÓN: evita sesgo por orden fijo (reparte prioridad de encolado)
        rot = t % len(clients) if clients else 0
        ordered_clients = clients[rot:] + clients[:rot]

        # Envío: intentar encolar hasta cwnd; si cola llena, cuenta pérdida
        drops = {client.id: 0 for client in clients}   # Contador de descartes por cliente en este paso
        attempted = {client.id: 0 for client in clients}
        for client in ordered_clients:
            to_send = int(max(1, client.algorithm.cwnd))  # No permitir 0; al menos 1 paquete
            attempted[client.id] = to_send
            for _ in range(to_send):
                if not link.enqueue(client, t):
                    drops[client.id] += 1  # pérdida por overflow de búfer

        # RTT en función de ocupación actual de cola (bufferbloat)
        occ_ratio = (len(link.buffer) / link.buffer_size) if link.buffer_size > 0 else 0.0  # Ocupación [0,1]
        base_rtt = 0.1                  # RTT base (100 ms)
        simulated_rtt = base_rtt + (occ_ratio * 0.05)  # Hasta +50 ms si la cola está llena

        # Procesar paquetes del paso
        processed_this_step = link.process_packets()  # Sirve hasta 'capacity' del frente de la cola
        processed_counts = {client.id: 0 for client in clients}
        for processed_client, send_time in processed_this_step:
            processed_counts[processed_client.id] += 1

        # Feedback a cada algoritmo
        for client in clients:
            acked = processed_counts[client.id]
            # aplicar acks
            for _ in range(acked):                 # Aplicar crecimiento por cada ACK
                client.algorithm.on_ack(simulated_rtt)
            # pérdidas solo si hubo drops (no por backlog no procesado)
            if drops[client.id] > 0:
                client.algorithm.on_packet_loss(simulated_rtt)

            history[client.id]["throughput"].append(acked)              # Paquetes efectivamente servidos
            history[client.id]["rtt"].append(simulated_rtt * 1000)       # RTT en ms
            history[client.id]["cwnd"].append(client.algorithm.cwnd)     # cwnd posterior al feedback
            history[client.id]["drops"].append(drops[client.id])         # Descargas por overflow

    return history

# --- Visualización de Resultados ---

def get_color(client_id):
    if 'Reno' in client_id: return 'r'
    if 'Cubic' in client_id: return 'g'
    if 'BBR' in client_id: return 'b'
    if 'Custom' in client_id: return 'm' # Magenta for Custom
    return 'k'

def plot_results(history, sim_time=SIMULATION_TIME, title_suffix=""):
    """Genera 3 gráficos principales y 2 adicionales (Fairness y Descartes).

    - Panel: Throughput por cliente, RTT promedio por algoritmo, cwnd por cliente
    - Fairness: Índice de Jain a lo largo del tiempo
    - Drops: Barras con paquetes descartados por algoritmo
    """
    time_axis = range(sim_time)            # Eje temporal (0..sim_time-1)
    plt.figure(figsize=(18, 10))           # Lienzo amplio para 3 subplots

    # Gráfico de Throughput
    plt.subplot(3, 1, 1)
    for client_id, data in history.items():
        linestyle = '--' if '1' in client_id else ':' if '2' in client_id else '-'
        plt.plot(time_axis, data["throughput"], label=client_id, color=get_color(client_id), linestyle=linestyle)
    plt.title(f"Throughput por Cliente{(' — ' + title_suffix) if title_suffix else ''}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Paquetes / s")
    plt.legend()
    plt.grid(True)

    # Gráfico de RTT (por cliente)
    plt.subplot(3, 1, 2)
    for client_id, data in history.items():
        linestyle = '--' if '1' in client_id else ':' if '2' in client_id else '-'
        plt.plot(time_axis, data["rtt"], label=client_id, color=get_color(client_id), linestyle=linestyle)
    plt.title(f"RTT Simulado{(' — ' + title_suffix) if title_suffix else ''}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("RTT (ms)")
    plt.legend()
    plt.grid(True)

    # Gráfico de Congestion Window (cwnd)
    plt.subplot(3, 1, 3)
    for client_id, data in history.items():
        linestyle = '--' if '1' in client_id else ':' if '2' in client_id else '-'
        plt.plot(time_axis, data["cwnd"], label=client_id, color=get_color(client_id), linestyle=linestyle)
    plt.title(f"Ventana de Congestión (cwnd) por Cliente{(' — ' + title_suffix) if title_suffix else ''}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Tamaño de la Ventana")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    out = f"congestion_simulation_results_with_custom{('_' + title_suffix.replace(' ', '_')) if title_suffix else ''}.png"
    plt.savefig(out)
    print(f"Gráficos guardados en '{out}'")

    # Fairness de Jain (sobre throughput por paso)
    def jain_index(values):
        s = sum(values)
        s2 = sum(v*v for v in values)
        n = len(values)
        return (s*s) / (n * s2) if s2 > 0 and n > 0 else 0.0

    # Construir fairness por tiempo usando througputs por cliente
    th_series = [data["throughput"] for _, data in history.items()]
    fairness_over_time = [jain_index([series[t] for series in th_series]) for t in range(sim_time)]
    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, fairness_over_time, color='purple')
    plt.ylim(0, 1.05)
    plt.title(f"Índice de Jain (Fairness){(' — ' + title_suffix) if title_suffix else ''}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("J(t)")
    plt.grid(True)
    out_fair = f"fairness_jain{('_' + title_suffix.replace(' ', '_')) if title_suffix else ''}.png"
    plt.tight_layout(); plt.savefig(out_fair)
    print(f"Gráfico de fairness guardado en '{out_fair}'")

    # Barras: paquetes descartados por algoritmo (agregado)
    algos = ["Reno", "Cubic", "BBR", "Custom"]
    drops_by_algo = {"Reno": 0, "Cubic": 0, "BBR": 0, "Custom": 0}
    for cid, data in history.items():
        total_drops = sum(data["drops"]) if "drops" in data else 0
        for name in algos:
            if name in cid:
                drops_by_algo[name] += total_drops
                break
    labels = [k for k, v in drops_by_algo.items() if v > 0]
    values = [drops_by_algo[k] for k in labels]
    if labels:
        plt.figure(figsize=(6, 4))
        plt.bar(labels, values, color=['r','g','b','m'][:len(labels)])
        plt.title(f"Paquetes descartados por algoritmo{(' — ' + title_suffix) if title_suffix else ''}")
        plt.ylabel("Paquetes descartados")
        plt.xlabel("Algoritmo")
        out_drops = f"drops_by_algorithm{('_' + title_suffix.replace(' ', '_')) if title_suffix else ''}.png"
        plt.tight_layout(); plt.savefig(out_drops)
        print(f"Gráfico de descartes guardado en '{out_drops}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulación de control de congestión TCP en enlace compartido")
    parser.add_argument("--sim-time", type=int, default=SIMULATION_TIME, help="Tiempo de simulación en segundos")
    parser.add_argument("--capacity", type=int, default=LINK_CAPACITY, help="Capacidad del enlace (pps)")
    parser.add_argument("--buffer", type=int, default=BUFFER_SIZE, help="Tamaño del búfer (paquetes)")
    parser.add_argument("--reno", type=int, default=NUM_CLIENTS_RENO, help="Número de clientes TCP Reno")
    parser.add_argument("--cubic", type=int, default=NUM_CLIENTS_CUBIC, help="Número de clientes TCP CUBIC")
    parser.add_argument("--bbr", type=int, default=NUM_CLIENTS_BBR, help="Número de clientes TCP BBR")
    parser.add_argument("--custom", type=int, default=NUM_CLIENTS_CUSTOM, help="Número de clientes TCPCustom")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--all-scenarios", action="store_true", help="Ejecuta 4 escenarios: (10/100 pps) x (10/30 buffer)")
    args = parser.parse_args()

    if args.all_scenarios:
        scenarios = [
            (10, 10), (10, 30), (100, 10), (100, 30)
        ]
        for cap, buf in scenarios:
            title = f"cap{cap}_buf{buf}"
            hist = run_simulation(sim_time=args.sim_time,
                                  capacity=cap, buffer_size=buf,
                                  n_reno=args.reno, n_cubic=args.cubic, n_bbr=args.bbr, n_custom=args.custom,
                                  seed=args.seed)
            plot_results(hist, sim_time=args.sim_time, title_suffix=title)
    else:
        hist = run_simulation(sim_time=args.sim_time, capacity=args.capacity, buffer_size=args.buffer,
                              n_reno=args.reno, n_cubic=args.cubic, n_bbr=args.bbr, n_custom=args.custom,
                              seed=args.seed)
        suffix = f"cap{args.capacity}_buf{args.buffer}"
        plot_results(hist, sim_time=args.sim_time, title_suffix=suffix)
