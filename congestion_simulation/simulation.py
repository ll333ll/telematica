import random
import matplotlib.pyplot as plt

# --- Parámetros de la Simulación ---
SIMULATION_TIME = 100  # segundos
LINK_CAPACITY = 100  # paquetes por segundo
BUFFER_SIZE = 30  # paquetes
NUM_CLIENTS_RENO = 3
NUM_CLIENTS_CUBIC = 3
NUM_CLIENTS_BBR = 3
NUM_CLIENTS_CUSTOM = 3

# --- Clases de Algoritmos de Control de Congestión ---

class CongestionControlAlgorithm:
    """Clase base para los algoritmos de control de congestión."""
    def __init__(self, ssthresh=64, cwnd=1):
        self.cwnd = cwnd
        self.ssthresh = ssthresh

    def on_ack(self, rtt):
        """Se llama cuando se recibe un ACK."""
        raise NotImplementedError

    def on_packet_loss(self, rtt):
        """Se llama cuando se detecta una pérdida de paquete."""
        raise NotImplementedError

class Reno(CongestionControlAlgorithm):
    """Implementación de TCP Reno."""
    def on_ack(self, rtt):
        if self.cwnd < self.ssthresh:
            self.cwnd *= 2
        else:
            self.cwnd += 1

    def on_packet_loss(self, rtt):
        self.ssthresh = self.cwnd / 2
        self.cwnd = self.ssthresh
        if self.cwnd < 1:
            self.cwnd = 1

class Cubic(CongestionControlAlgorithm):
    """Implementación simplificada de TCP CUBIC."""
    def __init__(self, c=0.4, beta=0.7, **kwargs):
        super().__init__(**kwargs)
        self.c = c
        self.beta = beta
        self.w_max = self.cwnd

    def on_ack(self, rtt):
        self.cwnd += 0.5

    def on_packet_loss(self, rtt):
        self.w_max = self.cwnd
        self.cwnd *= self.beta
        if self.cwnd < 1:
            self.cwnd = 1
        self.ssthresh = self.cwnd

class BBR(CongestionControlAlgorithm):
    """Implementación conceptual de BBR."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bottleneck_bw = float('inf')
        self.min_rtt = float('inf')

    def on_ack(self, rtt):
        self.min_rtt = min(self.min_rtt, rtt)
        self.cwnd += 0.75

    def on_packet_loss(self, rtt):
        self.cwnd *= 0.85
        if self.cwnd < 1:
            self.cwnd = 1

class TCPCustom(CongestionControlAlgorithm):
    """Implementación de un algoritmo custom 'Cautious Reno'."""
    def on_ack(self, rtt):
        if self.cwnd < self.ssthresh:
            self.cwnd *= 2
        else:
            self.cwnd += 1

    def on_packet_loss(self, rtt):
        # Reacciona basado en el RTT actual.
        # Si el RTT es alto (>130ms), indica congestión, reacciona más agresivamente.
        if rtt > 0.13:
            self.ssthresh = self.cwnd * 0.4
        else:
            # Si el RTT es bajo, es más conservador.
            self.ssthresh = self.cwnd * 0.6
        
        self.cwnd = self.ssthresh
        if self.cwnd < 1:
            self.cwnd = 1

# --- Clases de la Simulación ---

class Client:
    """Representa un cliente que envía datos."""
    def __init__(self, client_id, algorithm):
        self.id = client_id
        self.algorithm = algorithm
        self.packets_to_send = 0

class SharedLink:
    """Simula el enlace de red compartido."""
    def __init__(self, capacity, buffer_size):
        self.capacity = capacity
        self.buffer_size = buffer_size
        self.buffer = []

    def process_packets(self):
        processed_packets = self.buffer[:self.capacity]
        self.buffer = self.buffer[self.capacity:]
        return processed_packets

# --- Lógica Principal de la Simulación ---

def run_simulation():
    clients = []
    client_id_counter = 0
    for _ in range(NUM_CLIENTS_RENO):
        clients.append(Client(f"Reno_{client_id_counter}", Reno()))
        client_id_counter += 1
    for _ in range(NUM_CLIENTS_CUBIC):
        clients.append(Client(f"Cubic_{client_id_counter}", Cubic()))
        client_id_counter += 1
    for _ in range(NUM_CLIENTS_BBR):
        clients.append(Client(f"BBR_{client_id_counter}", BBR()))
        client_id_counter += 1
    for _ in range(NUM_CLIENTS_CUSTOM):
        clients.append(Client(f"Custom_{client_id_counter}", TCPCustom()))
        client_id_counter += 1

    link = SharedLink(LINK_CAPACITY, BUFFER_SIZE)
    history = {client.id: {"throughput": [], "rtt": [], "cwnd": []} for client in clients}

    for t in range(SIMULATION_TIME):
        for client in clients:
            client.packets_to_send = int(client.algorithm.cwnd)
            link.buffer.extend([(client, t) for _ in range(client.packets_to_send)])
        
        processed_this_step = link.process_packets()
        link.buffer = []

        current_buffer_fill_ratio = len(processed_this_step) / link.capacity
        base_rtt = 0.1
        simulated_rtt = base_rtt + (current_buffer_fill_ratio * 0.05)

        processed_counts = {client.id: 0 for client in clients}
        for processed_client, send_time in processed_this_step:
            processed_counts[processed_client.id] += 1

        for client in clients:
            acked_packets = processed_counts[client.id]
            lost_packets_for_client = client.packets_to_send - acked_packets

            for _ in range(acked_packets):
                client.algorithm.on_ack(simulated_rtt)
            
            if lost_packets_for_client > 0:
                client.algorithm.on_packet_loss(simulated_rtt)

            history[client.id]["throughput"].append(acked_packets)
            history[client.id]["rtt"].append(simulated_rtt * 1000)
            history[client.id]["cwnd"].append(client.algorithm.cwnd)

    return history

# --- Visualización de Resultados ---

def get_color(client_id):
    if 'Reno' in client_id: return 'r'
    if 'Cubic' in client_id: return 'g'
    if 'BBR' in client_id: return 'b'
    if 'Custom' in client_id: return 'm' # Magenta for Custom
    return 'k'

def plot_results(history):
    time_axis = range(SIMULATION_TIME)
    plt.figure(figsize=(18, 10))

    # Gráfico de Throughput
    plt.subplot(3, 1, 1)
    for client_id, data in history.items():
        linestyle = '--' if '1' in client_id else ':' if '2' in client_id else '-'
        plt.plot(time_axis, data["throughput"], label=client_id, color=get_color(client_id), linestyle=linestyle)
    plt.title("Throughput por Cliente")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Paquetes / s")
    plt.legend()
    plt.grid(True)

    # Gráfico de RTT
    plt.subplot(3, 1, 2)
    rtt_data = {
        "Reno": next(iter(data["rtt"] for cid, data in history.items() if 'Reno' in cid)),
        "Cubic": next(iter(data["rtt"] for cid, data in history.items() if 'Cubic' in cid)),
        "BBR": next(iter(data["rtt"] for cid, data in history.items() if 'BBR' in cid)),
        "Custom": next(iter(data["rtt"] for cid, data in history.items() if 'Custom' in cid))
    }
    for name, rtt_values in rtt_data.items():
        plt.plot(time_axis, rtt_values, label=f"RTT ({name})", color=get_color(name))
    plt.title("RTT Simulado")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("RTT (ms)")
    plt.legend()
    plt.grid(True)

    # Gráfico de Congestion Window (cwnd)
    plt.subplot(3, 1, 3)
    for client_id, data in history.items():
        linestyle = '--' if '1' in client_id else ':' if '2' in client_id else '-'
        plt.plot(time_axis, data["cwnd"], label=client_id, color=get_color(client_id), linestyle=linestyle)
    plt.title("Ventana de Congestión (cwnd) por Cliente")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Tamaño de la Ventana")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig("congestion_simulation_results_with_custom.png")
    print("Gráficos guardados en 'congestion_simulation_results_with_custom.png'")

if __name__ == "__main__":
    simulation_history = run_simulation()
    plot_results(simulation_history)
