#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Este script simula el comportamiento de diferentes algoritmos de control de congestión TCP
# en un enlace de red compartido. El objetivo es visualizar cómo cada algoritmo
# ajusta su ventana de congestión (cwnd) en respuesta a la congestión y cómo esto
# afecta el throughput y el Round-Trip Time (RTT).

# Importación de librerías necesarias
import random # Para generar números aleatorios, usado en la simulación de pérdidas o RTT.
import matplotlib.pyplot as plt # Para la creación de gráficos y visualización de los resultados.

# --- Parámetros Fundamentales de la Simulación ---
# Estos parámetros definen el entorno y la duración de nuestra simulación de red.

SIMULATION_TIME = 100  # Duración total de la simulación en "segundos" (pasos de tiempo).
                       # Cada paso de tiempo representa una unidad de tiempo en la simulación.

LINK_CAPACITY = 100  # Capacidad máxima del enlace compartido en "paquetes por segundo".
                     # Esto define cuántos paquetes puede procesar el enlace en cada paso de tiempo.

BUFFER_SIZE = 30  # Tamaño máximo del búfer (cola) del enlace en "paquetes".
                  # Si el número de paquetes que llegan al enlace excede su capacidad
                  # y el búfer está lleno, los paquetes adicionales se descartan (pérdida).

# Número de clientes para cada tipo de algoritmo de control de congestión.
# Esto permite observar la competencia entre diferentes algoritmos.
NUM_CLIENTS_RENO = 3    # Cantidad de clientes que usarán el algoritmo TCP Reno.
NUM_CLIENTS_CUBIC = 3   # Cantidad de clientes que usarán el algoritmo TCP CUBIC.
NUM_CLIENTS_BBR = 3     # Cantidad de clientes que usarán el algoritmo TCP BBR.
NUM_CLIENTS_CUSTOM = 3  # Cantidad de clientes que usarán nuestro algoritmo TCP Custom (Cautious Reno).


# --- Definición de Clases para Algoritmos de Control de Congestión ---
# Estas clases implementan la lógica central de cómo los algoritmos reaccionan
# a los ACKs (reconocimientos de paquetes) y a las pérdidas de paquetes para
# ajustar su ventana de congestión (cwnd).

class CongestionControlAlgorithm:
    """
    Clase base abstracta para todos los algoritmos de control de congestión.
    Define la interfaz común que todos los algoritmos deben implementar.
    """
    def __init__(self, ssthresh=64, cwnd=1):
        """
        Inicializa el algoritmo con una ventana de congestión (cwnd) y un umbral de slow start (ssthresh).
        :param ssthresh: Umbral de slow start inicial.
        :param cwnd: Ventana de congestión inicial.
        """
        self.cwnd = cwnd  # Congestion Window (cwnd): Número de paquetes que el emisor puede tener en tránsito (no reconocidos) en la red.
                          # Es la variable clave que el algoritmo ajusta para controlar la tasa de envío.
        self.ssthresh = ssthresh  # Slow Start Threshold (ssthresh): Umbral que determina cuándo el algoritmo
                                  # debe pasar de la fase de Slow Start a la fase de Congestion Avoidance.

    def on_ack(self, rtt):
        """
        Método abstracto que se llama cuando el emisor recibe un ACK (reconocimiento) de un paquete.
        Los algoritmos concretos implementarán cómo la cwnd debe crecer en respuesta a los ACKs.
        :param rtt: Round-Trip Time (tiempo de ida y vuelta) actual de la conexión.
        """
        raise NotImplementedError("El método on_ack debe ser implementado por las subclases.")

    def on_packet_loss(self, rtt):
        """
        Método abstracto que se llama cuando el emisor detecta una pérdida de paquete.
        Los algoritmos concretos implementarán cómo la cwnd debe reducirse en respuesta a las pérdidas.
        :param rtt: Round-Trip Time (tiempo de ida y vuelta) actual de la conexión.
        """
        raise NotImplementedError("El método on_packet_loss debe ser implementado por las subclases.")


class Reno(CongestionControlAlgorithm):
    """
    Implementación simplificada del algoritmo de control de congestión TCP Reno.
    Reno es uno de los algoritmos más influyentes y sirve como base para muchos otros.
    Se caracteriza por sus dos fases principales: Slow Start y Congestion Avoidance,
    y su reacción a la pérdida de paquetes.
    """
    def on_ack(self, rtt):
        """
        Define cómo TCP Reno incrementa su cwnd al recibir un ACK.
        :param rtt: RTT actual (no usado directamente por esta simplificación de Reno, pero se mantiene por la interfaz).
        """
        if self.cwnd < self.ssthresh:
            # Fase 1: Slow Start (Inicio Lento)
            # La cwnd crece exponencialmente (duplicándose por cada RTT, o incrementando en 1 por cada ACK).
            # Esto permite una rápida exploración del ancho de banda disponible al inicio de la conexión.
            self.cwnd *= 2
        else:
            # Fase 2: Congestion Avoidance (Evitación de Congestión)
            # Una vez que cwnd alcanza o supera ssthresh, el crecimiento se vuelve lineal.
            # La cwnd se incrementa en 1 por cada RTT (o una fracción por cada ACK),
            # buscando evitar la congestión de forma más cautelosa.
            self.cwnd += 1

    def on_packet_loss(self, rtt):
        """
        Define cómo TCP Reno reacciona a la detección de una pérdida de paquete.
        La pérdida de un paquete se interpreta como una señal de congestión en la red.
        :param rtt: RTT actual (no usado directamente por esta simplificación de Reno, pero se mantiene por la interfaz).
        """
        # Reducción multiplicativa:
        # El ssthresh se establece a la mitad de la cwnd actual.
        # Esto marca el nuevo umbral para la fase de Congestion Avoidance.
        self.ssthresh = self.cwnd / 2
        
        # La cwnd se reduce a la mitad del valor anterior (o al nuevo ssthresh).
        # Esta reducción drástica es la forma en que Reno alivia la congestión.
        self.cwnd = self.ssthresh
        
        # Asegura que la cwnd nunca sea menor que 1 paquete.
        if self.cwnd < 1:
            self.cwnd = 1

class Cubic(CongestionControlAlgorithm):
    """
    Implementación simplificada del algoritmo de control de congestión TCP CUBIC.
    CUBIC es una evolución de Reno, diseñada para funcionar mejor en redes de alta velocidad
    y largo retardo (Long Fat Networks - LFNs). Su principal característica es que su
    función de crecimiento de cwnd es cúbica, lo que le permite explorar el ancho de banda
    de forma más agresiva y justa en LFNs.
    """
    def __init__(self, c=0.4, beta=0.7, **kwargs):
        """
        Inicializa el algoritmo CUBIC.
        :param c: Constante de CUBIC que controla la agresividad del crecimiento.
        :param beta: Factor de reducción multiplicativa de la cwnd tras una pérdida.
        """
        super().__init__(**kwargs)
        self.c = c # Constante de CUBIC (valor típico 0.4)
        self.beta = beta # Factor de reducción multiplicativa (valor típico 0.7)
        self.w_max = self.cwnd # w_max: cwnd máxima alcanzada antes de la última reducción.
                               # Usado para calcular el punto de inflexión de la función cúbica.

    def on_ack(self, rtt):
        """
        Define cómo TCP CUBIC incrementa su cwnd al recibir un ACK.
        En una implementación completa, CUBIC usa una función cúbica basada en el tiempo
        desde la última pérdida para calcular el crecimiento. Aquí, para simplificar,
        usamos un crecimiento más rápido que Reno pero sin la complejidad de la función cúbica completa.
        :param rtt: RTT actual (usado en CUBIC real para ajustar el crecimiento, aquí simplificado).
        """
        # Simplificación: Crecimiento más rápido que Reno, pero sin la función cúbica completa.
        # En la realidad, CUBIC calcula un punto de inflexión y crece más rápido lejos de él.
        self.cwnd += 0.5 # Incremento de la ventana de congestión.

    def on_packet_loss(self, rtt):
        """
        Define cómo TCP CUBIC reacciona a la detección de una pérdida de paquete.
        :param rtt: RTT actual (no usado directamente en esta simplificación).
        """
        # w_max se actualiza a la cwnd actual antes de la reducción.
        # Esto es clave para la función cúbica de CUBIC, que usa w_max para su cálculo.
        self.w_max = self.cwnd
        
        # Reducción multiplicativa: La cwnd se reduce por un factor beta.
        # Esta reducción es menos drástica que la de Reno (0.5), lo que permite a CUBIC
        # recuperarse más rápidamente y mantener un mayor throughput en LFNs.
        self.cwnd *= self.beta
        
        # Asegura que la cwnd nunca sea menor que 1 paquete.
        if self.cwnd < 1:
            self.cwnd = 1
        
        # El ssthresh se establece al nuevo valor de la cwnd.
        self.ssthresh = self.cwnd


class BBR(CongestionControlAlgorithm):
    """
    Implementación conceptual del algoritmo de control de congestión TCP BBR (Bottleneck Bandwidth and Round-trip propagation time).
    BBR es un algoritmo más moderno desarrollado por Google. A diferencia de los algoritmos
    basados en pérdidas (como Reno y CUBIC), BBR se centra en estimar el ancho de banda
    del cuello de botella (bottleneck bandwidth) y el RTT mínimo de la ruta.
    Su objetivo es operar en el punto óptimo de la red, evitando llenar el búfer del enlace.
    """
    def __init__(self, **kwargs):
        """
        Inicializa el algoritmo BBR.
        """
        super().__init__(**kwargs)
        # Estimaciones clave de BBR:
        self.bottleneck_bw = float('inf') # Ancho de banda del cuello de botella estimado.
                                          # Inicialmente infinito, se refina durante la simulación.
        self.min_rtt = float('inf')       # RTT mínimo observado en la conexión.
                                          # Es una estimación del retardo de propagación puro.

    def on_ack(self, rtt):
        """
        Define cómo TCP BBR incrementa su cwnd al recibir un ACK.
        BBR ajusta su cwnd basándose en sus estimaciones de ancho de banda y RTT.
        Aquí, se simplifica a un crecimiento más estable.
        :param rtt: RTT actual, usado para actualizar la estimación de min_rtt.
        """
        # Actualiza la estimación del RTT mínimo.
        self.min_rtt = min(self.min_rtt, rtt)
        
        # En una implementación completa, BBR usaría el ancho de banda estimado
        # y el RTT mínimo para calcular la cwnd óptima. Aquí, se simplifica a
        # un crecimiento más estable y menos reactivo a las fluctuaciones.
        self.cwnd += 0.75 # Incremento de la ventana de congestión.

    def on_packet_loss(self, rtt):
        """
        Define cómo TCP BBR reacciona a la detección de una pérdida de paquete.
        A diferencia de Reno/CUBIC, BBR no reacciona tan drásticamente a la pérdida
        porque no la usa como la señal principal de congestión. Se enfoca en la
        utilización del ancho de banda y el RTT.
        :param rtt: RTT actual (no usado directamente en esta simplificación).
        """
        # BBR reduce la ventana, pero no tan drásticamente como los algoritmos
        # basados en pérdidas. Su objetivo es evitar llenar el búfer, no reaccionar
        # a la pérdida una vez que ya ocurrió.
        self.cwnd *= 0.85 # Reducción de la ventana de congestión.
        
        # Asegura que la cwnd nunca sea menor que 1 paquete.
        if self.cwnd < 1:
            self.cwnd = 1

class TCPCustom(CongestionControlAlgorithm):
    """
    Implementación de un algoritmo de control de congestión personalizado, denominado 'Cautious Reno'.
    Este algoritmo se basa en TCP Reno, pero introduce una lógica de reacción a la pérdida
    más sofisticada, que considera el Round-Trip Time (RTT) actual como un indicador
    adicional del nivel de congestión en la red.

    La idea es ser más agresivo en la reducción de la ventana cuando el RTT ya es alto
    (lo que sugiere que la red está muy congestionada), y ser más conservador
    cuando el RTT es bajo (lo que podría indicar que la pérdida fue esporádica o no
    debida a una congestión severa).
    """
    def on_ack(self, rtt):
        """
        Define cómo TCPCustom incrementa su cwnd al recibir un ACK.
        La fase de crecimiento es idéntica a la de TCP Reno.
        :param rtt: RTT actual (no usado directamente en esta fase).
        """
        if self.cwnd < self.ssthresh:
            # Fase 1: Slow Start (Inicio Lento)
            # Crecimiento exponencial para explorar rápidamente el ancho de banda.
            self.cwnd *= 2
        else:
            # Fase 2: Congestion Avoidance (Evitación de Congestión)
            # Crecimiento lineal para evitar sobrecargar la red.
            self.cwnd += 1

    def on_packet_loss(self, rtt):
        """
        Define cómo TCPCustom reacciona a la detección de una pérdida de paquete.
        La reacción es adaptativa, basándose en el RTT actual.
        :param rtt: RTT actual de la conexión, usado para determinar la agresividad de la reducción.
        """
        # Umbral de RTT para determinar la agresividad de la reducción.
        # Un RTT de 0.13 segundos (130 ms) se considera alto en este contexto simulado.
        if rtt > 0.13: 
            # Si el RTT es alto, la red está probablemente muy congestionada.
            # Reacción más agresiva: reduce ssthresh a un 40% de la cwnd actual.
            self.ssthresh = self.cwnd * 0.4
        else:
            # Si el RTT es bajo, la congestión podría no ser tan severa.
            # Reacción más conservadora: reduce ssthresh a un 60% de la cwnd actual.
            self.ssthresh = self.cwnd * 0.6
        
        # La cwnd se establece al nuevo ssthresh.
        self.cwnd = self.ssthresh
        
        # Asegura que la cwnd nunca sea menor que 1 paquete.
        if self.cwnd < 1:
            self.cwnd = 1

# --- Clases de la Simulación ---
# Estas clases modelan los componentes del entorno de red en el que operan
# los algoritmos de control de congestión.

class Client:
    """
    Representa un cliente individual que participa en la simulación de la red.
    Cada cliente tiene asociado un algoritmo de control de congestión que determina
    cuántos paquetes intenta enviar en cada paso de tiempo.
    """
    def __init__(self, client_id, algorithm):
        """
        Inicializa un nuevo cliente.
        :param client_id: Identificador único para el cliente (ej. "Reno_0", "Cubic_1").
        :param algorithm: Instancia del algoritmo de control de congestión asociado a este cliente.
        """
        self.id = client_id  # Identificador del cliente.
        self.algorithm = algorithm  # Instancia del algoritmo de control de congestión (Reno, CUBIC, BBR, Custom).
        self.packets_to_send = 0  # Número de paquetes que el cliente intenta enviar en el paso de tiempo actual.


class SharedLink:
    """
    Simula el enlace de red compartido, que actúa como el cuello de botella de la red.
    Este enlace tiene una capacidad limitada (paquetes por segundo) y un búfer de tamaño fijo.
    Los paquetes que exceden la capacidad del enlace o el tamaño del búfer se pierden.
    """
    def __init__(self, capacity, buffer_size):
        """
        Inicializa el enlace compartido.
        :param capacity: Capacidad máxima de paquetes que el enlace puede procesar por unidad de tiempo.
        :param buffer_size: Tamaño máximo del búfer (cola) del enlace.
        """
        self.capacity = capacity  # Capacidad del enlace (paquetes/segundo).
        self.buffer_size = buffer_size  # Tamaño máximo del búfer.
        self.buffer = []  # Lista que representa los paquetes actualmente en el búfer del enlace.

    def process_packets(self):
        """
        Procesa los paquetes que están en el búfer del enlace.
        Los paquetes se procesan hasta la capacidad del enlace en el paso de tiempo actual.
        Los paquetes restantes en el búfer (si los hay) se consideran perdidos por desbordamiento.
        :return: Una lista de los paquetes que fueron procesados exitosamente en este paso de tiempo.
        """
        # Los paquetes que se procesan son los que caben dentro de la capacidad del enlace.
        processed_packets = self.buffer[:self.capacity]
        
        # Los paquetes restantes en el búfer se descartan (simulando desbordamiento).
        # En una simulación más compleja, estos paquetes podrían ser retransmitidos.
        self.buffer = self.buffer[self.capacity:]
        
        return processed_packets


# --- Lógica Principal de la Simulación ---
# Esta sección contiene la función principal que orquesta la simulación,
# inicializando los componentes y ejecutando el bucle de tiempo.

def run_simulation():
    """
    Función principal que ejecuta la simulación de control de congestión.
    Orquesta la interacción entre los clientes, el enlace compartido y los algoritmos.
    :return: Un diccionario `history` que contiene las métricas (throughput, RTT, cwnd)
             para cada cliente a lo largo del tiempo de simulación.
    """
    # 1. Inicialización de Clientes y Enlace
    # Se crean las instancias de los clientes, cada uno con su algoritmo de control de congestión.
    # Se asigna un ID único a cada cliente para su seguimiento en las métricas.
    clients = []
    client_id_counter = 0 # Contador para asignar IDs únicos a los clientes.

    # Creación de clientes TCP Reno
    for _ in range(NUM_CLIENTS_RENO):
        clients.append(Client(f"Reno_{client_id_counter}", Reno()))
        client_id_counter += 1
    
    # Creación de clientes TCP CUBIC
    for _ in range(NUM_CLIENTS_CUBIC):
        clients.append(Client(f"Cubic_{client_id_counter}", Cubic()))
        client_id_counter += 1
    
    # Creación de clientes TCP BBR
    for _ in range(NUM_CLIENTS_BBR):
        clients.append(Client(f"BBR_{client_id_counter}", BBR()))
        client_id_counter += 1
    
    # Creación de clientes TCP Custom (nuestro algoritmo Cautious Reno)
    for _ in range(NUM_CLIENTS_CUSTOM):
        clients.append(Client(f"Custom_{client_id_counter}", TCPCustom()))
        client_id_counter += 1

    # Creación de la instancia del enlace compartido con su capacidad y tamaño de búfer.
    link = SharedLink(LINK_CAPACITY, BUFFER_SIZE)
    
    # Estructura para almacenar el historial de métricas de cada cliente a lo largo del tiempo.
    # Cada cliente tendrá listas para su throughput, RTT y cwnd en cada paso de tiempo.
    history = {client.id: {"throughput": [], "rtt": [], "cwnd": []} for client in clients}

    # 2. Bucle Principal de la Simulación
    # La simulación avanza paso a paso en el tiempo, simulando la interacción
    # de los clientes con el enlace y la reacción de los algoritmos.
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
