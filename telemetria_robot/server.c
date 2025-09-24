#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <time.h>

#define MAX_CLIENTS 10
#define BUFFER_SIZE 1024
#define ADMIN_PASSWORD "mysecretpassword"

// Estructura para la información del cliente
typedef struct {
    int socket;
    struct sockaddr_in address;
    char ip_str[INET_ADDRSTRLEN];
    int port;
    int is_admin;
} client_t;

client_t *clients[MAX_CLIENTS];
pthread_mutex_t clients_mutex = PTHREAD_MUTEX_INITIALIZER;

// Prototipos de funciones
void *handle_client(void *arg);
void *send_telemetry(void *arg);
void add_client(client_t *cl);
void remove_client(int client_socket);

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Uso: %s <puerto> <archivo_log>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int port = atoi(argv[1]);
    char *log_filename = argv[2];

    int server_socket, client_socket;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);
    pthread_t tid, telemetry_tid;

    // Crear socket del servidor
    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket == -1) {
        perror("Error al crear el socket");
        exit(EXIT_FAILURE);
    }

    // Configurar dirección del servidor
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    // Enlazar socket
    if (bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Error al enlazar");
        exit(EXIT_FAILURE);
    }

    // Escuchar
    listen(server_socket, 5);
    printf("Servidor escuchando en el puerto %d\n", port);

    // Crear hilo para enviar telemetría
    pthread_create(&telemetry_tid, NULL, send_telemetry, (void *)log_filename);

    while (1) {
        client_socket = accept(server_socket, (struct sockaddr *)&client_addr, &client_len);
        if (client_socket < 0) {
            perror("Error al aceptar cliente");
            continue;
        }

        // Crear estructura de cliente
        client_t *cli = (client_t *)malloc(sizeof(client_t));
        cli->socket = client_socket;
        cli->address = client_addr;
        cli->is_admin = 0;
        inet_ntop(AF_INET, &client_addr.sin_addr, cli->ip_str, INET_ADDRSTRLEN);
        cli->port = ntohs(client_addr.sin_port);

        // Añadir cliente a la lista y crear hilo para manejarlo
        add_client(cli);
        pthread_create(&tid, NULL, &handle_client, (void *)cli);
    }

    close(server_socket);
    return 0;
}

void *handle_client(void *arg) {
    client_t *cli = (client_t *)arg;
    char buffer[BUFFER_SIZE];
    int nbytes;

    printf("Cliente conectado: %s:%d\n", cli->ip_str, cli->port);

    while ((nbytes = read(cli->socket, buffer, sizeof(buffer) - 1)) > 0) {
        buffer[nbytes] = '\0';
        char *command = strtok(buffer, " \n");

        if (strcmp(command, "LOGIN") == 0) {
            char *role = strtok(NULL, " \n");
            char *password = strtok(NULL, " \n");
            if (role && strcmp(role, "ADMIN") == 0 && password && strcmp(password, ADMIN_PASSWORD) == 0) {
                cli->is_admin = 1;
                write(cli->socket, "LOGIN_SUCCESS ADMIN\n", 20);
            } else if (role && strcmp(role, "USER") == 0) {
                write(cli->socket, "LOGIN_SUCCESS USER\n", 19);
            } else {
                write(cli->socket, "LOGIN_FAIL\n", 11);
            }
        } else if (strcmp(command, "MOVE") == 0) {
            if (cli->is_admin) {
                char *direction = strtok(NULL, " \n");
                // Simulación de obstáculo
                if (rand() % 5 == 0) { 
                    char response[50];
                    sprintf(response, "MOVE_FAIL %s OBSTACLE\n", direction);
                    write(cli->socket, response, strlen(response));
                } else {
                    char response[50];
                    sprintf(response, "MOVE_SUCCESS %s\n", direction);
                    write(cli->socket, response, strlen(response));
                }
            } else {
                write(cli->socket, "ERROR No tienes permisos\n", 26);
            }
        } else if (strcmp(command, "LIST_USERS") == 0) {
            if (cli->is_admin) {
                char user_list[BUFFER_SIZE] = "USER_LIST ";
                pthread_mutex_lock(&clients_mutex);
                for (int i = 0; i < MAX_CLIENTS; i++) {
                    if (clients[i]) {
                        char client_info[50];
                        sprintf(client_info, "%s:%d;", clients[i]->ip_str, clients[i]->port);
                        strcat(user_list, client_info);
                    }
                }
                pthread_mutex_unlock(&clients_mutex);
                strcat(user_list, "\n");
                write(cli->socket, user_list, strlen(user_list));
            } else {
                write(cli->socket, "ERROR No tienes permisos\n", 26);
            }
        } else if (strcmp(command, "LOGOUT") == 0) {
            break;
        }
    }

    printf("Cliente desconectado: %s:%d\n", cli->ip_str, cli->port);
    remove_client(cli->socket);
    close(cli->socket);
    free(cli);
    pthread_detach(pthread_self());
    return NULL;
}

void *send_telemetry(void *arg) {
    char *log_filename = (char *)arg;
    while (1) {
        sleep(15);
        pthread_mutex_lock(&clients_mutex);

        time_t t = time(NULL);
        float temp = (rand() % 300) / 10.0; // 0.0 - 30.0
        float hum = (rand() % 1000) / 10.0; // 0.0 - 100.0

        char telemetry_data[BUFFER_SIZE];
        sprintf(telemetry_data, "DATA %ld TEMP=%.1f;HUM=%.1f\n", t, temp, hum);

        FILE *log_file = fopen(log_filename, "a");
        if (log_file) {
            fprintf(log_file, "%s", telemetry_data);
            fclose(log_file);
        }

        for (int i = 0; i < MAX_CLIENTS; i++) {
            if (clients[i]) {
                write(clients[i]->socket, telemetry_data, strlen(telemetry_data));
            }
        }
        pthread_mutex_unlock(&clients_mutex);
    }
    return NULL;
}

void add_client(client_t *cl) {
    pthread_mutex_lock(&clients_mutex);
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (!clients[i]) {
            clients[i] = cl;
            break;
        }
    }
    pthread_mutex_unlock(&clients_mutex);
}

void remove_client(int client_socket) {
    pthread_mutex_lock(&clients_mutex);
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (clients[i] && clients[i]->socket == client_socket) {
            clients[i] = NULL;
            break;
        }
    }
    pthread_mutex_unlock(&clients_mutex);
}
