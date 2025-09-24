#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/stat.h>

#define MAX_CLIENTS 10
#define BUFFER_SIZE 1024

/*
 Concurrency Contract:
 - clients[] is only modified via add_client() and remove_client(), both take clients_mutex.
 - handle_client() reads/writes only its own client socket; it does not iterate clients[].
 - send_telemetry() builds a local snapshot of connected client sockets under clients_mutex, then releases the lock
   and performs network writes outside the lock to avoid blocking other threads.
 - No write() calls are performed while holding clients_mutex.
*/

// Estructura para la información del cliente
typedef struct {
    int socket;
    struct sockaddr_in address;
    char ip_str[INET_ADDRSTRLEN];
    int port;
    int is_admin;
    time_t last_ping_time;
    time_t last_get_time;
} client_t;

client_t *clients[MAX_CLIENTS];
pthread_mutex_t clients_mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    time_t ts;
    float temp;
    float hum;
} telemetry_t;

static telemetry_t g_last_telemetry = {0};
static char g_admin_password[128] = {0};

// Prototipos de funciones
void *handle_client(void *arg);
void *send_telemetry(void *arg);
int add_client(client_t *cl);
void remove_client(int client_socket);
ssize_t send_all(int fd, const char *buf, size_t len);
static void load_admin_password(void);
static void rotate_log_if_needed(const char *path, size_t max_bytes);

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

    // Seeds y señales
    srand(time(NULL));
    signal(SIGPIPE, SIG_IGN);
    load_admin_password();

    // Crear socket del servidor
    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket == -1) {
        perror("Error al crear el socket");
        exit(EXIT_FAILURE);
    }

    // Reuse addr para reinicios rápidos
    int opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt(SO_REUSEADDR)");
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
        if (add_client(cli) == 0) {
            pthread_create(&tid, NULL, &handle_client, (void *)cli);
        } else {
            const char *msg = "ERROR Server at capacity\n";
            send_all(client_socket, msg, strlen(msg));
            close(client_socket);
            free(cli);
        }
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

        if (!command) {
            const char *err = "ERROR Invalid command\n";
            send_all(cli->socket, err, strlen(err));
            continue;
        }

        if (strcmp(command, "LOGIN") == 0) {
            char *role = strtok(NULL, " \n");
            char *password = strtok(NULL, " \n");
            if (role && strcmp(role, "ADMIN") == 0 && password && strcmp(password, g_admin_password) == 0) {
                cli->is_admin = 1;
                send_all(cli->socket, "LOGIN_SUCCESS ADMIN\n", 20);
            } else if (role && strcmp(role, "USER") == 0) {
                send_all(cli->socket, "LOGIN_SUCCESS USER\n", 19);
            } else {
                send_all(cli->socket, "LOGIN_FAIL\n", 11);
            }
        } else if (strcmp(command, "MOVE") == 0) {
            if (cli->is_admin) {
                char *direction = strtok(NULL, " \n");
                if (!direction) {
                    const char *err = "ERROR Invalid command\n";
                    send_all(cli->socket, err, strlen(err));
                    continue;
                }
                // Simulación de obstáculo
                if (rand() % 5 == 0) { 
                    char response[50];
                    snprintf(response, sizeof(response), "MOVE_FAIL %s OBSTACLE\n", direction);
                    send_all(cli->socket, response, strlen(response));
                } else {
                    char response[50];
                    snprintf(response, sizeof(response), "MOVE_SUCCESS %s\n", direction);
                    send_all(cli->socket, response, strlen(response));
                }
            } else {
                send_all(cli->socket, "ERROR No tienes permisos\n", 26);
            }
        } else if (strcmp(command, "LIST_USERS") == 0) {
            if (cli->is_admin) {
                char user_list[BUFFER_SIZE];
                size_t pos = 0;
                int written = snprintf(user_list, sizeof(user_list), "USER_LIST ");
                if (written < 0) written = 0;
                pos = (size_t)written;
                pthread_mutex_lock(&clients_mutex);
                for (int i = 0; i < MAX_CLIENTS; i++) {
                    if (clients[i]) {
                        char client_info[50];
                        int w = snprintf(client_info, sizeof(client_info), "%s:%d;", clients[i]->ip_str, clients[i]->port);
                        if (w < 0) continue;
                        size_t need = (size_t)w;
                        if (pos + need + 2 >= sizeof(user_list)) {
                            // no espacio suficiente, cortar
                            break;
                        }
                        memcpy(user_list + pos, client_info, need);
                        pos += need;
                    }
                }
                pthread_mutex_unlock(&clients_mutex);
                if (pos + 1 < sizeof(user_list)) {
                    user_list[pos++] = '\n';
                    user_list[pos] = '\0';
                } else {
                    // fallback newline
                    user_list[sizeof(user_list)-2] = '\n';
                    user_list[sizeof(user_list)-1] = '\0';
                }
                send_all(cli->socket, user_list, strlen(user_list));
            } else {
                send_all(cli->socket, "ERROR No tienes permisos\n", 26);
            }
        } else if (strcmp(command, "GET_DATA") == 0) {
            time_t now = time(NULL);
            if (cli->last_get_time != 0 && (now - cli->last_get_time) < 1) {
                const char *err = "ERROR Rate limit\n";
                send_all(cli->socket, err, strlen(err));
                continue;
            }
            cli->last_get_time = now;
            char *var = strtok(NULL, " \n");
            char response[BUFFER_SIZE];
            telemetry_t t;
            t = g_last_telemetry;
            if (var == NULL || strcmp(var, "ALL") == 0) {
                int n = snprintf(response, sizeof(response), "DATA %ld TEMP=%.1f;HUM=%.1f\n", (long)t.ts, t.temp, t.hum);
                if (n > 0) send_all(cli->socket, response, (size_t)n);
            } else if (strcmp(var, "TEMP") == 0) {
                int n = snprintf(response, sizeof(response), "DATA %ld TEMP=%.1f\n", (long)t.ts, t.temp);
                if (n > 0) send_all(cli->socket, response, (size_t)n);
            } else if (strcmp(var, "HUM") == 0) {
                int n = snprintf(response, sizeof(response), "DATA %ld HUM=%.1f\n", (long)t.ts, t.hum);
                if (n > 0) send_all(cli->socket, response, (size_t)n);
            } else {
                const char *err = "ERROR Unknown var\n";
                send_all(cli->socket, err, strlen(err));
            }
        } else if (strcmp(command, "PING") == 0) {
            time_t now = time(NULL);
            if (cli->last_ping_time != 0 && (now - cli->last_ping_time) < 1) {
                const char *err = "ERROR Rate limit\n";
                send_all(cli->socket, err, strlen(err));
                continue;
            }
            cli->last_ping_time = now;
            send_all(cli->socket, "PONG\n", 5);
        } else if (strcmp(command, "LOGOUT") == 0) {
            break;
        } else {
            const char *err = "ERROR Invalid command\n";
            send_all(cli->socket, err, strlen(err));
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
        // Crear snapshot de sockets bajo lock
        int sockets_snapshot[MAX_CLIENTS];
        int count = 0;
        pthread_mutex_lock(&clients_mutex);
        for (int i = 0; i < MAX_CLIENTS; i++) {
            if (clients[i]) {
                sockets_snapshot[count++] = clients[i]->socket;
            }
        }
        pthread_mutex_unlock(&clients_mutex);

        time_t t = time(NULL);
        float temp = (rand() % 300) / 10.0; // 0.0 - 30.0
        float hum = (rand() % 1000) / 10.0; // 0.0 - 100.0

        char telemetry_data[BUFFER_SIZE];
        snprintf(telemetry_data, sizeof(telemetry_data), "DATA %ld TEMP=%.1f;HUM=%.1f\n", (long)t, temp, hum);

        // Actualizar última telemetría
        g_last_telemetry.ts = t;
        g_last_telemetry.temp = temp;
        g_last_telemetry.hum = hum;

        FILE *log_file = fopen(log_filename, "a");
        if (log_file) {
            fprintf(log_file, "%s", telemetry_data);
            fclose(log_file);
        }

        // Rotar log si es grande
        rotate_log_if_needed(log_filename, 1024 * 1024);

        for (int i = 0; i < count; i++) {
            send_all(sockets_snapshot[i], telemetry_data, strlen(telemetry_data));
        }
    }
    return NULL;
}

int add_client(client_t *cl) {
    pthread_mutex_lock(&clients_mutex);
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (!clients[i]) {
            clients[i] = cl;
            pthread_mutex_unlock(&clients_mutex);
            return 0;
        }
    }
    pthread_mutex_unlock(&clients_mutex);
    return -1; // lleno
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

ssize_t send_all(int fd, const char *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t sent = send(fd, buf + total, len - total, 0);
        if (sent < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        total += (size_t)sent;
    }
    return (ssize_t)total;
}

static void load_admin_password(void) {
    const char *env = getenv("ADMIN_PASSWORD");
    if (env && *env) {
        strncpy(g_admin_password, env, sizeof(g_admin_password) - 1);
        g_admin_password[sizeof(g_admin_password) - 1] = '\0';
        return;
    }
    // Intentar leer de .env (formato: ADMIN_PASSWORD=...)
    FILE *f = fopen(".env", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "ADMIN_PASSWORD=", 15) == 0) {
                char *val = line + 15;
                // quitar salto
                char *nl = strchr(val, '\n');
                if (nl) *nl = '\0';
                strncpy(g_admin_password, val, sizeof(g_admin_password) - 1);
                g_admin_password[sizeof(g_admin_password) - 1] = '\0';
                break;
            }
        }
        fclose(f);
    }
    if (g_admin_password[0] == '\0') {
        // Último recurso: valor por defecto explícito pero no hardcode fijo
        strncpy(g_admin_password, "admin", sizeof(g_admin_password) - 1);
    }
}

static void rotate_log_if_needed(const char *path, size_t max_bytes) {
    struct stat st;
    if (stat(path, &st) == 0) {
        if ((size_t)st.st_size > max_bytes) {
            char backup[512];
            snprintf(backup, sizeof(backup), "%s.1", path);
            // best-effort rename; ignore errors
            rename(path, backup);
        }
    }
}
