/*
 * Linux-Optimized UDP Latency Test Client
 * Uses all available Linux features for minimum latency
 *
 * Features:
 * - SO_BUSY_POLL kernel busy polling
 * - Application-level busy polling
 * - CPU affinity pinning
 * - Real-time scheduling (SCHED_FIFO)
 * - High-resolution timing
 * - Socket timestamping
 *
 * Compile: gcc -O3 -march=native -o client_opt_linux client_opt_linux.c -lm -lpthread
 * Run: sudo ./client_opt_linux <server_ip> [options]
 *      --busy-poll <us>  Enable kernel busy polling (e.g., 50)
 *      --app-poll        Enable application-level busy polling
 *      --cpu <n>         Pin to specific CPU core
 *      --rt              Use real-time scheduling (SCHED_FIFO)
 *      --port <n>        Server port (default 4321)
 *      --size <n>        Payload size (default 32)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>
#include <sched.h>
#include <pthread.h>

#define DEFAULT_PORT 4321
#define DEFAULT_PAYLOAD 32
#define CLIENT_PORT 65400
#define BUFFER_SIZE 1024

#ifndef SO_BUSY_POLL
#define SO_BUSY_POLL 46
#endif

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

volatile sig_atomic_t running = 1;

void signal_handler(int sig) {
    running = 0;
}

// High precision timer
static inline double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000.0 + ts.tv_nsec / 1000.0;
}

// Statistics structure
typedef struct {
    double sum;
    double min;
    double max;
    double sum_sq;
    int count;
} stats_t;

static inline void stats_init(stats_t *s) {
    s->sum = 0;
    s->min = 1e9;
    s->max = 0;
    s->sum_sq = 0;
    s->count = 0;
}

static inline void stats_add(stats_t *s, double value) {
    s->sum += value;
    s->sum_sq += value * value;
    if (value < s->min) s->min = value;
    if (value > s->max) s->max = value;
    s->count++;
}

static inline void stats_print(stats_t *s) {
    if (s->count == 0) return;
    
    double avg = s->sum / s->count;
    double variance = (s->sum_sq / s->count) - (avg * avg);
    double stdev = sqrt(variance > 0 ? variance : 0);
    
    printf("pps=%5d avg=%7.2fus min=%7.2fus max=%7.2fus stdev=%6.2fus\n",
           s->count, avg, s->min, s->max, stdev);
}

// Set CPU affinity
int set_cpu_affinity(int cpu) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) < 0) {
        perror("sched_setaffinity");
        return -1;
    }
    
    printf("[+] Pinned to CPU %d\n", cpu);
    return 0;
}

// Set real-time scheduling
int set_realtime_priority() {
    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    
    if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
        perror("sched_setscheduler");
        printf("[!] Failed to set real-time priority (need sudo)\n");
        return -1;
    }
    
    printf("[+] Real-time scheduling enabled (priority %d)\n", param.sched_priority);
    return 0;
}

void print_usage(const char *prog) {
    printf("Usage: sudo %s <server_ip> [options]\n", prog);
    printf("Options:\n");
    printf("  --busy-poll <us>  Enable kernel busy polling (e.g., 50)\n");
    printf("  --app-poll        Enable application-level busy polling\n");
    printf("  --cpu <n>         Pin to specific CPU core\n");
    printf("  --rt              Use real-time scheduling (SCHED_FIFO)\n");
    printf("  --port <n>        Server port (default %d)\n", DEFAULT_PORT);
    printf("  --size <n>        Payload size (default %d)\n", DEFAULT_PAYLOAD);
    printf("\nExamples:\n");
    printf("  sudo %s 10.0.0.2 --busy-poll 50 --cpu 3 --rt\n", prog);
    printf("  sudo %s 10.0.0.2 --app-poll --cpu 5\n", prog);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    char *server_ip = argv[1];
    int port = DEFAULT_PORT;
    int payload_size = DEFAULT_PAYLOAD;
    int busy_poll_us = 0;
    int app_poll = 0;
    int cpu_pin = -1;
    int use_rt = 0;
    
    // Parse arguments
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--busy-poll") == 0 && i + 1 < argc) {
            busy_poll_us = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--app-poll") == 0) {
            app_poll = 1;
        } else if (strcmp(argv[i], "--cpu") == 0 && i + 1 < argc) {
            cpu_pin = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--rt") == 0) {
            use_rt = 1;
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            payload_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    signal(SIGINT, signal_handler);
    
    printf("=================================================\n");
    printf("Linux-Optimized UDP Latency Client\n");
    printf("=================================================\n");
    
    // Set CPU affinity if requested
    if (cpu_pin >= 0) {
        set_cpu_affinity(cpu_pin);
    }
    
    // Set real-time scheduling if requested
    if (use_rt) {
        set_realtime_priority();
    }
    
    // Create socket
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (unlikely(sockfd < 0)) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
    
    // Set large socket buffers
    int bufsize = 4 * 1024 * 1024;  // 4MB
    setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
    setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));
    
    // Enable SO_BUSY_POLL if requested
    if (busy_poll_us > 0) {
        if (setsockopt(sockfd, SOL_SOCKET, SO_BUSY_POLL, &busy_poll_us, sizeof(busy_poll_us)) < 0) {
            perror("setsockopt SO_BUSY_POLL");
            printf("[!] Warning: SO_BUSY_POLL not supported\n");
        } else {
            printf("[+] Kernel busy polling enabled (%d us)\n", busy_poll_us);
        }
    }
    
    // Bind to specific port
    struct sockaddr_in client_addr;
    memset(&client_addr, 0, sizeof(client_addr));
    client_addr.sin_family = AF_INET;
    client_addr.sin_addr.s_addr = INADDR_ANY;
    client_addr.sin_port = htons(CLIENT_PORT);
    
    if (unlikely(bind(sockfd, (const struct sockaddr *)&client_addr, sizeof(client_addr)) < 0)) {
        perror("bind failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    
    // Setup server address
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    
    if (inet_pton(AF_INET, server_ip, &server_addr.sin_addr) <= 0) {
        printf("Invalid address: %s\n", server_ip);
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    
    // Connect socket for slight performance improvement
    if (connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    
    // Set non-blocking for app-level busy polling
    if (app_poll) {
        int flags = fcntl(sockfd, F_GETFL, 0);
        fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);
        printf("[+] Application-level busy polling enabled (will use 100%% CPU)\n");
    }
    
    printf("[*] Sending to %s:%d (payload: %d bytes)\n", server_ip, port, payload_size);
    printf("[*] Press Ctrl+C to stop\n");
    printf("=================================================\n\n");
    
    // Allocate aligned buffers
    char *send_buffer = aligned_alloc(64, payload_size);
    char *recv_buffer = aligned_alloc(64, BUFFER_SIZE);
    memset(send_buffer, 0, payload_size);
    
    if (!send_buffer || !recv_buffer) {
        fprintf(stderr, "Failed to allocate buffers\n");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    
    stats_t stats;
    stats_init(&stats);
    double last_print_time = get_time_us();
    
    // Main loop
    while (likely(running)) {
        double t1 = get_time_us();
        
        // Send packet
        ssize_t sent = send(sockfd, send_buffer, payload_size, 0);
        if (unlikely(sent < 0)) {
            perror("send failed");
            break;
        }
        
        // Receive response
        ssize_t recv_len;
        if (app_poll) {
            // Application-level busy poll
            while (likely(running)) {
                recv_len = recv(sockfd, recv_buffer, BUFFER_SIZE, 0);
                if (likely(recv_len > 0)) break;
                if (unlikely(recv_len < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
                    perror("recv failed");
                    running = 0;
                    break;
                }
            }
        } else {
            // Blocking receive (or kernel busy poll)
            recv_len = recv(sockfd, recv_buffer, BUFFER_SIZE, 0);
            if (recv_len < 0) {
                if (errno == EINTR) continue;
                perror("recv failed");
                break;
            }
        }
        
        if (unlikely(!running)) break;
        
        double t2 = get_time_us();
        stats_add(&stats, t2 - t1);
        
        // Print stats every second
        double current_time = t2;
        if (unlikely(current_time - last_print_time >= 1000000.0)) {
            stats_print(&stats);
            stats_init(&stats);
            last_print_time = current_time;
        }
    }
    
    printf("\n[*] Stopped\n");
    
    free(send_buffer);
    free(recv_buffer);
    close(sockfd);
    return 0;
}