/*
 * Linux-Optimized UDP Echo Server
 * Uses Linux-specific features for minimum latency
 *
 * Features:
 * - SO_BUSY_POLL kernel busy polling
 * - CPU affinity pinning
 * - Real-time scheduling (SCHED_FIFO)
 * - Large socket buffers
 * - Timestamping support
 *
 * Compile: gcc -O3 -march=native -o server_linux server_linux.c -lpthread
 * Run: sudo ./server_linux [options]
 *      --busy-poll <us>  Enable kernel busy polling (e.g., 50)
 *      --cpu <n>         Pin to specific CPU core
 *      --rt              Use real-time scheduling (SCHED_FIFO)
 *      --port <n>        Listen port (default 4321)
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
#include <sched.h>
#include <pthread.h>

#define DEFAULT_PORT 4321
#define BUFFER_SIZE 65536

#ifndef SO_BUSY_POLL
#define SO_BUSY_POLL 46
#endif

volatile sig_atomic_t running = 1;

void signal_handler(int sig) {
    running = 0;
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
    printf("Usage: sudo %s [options]\n", prog);
    printf("Options:\n");
    printf("  --busy-poll <us>  Enable kernel busy polling (e.g., 50)\n");
    printf("  --cpu <n>         Pin to specific CPU core\n");
    printf("  --rt              Use real-time scheduling (SCHED_FIFO)\n");
    printf("  --port <n>        Listen port (default %d)\n", DEFAULT_PORT);
    printf("\nExample:\n");
    printf("  sudo %s --busy-poll 50 --cpu 3 --rt\n", prog);
}

int main(int argc, char *argv[]) {
    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);
    char buffer[BUFFER_SIZE];
    ssize_t recv_len;
    unsigned long packet_count = 0;
    
    int port = DEFAULT_PORT;
    int busy_poll_us = 0;
    int cpu_pin = -1;
    int use_rt = 0;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--busy-poll") == 0 && i + 1 < argc) {
            busy_poll_us = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpu") == 0 && i + 1 < argc) {
            cpu_pin = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--rt") == 0) {
            use_rt = 1;
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    signal(SIGINT, signal_handler);
    
    printf("=================================================\n");
    printf("Linux-Optimized UDP Echo Server\n");
    printf("=================================================\n");
    
    // Set CPU affinity if requested
    if (cpu_pin >= 0) {
        set_cpu_affinity(cpu_pin);
    }
    
    // Set real-time scheduling if requested
    if (use_rt) {
        set_realtime_priority();
    }
    
    // Create UDP socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
    
    // Set large socket buffers
    int sndbuf = 4 * 1024 * 1024;  // 4MB
    int rcvbuf = 4 * 1024 * 1024;
    if (setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf)) < 0) {
        perror("setsockopt SO_SNDBUF");
    }
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf)) < 0) {
        perror("setsockopt SO_RCVBUF");
    }
    
    // Enable SO_BUSY_POLL if requested (Linux 3.11+)
    if (busy_poll_us > 0) {
        if (setsockopt(sockfd, SOL_SOCKET, SO_BUSY_POLL, &busy_poll_us, sizeof(busy_poll_us)) < 0) {
            perror("setsockopt SO_BUSY_POLL");
            printf("[!] Warning: SO_BUSY_POLL not supported on this kernel\n");
        } else {
            printf("[+] Kernel busy polling enabled (%d us)\n", busy_poll_us);
        }
    }
    
    // Bind socket
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);
    
    if (bind(sockfd, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    
    printf("[*] Listening on port %d\n", port);
    printf("[*] Press Ctrl+C to stop\n");
    printf("=================================================\n\n");
    
    // Main echo loop
    while (running) {
        recv_len = recvfrom(sockfd, buffer, BUFFER_SIZE, 0,
                           (struct sockaddr *)&client_addr, &client_len);
        
        if (recv_len < 0) {
            if (errno == EINTR) continue;
            perror("recvfrom failed");
            break;
        }
        
        // Echo back immediately
        if (sendto(sockfd, buffer, recv_len, 0,
                  (struct sockaddr *)&client_addr, client_len) < 0) {
            perror("sendto failed");
        }
        
        packet_count++;
        
        if (packet_count % 100000 == 0) {
            printf("[*] Echoed %lu packets\n", packet_count);
        }
    }
    
    printf("\n[*] Server stopped. Total packets: %lu\n", packet_count);
    close(sockfd);
    return 0;
}