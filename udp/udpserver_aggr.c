/*
 * UDP Echo Server with Link Aggregation
 *
 * Features:
 * - Variable packet sizes (up to 64KB)
 * - Handles multiple network interfaces (2 separate ports)
 * - Minimal latency with optimized echo
 * - macOS compatible
 *
 * Compile: gcc -O3 -march=native -o udpserver_aggr udpserver_aggr.c -lm
 * Run: sudo ./udpserver_aggr
 */

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
#include <poll.h>

#define SERVER_PORT_BASE 4321
#define MAX_BUFFER_SIZE (64 * 1024)
#define NUM_LINKS 2

volatile sig_atomic_t running = 1;

// Per-link state
typedef struct {
    int sockfd;
    int link_id;
    int port;
    unsigned long packets_received;
    unsigned long packets_sent;
    unsigned long bytes_received;
    unsigned long bytes_sent;
} link_t;

void signal_handler(int sig) {
    running = 0;
}

// High precision timer
static inline double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000.0 + ts.tv_nsec / 1000.0;
}

// Set process to highest priority
void set_high_priority() {
    errno = 0;
    if (nice(-20) == -1 && errno != 0) {
        printf("[!] Warning: Could not set high priority (run with sudo)\n");
    } else {
        printf("[+] Process priority set to highest\n");
    }
}

// Initialize a server link
int init_server_link(link_t *link, int link_id) {
    link->link_id = link_id;
    link->port = SERVER_PORT_BASE + link_id;
    link->packets_received = 0;
    link->packets_sent = 0;
    link->bytes_received = 0;
    link->bytes_sent = 0;

    // Create UDP socket
    link->sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (link->sockfd < 0) {
        perror("socket creation failed");
        return -1;
    }

    // Set socket buffer sizes
    int sndbuf = 4 * 1024 * 1024;  // 4MB
    int rcvbuf = 4 * 1024 * 1024;
    if (setsockopt(link->sockfd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf)) < 0) {
        perror("setsockopt SO_SNDBUF");
    }
    if (setsockopt(link->sockfd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf)) < 0) {
        perror("setsockopt SO_RCVBUF");
    }

    // Enable address reuse
    int reuse = 1;
    setsockopt(link->sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    // Bind socket
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(link->port);

    if (bind(link->sockfd, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(link->sockfd);
        return -1;
    }

    printf("[+] Link %d listening on port %d\n", link_id, link->port);
    return 0;
}

int main(int argc, char *argv[]) {
    signal(SIGINT, signal_handler);
    set_high_priority();

    // Initialize links
    link_t links[NUM_LINKS];
    for (int i = 0; i < NUM_LINKS; i++) {
        if (init_server_link(&links[i], i) < 0) {
            // Cleanup previously initialized links
            for (int j = 0; j < i; j++) {
                close(links[j].sockfd);
            }
            return 1;
        }
    }

    printf("\n[*] UDP server with %d links ready\n", NUM_LINKS);
    printf("[*] Press Ctrl+C to stop\n\n");

    // Setup poll structures for multiplexing
    struct pollfd fds[NUM_LINKS];
    for (int i = 0; i < NUM_LINKS; i++) {
        fds[i].fd = links[i].sockfd;
        fds[i].events = POLLIN;
    }

    // Allocate buffer
    char *buffer = aligned_alloc(64, MAX_BUFFER_SIZE);
    if (!buffer) {
        perror("aligned_alloc failed");
        for (int i = 0; i < NUM_LINKS; i++) {
            close(links[i].sockfd);
        }
        return 1;
    }

    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);

    double last_print_time = get_time_us();
    unsigned long total_packets = 0;

    // Main echo loop
    while (running) {
        // Poll all sockets with minimal timeout
        int ready = poll(fds, NUM_LINKS, 100);  // 100ms timeout

        if (ready < 0) {
            if (errno == EINTR) continue;
            perror("poll failed");
            break;
        }

        if (ready == 0) {
            // Timeout - check if we should print stats
            double current_time = get_time_us();
            if (current_time - last_print_time >= 1000000.0) {
                if (total_packets > 0) {
                    printf("[*] Link stats:\n");
                    for (int i = 0; i < NUM_LINKS; i++) {
                        printf("    Link %d (port %d): rx=%lu tx=%lu bytes_rx=%lu bytes_tx=%lu\n",
                               i, links[i].port,
                               links[i].packets_received, links[i].packets_sent,
                               links[i].bytes_received, links[i].bytes_sent);
                    }

                    // Calculate total bandwidth
                    double elapsed = (current_time - last_print_time) / 1000000.0;
                    unsigned long total_bytes = 0;
                    for (int i = 0; i < NUM_LINKS; i++) {
                        total_bytes += links[i].bytes_received + links[i].bytes_sent;
                    }
                    double bandwidth_mbps = (total_bytes * 8) / (elapsed * 1000000.0);
                    printf("    Bandwidth: %.2f Mbps (%.2f MB/s)\n\n",
                           bandwidth_mbps, bandwidth_mbps / 8.0);

                    // Reset byte counters
                    for (int i = 0; i < NUM_LINKS; i++) {
                        links[i].bytes_received = 0;
                        links[i].bytes_sent = 0;
                    }
                }
                last_print_time = current_time;
            }
            continue;
        }

        // Check which sockets have data
        for (int i = 0; i < NUM_LINKS; i++) {
            if (fds[i].revents & POLLIN) {
                ssize_t recv_len = recvfrom(links[i].sockfd, buffer, MAX_BUFFER_SIZE, 0,
                                           (struct sockaddr *)&client_addr, &client_len);

                if (recv_len < 0) {
                    if (errno == EINTR) continue;
                    perror("recvfrom failed");
                    continue;
                }

                links[i].packets_received++;
                links[i].bytes_received += recv_len;

                // Echo back immediately
                ssize_t sent = sendto(links[i].sockfd, buffer, recv_len, 0,
                                     (struct sockaddr *)&client_addr, client_len);

                if (sent < 0) {
                    perror("sendto failed");
                } else {
                    links[i].packets_sent++;
                    links[i].bytes_sent += sent;
                }

                total_packets++;
            }

            if (fds[i].revents & (POLLERR | POLLHUP | POLLNVAL)) {
                printf("[!] Error on link %d socket\n", i);
                running = 0;
                break;
            }
        }
    }

    printf("\n[*] Server stopped\n");
    printf("[*] Final statistics:\n");
    for (int i = 0; i < NUM_LINKS; i++) {
        printf("    Link %d (port %d): received=%lu sent=%lu\n",
               i, links[i].port,
               links[i].packets_received, links[i].packets_sent);
    }

    // Cleanup
    free(buffer);
    for (int i = 0; i < NUM_LINKS; i++) {
        close(links[i].sockfd);
    }

    return 0;
}
