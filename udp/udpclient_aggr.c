/*
 * UDP Latency Test Client with Link Aggregation
 *
 * Features:
 * - Variable packet sizes (configurable, default 20KB for 5120 floats)
 * - Application-level link aggregation across 2 network interfaces
 * - Busy polling for minimal latency
 * - macOS compatible (no Linux-specific optimizations)
 *
 * Compile: gcc -O3 -march=native -o udpclient_aggr udpclient_aggr.c -lm
 * Run: sudo ./udpclient_aggr 10.0.0.2 10.0.1.2 [packet_size_bytes]
 *      sudo ./udpclient_aggr 10.0.0.2 10.0.1.2 20480
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
#include <math.h>
#include <pthread.h>

#define DEFAULT_PAYLOAD_SIZE (5120 * sizeof(float))  // 20480 bytes
#define MAX_PAYLOAD_SIZE (64 * 1024)                  // 64KB max
#define CLIENT_PORT_BASE 65400
#define SERVER_PORT 4321
#define NUM_LINKS 2

// Branch prediction hints
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

volatile sig_atomic_t running = 1;

// Per-link state
typedef struct {
    int sockfd;
    int link_id;
    struct sockaddr_in server_addr;
    unsigned long packets_sent;
    unsigned long packets_received;
    unsigned long bytes_sent;
    unsigned long bytes_received;
} link_t;

// Statistics structure
typedef struct {
    double sum;
    double min;
    double max;
    double sum_sq;
    int count;
} stats_t;

void signal_handler(int sig) {
    running = 0;
}

// High precision timer
static inline double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000.0 + ts.tv_nsec / 1000.0;
}

// Statistics functions
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

static inline void stats_print(const char *label, stats_t *s) {
    if (s->count == 0) return;

    double avg = s->sum / s->count;
    double variance = (s->sum_sq / s->count) - (avg * avg);
    double stdev = sqrt(variance > 0 ? variance : 0);

    printf("%s: pps=%5d avg=%7.2fus min=%7.2fus max=%7.2fus stdev=%6.2fus\n",
           label, s->count, avg, s->min, s->max, stdev);
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

// Initialize a network link
int init_link(link_t *link, const char *server_ip, int link_id) {
    link->link_id = link_id;
    link->packets_sent = 0;
    link->packets_received = 0;
    link->bytes_sent = 0;
    link->bytes_received = 0;

    // Create socket
    link->sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (unlikely(link->sockfd < 0)) {
        perror("socket creation failed");
        return -1;
    }

    // Set large socket buffers
    int bufsize = 4 * 1024 * 1024;  // 4MB
    setsockopt(link->sockfd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
    setsockopt(link->sockfd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));

    // Enable address reuse
    int reuse = 1;
    setsockopt(link->sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    // Bind to specific port
    struct sockaddr_in client_addr;
    memset(&client_addr, 0, sizeof(client_addr));
    client_addr.sin_family = AF_INET;
    client_addr.sin_addr.s_addr = INADDR_ANY;
    client_addr.sin_port = htons(CLIENT_PORT_BASE + link_id);

    if (unlikely(bind(link->sockfd, (const struct sockaddr *)&client_addr, sizeof(client_addr)) < 0)) {
        perror("bind failed");
        close(link->sockfd);
        return -1;
    }

    // Setup server address
    memset(&link->server_addr, 0, sizeof(link->server_addr));
    link->server_addr.sin_family = AF_INET;
    link->server_addr.sin_port = htons(SERVER_PORT + link_id);

    if (inet_pton(AF_INET, server_ip, &link->server_addr.sin_addr) <= 0) {
        printf("Invalid address: %s\n", server_ip);
        close(link->sockfd);
        return -1;
    }

    // Use connect() for slight performance improvement
    if (connect(link->sockfd, (struct sockaddr *)&link->server_addr, sizeof(link->server_addr)) < 0) {
        perror("connect failed");
        close(link->sockfd);
        return -1;
    }

    // Set non-blocking for busy polling
    int flags = fcntl(link->sockfd, F_GETFL, 0);
    fcntl(link->sockfd, F_SETFL, flags | O_NONBLOCK);

    printf("[+] Link %d initialized: %s:%d -> local port %d\n",
           link_id, server_ip, SERVER_PORT + link_id, CLIENT_PORT_BASE + link_id);

    return 0;
}

// Round-robin link selection
static inline int select_link(unsigned long counter) {
    return counter % NUM_LINKS;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: sudo %s <server_ip1> <server_ip2> [packet_size_bytes]\n", argv[0]);
        printf("Example: sudo %s 10.0.0.2 10.0.1.2\n", argv[0]);
        printf("         sudo %s 10.0.0.2 10.0.1.2 20480\n", argv[0]);
        printf("\nDefault packet size: %zu bytes (5120 floats)\n", DEFAULT_PAYLOAD_SIZE);
        return 1;
    }

    char *server_ip1 = argv[1];
    char *server_ip2 = argv[2];

    size_t payload_size = DEFAULT_PAYLOAD_SIZE;
    if (argc > 3) {
        payload_size = atoi(argv[3]);
        if (payload_size == 0 || payload_size > MAX_PAYLOAD_SIZE) {
            printf("Invalid packet size. Must be between 1 and %d bytes\n", MAX_PAYLOAD_SIZE);
            return 1;
        }
    }

    signal(SIGINT, signal_handler);
    set_high_priority();

    // Initialize links
    link_t links[NUM_LINKS];
    if (init_link(&links[0], server_ip1, 0) < 0) {
        return 1;
    }
    if (init_link(&links[1], server_ip2, 1) < 0) {
        close(links[0].sockfd);
        return 1;
    }

    printf("\n[*] Link aggregation active with %d links\n", NUM_LINKS);
    printf("[*] Packet size: %zu bytes (%.2f KB)\n", payload_size, payload_size / 1024.0);
    printf("[*] BUSY POLLING MODE - will use 100%% of one CPU core\n");
    printf("[*] Press Ctrl+C to stop\n\n");

    // Allocate aligned buffers
    char *send_buffer = aligned_alloc(64, payload_size);
    char *recv_buffer = aligned_alloc(64, MAX_PAYLOAD_SIZE);
    if (!send_buffer || !recv_buffer) {
        perror("aligned_alloc failed");
        return 1;
    }
    memset(send_buffer, 0, payload_size);

    stats_t overall_stats;
    stats_t link_stats[NUM_LINKS];
    stats_init(&overall_stats);
    for (int i = 0; i < NUM_LINKS; i++) {
        stats_init(&link_stats[i]);
    }

    double last_print_time = get_time_us();
    unsigned long total_packets = 0;

    // Main loop
    while (likely(running)) {
        // Select link using round-robin
        int link_idx = select_link(total_packets);
        link_t *link = &links[link_idx];

        double t1 = get_time_us();

        // Send packet
        ssize_t sent = send(link->sockfd, send_buffer, payload_size, 0);
        if (unlikely(sent < 0)) {
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                perror("send failed");
                break;
            }
            continue;
        }

        link->packets_sent++;
        link->bytes_sent += sent;

        // Busy poll for response
        ssize_t recv_len;
        while (likely(running)) {
            recv_len = recv(link->sockfd, recv_buffer, MAX_PAYLOAD_SIZE, 0);
            if (likely(recv_len > 0)) break;
            if (unlikely(recv_len < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
                perror("recv failed");
                running = 0;
                break;
            }
        }

        if (unlikely(!running)) break;

        link->packets_received++;
        link->bytes_received += recv_len;

        double t2 = get_time_us();
        double rtt = t2 - t1;

        stats_add(&overall_stats, rtt);
        stats_add(&link_stats[link_idx], rtt);

        total_packets++;

        // Print stats every second
        if (unlikely(t2 - last_print_time >= 1000000.0)) {
            stats_print("Overall", &overall_stats);
            stats_print("Link 0 ", &link_stats[0]);
            stats_print("Link 1 ", &link_stats[1]);
            printf("\n");

            // Reset stats and counters
            stats_init(&overall_stats);
            for (int i = 0; i < NUM_LINKS; i++) {
                stats_init(&link_stats[i]);
                links[i].bytes_sent = 0;
                links[i].bytes_received = 0;
            }
            last_print_time = t2;
        }
    }

    printf("\n[*] Stopped\n");
    printf("[*] Total packets sent: %lu\n", total_packets);
    for (int i = 0; i < NUM_LINKS; i++) {
        printf("[*] Link %d: sent=%lu received=%lu\n",
               i, links[i].packets_sent, links[i].packets_received);
    }

    // Cleanup
    free(send_buffer);
    free(recv_buffer);
    for (int i = 0; i < NUM_LINKS; i++) {
        close(links[i].sockfd);
    }

    return 0;
}
