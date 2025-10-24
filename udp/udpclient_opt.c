/*
 * Maximally Optimized UDP Latency Test Client
 * 
 * Additional optimizations:
 * - Pre-connected socket
 * - Minimal system calls
 * - Cache-friendly data structures
 * - Compiler hints for branch prediction
 * 
 * Compile: gcc -O3 -march=native -o udpclient_optimized udpclient_optimized.c
 * Run: sudo ./udpclient_optimized 10.0.0.2
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

#define PAYLOAD_SIZE 32
#define BUFFER_SIZE 1024
#define CLIENT_PORT 65400
#define SERVER_PORT 4321
#define STATS_WINDOW 10000

// Branch prediction hints (GCC)
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

volatile sig_atomic_t running = 1;

void signal_handler(int sig) {
    running = 0;
}

// Inline high precision timer
static inline double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000.0 + ts.tv_nsec / 1000.0;
}

// Fast statistics calculation
typedef struct {
    double sum;
    double min;
    double max;
    double sum_sq;  // For variance
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
    double stdev = sqrt(variance);
    
    printf("pps=%5d avg=%7.2fus min=%7.2fus max=%7.2fus stdev=%6.2fus\n",
           s->count, avg, s->min, s->max, stdev);
}

// Set process to highest priority (requires sudo)
void set_high_priority() {
    // On macOS, just use nice() - no RLIMIT_RTPRIO available
    errno = 0;
    if (nice(-20) == -1 && errno != 0) {
        printf("[!] Warning: Could not set high priority (run with sudo)\n");
    } else {
        printf("[+] Process priority set to highest\n");
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: sudo %s <server_ip>\n", argv[0]);
        printf("Example: sudo %s 10.0.0.2\n", argv[0]);
        return 1;
    }
    
    char *server_ip = argv[1];
    
    signal(SIGINT, signal_handler);
    set_high_priority();
    
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
    
    // Enable address reuse
    int reuse = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    
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
    server_addr.sin_port = htons(SERVER_PORT);
    
    if (inet_pton(AF_INET, server_ip, &server_addr.sin_addr) <= 0) {
        printf("Invalid address: %s\n", server_ip);
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    
    // Use connect() for slight performance improvement
    if (connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    
    // Set non-blocking for busy polling
    int flags = fcntl(sockfd, F_GETFL, 0);
    fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);
    
    printf("[*] Sending to %s:%d (BUSY POLLING MODE)\n", server_ip, SERVER_PORT);
    printf("[*] This will use 100%% of one CPU core\n");
    printf("[*] Press Ctrl+C to stop\n\n");
    
    // Pre-allocate buffers
    char send_buffer[PAYLOAD_SIZE] __attribute__((aligned(64)));
    char recv_buffer[BUFFER_SIZE] __attribute__((aligned(64)));
    memset(send_buffer, 0, PAYLOAD_SIZE);
    
    stats_t stats;
    stats_init(&stats);
    double last_print_time = get_time_us();
    
    // Main loop
    while (likely(running)) {
        double t1 = get_time_us();
        
        // Send packet (using send instead of sendto since we're connected)
        ssize_t sent = send(sockfd, send_buffer, PAYLOAD_SIZE, 0);
        if (unlikely(sent < 0)) {
            perror("send failed");
            break;
        }
        
        // Busy poll for response
        ssize_t recv_len;
        while (likely(running)) {
            recv_len = recv(sockfd, recv_buffer, BUFFER_SIZE, 0);
            if (likely(recv_len > 0)) break;
            if (unlikely(recv_len < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
                perror("recv failed");
                running = 0;
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
    close(sockfd);
    return 0;
}
