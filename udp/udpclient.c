/*
 * UDP Latency Test Client in C
 * Optimized for low latency with busy polling
 * 
 * Compile: gcc -O3 -o udpclient udpclient.c
 * Run: ./udpclient 10.0.0.2
 *      ./udpclient 10.0.0.2 --busy-poll  (for busy polling mode)
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

// Calculate statistics
void print_stats(double *rtts, int count) {
    if (count == 0) return;
    
    double sum = 0, min = rtts[0], max = rtts[0];
    
    for (int i = 0; i < count; i++) {
        sum += rtts[i];
        if (rtts[i] < min) min = rtts[i];
        if (rtts[i] > max) max = rtts[i];
    }
    
    double avg = sum / count;
    
    // Calculate standard deviation
    double variance = 0;
    for (int i = 0; i < count; i++) {
        double diff = rtts[i] - avg;
        variance += diff * diff;
    }
    double stdev = sqrt(variance / count);
    
    printf("pps=%5d avg=%7.2fus min=%7.2fus max=%7.2fus stdev=%6.2fus\n",
           count, avg, min, max, stdev);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <server_ip> [--busy-poll]\n", argv[0]);
        printf("Example: %s 10.0.0.2\n", argv[0]);
        printf("         %s 10.0.0.2 --busy-poll\n", argv[0]);
        return 1;
    }
    
    char *server_ip = argv[1];
    int busy_poll = 0;
    
    if (argc > 2 && strcmp(argv[2], "--busy-poll") == 0) {
        busy_poll = 1;
    }
    
    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    char send_buffer[PAYLOAD_SIZE];
    char recv_buffer[BUFFER_SIZE];
    
    signal(SIGINT, signal_handler);
    
    // Create socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
    
    // Set socket buffer sizes
    int sndbuf = 2 * 1024 * 1024;
    int rcvbuf = 2 * 1024 * 1024;
    setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
    setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    
    // Bind to specific port
    memset(&client_addr, 0, sizeof(client_addr));
    client_addr.sin_family = AF_INET;
    client_addr.sin_addr.s_addr = INADDR_ANY;
    client_addr.sin_port = htons(CLIENT_PORT);
    
    if (bind(sockfd, (const struct sockaddr *)&client_addr, sizeof(client_addr)) < 0) {
        perror("bind failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    
    // Setup server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(SERVER_PORT);
    
    if (inet_pton(AF_INET, server_ip, &server_addr.sin_addr) <= 0) {
        printf("Invalid address: %s\n", server_ip);
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    
    // Set non-blocking for busy poll mode
    if (busy_poll) {
        int flags = fcntl(sockfd, F_GETFL, 0);
        fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);
        printf("[*] Busy polling mode enabled (will use 100%% CPU)\n");
    }
    
    // Initialize send buffer
    memset(send_buffer, 0, PAYLOAD_SIZE);
    
    printf("[*] Sending to %s:%d from port %d\n", server_ip, SERVER_PORT, CLIENT_PORT);
    printf("[*] Press Ctrl+C to stop\n\n");
    
    double *rtts = malloc(100000 * sizeof(double));  // Store up to 100k RTTs
    int rtt_count = 0;
    double last_print_time = get_time_us();
    
    while (running) {
        double t1 = get_time_us();
        
        // Send packet
        if (sendto(sockfd, send_buffer, PAYLOAD_SIZE, 0,
                  (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
            perror("sendto failed");
            break;
        }
        
        // Receive response
        ssize_t recv_len;
        if (busy_poll) {
            // Busy poll for response
            while (running) {
                recv_len = recvfrom(sockfd, recv_buffer, BUFFER_SIZE, 0, NULL, NULL);
                if (recv_len > 0) break;
                if (recv_len < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
                    perror("recvfrom failed");
                    running = 0;
                    break;
                }
            }
        } else {
            // Blocking receive
            recv_len = recvfrom(sockfd, recv_buffer, BUFFER_SIZE, 0, NULL, NULL);
            if (recv_len < 0) {
                if (errno == EINTR) continue;
                perror("recvfrom failed");
                break;
            }
        }
        
        if (!running) break;
        
        double t2 = get_time_us();
        double rtt = t2 - t1;
        
        rtts[rtt_count++] = rtt;
        
        // Print stats every second
        if (t2 - last_print_time >= 1000000.0) {  // 1 second
            print_stats(rtts, rtt_count);
            rtt_count = 0;
            last_print_time = t2;
        }
    }
    
    printf("\n[*] Stopped\n");
    
    free(rtts);
    close(sockfd);
    return 0;
}
