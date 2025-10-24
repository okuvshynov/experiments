/*
 * UDP Echo Server in C
 * Optimized for low latency
 * 
 * Compile: gcc -O3 -o udpserver udpserver.c
 * Run: ./udpserver
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

#define PORT 4321
#define BUFFER_SIZE 65536

volatile sig_atomic_t running = 1;

void signal_handler(int sig) {
    running = 0;
}

int main(int argc, char *argv[]) {
    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);
    char buffer[BUFFER_SIZE];
    ssize_t recv_len;
    unsigned long packet_count = 0;
    
    // Handle Ctrl+C gracefully
    signal(SIGINT, signal_handler);
    
    // Create UDP socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
    
    // Set socket buffer sizes
    int sndbuf = 2 * 1024 * 1024;  // 2MB
    int rcvbuf = 2 * 1024 * 1024;
    if (setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf)) < 0) {
        perror("setsockopt SO_SNDBUF");
    }
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf)) < 0) {
        perror("setsockopt SO_RCVBUF");
    }
    
    // Bind socket
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);
    
    if (bind(sockfd, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    
    printf("[*] UDP server listening on port %d\n", PORT);
    printf("[*] Press Ctrl+C to stop\n\n");
    
    // Main echo loop
    while (running) {
        recv_len = recvfrom(sockfd, buffer, BUFFER_SIZE, 0,
                           (struct sockaddr *)&client_addr, &client_len);
        
        if (recv_len < 0) {
            if (errno == EINTR) continue;  // Interrupted by signal
            perror("recvfrom failed");
            break;
        }
        
        // Echo back immediately
        if (sendto(sockfd, buffer, recv_len, 0,
                  (struct sockaddr *)&client_addr, client_len) < 0) {
            perror("sendto failed");
        }
        
        packet_count++;
        
        if (packet_count % 10000 == 0) {
            printf("[*] Echoed %lu packets\n", packet_count);
        }
    }
    
    printf("\n[*] Server stopped. Total packets: %lu\n", packet_count);
    close(sockfd);
    return 0;
}
