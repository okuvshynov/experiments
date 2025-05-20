// Sample C file for testing

#include <stdio.h>
#include <stdlib.h>

// Global variable
const double PI = 3.14159;
int counter = 0;

// Structure definition
struct Point {
    double x;
    double y;
};

// Function to add two numbers
double add(double a, double b) {
    return a + b;
}

// Function to calculate distance between two points
double distance(struct Point p1, struct Point p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    return sqrt(dx*dx + dy*dy);
}

// Function to square a number
double square(double x) {
    return x * x;
}

// Main function
int main(int argc, char *argv[]) {
    struct Point p1 = {0.0, 0.0};
    struct Point p2 = {3.0, 4.0};
    
    printf("Distance between points: %f\n", distance(p1, p2));
    printf("Square of 5: %f\n", square(5.0));
    
    return 0;
}