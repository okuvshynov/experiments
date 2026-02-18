// Topswops: Find the maximum number of reversals for permutations of 1..n
// Uses Knuth's backward search algorithm (TAOCP 7.2.1.2, Exercise 107)
//
// The key insight: instead of simulating forward from all n! permutations,
// work backward from the terminal state (card 1 on top). A reversal is its
// own inverse, so we can enumerate all permutations reachable in exactly
// k backward steps, finding the maximum depth.
//
// Pruning: if current_depth + best[s] <= best[n], we can't beat the current
// record, so we prune the branch. best[s] is the known optimum for smaller s,
// computed incrementally.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>

static constexpr int MAX_N = 20;

// Partial permutation: -1 means unfilled
struct Deck {
    int8_t c[MAX_N];
};

static int best[MAX_N + 1];  // best[k] = f(k), the maximum topswops for size k
static int current_n;
static int depth;

// Try all backward reversals from the current partial permutation.
//   deck:       current partial permutation (card values 0-indexed: 0..n-1)
//   placed:     bitmask of which card values have been assigned
//   max_swap:   highest position to try swapping (limits search)
static void tryswaps(const Deck& deck, uint32_t placed, int max_swap) {
    if (depth > best[current_n])
        best[current_n] = depth;

    // Find the highest feasible swap position, pruning as we go.
    // We iterate downward from max_swap: this lets us prune using
    // best[s] which decreases as s decreases.
    int s = max_swap;
    while (s >= 1) {
        // Position s can be swapped if:
        //   - deck[s] == s (card s is already there, meaning card s+1 in 1-indexed), OR
        //   - deck[s] == -1 and card s is not yet placed
        bool feasible = (deck.c[s] == s) ||
                        (deck.c[s] == -1 && !(placed & (1u << s)));

        if (feasible) {
            // Check if this branch can potentially improve the best
            if (depth + best[s] >= best[current_n] || deck.c[s] == -1)
                break;
        }

        // Prune: even the best possible continuation from position s can't beat current best
        if (depth + best[s] <= best[current_n])
            return;

        s--;
    }

    if (s < 1)
        return;

    depth++;

    // Try each reversal length i (reverse positions 0..i)
    for (int i = 1; i <= s; i++) {
        // For a backward reversal of length i+1:
        //   The card at position i will move to position 0, and its value
        //   must equal i (0-indexed) so that it represents the "top card"
        //   that triggered this reversal in the forward direction.
        //
        // Check feasibility for position i:
        if (deck.c[i] != i) {
            if (deck.c[i] != -1 || (placed & (1u << i)))
                continue;
        }

        // Apply the backward reversal: reverse positions 0..i
        Deck next;
        memcpy(&next, &deck, sizeof(Deck));

        next.c[0] = i;  // card i moves to top
        for (int j = 1; j <= i; j++)
            next.c[j] = deck.c[i - j];

        tryswaps(next, placed | (1u << i), s);
    }

    depth--;
}

int main(int argc, char* argv[]) {
    int max_n = 14;
    if (argc > 1) max_n = atoi(argv[1]);
    if (max_n > MAX_N) max_n = MAX_N;

    setbuf(stdout, nullptr);

    auto t0 = std::chrono::steady_clock::now();

    printf("Topswops: backward search (Knuth's algorithm)\n");
    printf("%3s  %6s  %10s\n", "n", "f(n)", "time(s)");
    printf("---  ------  ----------\n");

    best[0] = 0;
    best[1] = 0;

    for (int n = 1; n <= max_n; n++) {
        current_n = n;
        best[n] = 0;
        depth = 0;

        // Terminal state: card 0 (i.e., card "1" in 1-indexed) at position 0
        Deck start;
        memset(&start, -1, sizeof(Deck));
        start.c[0] = 0;

        // placed=1 means card 0 is placed
        tryswaps(start, 1u, n - 1);

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        printf("%3d  %6d  %10.3f\n", n, best[n], elapsed);
    }

    return 0;
}
