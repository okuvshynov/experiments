#include <random>
#include <cstdlib>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <string_view>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <iterations>\n", argv[0]);
        return 1;
    }

    std::string_view arg{argv[1]};
    uint64_t iterations;

    auto [p, e] = std::from_chars(arg.data(), arg.data() + arg.size(), iterations);

    if (e != std::errc{} || p != arg.data() + arg.size()) {
        std::fprintf(stderr, "%s is not a positive integer\n", argv[1]);
    }
    
    const uint64_t kSeed = 1231;

    std::mt19937_64 rng{kSeed};
    std::uniform_int_distribution<int> die{1, 6};

    auto throw_die = [&]() {
        return die(rng);
    };

    double n = 0.0;
    for (uint64_t i = 0; i < iterations; i++) {
        while (true) {
            n += 1.0;
            int a = throw_die(), b = throw_die(), c = throw_die();
            if (a == b && b == c) {
                break;
            }
            if (a == b || b == c || a == c) {
                n += 6.0;
                break;
            }
        }
        uint64_t j = i + 1;
        if (j % 1000000 == 0 || j == iterations) {
            std::printf("%lld: %lf\n", j, n / j);
        }
    }

    return 0;
}