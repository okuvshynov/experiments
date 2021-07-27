#include "include/rtdigest.h"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

int main(int argc, char **argv) {
  const auto digest_sizes = {50, 100, 200, 400, 800};
  const size_t n = 10000000;

  std::random_device rd;
  std::mt19937 gen(rd());

  std::normal_distribution<> nd{0, 10};

  std::vector<double> v;
  v.resize(n);
  std::generate(v.begin(), v.end(), [&]() { return nd(gen); });

  // so it won't get optimized away
  double total = 0.0;

  printf("digest_size operation time_ns_per_op\n");
  for (auto digest_size : digest_sizes) {
    auto d = rtdigest::DigestUpper(digest_size, 100);
    {
      auto start = std::chrono::steady_clock::now();

      for (auto vv : v) {
        d.add(vv);
      }
      auto end = std::chrono::steady_clock::now();
      std::chrono::duration<double> diff = end - start;
      printf("%d add %lf\n", digest_size, 1.0e9 * diff.count() / n);
    }
    {
      auto start = std::chrono::steady_clock::now();

      double step = 0.1 / n;
      for (double q = 0.9; q < 1.0; q += step) {
        auto p = d.range(q).value();
        total += p.first + p.second;
      }
      auto end = std::chrono::steady_clock::now();
      std::chrono::duration<double> diff = end - start;
      printf("%d range %lf\n", digest_size, 1.0e9 * diff.count() / n);
    }
  }
  printf("%lf\n", total);
  return 0;
}
