#include "bswaps.h"

#include <b63/b63.h>
#include <b63/counters/osx_kperf.h>
#include <b63/counters/cycles.h>

#include <cstdint>
#include <vector>

B63_BASELINE(naive, n) {
  int res = 0;
  for (int i = 0; i < n; i++) {
    uint8_t y[16] = {2, 9, 4, 5, 11, 12, 10, 1, 8, 13, 3, 6, 7, 0, 0, 0};
    res += topswops_naive(y);
  }
  B63_KEEP(res);
}

B63_BENCHMARK(unrolled, n) {
  int res = 0;
  for (int i = 0; i < n; i++) {
    uint8_t y[16] = {2, 9, 4, 5, 11, 12, 10, 1, 8, 13, 3, 6, 7, 0, 0, 0};
    res += topswops_unrolled(y);
  }
  B63_KEEP(res);
}

B63_BENCHMARK(unrolled_, n) {
  int res = 0;
  for (int i = 0; i < n; i++) {
    uint8_t y[16] = {2, 9, 4, 5, 11, 12, 10, 1, 8, 13, 3, 6, 7, 0, 0, 0};
    res += topswops_unrolled_(y);
  }
  B63_KEEP(res);
}

B63_BENCHMARK(neon2, n) {
  load_neon_table();
  int res = 0;
  for (int i = 0; i < n; i++) {
    uint8_t y[16] = {2, 9, 4, 5, 11, 12, 10, 1, 8, 13, 3, 6, 7, 0, 0, 0};
    res += topswops_neon2(y);
  }
  B63_KEEP(res);
}

B63_BENCHMARK(neon, n) {
  load_neon_table();
  int res = 0;
  for (int i = 0; i < n; i++) {
    uint8_t y[16] = {2, 9, 4, 5, 11, 12, 10, 1, 8, 13, 3, 6, 7, 0, 0, 0};
    res += topswops_neon(y);
  }
  B63_KEEP(res);
}

int main(int argc, char **argv) {
  B63_RUN_WITH("time", argc, argv);
  return 0;
}
