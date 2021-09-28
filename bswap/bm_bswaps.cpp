#include "bswaps.h"

#include <b63/b63.h>
#include <b63/counters/osx_kperf.h>

#include <cstdint>
#include <vector>

B63_BENCHMARK(bswaps_shifts, n) {
  std::vector<uint64_t> v;
  uint64_t res = 0;
  B63_SUSPEND {
    for (int i = 0; i < 1001002; i++) {
      v.push_back(i);
    }
  }
  for (int i = 0; i < n; i++) {
    for (auto vv: v) {
      bswap8b(reinterpret_cast<uint8_t*>(&vv), 1);
      res += vv;
      bswap8b(reinterpret_cast<uint8_t*>(&vv), 2);
      res += vv;
      bswap8b(reinterpret_cast<uint8_t*>(&vv), 3);
      res += vv;
      bswap8b(reinterpret_cast<uint8_t*>(&vv), 4);
      res += vv;
      bswap8b(reinterpret_cast<uint8_t*>(&vv), 6);
      res += vv;
      bswap8b(reinterpret_cast<uint8_t*>(&vv), 7);
      res += vv;
      bswap8b(reinterpret_cast<uint8_t*>(&vv), 8);
      res += vv;
    }
  }
  B63_KEEP(res);

}


B63_BENCHMARK(bswaps_naive, n) {
  std::vector<uint64_t> v;
  uint64_t res = 0;
  B63_SUSPEND {
    for (int i = 0; i < 1001002; i++) {
      v.push_back(i);
    }
  }
  for (int i = 0; i < n; i++) {
    for (auto vv: v) {
      bswap_naive(reinterpret_cast<uint8_t*>(&vv), 1);
      res += vv;
      bswap_naive(reinterpret_cast<uint8_t*>(&vv), 2);
      res += vv;
      bswap_naive(reinterpret_cast<uint8_t*>(&vv), 3);
      res += vv;
      bswap_naive(reinterpret_cast<uint8_t*>(&vv), 4);
      res += vv;
      bswap_naive(reinterpret_cast<uint8_t*>(&vv), 6);
      res += vv;
      bswap_naive(reinterpret_cast<uint8_t*>(&vv), 7);
      res += vv;
      bswap_naive(reinterpret_cast<uint8_t*>(&vv), 8);
      res += vv;
    }
  }
  B63_KEEP(res);

}

int main(int argc, char **argv) {
  B63_RUN_WITH("time", argc, argv);
  return 0;
}
