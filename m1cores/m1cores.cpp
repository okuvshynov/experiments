#include <b63/b63.h>
#include <b63/counters/osx_kperf.h>

#include <algorithm>
#include <cstdint>
#include <pthread.h>
#include <random>
#include <vector>

constexpr bool is_pow2(int a) {
  return a && ((a & (a - 1)) == 0);
}

const size_t kSize = (1 << 10);

struct PointerChasingTest {
  struct A {
    A* next;
    int64_t payload;
  };

  PointerChasingTest(int64_t rep) : rep(rep * kSize) {
    l.resize(kSize);
    for (int64_t i = 0; i < kSize; i++) {
      l[i].payload = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(l.begin(), l.end(), g);

    for (int64_t i = 0; i + 1 < kSize; i++) {
      l[i].next = &l[i+1];
    }
    l[kSize - 1].next = &l[0];

    std::sort(l.begin(), l.end(), [](const A& a, const A& b) {
      return a.payload < b.payload;
    });
  }

  template<int Unroll=2>
  int64_t run() {
    static_assert(is_pow2(Unroll), "unroll factor must be power of 2");
    A* curr = &l[0];
    int64_t res = 0;
    #pragma clang loop unroll_count(Unroll)
    for (int64_t r = 0; r < rep; r++) {
      curr = curr->next;
      if (curr->payload % 3 == 0) {
        res += curr->payload;
      }
      if (curr->payload % 5 == 0) {
        res += curr->payload;
      }
      if (curr->payload % 7 == 0) {
        res += curr->payload;
      }
      if (curr->payload % 11 == 0) {
        res += curr->payload;
      }
    }
    return res;
  }
 private:
  std::vector<A> l;
  int64_t rep;
};

#define BM_UNROLLED(name, qos, unroll)     \
  B63_BENCHMARK(name##_##unroll, n) {      \
    pthread_set_qos_class_self_np(qos, 0); \
    PointerChasingTest* t;                 \
    B63_SUSPEND {                          \
      t = new PointerChasingTest(n);       \
    }                                      \
    int64_t res = t->run<unroll>();        \
    B63_KEEP(res);                         \
    B63_SUSPEND {                          \
      delete t;                            \
    }                                      \
  }

#define FIRESTORM_UNROLLED(unroll)         \
  BM_UNROLLED(firestorm, QOS_CLASS_USER_INTERACTIVE, unroll)
#define ICESTORM_UNROLLED(unroll)          \
  BM_UNROLLED(icestorm, QOS_CLASS_BACKGROUND, unroll)
#define BOTH_UNROLLED(unroll)              \
  FIRESTORM_UNROLLED(unroll)               \
  ICESTORM_UNROLLED(unroll)

BOTH_UNROLLED(1)
BOTH_UNROLLED(2)
BOTH_UNROLLED(4)
BOTH_UNROLLED(8)
BOTH_UNROLLED(16)
BOTH_UNROLLED(32)
BOTH_UNROLLED(64)
BOTH_UNROLLED(128)

int main(int argc, char **argv) {
  B63_RUN_WITH("time,kperf:cycles,kperf:instructions", argc, argv);
  return 0;
}
