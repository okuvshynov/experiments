#include "bswaps.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

TEST(naive, vs_hardcoded) {
  uint64_t v = 0x12345678;
  bswap_naive(reinterpret_cast<uint8_t*>(&v), 4);
  EXPECT_EQ(v, 0x78563412);
  bswap_naive(reinterpret_cast<uint8_t*>(&v), 3);
  EXPECT_EQ(v, 0x78123456);
}

TEST(bswap8b_masks, vs_hardcoded) {
  uint64_t v = 0x12345678;
  bswap8b(reinterpret_cast<uint8_t*>(&v), 4);
  EXPECT_EQ(v, 0x78563412);
  bswap8b(reinterpret_cast<uint8_t*>(&v), 3);
  EXPECT_EQ(v, 0x78123456);
}
TEST(topswops, neon2) {
  load_neon_table();
  uint8_t y[16] = {2, 9, 4, 5, 11, 12, 10, 1, 8, 13, 3, 6, 7, 0, 0, 0};
  EXPECT_EQ(topswops_neon2(y), 80);
}

TEST(topswops, neon) {
  load_neon_table();
  uint8_t y[16] = {2, 9, 4, 5, 11, 12, 10, 1, 8, 13, 3, 6, 7, 0, 0, 0};
  EXPECT_EQ(topswops_neon(y), 80);
}

TEST(topswops, unrolled) {
  uint8_t x[] = {3, 2, 4, 5, 1};
  EXPECT_EQ(topswops_unrolled(x), 3);
  uint8_t y[] = {2, 9, 4, 5, 11, 12, 10, 1, 8, 13, 3, 6, 7};
  EXPECT_EQ(topswops_unrolled(y), 80);
}


TEST(topswops, naive) {
  uint8_t x[] = {3, 2, 4, 5, 1};
  EXPECT_EQ(topswops_naive(x), 3);
  uint8_t y[] = {2, 9, 4, 5, 11, 12, 10, 1, 8, 13, 3, 6, 7};
  EXPECT_EQ(topswops_naive(y), 80);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
