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

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
