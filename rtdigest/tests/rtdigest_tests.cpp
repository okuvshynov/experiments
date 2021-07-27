#include "include/rtdigest.h"

#include <cstdio>
#include <gtest/gtest.h>

TEST(Scaling, TwoSide) {
  double q;
  q = rtdigest::ScalePowTwoSide<1000>()(0.0, 1.0);
  EXPECT_NEAR(q, 0.0, 1e-8);
  q = rtdigest::ScalePowTwoSide<2000>()(0.0, 1.0);
  EXPECT_NEAR(q, 0.0, 1e-8);
  q = rtdigest::ScalePowTwoSide<3000>()(0.0, 1.0);
  EXPECT_NEAR(q, 0.0, 1e-8);
  q = rtdigest::ScalePowTwoSide<4000>()(0.0, 1.0);
  EXPECT_NEAR(q, 0.0, 1e-8);

  q = rtdigest::ScalePowTwoSide<1000>()(0.5, 1.0);
  EXPECT_NEAR(q, 0.5, 1e-8);
  q = rtdigest::ScalePowTwoSide<2000>()(0.5, 1.0);
  EXPECT_NEAR(q, 0.5, 1e-8);
  q = rtdigest::ScalePowTwoSide<3000>()(0.5, 1.0);
  EXPECT_NEAR(q, 0.5, 1e-8);
  q = rtdigest::ScalePowTwoSide<4000>()(0.5, 1.0);
  EXPECT_NEAR(q, 0.5, 1e-8);

  q = rtdigest::ScalePowTwoSide<1000>()(1.0, 1.0);
  EXPECT_NEAR(q, 1.0, 1e-8);
  q = rtdigest::ScalePowTwoSide<2000>()(1.0, 1.0);
  EXPECT_NEAR(q, 1.0, 1e-8);
  q = rtdigest::ScalePowTwoSide<3000>()(1.0, 1.0);
  EXPECT_NEAR(q, 1.0, 1e-8);
  q = rtdigest::ScalePowTwoSide<4000>()(1.0, 1.0);
  EXPECT_NEAR(q, 1.0, 1e-8);
}

TEST(Scaling, Ones_Linear) {
  double q;
  q = rtdigest::ScalePowCustom<950000, 1000>()(1.0, 1.0);
  EXPECT_NEAR(q, 1.0, 1e-8);
  q = rtdigest::ScalePowCustom<990000, 1000>()(1.0, 1.0);
  EXPECT_NEAR(q, 1.0, 1e-8);
  q = rtdigest::ScalePowCustom<999000, 1000>()(1.0, 1.0);
  EXPECT_NEAR(q, 1.0, 1e-8);
  q = rtdigest::ScalePowCustom<999900, 1000>()(1.0, 1.0);
  EXPECT_NEAR(q, 1.0, 1e-8);
  q = rtdigest::ScalePowCustom<1000000, 1000>()(1.0, 1.0);
  EXPECT_NEAR(q, 1.0, 1e-8);
}

TEST(Scaling, Ones_Square) {
  double q;
  q = rtdigest::ScalePowCustom<950000, 2000>()(1.0, 1.0);
  EXPECT_NEAR(q, 1.0, 1e-8);
  q = rtdigest::ScalePowCustom<990000, 2000>()(1.0, 1.0);
  EXPECT_NEAR(q, 1.0, 1e-8);
  q = rtdigest::ScalePowCustom<999000, 2000>()(1.0, 1.0);
  EXPECT_NEAR(q, 1.0, 1e-8);
  q = rtdigest::ScalePowCustom<999900, 2000>()(1.0, 1.0);
  EXPECT_NEAR(q, 1.0, 1e-8);
  q = rtdigest::ScalePowCustom<1000000, 2000>()(1.0, 1.0);
  EXPECT_NEAR(q, 1.0, 1e-8);
}

TEST(RTDigestTest, Empty) {
  rtdigest::DigestP99 d(100, 100);
  auto p = d.range(0.99);
  EXPECT_FALSE(p.has_value());
}

TEST(RTDigestTest, One) {
  rtdigest::DigestP99 d(100, 100);
  d.add(1);
  auto p = d.range(0.99).value();
  EXPECT_NEAR(1.0, p.first, 1e-8);
  EXPECT_NEAR(1.0, p.second, 1e-8);
}

TEST(RTDigestTest, Bounds) {
  rtdigest::DigestP99 d(100, 100);
  d.add(0);
  d.add(1);
  {
    auto p = d.range(100.01).value();
    EXPECT_NEAR(1.0, p.first, 1e-8);
    EXPECT_NEAR(1.0, p.second, 1e-8);
  }
  {
    auto p = d.range(1.01).value();
    EXPECT_NEAR(1.0, p.first, 1e-8);
    EXPECT_NEAR(1.0, p.second, 1e-8);
  }
  {
    auto p = d.range(0.0).value();
    EXPECT_NEAR(0.0, p.first, 1e-8);
    EXPECT_NEAR(0.0, p.second, 1e-8);
  }
  {
    auto p = d.range(-1000.0).value();
    EXPECT_NEAR(0.0, p.first, 1e-8);
    EXPECT_NEAR(0.0, p.second, 1e-8);
  }
}

TEST(RTDigestTest, Merge) {
  rtdigest::DigestP99 d1(100, 100);
  rtdigest::DigestP99 d2(100, 100);
  d1.add(1);
  d2.add(2);
  d1.add(d2);
  auto p = d1.range(0.99).value();
  EXPECT_NEAR(2.0, p.first, 1e-8);
  EXPECT_NEAR(2.0, p.second, 1e-8);
}

TEST(RTDigestTest, Basic) {
  rtdigest::DigestP99 d(100, 100);
  for (int i = 1; i <= 10; i++) {
    d.add(i);
  }
  auto p = d.range(0.9).value();
  EXPECT_NEAR(9.0, p.first, 1e-8);
  EXPECT_NEAR(9.0, p.second, 1e-8);
}

TEST(RTDigestTest, SmallBuffer) {
  rtdigest::DigestP99 d(100, 1);
  for (int i = 1; i <= 10; i++) {
    d.add(i);
  }
  auto p = d.range(0.9).value();
  EXPECT_NEAR(9.0, p.first, 1e-8);
  EXPECT_NEAR(9.0, p.second, 1e-8);
}

TEST(RTDigestTest, SingleCluster) {
  rtdigest::DigestP99 d(1, 1);
  for (int i = 1; i <= 10; i++) {
    d.add(i);
  }
  auto p = d.range(0.9).value();
  EXPECT_NEAR(1.0, p.first, 1e-8);
  EXPECT_NEAR(10.0, p.second, 1e-8);
}

TEST(RTDigestTest, Equal) {
  rtdigest::Digest d(100, 100);
  for (int i = 1; i <= 1000000; i++) {
    d.add(1000.0);
  }
  {
    auto p = d.range(0.01).value();
    // with generic function is not too narrow
    EXPECT_DOUBLE_EQ(p.first, 1000.0);
    EXPECT_DOUBLE_EQ(p.second, 1000.0);
  }
  {
    auto p = d.range(0.99).value();
    // with generic function is not too narrow
    EXPECT_DOUBLE_EQ(p.first, 1000.0);
    EXPECT_DOUBLE_EQ(p.second, 1000.0);
  }
}

TEST(RTDigestTest, Monotonic) {
  rtdigest::Digest d(100, 100);
  rtdigest::DigestP99 d99(100, 100);
  for (int i = 1; i <= 1000000; i++) {
    d.add(i);
    d99.add(i);
  }
  {
    auto p = d.range(0.99).value();
    // with generic function is not too narrow
    EXPECT_NEAR(p.first, 990000, 1500);
    EXPECT_NEAR(p.second, 990000, 1500);
  }
  {
    auto p = d99.range(0.99).value();
    // with custom it is much better
    EXPECT_NEAR(p.first, 990000, 150);
    EXPECT_NEAR(p.second, 990000, 150);
  }
}

template <typename DigestT>
bool run(DigestT d, const std::vector<double> &values, uint32_t quantile,
         double expected) {

  for (double v : values) {
    d.add(v);
  }
  auto p = d.range(quantile / 1000000.0).value();
  return (p.first <= expected && p.second >= expected);
}

// loads data from a file
TEST(RTDigestTest, VsReference) {
  const auto digest_sizes = {50, 100, 150, 200, 250};
  size_t n;
  uint32_t quantile;
  double expected;
  std::vector<double> values;

  // this file is generated with
  // Rscript tests/reference_impl.r 20000 2 > tests/data/medium.in
  FILE *f = std::fopen("tests/data/medium.in", "r");
  if (f == NULL) {
    return;
  }

  while (fscanf(f, "%zu", &n) == 1) {
    values.resize(n);
    fscanf(f, "%u", &quantile);
    fscanf(f, "%lf", &expected);
    for (size_t i = 0; i < n; i++) {
      fscanf(f, "%lf", &values[i]);
    }
    for (auto digest_size : digest_sizes) {
      EXPECT_TRUE(
          run(rtdigest::Digest(digest_size, 100), values, quantile, expected));

      EXPECT_TRUE(run(rtdigest::DigestUpper(digest_size, 100), values, quantile,
                      expected));
      EXPECT_TRUE(run(rtdigest::DigestP95(digest_size, 100), values, quantile,
                      expected));
      EXPECT_TRUE(run(rtdigest::DigestP99(digest_size, 100), values, quantile,
                      expected));
      EXPECT_TRUE(run(rtdigest::DigestP99_9(digest_size, 100), values, quantile,
                      expected));
      EXPECT_TRUE(run(rtdigest::DigestP99_99(digest_size, 100), values,
                      quantile, expected));
    }
  }
  // TODO: fix this, move to setUp/tearDown
  fclose(f);
}

TEST(RTDigestTest, Custom4) {
  using ScaleF4 = rtdigest::ScalePowCustom<990000, 4000>;
  rtdigest::Digest<ScaleF4> d(100, 100);
  for (int i = 1; i <= 10; i++) {
    d.add(i);
  }
  auto p = d.range(0.9).value();
  EXPECT_NEAR(9.0, p.first, 1e-8);
  EXPECT_NEAR(9.0, p.second, 1e-8);
}

TEST(RTDigestTest, Custom3) {
  using ScaleF4 = rtdigest::ScalePowCustom<990000, 3000>;
  rtdigest::Digest<ScaleF4> d(100, 100);
  for (int i = 1; i <= 10; i++) {
    d.add(i);
  }
  auto p = d.range(0.9).value();
  EXPECT_NEAR(9.0, p.first, 1e-8);
  EXPECT_NEAR(9.0, p.second, 1e-8);
}

TEST(RTDigestTest, Custom2) {
  using ScaleF4 = rtdigest::ScalePowCustom<990000, 2000>;
  rtdigest::Digest<ScaleF4> d(100, 100);
  for (int i = 1; i <= 10; i++) {
    d.add(i);
  }
  auto p = d.range(0.9).value();
  EXPECT_NEAR(9.0, p.first, 1e-8);
  EXPECT_NEAR(9.0, p.second, 1e-8);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
