#include "include/rtdigest.h"

#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

// what do we compare here? 
// input dataset has numbers and target quantile.
// we need to produce 2 numbers (if available):
//  - tdigest_custom (for that quantile)
//  - tdigest_generic (for that quantile)
//  the output:
//  - implementation_name: rtdigest, rtdigest_custom
//  - size (how many clusters)
//  - target quantile
//  - interval width

template <typename DigestT>
void run_digest(const std::string& prefix, const std::vector<double> &values, size_t size, uint32_t quantile) {

  DigestT d(size, 100);

  if (values.size() == 0) {
    return;
  }
  for (double v : values) {
    d.add(v);
  }
  auto p = d.range(quantile / 1000000.0).value();
  // print out the log we use for plotting the accuracy report
  printf("%s_rtdigest,%zu,%lf,%lf\n",
        prefix.c_str(),
         // how many clusters
         d.max_merged_clusters(),
         // target quantile
         quantile / 1000000.0,
         // how narrow was the returned interval
         p.second - p.first);
}

void run(const std::vector<double> &values, size_t size, uint32_t quantile) {
  if (quantile == 500000) {
    run_digest<rtdigest::Digest<rtdigest::ScalePowCustom<500000, 4000>>>(
        "custom", values, size, quantile);
  }
  if (quantile == 950000) {
    run_digest<rtdigest::Digest<rtdigest::ScalePowCustom<950000, 4000>>>(
        "custom", values, size, quantile);
  }
  if (quantile == 990000) {
    run_digest<rtdigest::Digest<rtdigest::ScalePowCustom<990000, 4000>>>(
        "custom", values, size, quantile);
  }
  if (quantile == 999000) {
    run_digest<rtdigest::Digest<rtdigest::ScalePowCustom<999000, 4000>>>(
        "custom", values, size, quantile);
  }
  if (quantile == 999900) {
    run_digest<rtdigest::Digest<rtdigest::ScalePowCustom<999900, 4000>>>(
        "custom", values, size, quantile);
  }
  run_digest<rtdigest::Digest<rtdigest::ScalePowTwoSide<1000>>>(
      "linear", values, size, quantile);
  run_digest<rtdigest::Digest<rtdigest::ScalePowTwoSide<2000>>>(
      "2nd_degree", values, size, quantile);
  run_digest<rtdigest::Digest<rtdigest::ScalePowTwoSide<4000>>>(
      "4th_degree", values, size, quantile);
}

int main(int argc, char **argv) {
  const auto digest_sizes = {50, 100, 200};
  size_t n;
  double quantile;
  double expected;
  std::vector<double> values;

  // this file is generated with
  // Rscript tests/reference_impl.r 20000 2 > tests/data/medium.in
  FILE *f = std::fopen("tests/data/medium2.in", "r");
  if (f == NULL) {
    return 0;
  }

  while (fscanf(f, "%zu", &n) == 1) {
    values.resize(n);
    fscanf(f, "%lf", &quantile);
    fscanf(f, "%lf", &expected);
    for (size_t i = 0; i < n; i++) {
      fscanf(f, "%lf", &values[i]);
    }
    for (size_t digest_size = 25; digest_size <= 100; digest_size++) {
      run(values, digest_size, static_cast<uint32_t>(quantile));
    }
  }
  fclose(f);
  return 0;
}
