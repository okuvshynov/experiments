#include "include/rtdigest.h"

#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

// this test is a quick way to measure precision - how tight the
// interval is. It does not check for the 'correct' answer.
template <typename DigestT>
std::pair<double, double> run_silent(const std::string &dist_name, size_t digest_size,
                const std::vector<double> &values, uint32_t quantile) {

  DigestT d(digest_size, 100);

  for (double v : values) {
    d.add(v);
  }
  return d.range(quantile / 1000000.0).value();
}

template <typename DigestT>
void run_digest(const std::string &dist_name, size_t digest_size,
                const std::vector<double> &values, uint32_t quantile) {

  DigestT d(digest_size, 100);

  if (values.size() == 0) {
    return;
  }
  double mn = values[0], mx = values[0];
  for (double v : values) {
    d.add(v);
    mn = std::min(mn, v);
    mx = std::max(mx, v);
  }
  auto p = d.range(quantile / 1000000.0).value();
  double interval_width = (p.second - p.first) / (mx - mn);
  // print out the log we use for plotting the accuracy report
  printf("%s %lf %zu %zu %lf %lf %lf %lf %u\n",
         // distribution used; comes from the input
         dist_name.c_str(),
         // how narrow was the returned interval compared to entire data range.
         interval_width,
         // how many clusters
         d.max_merged_clusters(),
         // input size
         values.size(),
         // returned interval
         p.first, p.second,
         // scale function used
         DigestT::ScaleFunction::Pow, DigestT::ScaleFunction::TargetQuantile,
         // quantile queried
         quantile);
}

template <uint32_t TargetQuantile1M>
using DigestP2 =
    rtdigest::Digest<rtdigest::ScalePowCustom<TargetQuantile1M, 2000>>;

template <uint32_t TargetQuantile1M>
using DigestP3 =
    rtdigest::Digest<rtdigest::ScalePowCustom<TargetQuantile1M, 3000>>;

template <uint32_t TargetQuantile1M>
using DigestP4 =
    rtdigest::Digest<rtdigest::ScalePowCustom<TargetQuantile1M, 4000>>;

template <uint32_t TargetQuantile1M>
using DigestP5 =
    rtdigest::Digest<rtdigest::ScalePowCustom<TargetQuantile1M, 5000>>;

template <uint32_t TargetQuantile1M>
using DigestP6 =
    rtdigest::Digest<rtdigest::ScalePowCustom<TargetQuantile1M, 6000>>;

template <uint32_t TargetQuantile1M>
using DigestP7 =
    rtdigest::Digest<rtdigest::ScalePowCustom<TargetQuantile1M, 7000>>;

template <uint32_t TargetQuantile1M>
using DigestP8 =
    rtdigest::Digest<rtdigest::ScalePowCustom<TargetQuantile1M, 8000>>;

template <uint32_t TargetQuantile1M>
void run_ensemble(const std::string &dist_name,
                      const std::vector<double> &values, size_t digest_size,
                      uint32_t quantile) {
  std::vector<double> a, b;
  auto p = run_silent<DigestP2<TargetQuantile1M>>(dist_name, digest_size, values,
                                         quantile);
  a.push_back(p.first);
  b.push_back(p.second);
  p = run_silent<DigestP3<TargetQuantile1M>>(dist_name, digest_size, values,
                                         quantile);
  a.push_back(p.first);
  b.push_back(p.second);
  p = run_silent<DigestP4<TargetQuantile1M>>(dist_name, digest_size, values,
                                         quantile);
  a.push_back(p.first);
  b.push_back(p.second);
  p = run_silent<DigestP5<TargetQuantile1M>>(dist_name, digest_size, values,
                                         quantile);
  a.push_back(p.first);
  b.push_back(p.second);
  p = run_silent<DigestP6<TargetQuantile1M>>(dist_name, digest_size, values,
                                         quantile);
  a.push_back(p.first);
  b.push_back(p.second);
  p = run_silent<DigestP7<TargetQuantile1M>>(dist_name, digest_size, values,
                                         quantile);
  a.push_back(p.first);
  b.push_back(p.second);
  p = run_silent<DigestP8<TargetQuantile1M>>(dist_name, digest_size, values,
                                         quantile);
  a.push_back(p.first);
  b.push_back(p.second);
  double a0 = *std::max_element(a.begin(), a.end());
  double b0 = *std::min_element(b.begin(), b.end());
  if (values.size() == 0) {
    return;
  }
  double mn = values[0], mx = values[0];
  for (double v : values) {
    mn = std::min(mn, v);
    mx = std::max(mx, v);
  }
  double interval_width = (b0 - a0) / (mx - mn);
  // print out the log we use for plotting the accuracy report
  printf("%s %lf %zu %zu %lf %lf %lf %lf %u\n",
         // distribution used; comes from the input
         dist_name.c_str(),
         // how narrow was the returned interval compared to entire data range.
         interval_width,
         // how many clusters
         digest_size,
         // input size
         values.size(),
         // returned interval
         p.first, p.second,
         // scale function used
         3.14, TargetQuantile1M / 1000000.0,
         // quantile queried
         quantile);

}


template <uint32_t TargetQuantile1M>
void run_for_quantile(const std::string &dist_name,
                      const std::vector<double> &values, size_t digest_size,
                      uint32_t quantile) {
  run_digest<DigestP2<TargetQuantile1M>>(dist_name, digest_size, values,
                                         quantile);
  run_digest<DigestP3<TargetQuantile1M>>(dist_name, digest_size, values,
                                         quantile);
  run_digest<DigestP4<TargetQuantile1M>>(dist_name, digest_size, values,
                                         quantile);
  run_digest<DigestP5<TargetQuantile1M>>(dist_name, digest_size, values,
                                         quantile);
  run_digest<DigestP6<TargetQuantile1M>>(dist_name, digest_size, values,
                                         quantile);
  run_digest<DigestP7<TargetQuantile1M>>(dist_name, digest_size, values,
                                         quantile);
  run_digest<DigestP8<TargetQuantile1M>>(dist_name, digest_size, values,
                                         quantile);
  run_ensemble<TargetQuantile1M>(dist_name, values, digest_size, quantile);
}

void run(const std::string &dist_name, const std::vector<double> &values,
         size_t digest_size, uint32_t quantile) {
  run_for_quantile<500000>(dist_name, values, digest_size, quantile);
  run_for_quantile<950000>(dist_name, values, digest_size, quantile);
  run_for_quantile<990000>(dist_name, values, digest_size, quantile);
  run_for_quantile<999000>(dist_name, values, digest_size, quantile);
  run_for_quantile<999900>(dist_name, values, digest_size, quantile);
  run_for_quantile<1000000>(dist_name, values, digest_size, quantile);
  run_digest<rtdigest::Digest<rtdigest::ScalePowTwoSide<2000>>>(
      dist_name, digest_size, values, quantile);
  run_digest<rtdigest::Digest<rtdigest::ScalePowTwoSide<3000>>>(
      dist_name, digest_size, values, quantile);
  run_digest<rtdigest::Digest<rtdigest::ScalePowTwoSide<4000>>>(
      dist_name, digest_size, values, quantile);
  run_digest<rtdigest::Digest<rtdigest::ScalePowTwoSide<5000>>>(
      dist_name, digest_size, values, quantile);
  run_digest<rtdigest::Digest<rtdigest::ScalePowTwoSide<6000>>>(
      dist_name, digest_size, values, quantile);
  run_digest<rtdigest::Digest<rtdigest::ScalePowTwoSide<7000>>>(
      dist_name, digest_size, values, quantile);
  run_digest<rtdigest::Digest<rtdigest::ScalePowTwoSide<8000>>>(
      dist_name, digest_size, values, quantile);
}

int main(int argc, char **argv) {
  const auto digest_sizes = {50, 75, 100};
  const auto quantiles = {500000, 950000, 990000, 999000, 999900};

  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> normal{0.0, 1.0};
  std::uniform_real_distribution<> uniform{0.0, 1.0};
  // random size
  std::uniform_int_distribution<size_t> ll(100, 500000);

  std::vector<double> normal_values, uniform_values, asc, desc;

  // tests cases
  int32_t n = 20;

  printf("dist width clusters input_size a b scale_power scale_quantile "
         "quantile\n");
  for (int32_t ni = 0; ni < n; ni++) {
    fprintf(stderr, "Running iteration %d\n", ni);
    // running test ni
    auto s = ll(gen);
    normal_values.resize(s);
    uniform_values.resize(s);
    asc.resize(s);
    desc.resize(s);
    std::iota(asc.begin(), asc.end(), 1.0);
    std::iota(desc.begin(), desc.end(), 1.0);
    std::reverse(desc.begin(), desc.end());
    std::generate(normal_values.begin(), normal_values.end(),
                  [&]() { return normal(gen); });
    std::generate(uniform_values.begin(), uniform_values.end(),
                  [&]() { return uniform(gen); });
    for (auto digest_size : digest_sizes) {
      for (auto quantile : quantiles) {
        run("normal", normal_values, digest_size, quantile);
        run("uniform", uniform_values, digest_size, quantile);
        run("asc", asc, digest_size, quantile);
        run("desc", desc, digest_size, quantile);
      }
    }
  }
}
