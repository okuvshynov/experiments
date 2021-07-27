#ifndef _R_T_DIGEST_H_
#define _R_T_DIGEST_H_

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace rtdigest {
// Scale functions
// This is for specific percentile/one side
// https://www.desmos.com/calculator/graaiz1mpz
template <uint32_t TargetQuantile1M, uint32_t Power1K> struct ScalePowCustom {
  static constexpr double Pow = Power1K / 1000.0;
  static constexpr double TargetQuantile = TargetQuantile1M / 1000000.0;
  static_assert(TargetQuantile1M >= 0 && TargetQuantile1M <= 1000000,
                "quantile must be between 0 and 1");
  static_assert(Power1K >= 1 && Power1K <= 15000,
                "higher power results in floating point accuracy problems");
  double operator()(double kd, double d) {
    const double k = kd / d;
    static constexpr double p = TargetQuantile1M / 1000000.0;
    static constexpr double w = Power1K / 1000.0;
    static const double a = std::pow(p, 1.0 / w);
    static const double b = std::pow(1.0 - p, 1.0 / w);

    if (k * (a + b) > a) {
      return p + pow(k * (a + b) - a, w);
    } else {
      return p - pow(a - k * (a + b), w);
    }
  }
};

// this is generic one, with focus on both ends
// Inverse power-based scale function. It is similar to the one used in
// https://github.com/facebook/folly/blob/master/folly/stats/TDigest.cpp
// but is more flexible and can use variable power instead of square.
// https://www.desmos.com/calculator/vccd3eeh0l
template <uint32_t Power1K> struct ScalePowTwoSide {
  static constexpr double Pow = Power1K / 1000.0;
  static constexpr double TargetQuantile = -1.0;
  static_assert(Power1K >= 1 && Power1K <= 15000,
                "higher power results in floating point accuracy problems");
  double operator()(double kd, double d) {
    static constexpr double w = Power1K / 1000.0;
    double k = kd / d;
    if (k >= 0.5) {
      k = 1.0 - k;
      return 1.0 - 0.5 * std::pow(2.0 * k, w);
    } else {
      return 0.5 * std::pow(2.0 * k, w);
    }
  }
};

template <typename ScaleF = ScalePowTwoSide<4000>> class Digest {
public:
  using SelfT = Digest<ScaleF>;
  using ScaleFunction = ScaleF;
  // clusters_size -- how many clusters to store
  // buffer_size   -- how many elements to buffer before merge
  Digest(size_t clusters_size, size_t buffer_size)
      : max_merged(clusters_size), size(clusters_size + buffer_size) {
    clusters.resize(size);
    clusters.shrink_to_fit();
  }

  // adds a new data point; might trigger merge, if buffer is full
  void add(double v) {
    clusters[unmerged_index++] = Cluster(v);
    if (unmerged_index >= size) {
      merge();
    }
  }

  size_t total_buffer_size() const { return size; }

  size_t max_merged_clusters() const { return max_merged; }

  // adds everything from another digest; it's a copy, not move operation
  void add(const SelfT &src) {
    // copy both merged and unmerged from other digest to
    // local buffer in batches, and then call merge until finished.
    auto src_begin = src.clusters.begin();
    auto src_end = src.clusters.begin() + src.unmerged_index;
    while (src_begin != src_end) {
      // how much space do we have to copy elements
      // TODO: clean this up
      const ssize_t capacity = size - unmerged_index;
      // min of 'how many elements we still have to process'
      // and 'how much space do we have'
      const size_t to_copy =
          std::min(capacity, std::distance(src_begin, src_end));
      std::copy(src_begin, src_begin + to_copy,
                clusters.begin() + unmerged_index);

      // advance pointers
      unmerged_index += to_copy;
      src_begin += to_copy;
      merge();
    }
  }

  // compute lower and upper bounds of the quantile
  std::optional<std::pair<double, double>> range(double quantile) {
    // merge clusters if needed
    merge();

    // digest is empty
    if (unmerged_start == 0) {
      return std::nullopt;
    }

    if (quantile >= 1.0) {
      return std::make_pair(this->max, this->max);
    }
    if (quantile <= 0.0) {
      return std::make_pair(this->min, this->min);
    }

    // how many points are in the quantile?
    int64_t target_count = ceil(quantile * count);
    // how many points have we encountered so far?
    int64_t current_count = 0LL;
    // max right cluster boundary observed so far
    double maxb = this->min;

    // iterate over all 'merged' clusters
    for (size_t i = 0; i < unmerged_start; i++) {
      maxb = std::max(maxb, clusters[i].max);
      current_count += clusters[i].count;
      if (current_count >= target_count) {
        // clusters can overlap, thus we have already added elements up to 
        // maxb to the current_count. Because of that, the interval right
        // boundary is maxb rather than clusters[i].max
        return std::make_pair(clusters[i].min, maxb);
      }
    }
    return std::make_pair(this->max, this->max);
  }

private:
  struct Cluster {
    double min, max;
    int64_t count;
    explicit Cluster(double v) : min(v), max(v), count(1LL) {}
    Cluster() : Cluster(0.0) {}

    void add(const Cluster &from) {
      min = std::min(min, from.min);
      max = std::max(max, from.max);
      count += from.count;
    }

    friend bool operator<(const Cluster &a, const Cluster &b) {
      return a.min < b.min;
    }
  };

  // merges 'unmerged' section of the buffer to the cluster list
  void merge() {
    const size_t to_merge = unmerged_index - unmerged_start;
    if (to_merge == 0) {
      return;
    }

    double new_count = count;

    // TODO: can instead maintain 'unmerged count'?
    for (size_t i = unmerged_start; i < unmerged_index; i++) {
      new_count += clusters[i].count;
    }

    // After the sort, we have an ordered list of clusters to merge.
    // There are ways to speed up this sort:
    //  - use radix sort
    //  - utilize the fact that merged clusters are already sorted
    // Note: Theoretically this might allocate
    std::sort(clusters.begin(), clusters.begin() + unmerged_index);

    double new_min = clusters[0].min;
    double new_max = clusters[0].max;

    double k = 1.0;
    // at most q points can go to next cluster
    double q = ScaleF()(k++, max_merged) * new_count;
    double current = clusters[0].count;

    // i - index where we write to
    size_t i = 0;

    // j - index where we read from
    for (size_t j = 1; j < unmerged_index; j++) {
      current += clusters[j].count;
      new_min = std::min(new_min, clusters[j].min);
      new_max = std::max(new_max, clusters[j].max);
      if (current > q) { // not merging, size would violate the constraint
        // find new boundary
        q = ScaleF()(k++, max_merged) * new_count;
        // advance the write pointer and copy the Cj cluster
        clusters[++i] = clusters[j];
      } else {
        clusters[i].add(clusters[j]);
      }
    }
    unmerged_index = unmerged_start = i + 1;
    count = new_count;
    min = new_min;
    max = new_max;
  }

  size_t max_merged, size, unmerged_start = 0, unmerged_index = 0;
  double min, max = 0.0;
  int64_t count = 0LL;
  std::vector<Cluster> clusters;
};

// Shortcuts for common quantiles
using DigestP95 = Digest<ScalePowCustom<950000, 4000>>;
using DigestP99 = Digest<ScalePowCustom<990000, 4000>>;
using DigestP99_9 = Digest<ScalePowCustom<999000, 4000>>;
using DigestP99_99 = Digest<ScalePowCustom<999900, 4000>>;
using DigestUpper = Digest<ScalePowCustom<1000000, 4000>>;
} // namespace rtdigest

#endif
