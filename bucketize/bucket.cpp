#include <cstdint>
#include <cmath>
#include <limits>
#include <iostream>
#include <type_traits>

// c++ -std=c++2a ./bucketize/bucket.cpp -o /tmp/bar_demo && /tmp/bar_demo

template<typename num_t>
size_t bin_index_float(num_t mn, num_t mx, size_t bins, num_t v) {
    static_assert(std::is_floating_point_v<num_t> == true);

    // sanity checks:
    if (bins == 0 || mn > mx || std::isnan(v)) {
        // do something, return std::optional, etc.
        return 0;
    }

    // TODO: how to handle mn or mx being inf?

    if (std::isinf(v)) {
        return std::signbit(v) ? 0 : bins - 1;
    }

    if (v < mn) {
        return 0;
    }

    if (v > mx) {
        return bins - 1;
    }

    if (mn == mx) {
        return bins / 2;
    }

    if (bins == 1) {
        return 0;
    }

    auto bin_size = (mx / bins) - (mn / bins);

    std::cout << "bin_size = " << bin_size << std::endl;

    // just compute
    auto bin_index = static_cast<size_t>(std::max(0.0, (v / bin_size) - (mn / bin_size)));

    // another way:
    size_t index = 0;
    auto curr = mn + bin_size;
    while (true) {
        if (v < curr) {
            break;
        }
        index++;
        // what would be the error at the last step?
        curr += bin_size;
    }

    std::cout << "bin index: compute = " << bin_index << " | O(N) iterate = " << index << std::endl;


    return std::min(bins - 1, bin_index);
}

int main() {
    std::cout << bin_index_float(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max(), 4, -100000.0) << std::endl;
    return 0;
}