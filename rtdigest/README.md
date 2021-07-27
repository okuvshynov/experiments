# rtdigest - Range t-digest

Work-in-progress

## Overview

[t-digest](https://github.com/tdunning/t-digest) is a compact data structure used for quantile estimation. While it provides a great tradeoff between accuracy and resource usage, it doesn't have strict accuracy guarantees.

Ranged t-digest attempts to work around this issue. There are two notable changes:
* rather than providing a scalar estimate of a quantile, a range [a; b] is returned. Quantile value is guaranteed to be within that range, but there's still no guarantee on the size of the interval. This comes at a cost of storing min/max values per cluster instead of mean.
* introduction of custom scaling functions, which allocate more space to a specific quantile neighborhood. In many use-cases, only one specific quantile needs to be estimated with a query like ```SELECT service, APPROX_PERCENTILE(latency, 0.99) FROM service_health GROUP BY service```. Custom scaling function dramatically improves accuracy in these cases at the expense of accuracy loss at other quantiles. One special case is the one-side function focusing on one of the tails.

Other properties stay roughly the same: 
* small size
* no dynamic memory allocation after initialization;
* simple implementation (single header file with < 250 lines of code);
* easy to merge digests and thus easy to distribute the computation across nodes/services/threads;
* good add/estimate performance.

## Usage

Entire library is a [single header file](/rtdigest/include/rtdigest.h).

API is very simple:
* constructor: takes max clusters count and extra buffer size as parameters. These parameters allow making CPU/memory/accuracy tradeoffs;
* void add(double) - add a point to the digest;
* pair<double, double> range(double) - get an interval for the quantile;
* void add(Digest) - merge two digests.

API for getting the 'best estimate' of a quantile as a single scalar value is not provided. 

## Implementation
### What do we mean by quantile?
There are multiple ways to estimate distribution quantiles from the data sample. A summary of these methods can be found at [Sample Quantiles in Statistical Packages](https://www.researchgate.net/publication/222105754_Sample_Quantiles_in_Statistical_Packages). We use definition equivalent to Type-1 from [R's quantile](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/quantile) in our implementation and make [reference implementation](/tests/reference_impl.r) use the same definition.

### Scaling functions use
#### Custom power functions
Facebook's [folly implementation](https://github.com/facebook/folly/blob/master/folly/stats/TDigest.h) is using sqrt-based scaling function. Here we generalize this approach to a more flexible function which can use arbitrary power to define slope.

This chart shows 4th degree scaling two-tail function:

![two-tail 4th degree function](/rtdigest/tests/data/pow4_two_side.png)

[interactive chart](https://www.desmos.com/calculator/oivhqztran)

Additionally, we can shift the plateu to improve accuracy at that quantile (0.99 in this case): 

![p99-focused 4th degree function](/rtdigest/tests/data/pow4_q99.png)

[interactive chart](https://www.desmos.com/calculator/ubccsunfpd).


### Interval estimation algorithm
t-digest does not provide guarantees on cluster overlap, thus, estimating interval is not as trivial as 'find the cluster and use its boundaries'. 
We iterate over clusters and keep track of the total number of samples encountered. Because clusters can overlap, even if the current cluster Ci is 'enough' to get to the needed count, cluster Ck with k < i might have a right boundary higher than that of Ci, and we have already added all samples from Ck. In this case, the correct range would be [Ci.a, Ck.b].

### Merge algorithm
The biggest difference here is the sort order. We do not store the mean value per cluster and sort by min instead. It works empirically but likely this can be improved.

## Tests/Evaluation
 
### Unit tests
```make test```

Requires [googletest](https://github.com/google/googletest). Part of the test suite is using a reference dataset generated with R.

### Performance test
```make perf```

Perf test averages the cost across many iterations. In the case of 'add' testing, we average out very fast regular adds and merges.

```
% make perf
c++ -Wall tests/performance/rtdigest_performance.cpp -O2 -o _build/performance -I . -std=c++17
_build/performance
digest_size operation time_ns_per_op
50 add 41.096812
50 range 31.848150
100 add 53.618292
100 range 66.232362
200 add 72.382475
200 range 157.189425
400 add 101.386517
400 range 346.221967
800 add 145.603904
800 range 722.094362
```

There's no detailed study done, just the basic check that performance is in the right ballpark.


### Precision test

```make precision```

Precision test measures the relative width of the interval. The test consists of two parts:
* [c++ app](/rtdigest/tests/rtdigest_precision.cpp) which generates data and logs the outcome;
* [R plot](/rtdigest/tests/precision_plot.r) which plots the images below

![normal](/rtdigest/tests/data/precision_normal.png)

Some observations:
* Interval width shrinks as the number of clusters increase;
* the worst cases are when we use custom quantile scaling function but query for more extreme one;
* the best cases are when we use custom quantile and query for it;
* Interval shrinks as we increase the power for the scaling function and 4-6 power seems a good fit.

## Dependencies

C++17 is used.

* Library itself, performance test and precision test have no external dependencies; visualizing precision test results require R + ggplot2;
* Unit tests require googletest. Generating new data for the VsReference test case requires R.

## Potential next steps:
* Precision:
  * Experiment with sort order;
  * Consider reintroducing mean + implement interpolation;
  * study cluster overlap;
  * we can narrow down interval significantly in case of small clusters if we reintroduce mean;
  * Collect complicated samples, where the returned range will be large. Likely, insertion order matters.
* Performance:
  * do profiling;
  * allow using single-precision for min/max + int32 for count;
  * add()/merge(): improve sort performance; consider radix sort and take into account that merged part is already sorted;
  * range(): choose iteration direction based on queried percentile.

## References
* [t-digest source](https://github.com/tdunning/t-digest/)
* [t-digest paper](https://arxiv.org/abs/1902.04023)
* [Sample Quantiles in Statistical Packages](https://www.researchgate.net/publication/222105754_Sample_Quantiles_in_Statistical_Packages)
* [folly TDigest](https://github.com/facebook/folly/blob/master/folly/stats/TDigest.h)
